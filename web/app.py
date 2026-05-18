"""
ECABSD Web API — FastAPI application for serving predictions.

Endpoints:
    GET  /          → Serves the frontend HTML
    GET  /health    → Health check
    POST /predict   → Upload PDB, get per-residue binding predictions
    POST /explain   → Upload PDB, get attention rollout scores
"""

import os
import sys
import json
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph, get_residues
from Bio.PDB import PDBParser

# Global model instance
_model = None
_device = None
_config = None


def save_heatmap_plot(probs, out_path, title):
    try:
        probs_np = np.array(probs)
        heatmap = probs_np.reshape(1, -1)
        
        plt.figure(figsize=(14, 2))
        plt.imshow(heatmap, aspect="auto", cmap="viridis")
        plt.colorbar(label="Binding Probability")
        plt.title(title)
        plt.xlabel("Residue Index")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Web] Heatmap saved to: {out_path}")
    except Exception as e:
        print(f"[Web] Failed to save heatmap plot: {e}")


def save_gradcam_plot(saliency, out_path, title):
    try:
        saliency_np = np.array(saliency)
        heatmap = saliency_np.reshape(1, -1)
        
        plt.figure(figsize=(14, 2))
        plt.imshow(heatmap, aspect="auto", cmap="plasma")
        plt.colorbar(label="Grad-CAM Importance")
        plt.title(title)
        plt.xlabel("Residue Index")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Web] Grad-CAM saved to: {out_path}")
    except Exception as e:
        print(f"[Web] Failed to save Grad-CAM plot: {e}")


def load_config(config_path: str = "config.yaml") -> dict:
    # Resolve relative to the project root (one level above web/)
    if not os.path.isabs(config_path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(root, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(config_path: str = "config.yaml"):
    """Load model (singleton)."""
    global _model, _device, _config
    if _model is None:
        _config = load_config(config_path)
        mcfg = _config["model"]
        wcfg = _config.get("web", {})
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ECABSDModel with stable V1/V2+ Overboost constructor parameters
        _model = ECABSDModel(
            esm_dim=mcfg.get("esm_dim", 1280),
            hidden_dim=mcfg.get("hidden_dim", 128),
            num_heads=mcfg.get("num_heads", 4),
            dropout=0.0,
            num_layers=mcfg.get("num_gcn_layers", 3),
            cross_attention=True,
        ).to(_device)

        checkpoint_path = wcfg.get("checkpoint", "checkpoints/best_model.pt")
        # Resolve relative to project root
        if not os.path.isabs(checkpoint_path):
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            checkpoint_path = os.path.join(root, checkpoint_path)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            _model.best_threshold = ckpt.get("best_threshold", 0.5)
            print(f"[Web] Model loaded from: {checkpoint_path}")
        else:
            _model.best_threshold = 0.5
            print(f"[Web] WARNING: No checkpoint at '{checkpoint_path}'. Using random weights.")

        _model.eval()
    return _model, _device, _config


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    get_model(config_path)  # Pre-load model

    app = FastAPI(
        title="ECABSD — Binding Site Detection",
        description="Equivariant Cross-Attention for Protein-Protein Binding Site Detection",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Static results files mounting
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    app.mount("/results", StaticFiles(directory=results_dir), name="results")

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the frontend."""
        html_path = os.path.join(templates_dir, "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h1>ECABSD Web Interface</h1><p>Frontend not found.</p>")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        model, device, cfg = get_model()
        return {
            "status": "ok",
            "model": "ECABSDModel",
            "device": str(device),
            "version": "1.0.0",
        }

    @app.post("/predict")
    async def predict(
        pdb_file: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form(None),
        threshold: float = Form(0.5),
    ):
        """
        Predict binding sites from an uploaded PDB file or a 4-letter PDB ID.
        """
        model, device, cfg = get_model()

        # Resolve PDB input
        tmp_path = None
        filename = ""
        try:
            if pdb_file and pdb_file.filename:
                # Save uploaded PDB to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                    shutil.copyfileobj(pdb_file.file, tmp)
                    tmp_path = tmp.name
                    filename = pdb_file.filename
            elif pdb_id and pdb_id.strip():
                pid = pdb_id.strip().upper()
                if len(pid) == 4:
                    os.makedirs("data/raw/pdbs", exist_ok=True)
                    local_pdb = f"data/raw/pdbs/{pid}.pdb"
                    if not os.path.exists(local_pdb):
                        import urllib.request
                        url = f"https://files.rcsb.org/download/{pid}.pdb"
                        urllib.request.urlretrieve(url, local_pdb)
                    
                    # Create a copy in temp file
                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                        with open(local_pdb, "rb") as src:
                            shutil.copyfileobj(src, tmp)
                        tmp_path = tmp.name
                    filename = f"{pid}.pdb"
                else:
                    raise HTTPException(status_code=400, detail="Invalid PDB ID format. Must be a 4-character ID.")
            else:
                raise HTTPException(status_code=400, detail="Please upload a PDB file or provide a 4-letter PDB ID.")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Failed to load PDB resource: {str(e)}")

        try:
            # Build graphs — v2 model requires edge_attr (5-dim)
            try:
                data_a = build_residue_graph(tmp_path, chain_a)
                if data_a.edge_attr is None:
                    raise ValueError("Graph has no edge_attr — check graph_construction.py")
                data_a = data_a.to(device)
            except (ValueError, KeyError) as e:
                raise HTTPException(status_code=400, detail=f"Chain {chain_a}: {str(e)}")

            data_b = None
            if chain_b and chain_b.strip():
                try:
                    data_b = build_residue_graph(tmp_path, chain_b)
                    if data_b.edge_attr is not None:
                        data_b = data_b.to(device)
                    else:
                        data_b = None
                except Exception:
                    data_b = None

            # Resolve threshold: negative value indicates "Auto"
            threshold_val = threshold
            if threshold < 0:
                threshold_val = getattr(model, "best_threshold", 0.5819)

            # Predict
            probs, labels, attn = model.predict(data_a, data_b, threshold=threshold_val)
            probs_np = probs.squeeze(-1).cpu().tolist()
            labels_np = labels.cpu().tolist()

            # Get residue info for labelling results
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", tmp_path)
            chain_obj = structure[0][chain_a]
            # get_residues returns (residue_list, coords) — only need list
            residue_list = get_residues(chain_obj)
            if isinstance(residue_list, tuple):
                residue_list = residue_list[0]

            results = []
            for i, r in enumerate(residue_list):
                if i >= len(probs_np):
                    break
                results.append({
                    "index": i,
                    "resname": r.get_resname(),
                    "resid": r.get_id()[1],
                    "chain": chain_a,
                    "probability": round(probs_np[i], 4),
                    "is_binding": bool(labels_np[i]),
                })

            binding_count = sum(1 for r in results if r["is_binding"])
            total_count = len(results)
            binding_ratio = binding_count / total_count if total_count > 0 else 0.0

            # Determine prediction quality classification
            quality = "Unknown"
            if total_count > 0:
                if binding_ratio < 0.08:
                    quality = "Too strict / too few predicted binding residues"
                elif 0.08 <= binding_ratio <= 0.20:
                    quality = "Good realistic range (Perfect Sample)"
                elif 0.21 <= binding_ratio <= 0.40:
                    quality = "Broad interface prediction"
                else:
                    quality = "Overprediction - use higher threshold or exclude"

            # Setup results directory and filenames
            clean_filename = os.path.basename(filename)
            pdb_name = os.path.splitext(clean_filename)[0]
            results_dir = _config.get("paths", {}).get("results_dir", "results")
            out_dir = os.path.join(results_dir, pdb_name)
            os.makedirs(out_dir, exist_ok=True)
            
            heatmap_url = ""
            gradcam_url = ""
            
            if total_count > 0:
                # Generate and save Heatmap
                try:
                    heatmap_filename = f"heatmap_{chain_a}.png"
                    heatmap_path = os.path.join(out_dir, heatmap_filename)
                    save_heatmap_plot(probs_np, heatmap_path, f"Binding Probability Heatmap - {pdb_name} Chain {chain_a}")
                    heatmap_url = f"/results/{pdb_name}/{heatmap_filename}"
                except Exception as e:
                    print(f"[Web] Error generating Heatmap: {e}")

                # Generate and save Grad-CAM
                try:
                    data_a_grad = data_a.clone()
                    data_a_grad.x = data_a_grad.x.float().detach().clone()
                    data_a_grad.x.requires_grad_(True)
                    
                    model.zero_grad()
                    logits, _ = model(data_a_grad, data_b)
                    score = logits.squeeze(-1).sum()
                    score.backward()
                    
                    if data_a_grad.x.grad is not None:
                        grads = data_a_grad.x.grad.detach().cpu().numpy()
                        saliency_raw = np.abs(grads).mean(axis=1)
                        saliency = ((saliency_raw - saliency_raw.min()) / (saliency_raw.max() - saliency_raw.min() + 1e-8)).tolist()
                        
                        gradcam_filename = f"gradcam_{chain_a}.png"
                        gradcam_path = os.path.join(out_dir, gradcam_filename)
                        save_gradcam_plot(saliency, gradcam_path, f"Grad-CAM Saliency Map - {pdb_name} Chain {chain_a}")
                        gradcam_url = f"/results/{pdb_name}/{gradcam_filename}"
                        
                        gradcam_json_path = os.path.join(out_dir, f"gradcam_{chain_a}.json")
                        gradcam_residues = [{"index": int(i), "gradcam_score": float(s)} for i, s in enumerate(saliency)]
                        with open(gradcam_json_path, "w") as f:
                            json.dump({
                                "pdb_file": clean_filename,
                                "chain": chain_a,
                                "method": "gradcam_saliency",
                                "residues": gradcam_residues
                            }, f, indent=2)
                except Exception as e:
                    print(f"[Web] Error generating Grad-CAM: {e}")

            # Auto-save "perfect" samples (Good realistic range)
            saved_to_results = False
            saved_path = ""
            if 0.08 <= binding_ratio <= 0.20:
                try:
                    saved_path = os.path.join(out_dir, f"web_perfect_prediction_{chain_a}.json")
                    payload = {
                        "pdb_file": clean_filename,
                        "chain_a": chain_a,
                        "chain_b": chain_b,
                        "threshold": threshold_val,
                        "total_residues": total_count,
                        "binding_residues_count": binding_count,
                        "binding_ratio": round(binding_ratio, 4),
                        "prediction_quality": quality,
                        "residues": results,
                    }
                    with open(saved_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    saved_to_results = True
                    print(f"[Web] Perfect prediction auto-saved to: {saved_path}")
                except Exception as e:
                    print(f"[Web] Error auto-saving perfect prediction: {e}")

            return JSONResponse({
                "status": "success",
                "pdb_file": filename,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "threshold": threshold_val,
                "total_residues": total_count,
                "binding_residues_count": binding_count,
                "binding_ratio": round(binding_ratio, 4),
                "prediction_quality": quality,
                "saved_to_results": saved_to_results,
                "saved_path": saved_path,
                "heatmap_url": heatmap_url,
                "gradcam_url": gradcam_url,
                "residues": results,
            })

        finally:
            os.unlink(tmp_path)

    @app.post("/explain")
    async def explain(
        pdb_file: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form(None),
    ):
        """
        Get attention rollout explanation for a prediction.
        """
        model, device, cfg = get_model()

        # Resolve PDB input
        tmp_path = None
        filename = ""
        try:
            if pdb_file and pdb_file.filename:
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                    shutil.copyfileobj(pdb_file.file, tmp)
                    tmp_path = tmp.name
                    filename = pdb_file.filename
            elif pdb_id and pdb_id.strip():
                pid = pdb_id.strip().upper()
                if len(pid) == 4:
                    os.makedirs("data/raw/pdbs", exist_ok=True)
                    local_pdb = f"data/raw/pdbs/{pid}.pdb"
                    if not os.path.exists(local_pdb):
                        import urllib.request
                        url = f"https://files.rcsb.org/download/{pid}.pdb"
                        urllib.request.urlretrieve(url, local_pdb)
                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                        with open(local_pdb, "rb") as src:
                            shutil.copyfileobj(src, tmp)
                        tmp_path = tmp.name
                    filename = f"{pid}.pdb"
                else:
                    raise HTTPException(status_code=400, detail="Invalid PDB ID format. Must be a 4-character ID.")
            else:
                raise HTTPException(status_code=400, detail="Please upload a PDB file or provide a 4-letter PDB ID.")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Failed to load PDB resource: {str(e)}")

        try:
            from explainability.attention_rollout import AttentionRollout

            data_a = build_residue_graph(tmp_path, chain_a).to(device)
            data_b = None
            if chain_b and chain_b.strip():
                try:
                    data_b = build_residue_graph(tmp_path, chain_b).to(device)
                except Exception:
                    pass

            rollout = AttentionRollout(model)
            scores, attn_matrix = rollout.compute(data_a, data_b)
            rollout.remove_hook()

            return JSONResponse({
                "status": "success",
                "attention_scores": scores.tolist(),
                "attention_matrix_shape": list(attn_matrix.shape),
            })
        finally:
            os.unlink(tmp_path)

    return app


# App instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
