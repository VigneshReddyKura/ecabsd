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
from models.graph_construction import build_residue_graph, get_residues, compute_binding_labels
from Bio.PDB import PDBParser

# Global model instances (V3)
_model    = None   # V3 (primary)
_device   = None
_config   = None


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
    """Load V3 model (primary, singleton)."""
    global _model, _device, _config
    if _model is None:
        _config = load_config(config_path)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _model = ECABSDModel(
            input_dim=33,
            hidden_dim=256,
            num_heads=4,
            dropout=0.0,
            edge_dim=5,
            num_gcn_layers=6,
        ).to(_device)

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ckpt_path = os.path.join(root, "checkpoints", "best_model_v3.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            _model.best_threshold = ckpt.get("best_threshold", 0.52)
            print(f"[Web] V3 model loaded from: {ckpt_path}")
        else:
            _model.best_threshold = 0.52
            print(f"[Web] WARNING: V3 checkpoint not found at {ckpt_path}")
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
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        v2_ready = os.path.exists(os.path.join(root, "checkpoints", "best_model.pt"))
        v3_ready = os.path.exists(os.path.join(root, "checkpoints", "best_model_v3.pt"))
        return {
            "status": "ok",
            "device": str(device),
            "v2_available": v2_ready,
            "v3_available": v3_ready,
        }

    @app.post("/predict")
    async def predict(
        pdb_file: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form(None),
        threshold: str = Form("auto"),
        mode: str = Form("threshold"),
        top_k_percent: float = Form(15.0),
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

            # Resolve threshold
            threshold_val = 0.5
            if threshold.lower() == "auto":
                threshold_val = getattr(model, "best_threshold", 0.5819)
            else:
                try:
                    val = float(threshold)
                    if val < 0:
                        threshold_val = getattr(model, "best_threshold", 0.5819)
                    else:
                        threshold_val = val
                except ValueError:
                    threshold_val = 0.5

            # Predict
            logits, attn = model(data_a, data_b)
            probs = torch.sigmoid(logits).squeeze(-1)
            probs_np = probs.cpu().tolist()
            
            # Apply mode logic
            if mode == "topk":
                k = max(1, int(len(probs_np) * (top_k_percent / 100.0)))
                top_indices = torch.topk(probs, k).indices
                labels_np = [0] * len(probs_np)
                for idx in top_indices:
                    labels_np[idx] = 1
                threshold_val = min([probs_np[i] for i in top_indices]) if len(top_indices) > 0 else threshold_val
            else:
                labels_np = (probs >= threshold_val).cpu().numpy().astype(int).tolist()

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

            true_labels = []
            overlap_stats = None
            quality = "Unknown"
            
            # If partner chain is provided, we can compute ground truth and actual overlap
            if chain_b and chain_b.strip():
                try:
                    true_labels = compute_binding_labels(tmp_path, chain_a, chain_b, distance_cutoff=5.0)
                except Exception as e:
                    print(f"Failed to compute ground truth: {e}")
                    true_labels = []
                
                if true_labels and len(true_labels) == len(probs_np):
                    true_labels_np = np.array(true_labels)
                    pred_labels_np = np.array(labels_np)
                    
                    true_positives = int(np.sum((true_labels_np == 1) & (pred_labels_np == 1)))
                    false_positives = int(np.sum((true_labels_np == 0) & (pred_labels_np == 1)))
                    false_negatives = int(np.sum((true_labels_np == 1) & (pred_labels_np == 0)))
                    
                    precision = true_positives / max(true_positives + false_positives, 1)
                    recall = true_positives / max(true_positives + false_negatives, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    
                    overlap_stats = {
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "actual_binding_count": int(np.sum(true_labels_np))
                    }
                    
                    if f1 >= 0.70:
                        quality = "Excellent Experimental Overlap"
                    elif f1 >= 0.50:
                        quality = "Good Experimental Overlap"
                    elif precision > recall:
                        quality = "Underprediction - High Precision, Low Recall (Needs Review)"
                    else:
                        quality = "Overprediction - Low Precision, High Recall (Needs Review)"
                else:
                    # Fallback if computation failed
                    if binding_ratio < 0.10:
                        quality = f"Mode: {mode.upper()} | Tight Interface / Possible Underprediction (Ratio: {round(binding_ratio*100, 1)}%)"
                    elif 0.10 <= binding_ratio <= 0.30:
                        quality = f"Mode: {mode.upper()} | Healthy Moderate Interface (Ratio: {round(binding_ratio*100, 1)}%)"
                    elif 0.31 <= binding_ratio <= 0.40:
                        quality = f"Mode: {mode.upper()} | Broad Interface (Ratio: {round(binding_ratio*100, 1)}%)"
                    else:
                        quality = f"Mode: {mode.upper()} | Possible Overprediction (Ratio: {round(binding_ratio*100, 1)}%)"
            else:
                if binding_ratio < 0.10:
                    quality = f"Mode: {mode.upper()} | Tight Interface / Possible Underprediction (Ratio: {round(binding_ratio*100, 1)}%)"
                elif 0.10 <= binding_ratio <= 0.30:
                    quality = f"Mode: {mode.upper()} | Healthy Moderate Interface (Ratio: {round(binding_ratio*100, 1)}%)"
                elif 0.31 <= binding_ratio <= 0.40:
                    quality = f"Mode: {mode.upper()} | Broad Interface (Ratio: {round(binding_ratio*100, 1)}%)"
                else:
                    quality = f"Mode: {mode.upper()} | Possible Overprediction (Ratio: {round(binding_ratio*100, 1)}%)"

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
                    heatmap_filename = f"Binding_Probability_Heatmap_Chain_{chain_a}.png"
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
                        
                        gradcam_filename = f"GradCAM_Saliency_Map_Chain_{chain_a}.png"
                        gradcam_path = os.path.join(out_dir, gradcam_filename)
                        save_gradcam_plot(saliency, gradcam_path, f"Grad-CAM Saliency Map - {pdb_name} Chain {chain_a}")
                        gradcam_url = f"/results/{pdb_name}/{gradcam_filename}"
                        
                        gradcam_json_path = os.path.join(out_dir, f"GradCAM_Scores_Chain_{chain_a}.json")
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

            # Auto-save "perfect" samples or "Excellent" overlap
            saved_to_results = False
            saved_path = ""
            
            is_excellent_overlap = (overlap_stats is not None and overlap_stats.get("f1", 0) >= 0.5)
            
            if is_excellent_overlap:
                try:
                    saved_path = os.path.join(out_dir, f"High_Confidence_Prediction_Chain_{chain_a}.json")
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

            response_payload = {
                "status": "success",
                "pdb_file": filename,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "threshold": threshold_val,
                "mode": mode,
                "total_residues": total_count,
                "binding_residues_count": binding_count,
                "binding_ratio": round(binding_ratio, 4),
                "prediction_quality": quality,
                "saved_to_results": saved_to_results,
                "saved_path": saved_path,
                "heatmap_url": heatmap_url,
                "gradcam_url": gradcam_url,
                "residues": results,
            }
            if overlap_stats:
                response_payload["experimental_overlap"] = overlap_stats

            return JSONResponse(response_payload)

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
