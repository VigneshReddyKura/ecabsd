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
import gc

# ==========================================
# BULLETPROOF RENDER PORT & HOST MONKEYPATCH
# ==========================================
# Forces Uvicorn to bind to Render's dynamic $PORT and listen on 0.0.0.0,
# regardless of what start command or parameters were configured in the dashboard.
try:
    import uvicorn
    
    # 1. Patch any future Config instantiations
    original_init = uvicorn.Config.__init__
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if "PORT" in os.environ:
            self.port = int(os.environ["PORT"])
        self.host = "0.0.0.0"
        
    uvicorn.Config.__init__ = patched_init
    
    # 2. Override any already existing Config objects in memory (from CLI startup)
    for obj in gc.get_objects():
        if isinstance(obj, uvicorn.Config):
            if "PORT" in os.environ:
                obj.port = int(os.environ["PORT"])
                print(f"[ECABSD Patch] Found existing Config in memory: Overrode port to {obj.port}")
            obj.host = "0.0.0.0"
            print(f"[ECABSD Patch] Found existing Config in memory: Overrode host to 0.0.0.0")
except Exception as e:
    print(f"[ECABSD Patch] Exception applying dynamic port override: {e}")
# ==========================================

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


import io
import base64

def get_heatmap_plot_base64(probs, title):
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
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"[Web] Failed to generate heatmap plot: {e}")
        return ""


def get_gradcam_plot_base64(saliency, title):
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
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"[Web] Failed to generate Grad-CAM plot: {e}")
        return ""


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
    
    # Absolute minimum PyTorch memory footprint settings (crucial for 512MB RAM container)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception as e:
        print(f"[Web] Failed to limit PyTorch threads: {e}")
        
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
    # get_model(config_path)  # Lazy-loaded on first prediction to speed up server startup and avoid timeouts

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        v2_ready = os.path.exists(os.path.join(root, "checkpoints", "best_model.pt"))
        v3_ready = os.path.exists(os.path.join(root, "checkpoints", "best_model_v3.pt"))
        return {
            "status": "ok",
            "device": device,
            "v2_available": v2_ready,
            "v3_available": v3_ready,
        }

    @app.post("/predict")
    async def predict(
        pdb_file: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form("B"),
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
                    
                    # Validate existing file to prevent reading empty/corrupted 404 pages
                    is_corrupted = False
                    if os.path.exists(local_pdb):
                        if os.path.getsize(local_pdb) < 5000:
                            is_corrupted = True
                        else:
                            try:
                                with open(local_pdb, "r", encoding="utf-8", errors="ignore") as f:
                                    first_lines = "".join([f.readline() for _ in range(5)]).strip()
                                    if first_lines.startswith("<!DOCTYPE") or "<html" in first_lines.lower() or "404 not found" in first_lines.lower():
                                        is_corrupted = True
                            except Exception:
                                pass
                        if is_corrupted:
                            print(f"[Web] Corrupted PDB file found at {local_pdb}. Deleting and re-downloading...")
                            try:
                                os.remove(local_pdb)
                            except Exception:
                                pass

                    if not os.path.exists(local_pdb):
                        import urllib.request
                        url = f"https://files.rcsb.org/download/{pid}.pdb"
                        print(f"[Web] Downloading PDB from: {url}")
                        try:
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req) as response:
                                content_type = response.info().get_content_type()
                                if "html" in content_type.lower():
                                    raise ValueError("RCSB PDB archive returned HTML/error page instead of PDB coordinate data.")
                                data = response.read()
                                if len(data) < 5000:
                                    text_sample = data[:500].decode('utf-8', errors='ignore').strip()
                                    if text_sample.startswith("<!DOCTYPE") or "<html" in text_sample.lower() or "404" in text_sample:
                                        raise ValueError("RCSB returned HTML error page (404 Not Found).")
                                with open(local_pdb, "wb") as f:
                                    f.write(data)
                        except Exception as e:
                            if os.path.exists(local_pdb):
                                try:
                                    os.remove(local_pdb)
                                except Exception:
                                    pass
                            raise HTTPException(status_code=400, detail=f"Failed to retrieve PDB '{pid}' from RCSB: {str(e)}")
                    
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
            # Clean and auto-capitalize chain inputs
            chain_a = chain_a.strip().upper() if chain_a else "A"
            chain_b = chain_b.strip().upper() if chain_b and chain_b.strip() else None

            # Build graphs — v2 model requires edge_attr (5-dim)
            try:
                data_a = build_residue_graph(tmp_path, chain_a)
                if data_a.edge_attr is None:
                    raise ValueError("Graph has no edge_attr — check graph_construction.py")
                data_a = data_a.to(device)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to build graph for Chain {chain_a}: {str(e)}")

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

            # Predict with absolute minimum memory footprint
            with torch.no_grad():
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

            clean_filename = os.path.basename(filename)
            pdb_name = os.path.splitext(clean_filename)[0]
            
            heatmap_url = ""
            gradcam_url = ""
            
            if total_count > 0:
                # Generate in-memory Heatmap (Base64 data URL)
                try:
                    heatmap_url = get_heatmap_plot_base64(probs_np, f"Binding Probability Heatmap - {pdb_name} Chain {chain_a}")
                except Exception as e:
                    print(f"[Web] Error generating Heatmap: {e}")

                # Generate in-memory Grad-CAM with extreme memory optimization
                if os.environ.get("RENDER") == "true" or os.environ.get("DISABLE_GRADCAM") == "true":
                    print("[Web] Skipping Grad-CAM calculation on Render to prevent 512MB RAM OOM crash.")
                else:
                    try:
                        data_a_grad = data_a.clone()
                        data_a_grad.x = data_a_grad.x.float().detach().clone()
                        data_a_grad.x.requires_grad_(True)
                        
                        model.zero_grad(set_to_none=True)
                        logits, _ = model(data_a_grad, data_b)
                        score = logits.squeeze(-1).sum()
                        score.backward()
                        
                        if data_a_grad.x.grad is not None:
                            grads = data_a_grad.x.grad.detach().cpu().numpy()
                            saliency_raw = np.abs(grads).mean(axis=1)
                            saliency = ((saliency_raw - saliency_raw.min()) / (saliency_raw.max() - saliency_raw.min() + 1e-8)).tolist()
                            
                            gradcam_url = get_gradcam_plot_base64(saliency, f"Grad-CAM Saliency Map - {pdb_name} Chain {chain_a}")
                            
                            # Clean up intermediate arrays immediately
                            del grads, saliency_raw, saliency
                        
                        # Force delete gradient graph and clear gradients
                        del data_a_grad, logits, score
                        model.zero_grad(set_to_none=True)
                        import gc
                        gc.collect()
                    except Exception as e:
                        print(f"[Web] Error generating Grad-CAM: {e}")
                        model.zero_grad(set_to_none=True)
                        import gc
                        gc.collect()

            # Auto-save "perfect" samples or "Excellent" overlap
            saved_to_results = False
            saved_path = ""
            
            is_excellent_overlap = (overlap_stats is not None and overlap_stats.get("f1", 0) >= 0.5)
            
            if is_excellent_overlap:
                try:
                    # Disabled JSON result saving to prevent local file generation on the deployed site
                    # saved_path = os.path.join(out_dir, f"High_Confidence_Prediction_Chain_{chain_a}.json")
                    # payload = {
                    #     "pdb_file": clean_filename,
                    #     "chain_a": chain_a,
                    #     "chain_b": chain_b,
                    #     "threshold": threshold_val,
                    #     "total_residues": total_count,
                    #     "binding_residues_count": binding_count,
                    #     "binding_ratio": round(binding_ratio, 4),
                    #     "prediction_quality": quality,
                    #     "residues": results,
                    # }
                    # with open(saved_path, "w") as f:
                    #     json.dump(payload, f, indent=2)
                    # saved_to_results = True
                    # print(f"[Web] Perfect prediction auto-saved to: {saved_path}")
                    pass
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Prediction API failure: {str(e)}")
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
                import gc
                gc.collect()
            except Exception:
                pass

    @app.post("/explain")
    async def explain(
        pdb_file: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form("B"),
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
                    
                    # Validate existing file to prevent reading empty/corrupted 404 pages
                    is_corrupted = False
                    if os.path.exists(local_pdb):
                        if os.path.getsize(local_pdb) < 5000:
                            is_corrupted = True
                        else:
                            try:
                                with open(local_pdb, "r", encoding="utf-8", errors="ignore") as f:
                                    first_lines = "".join([f.readline() for _ in range(5)]).strip()
                                    if first_lines.startswith("<!DOCTYPE") or "<html" in first_lines.lower() or "404 not found" in first_lines.lower():
                                        is_corrupted = True
                            except Exception:
                                pass
                        if is_corrupted:
                            print(f"[Web] Corrupted PDB file found at {local_pdb}. Deleting and re-downloading...")
                            try:
                                os.remove(local_pdb)
                            except Exception:
                                pass

                    if not os.path.exists(local_pdb):
                        import urllib.request
                        url = f"https://files.rcsb.org/download/{pid}.pdb"
                        print(f"[Web] Downloading PDB from: {url}")
                        try:
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req) as response:
                                content_type = response.info().get_content_type()
                                if "html" in content_type.lower():
                                    raise ValueError("RCSB PDB archive returned HTML/error page instead of PDB coordinate data.")
                                data = response.read()
                                if len(data) < 5000:
                                    text_sample = data[:500].decode('utf-8', errors='ignore').strip()
                                    if text_sample.startswith("<!DOCTYPE") or "<html" in text_sample.lower() or "404" in text_sample:
                                        raise ValueError("RCSB returned HTML error page (404 Not Found).")
                                with open(local_pdb, "wb") as f:
                                    f.write(data)
                        except Exception as e:
                            if os.path.exists(local_pdb):
                                try:
                                    os.remove(local_pdb)
                                except Exception:
                                    pass
                            raise HTTPException(status_code=400, detail=f"Failed to retrieve PDB '{pid}' from RCSB: {str(e)}")
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
            # Clean and auto-capitalize chain inputs
            chain_a = chain_a.strip().upper() if chain_a else "A"
            chain_b = chain_b.strip().upper() if chain_b and chain_b.strip() else None

            from explainability.attention_rollout import AttentionRollout

            try:
                data_a = build_residue_graph(tmp_path, chain_a).to(device)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to build graph for Chain {chain_a}: {str(e)}")
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Explainability API failure: {str(e)}")
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
                import gc
                gc.collect()
            except Exception:
                pass


    return app



# App instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
