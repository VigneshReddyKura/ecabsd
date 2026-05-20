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
        
        cb_label = "Attention Weight" if "Attention" in title else "Grad-CAM Importance"
        plt.colorbar(label=cb_label)
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

            # Predict with absolute minimum memory footprint
            with torch.no_grad():
                logits, attn = model(data_a, data_b)
                probs = torch.sigmoid(logits).squeeze(-1)
                probs_np = probs.cpu().tolist()

            max_prob = max(probs_np) if len(probs_np) > 0 else 0.0

            # Resolve threshold
            is_auto = False
            if threshold.lower() == "auto":
                is_auto = True
            else:
                try:
                    val = float(threshold)
                    if val < 0:
                        is_auto = True
                    else:
                        threshold_val = val
                except ValueError:
                    threshold_val = 0.5

            if is_auto:
                default_thresh = getattr(model, "best_threshold", 0.52)
                if max_prob < default_thresh:
                    # Adaptive percentile threshold: Use the 90th percentile of predicted probabilities
                    # to isolate the top 10% highest-confidence relative peaks for low-probability samples.
                    # A floor of 0.01 prevents spurious predictions on absolute flat noise.
                    threshold_val = max(0.01, float(np.percentile(probs_np, 90)))
                else:
                    threshold_val = default_thresh

            # Apply mode logic
            if mode == "topk":
                k = max(1, int(len(probs_np) * (top_k_percent / 100.0)))
                # Get top k indices
                top_indices = np.argsort(probs_np)[::-1][:k].tolist()
                labels_np = [0] * len(probs_np)
                for idx in top_indices:
                    labels_np[idx] = 1
                threshold_val = min([probs_np[i] for i in top_indices]) if len(top_indices) > 0 else threshold_val
            else:
                labels_np = [1 if p >= threshold_val else 0 for p in probs_np]

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
                    elif true_positives == 0 and false_positives == 0:
                        quality = "Underprediction - No Binding Residues Predicted (Needs Review)"
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
            
            if total_count > 0:
                # Generate in-memory Heatmap (Base64 data URL)
                try:
                    heatmap_url = get_heatmap_plot_base64(probs_np, f"Binding Probability Heatmap - {pdb_name} Chain {chain_a}")
                except Exception as e:
                    print(f"[Web] Error generating Heatmap: {e}")

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
        chain_b: Optional[str] = Form(None),
        threshold: Optional[float] = Form(None),
    ):
        """
        Get Grad-CAM or Attention explanation for a prediction.
        """
        import gc
        gc.collect()

        tmp_path = None
        model = None
        device = None

        try:
            model, device, cfg = get_model()

            # Resolve PDB input
            filename = ""
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
                            try:
                                os.remove(local_pdb)
                            except Exception:
                                pass

                    if not os.path.exists(local_pdb):
                        import urllib.request
                        url = f"https://files.rcsb.org/download/{pid}.pdb"
                        try:
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req) as response:
                                content_type = response.info().get_content_type()
                                if "html" in content_type.lower():
                                    raise ValueError("RCSB returned HTML page instead of PDB.")
                                data = response.read()
                                if len(data) < 5000:
                                    text_sample = data[:500].decode('utf-8', errors='ignore').strip()
                                    if text_sample.startswith("<!DOCTYPE") or "<html" in text_sample.lower():
                                        raise ValueError("RCSB returned HTML page.")
                                with open(local_pdb, "wb") as f:
                                    f.write(data)
                        except Exception as e:
                            if os.path.exists(local_pdb):
                                try:
                                    os.remove(local_pdb)
                                except Exception:
                                    pass
                            return JSONResponse({
                                "status": "error",
                                "error": f"Failed to retrieve PDB '{pid}' from RCSB: {str(e)}"
                            })

                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                        with open(local_pdb, "rb") as src:
                            shutil.copyfileobj(src, tmp)
                        tmp_path = tmp.name
                    filename = f"{pid}.pdb"
                else:
                    return JSONResponse({
                        "status": "error",
                        "error": "Invalid PDB ID format. Must be a 4-character ID."
                    })
            else:
                return JSONResponse({
                    "status": "error",
                    "error": "Please upload a PDB file or provide a 4-letter PDB ID."
                })

            # Clean and auto-capitalize chain inputs
            chain_a = chain_a.strip().upper() if chain_a else "A"
            chain_b = chain_b.strip().upper() if chain_b and chain_b.strip() else None

            # Temporarily move model to CPU to run explanation memory-safely
            model.to('cpu')

            # Build graph on CPU
            try:
                data_a = build_residue_graph(tmp_path, chain_a)
                if data_a.edge_attr is None:
                    raise ValueError("Graph has no edge_attr — check graph_construction.py")
                data_a = data_a.to('cpu')
            except Exception as e:
                return JSONResponse({
                    "status": "error",
                    "error": f"Failed to build graph for Chain {chain_a}: {str(e)}"
                })

            data_b = None
            if chain_b and chain_b.strip():
                try:
                    data_b = build_residue_graph(tmp_path, chain_b)
                    if data_b.edge_attr is not None:
                        data_b = data_b.to('cpu')
                except Exception:
                    pass

            num_nodes = data_a.num_nodes

            # Limit residues to max 512
            if num_nodes > 512:
                return JSONResponse({
                    "status": "error",
                    "error": "Grad-CAM unavailable for large proteins (>512 residues) on free hosting. Use local version."
                })

            saliency_gradcam = None
            gradcam_image = None
            gradcam_error = None

            # 1. Try Grad-CAM on CPU first
            try:
                print("[Web] Calculating Grad-CAM explanation on CPU.")
                data_a_grad = data_a.clone()
                data_a_grad.x = data_a_grad.x.float().detach().clone()
                data_a_grad.x.requires_grad_(True)

                model.zero_grad(set_to_none=True)

                # Temporarily disable requires_grad for all model parameters to save memory
                orig_requires_grad = {}
                for name, param in model.named_parameters():
                    orig_requires_grad[name] = param.requires_grad
                    param.requires_grad = False

                try:
                    logits, _ = model(data_a_grad, data_b)

                    if logits.ndim == 0:
                        logits = logits.unsqueeze(0)

                    if logits.ndim > 1:
                        score_logits = logits.squeeze(-1)
                    else:
                        score_logits = logits

                    score = score_logits.sum()
                    score.backward()
                finally:
                    # Restore original requires_grad settings
                    for name, param in model.named_parameters():
                        if name in orig_requires_grad:
                            param.requires_grad = orig_requires_grad[name]

                if data_a_grad.x.grad is not None:
                    grad_tensor = data_a_grad.x.grad
                    if grad_tensor.ndim == 1:
                        grad_tensor = grad_tensor.unsqueeze(0)

                    grads = grad_tensor.detach().cpu().numpy()
                    features = data_a_grad.x.detach().cpu().numpy()

                    # Gradient-based Grad-CAM calculation
                    weights = np.mean(grads, axis=0)
                    saliency_raw = np.sum(features * weights, axis=1)
                    saliency_raw = np.maximum(saliency_raw, 0) # ReLU positive contribution

                    if np.all(saliency_raw == 0) or np.max(saliency_raw) == 0:
                        saliency_raw = np.abs(np.sum(features * grads, axis=1))

                    denom = (saliency_raw.max() - saliency_raw.min() + 1e-8)
                    saliency_gradcam = ((saliency_raw - saliency_raw.min()) / denom).tolist()

                    pdb_name = os.path.splitext(filename)[0]
                    gradcam_image = get_gradcam_plot_base64(saliency_gradcam, f"Grad-CAM Saliency Map - {pdb_name} Chain {chain_a}")

                    del grads, features, saliency_raw
                else:
                    raise ValueError("No gradients computed on node features.")
            except (MemoryError, RuntimeError, Exception) as gradcam_err:
                print(f"[Web] Grad-CAM failed: {gradcam_err}")
                gradcam_error = f"Grad-CAM unavailable ({str(gradcam_err) or gradcam_err.__class__.__name__}), attention saliency shown separately."
                saliency_gradcam = None
                gradcam_image = None
                model.zero_grad(set_to_none=True)
                gc.collect()

            # 2. Always compute Attention Saliency Map separately
            attention_image = None
            saliency_attn = None
            try:
                print("[Web] Calculating Attention-based explanation on CPU.")
                model.eval()
                with torch.no_grad():
                    logits, attn_list = model(data_a, data_b)
                    if attn_list and len(attn_list) > 0:
                        attn = attn_list[0].detach().cpu().float()
                        if attn.ndim == 3:
                            attn = attn.mean(dim=0)
                        
                        if attn.ndim == 2:
                            scores_attn = attn.sum(dim=1).numpy()
                        elif attn.ndim == 1:
                            scores_attn = attn.numpy()
                        else:
                            scores_attn = attn.flatten().numpy()
                            
                        saliency_attn = ((scores_attn - scores_attn.min()) / (scores_attn.max() - scores_attn.min() + 1e-8)).tolist()

                        pdb_name = os.path.splitext(filename)[0]
                        attention_image = get_gradcam_plot_base64(saliency_attn, f"Attention Saliency Map - {pdb_name} Chain {chain_a}")
                    else:
                        raise ValueError("Attention weights unavailable.")
            except Exception as attn_err:
                print(f"[Web] Attention calculation failed: {attn_err}")

            # 3. Add overlap check (between top 10 Grad-CAM residues and top predicted binding residues)
            overlap_pct = 0.0
            sorted_gc_indices = []
            predicted_binding_indices = []

            # Compute prediction probabilities for overlap check
            try:
                model.eval()
                with torch.no_grad():
                    logits, _ = model(data_a, data_b)
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                    
                    max_prob = float(probs.max()) if len(probs) > 0 else 0.0
                    
                    is_auto = False
                    if threshold is None or threshold < 0:
                        is_auto = True
                    else:
                        threshold_val = threshold
                        
                    if is_auto:
                        default_thresh = getattr(model, "best_threshold", 0.52)
                        if max_prob < default_thresh:
                            threshold_val = max(0.01, float(np.percentile(probs, 90)))
                        else:
                            threshold_val = default_thresh
                            
                    predicted_binding_indices = np.where(probs >= threshold_val)[0].tolist()
            except Exception as e:
                print(f"[Web] Prediction check failed for overlap calculation: {e}")

            random_overlap_pct = 0.0
            if saliency_gradcam is not None:
                total_n = len(saliency_gradcam)
                if total_n > 0:
                    # Hypergeometric random expected baseline overlap % for 10 chosen residues
                    random_overlap_pct = round((len(predicted_binding_indices) / total_n) * 100, 1)

            if saliency_gradcam is not None and len(predicted_binding_indices) > 0:
                sorted_gc_indices = np.argsort(saliency_gradcam)[::-1][:10].tolist()
                intersection = set(sorted_gc_indices).intersection(set(predicted_binding_indices))
                # Compute percentage overlap based on top 10 GC
                overlap_pct = round((len(intersection) / 10.0) * 100, 1)

            # Garbage collect
            gc.collect()

            return JSONResponse({
                "status": "success",
                "gradcam_image": gradcam_image,
                "gradcam_error": gradcam_error,
                "attention_image": attention_image,
                "gradcam_scores": saliency_gradcam,
                "attention_scores": saliency_attn,
                "overlap_percentage": overlap_pct,
                "random_overlap_percentage": random_overlap_pct,
                "top_gradcam_residues": sorted_gc_indices,
                "predicted_binding_residues": predicted_binding_indices
            })

        except BaseException as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e) if str(e) else e.__class__.__name__
            return JSONResponse({
                "status": "error",
                "error": f"Explanation failed: {err_msg}"
            }, status_code=500)
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                if 'data_a' in locals(): del data_a
                if 'data_b' in locals(): del data_b
                if 'data_a_grad' in locals(): del data_a_grad
                if 'logits' in locals(): del logits
                if 'score' in locals(): del score
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
