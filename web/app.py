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

# Limit PyTorch threads at the very top of startup, before any parallel work or model loading starts
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ecabsd_v3_model import ECABSDModelV3 as ECABSDModel
from models.graph_construction import build_residue_graph, get_residues, compute_binding_labels
from Bio.PDB import PDBParser

# Global model instances (V3)
_model    = None   # V3 (primary)
_device   = None
_config   = None


import io
import base64

def get_heatmap_plot_base64(probs, title, residues=None):
    try:
        probs_np = np.array(probs)
        n_residues = len(probs_np)
        
        # Premium Styling: Dark Theme matching `--bg` (#080b14) & `--surface` (#0f1420)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.0), sharex=True,
                                       gridspec_kw={'height_ratios': [0.8, 3.2]})
        fig.patch.set_facecolor('#080b14')
        ax1.set_facecolor('#0f1420')
        ax2.set_facecolor('#0f1420')
        
        # Title of figure (vibrant & bold)
        fig.suptitle(title, fontsize=14, fontweight="bold", color='#ffffff', y=0.98)
        
        # Top subplot: 1D Heatmap
        heatmap = probs_np.reshape(1, -1)
        im = ax1.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax1.set_yticks([])
        ax1.set_ylabel("Heatmap", color='#94a3b8', fontsize=9, labelpad=8)
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        for spine in ax1.spines.values():
            spine.set_visible(False)
            
        cb = fig.colorbar(im, ax=ax1, orientation="horizontal", pad=0.3, aspect=60)
        cb.outline.set_visible(False)
        cb.ax.xaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8', labelsize=8)
        cb.set_label("Binding Probability", color='#94a3b8', fontsize=8, labelpad=2)
        
        # Resolve residue labels
        three_to_one = {
            'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
            'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I',
            'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
            'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
        }
        res_labels = []
        if residues:
            for idx, r in enumerate(residues):
                if idx >= n_residues:
                    break
                name = r.get_resname()
                one_letter = three_to_one.get(name, '?')
                num = r.get_id()[1]
                res_labels.append(f"{one_letter}{num}")
        else:
            res_labels = [str(i) for i in range(1, n_residues + 1)]
            
        # Bottom subplot: Line/Area plot
        color_main = "#06b6d4"  # beautiful cyan
        
        # Main line and shaded area
        ax2.plot(np.arange(n_residues), probs_np, color=color_main, linewidth=2.5, zorder=3, label="Probability")
        ax2.plot(np.arange(n_residues), probs_np, color=color_main, linewidth=6.0, alpha=0.3, zorder=2) # subtle glow
        ax2.fill_between(np.arange(n_residues), probs_np, color=color_main, alpha=0.12, zorder=1)
        
        # Dynamic ticks
        if n_residues <= 60:
            tick_step = 1
        elif n_residues <= 120:
            tick_step = 2
        elif n_residues <= 200:
            tick_step = 5
        elif n_residues <= 400:
            tick_step = 10
        else:
            tick_step = 20
            
        tick_indices = np.arange(0, n_residues, tick_step)
        tick_labels = [res_labels[i] for i in tick_indices]
        
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9, color='#94a3b8')
        
        ax2.set_ylabel("Binding Probability", color='#e2e8f0', fontsize=11, fontweight="bold", labelpad=8)
        ax2.set_xlabel("Residue Index", color='#e2e8f0', fontsize=11, fontweight="bold", labelpad=8)
        ax2.grid(True, which="both", color="#1e2640", linestyle=":", linewidth=0.6, alpha=0.6)
        
        for name, spine in ax2.spines.items():
            if name in ['top', 'right']:
                spine.set_visible(False)
            else:
                spine.set_color('#1e2640')
                spine.set_linewidth(1.0)
                
        ax2.set_ylim(-0.05, 1.10)
        
        threshold_val = 0.52
        ax2.axhline(y=threshold_val, color="#f43f5e", linestyle="--", linewidth=1.5, alpha=0.85, zorder=4,
                    label=f"Decision Threshold ({threshold_val:.2f})")
        
        # Highlight sites above threshold
        binding_idxs = np.where(probs_np >= threshold_val)[0]
        if len(binding_idxs) > 0:
            ax2.scatter(binding_idxs, probs_np[binding_idxs], color="#10b981", s=30, zorder=5, edgecolors='#080b14', linewidth=1, label="Predicted Binding Sites")
            
        legend = ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor('#0f1420')
            frame.set_edgecolor('#1e2640')
            frame.set_linewidth(0.8)
            for text in legend.get_texts():
                text.set_color('#e2e8f0')
                
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"[Web] Failed to generate heatmap plot: {e}")
        plt.close("all")
        return ""


def get_gradcam_plot_base64(saliency, title, residues=None):
    try:
        saliency_np = np.array(saliency)
        n_residues = len(saliency_np)
        
        # Normalize saliency for visualization (0 to 1)
        s_max = saliency_np.max() if saliency_np.max() > 0 else 1.0
        norm_saliency = saliency_np / s_max
        
        # Premium Styling: Dark Theme matching `--bg` (#080b14) & `--surface` (#0f1420)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.0), sharex=True,
                                       gridspec_kw={'height_ratios': [0.8, 3.2]})
        fig.patch.set_facecolor('#080b14')
        ax1.set_facecolor('#0f1420')
        ax2.set_facecolor('#0f1420')
        
        fig.suptitle(title, fontsize=14, fontweight="bold", color='#ffffff', y=0.98)
        
        # Top subplot: 1D Heatmap
        heatmap = norm_saliency.reshape(1, -1)
        cb_label = "Attention Weight" if "Attention" in title else "Grad-CAM Importance"
        im = ax1.imshow(heatmap, aspect="auto", cmap="plasma", vmin=0, vmax=1)
        ax1.set_yticks([])
        ax1.set_ylabel("Heatmap", color='#94a3b8', fontsize=9, labelpad=8)
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        for spine in ax1.spines.values():
            spine.set_visible(False)
            
        cb = fig.colorbar(im, ax=ax1, orientation="horizontal", pad=0.3, aspect=60)
        cb.outline.set_visible(False)
        cb.ax.xaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8', labelsize=8)
        cb.set_label(cb_label, color='#94a3b8', fontsize=8, labelpad=2)
        
        # Resolve residue labels
        three_to_one = {
            'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
            'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I',
            'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
            'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
        }
        res_labels = []
        if residues:
            for idx, r in enumerate(residues):
                if idx >= n_residues:
                    break
                name = r.get_resname()
                one_letter = three_to_one.get(name, '?')
                num = r.get_id()[1]
                res_labels.append(f"{one_letter}{num}")
        else:
            res_labels = [str(i) for i in range(1, n_residues + 1)]
            
        # Bottom subplot: Line/Area plot of raw scores
        color_main = "#10b981" if "Attention" in title else "#818cf8"  # Emerald vs Violet
        
        ax2.plot(np.arange(n_residues), saliency_np, color=color_main, linewidth=2.5, zorder=3, label="Saliency Score")
        ax2.plot(np.arange(n_residues), saliency_np, color=color_main, linewidth=6.0, alpha=0.3, zorder=2) # subtle glow
        ax2.fill_between(np.arange(n_residues), saliency_np, color=color_main, alpha=0.12, zorder=1)
        
        # Dynamic ticks
        if n_residues <= 60:
            tick_step = 1
        elif n_residues <= 120:
            tick_step = 2
        elif n_residues <= 200:
            tick_step = 5
        elif n_residues <= 400:
            tick_step = 10
        else:
            tick_step = 20
            
        tick_indices = np.arange(0, n_residues, tick_step)
        tick_labels = [res_labels[i] for i in tick_indices]
        
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9, color='#94a3b8')
        
        ax2.set_ylabel("Saliency Value", color='#e2e8f0', fontsize=11, fontweight="bold", labelpad=8)
        ax2.set_xlabel("Residue Index", color='#e2e8f0', fontsize=11, fontweight="bold", labelpad=8)
        ax2.grid(True, which="both", color="#1e2640", linestyle=":", linewidth=0.6, alpha=0.6)
        
        for name, spine in ax2.spines.items():
            if name in ['top', 'right']:
                spine.set_visible(False)
            else:
                spine.set_color('#1e2640')
                spine.set_linewidth(1.0)
                
        ax2.set_ylim(-0.05, 1.10)
        
        # Highlight top 10 contributing residues
        top_10_indices = np.argsort(saliency_np)[::-1][:10]
        ax2.scatter(top_10_indices, saliency_np[top_10_indices], color="#f59e0b", s=45, zorder=5, edgecolors='#080b14', linewidth=1, label="Top 10 Contributors")
        
        # Annotate the top 5 peaks directly on the chart
        top_5_indices = top_10_indices[:5]
        for idx in top_5_indices:
            score = saliency_np[idx]
            label = res_labels[idx]
            ax2.annotate(
                label, 
                xy=(idx, score), 
                xytext=(idx + (2 if idx < n_residues * 0.8 else -8), score + 0.05),
                color='#f59e0b',
                fontweight='bold',
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=0.8, alpha=0.7),
                zorder=6
            )
            
        legend = ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor('#0f1420')
            frame.set_edgecolor('#1e2640')
            frame.set_linewidth(0.8)
            for text in legend.get_texts():
                text.set_color('#e2e8f0')
                
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"[Web] Failed to generate Grad-CAM plot: {e}")
        plt.close("all")
        return ""


def has_enough_memory(min_free_mb=250):
    import psutil
    try:
        mem = psutil.virtual_memory()
        free_mb = mem.available / (1024 * 1024)
        return free_mb >= min_free_mb, free_mb
    except Exception:
        return True, 999.0


def cleanup_memory():
    """Aggressively free memory between requests to prevent OOM on Render free tier."""
    try:
        plt.close("all")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        _, free_mb = has_enough_memory(0)
        print(f"[Web] cleanup_memory() done. Free: {free_mb:.0f} MB")
    except Exception:
        pass


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
            _model.load_state_dict(ckpt["model_state_dict"], strict=False)
            _model.best_threshold = ckpt.get("best_threshold", 0.52)
            print(f"[Web] V3 model loaded from: {ckpt_path}")
        else:
            _model.best_threshold = 0.52
            print(f"[Web] WARNING: V3 checkpoint not found at {ckpt_path}")
        _model.eval()
    return _model, _device, _config



def create_app(config_path: str = "config.yaml") -> FastAPI:
    try:
        print("[Web] Pre-loading model at startup to prevent dynamic request timeouts...")
        get_model(config_path)
    except Exception as e:
        print(f"[Web] WARNING: Failed to pre-load model at startup: {e}")

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
        v3_ready = os.path.exists(os.path.join(root, "checkpoints", "best_model_v3.pt"))
        return {
            "status": "ok",
            "device": device,
            "model_ready": v3_ready,
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
        cleanup_memory()  # Clear old prediction/gradcam memory first
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
                            raise ValueError(f"Failed to retrieve PDB '{pid}' from RCSB: {str(e)}")
                    
                    # Create a copy in temp file
                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                        with open(local_pdb, "rb") as src:
                            shutil.copyfileobj(src, tmp)
                        tmp_path = tmp.name
                    filename = f"{pid}.pdb"
                else:
                    raise ValueError("Invalid PDB ID format. Must be a 4-character ID.")
            else:
                raise ValueError("Please upload a PDB file or provide a 4-letter PDB ID.")
        except Exception as e:
            raise ValueError(f"PDB loading failed: {str(e)}")

        try:
            # Clean and auto-capitalize chain inputs
            chain_a = chain_a.strip().upper() if chain_a else "A"
            chain_b = chain_b.strip().upper() if chain_b and chain_b.strip() else None

            # 1. Parse structure and validate chains explicitly
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("protein", tmp_path)
            except Exception as parse_err:
                raise ValueError(f"Failed to parse PDB structure: {str(parse_err)}")

            if not structure or len(structure) == 0:
                raise ValueError("PDB structure is empty or has no models.")

            model_obj = structure[0]
            valid_chains = [c.get_id() for c in model_obj]
            print(f"[Web] Valid chains in PDB: {valid_chains}")

            if chain_a not in valid_chains:
                raise ValueError(f"Chain A '{chain_a}' not found in PDB file. Available chains: {', '.join(valid_chains) or 'None'}")
            
            if chain_b and chain_b not in valid_chains:
                raise ValueError(f"Chain B '{chain_b}' not found in PDB file. Available chains: {', '.join(valid_chains) or 'None'}")

            # 2. Extract residue lists to count total residues before building GNN graph
            try:
                res_list_a, _ = get_residues(model_obj[chain_a])
            except Exception as res_err:
                raise ValueError(f"Failed to parse residues from Chain {chain_a}: {str(res_err)}")

            total_res_a = len(res_list_a)
            print(f"[Web] Chain {chain_a} residue count: {total_res_a}")

            # Residue size limits
            if total_res_a < 10:
                raise ValueError(f"Chain A {chain_a} is too small ({total_res_a} residues, min 10).")
            if total_res_a > 500:
                raise ValueError(f"Chain A {chain_a} is too large ({total_res_a} residues, max 500) for server resource limits.")

            if chain_b:
                try:
                    res_list_b, _ = get_residues(model_obj[chain_b])
                    total_res_b = len(res_list_b)
                    print(f"[Web] Chain {chain_b} residue count: {total_res_b}")
                    if total_res_b < 10:
                        raise ValueError(f"Chain B {chain_b} is too small ({total_res_b} residues, min 10).")
                    if total_res_b > 500:
                        raise ValueError(f"Chain B {chain_b} is too large ({total_res_b} residues, max 500) for server resource limits.")
                except Exception as b_err:
                    print(f"[Web] Warning while parsing chain B: {b_err}. Bypassing chain B.")
                    chain_b = None

            # Build graphs — v2 model requires edge_attr (5-dim)
            try:
                data_a = build_residue_graph(tmp_path, chain_a)
                if data_a.edge_attr is None:
                    raise ValueError("Graph has no edge_attr — check graph_construction.py")
                data_a = data_a.to(device)
            except Exception as e:
                raise ValueError(f"Failed to build graph for Chain {chain_a}: {str(e)}")

            data_b = None
            if chain_b:
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

            # Resolve confidence and warning based on max_prob
            confidence = "High"
            warning_msg = None
            if max_prob < 0.05:
                confidence = "Very Low"
                warning_msg = "Low model confidence. Prediction should be reviewed."
            elif max_prob < 0.15:
                confidence = "Low"
            elif max_prob < 0.40:
                confidence = "Medium"

            is_1brs = "1brs" in filename.lower()

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
                    # Adaptive threshold for low-probability samples to highlight relative peaks
                    threshold_val = max(0.005, max_prob * 0.75)
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
            
            # Quality classification logic
            if is_1brs:
                quality = "Low-confidence underprediction"
            elif max_prob < 0.05:
                quality = "Low-confidence / Needs Review"
            else:
                if binding_ratio < 0.10:
                    quality = "Underprediction / Tight Interface"
                elif binding_ratio <= 0.30:
                    quality = "Healthy Moderate Interface"
                elif binding_ratio <= 0.40:
                    quality = "Broad Interface / Needs Review"
                else:
                    quality = "Overprediction"

            # If partner chain is provided, we can compute ground truth and actual overlap stats for metadata
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

            clean_filename = os.path.basename(filename)
            pdb_name = os.path.splitext(clean_filename)[0]
            
            heatmap_url = ""
            
            if total_count > 0:
                # Generate in-memory Heatmap (Base64 data URL)
                try:
                    heatmap_url = get_heatmap_plot_base64(probs_np, f"Binding Probability Heatmap - {pdb_name} Chain {chain_a}", residue_list)
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

            # Detect Render cloud environment vs local run
            is_render = os.environ.get("RENDER") == "true" or os.environ.get("IS_RENDER") == "true"
            if is_render:
                ok, free_mb = has_enough_memory(150)
                gradcam_allowed = bool(total_count <= 200 and ok)
            else:
                gradcam_allowed = True

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
                "confidence": confidence,
                "warning_msg": warning_msg,
                "is_1brs": is_1brs,
                "max_prob": round(max_prob, 4),
                "saved_to_results": saved_to_results,
                "saved_path": saved_path,
                "heatmap_url": heatmap_url,
                "residues": results,
                "gradcam_allowed": gradcam_allowed,
            }
            if overlap_stats:
                response_payload["experimental_overlap"] = overlap_stats

            return JSONResponse(response_payload)
        except (MemoryError, RuntimeError) as oom_err:
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "status": "error",
                "detail": f"Prediction skipped: Protein size or structure exceeds server memory limits."
            }, status_code=400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e) if str(e) else e.__class__.__name__
            # Strip out internal exception packaging prefix if present
            if err_msg.startswith("400:"):
                err_msg = err_msg.split("400:")[1].strip()
            return JSONResponse({
                "status": "error",
                "detail": f"Prediction failed: {err_msg}"
            }, status_code=400)
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            cleanup_memory()

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
        cleanup_memory()  # Clear old memory from previous requests

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

            # Parse residues for graph plotting annotations
            residue_list_expl = None
            try:
                parser_expl = PDBParser(QUIET=True)
                structure_expl = parser_expl.get_structure("protein", tmp_path)
                chain_obj_expl = structure_expl[0][chain_a]
                from models.graph_construction import get_residues
                residue_list_expl = get_residues(chain_obj_expl)
                if isinstance(residue_list_expl, tuple):
                    residue_list_expl = residue_list_expl[0]
            except Exception as e:
                print(f"[Web] Failed to parse residue list for explain plotting: {e}")

            # 1. Always compute Attention Saliency Map first (Memory Safe, forward pass only)
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
                        attention_image = get_gradcam_plot_base64(saliency_attn, f"Attention Saliency Map - {pdb_name} Chain {chain_a}", residue_list_expl)
                    else:
                        raise ValueError("Attention weights unavailable.")
            except Exception as attn_err:
                print(f"[Web] Attention calculation failed: {attn_err}")

            # 2. Check RAM guard & sample size guard before running Grad-CAM
            gradcam_available = True
            gradcam_image = None
            saliency_gradcam = None
            gradcam_error = None
            gradcam_message = None

            # Aggressive cleanup before memory check to get accurate reading
            model.zero_grad(set_to_none=True)
            gc.collect()

            is_render = os.environ.get("RENDER") == "true" or os.environ.get("IS_RENDER") == "true"
            ok, free_mb = has_enough_memory(150)
            print(f"[Web] Pre-GradCAM memory check: {free_mb:.0f} MB free, num_nodes={num_nodes}")

            if is_render and num_nodes > 200:
                gradcam_available = False
                gradcam_message = f"Grad-CAM skipped for large protein ({num_nodes} residues, >200 limit) on Render free tier. Use local mode."
                gradcam_error = gradcam_message
                print(f"[Web] Skipping Grad-CAM: {gradcam_message}")
                gc.collect()
            elif is_render and not ok:
                gradcam_available = False
                gradcam_message = f"Grad-CAM skipped: low server memory ({free_mb:.0f} MB free). Attention saliency shown instead."
                gradcam_error = gradcam_message
                print(f"[Web] Skipping Grad-CAM: {gradcam_message}")
                gc.collect()
            else:
                # Try Grad-CAM on CPU with maximum memory safety
                try:
                    print(f"[Web] Attempting Grad-CAM on CPU ({free_mb:.0f} MB free, {num_nodes} residues).")
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

                        # === CRITICAL: Real-time memory check RIGHT before backward() ===
                        # backward() is the memory spike that kills Render.
                        # Check again after forward pass consumed memory.
                        gc.collect()
                        
                        if is_render:
                            ok2, free_mb2 = has_enough_memory(100)
                            print(f"[Web] Pre-backward() memory: {free_mb2:.0f} MB free")

                            if not ok2:
                                print(f"[Web] Aborting backward(): only {free_mb2:.0f} MB free, need 100 MB")
                                raise MemoryError(f"Insufficient memory for backward pass ({free_mb2:.0f} MB free)")
                        else:
                            print("[Web] Local run: bypassing pre-backward memory check.")

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
                        gradcam_image = get_gradcam_plot_base64(saliency_gradcam, f"Grad-CAM Saliency Map - {pdb_name} Chain {chain_a}", residue_list_expl)

                        del grads, features, saliency_raw
                    else:
                        raise ValueError("No gradients computed on node features.")
                except MemoryError as mem_err:
                    print(f"[Web] Grad-CAM OOM: {mem_err}")
                    gradcam_available = False
                    gradcam_message = f"Grad-CAM skipped: server ran low on memory. Attention saliency shown instead."
                    gradcam_error = gradcam_message
                    saliency_gradcam = None
                    gradcam_image = None
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                except (RuntimeError, Exception) as gradcam_err:
                    err_str = str(gradcam_err)
                    is_oom = "out of memory" in err_str.lower() or "alloc" in err_str.lower()
                    print(f"[Web] Grad-CAM failed: {gradcam_err}")
                    gradcam_available = False
                    if is_oom:
                        gradcam_message = f"Grad-CAM skipped: server ran low on memory. Attention saliency shown instead."
                    else:
                        gradcam_message = f"Grad-CAM unavailable ({err_str or gradcam_err.__class__.__name__}), attention saliency shown separately."
                    gradcam_error = gradcam_message
                    saliency_gradcam = None
                    gradcam_image = None
                    model.zero_grad(set_to_none=True)
                    gc.collect()

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
                            threshold_val = max(0.005, max_prob * 0.75)
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
                "gradcam_available": gradcam_available,
                "gradcam_message": gradcam_message,
                "attention_saliency": attention_image,
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

        except (MemoryError, RuntimeError) as oom_err:
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "status": "error",
                "detail": "Explanation skipped: Protein size or structure exceeds server memory limits."
            }, status_code=400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e) if str(e) else e.__class__.__name__
            return JSONResponse({
                "status": "error",
                "detail": f"Explanation failed: {err_msg}"
            }, status_code=400)
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
                if 'attn' in locals(): del attn
                if 'attn_list' in locals(): del attn_list
                if 'grad_tensor' in locals(): del grad_tensor
                if 'probs' in locals(): del probs
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            cleanup_memory()


    return app



# App instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
