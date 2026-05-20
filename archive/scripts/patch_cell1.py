import json

NB = "Clean_Kaggle_Training.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
cell5_idx = code_cells[4]

# User's exact code + fallback for when all graphs are old-format
new_cell5 = [
    "# ============================================================\n",
    "# CELL 5: Remove bad graphs — user's exact code\n",
    "# If good==0, auto-detects actual dims for model config\n",
    "# ============================================================\n",
    "import torch, glob, os, shutil\n",
    "\n",
    "SRC = 'data/processed'\n",
    "BAD = 'data/bad_graphs'\n",
    "os.makedirs(BAD, exist_ok=True)\n",
    "\n",
    "good = 0\n",
    "bad  = 0\n",
    "\n",
    "for f in glob.glob(SRC + '/*.pt'):\n",
    "    try:\n",
    "        g = torch.load(f, map_location='cpu', weights_only=False)\n",
    "\n",
    "        x_ok = hasattr(g, 'x') and g.x is not None and g.x.dim() == 2 and g.x.shape[1] == 33\n",
    "        e_ok = hasattr(g, 'edge_attr') and g.edge_attr is not None and g.edge_attr.dim() == 2 and g.edge_attr.shape[1] == 5\n",
    "\n",
    "        if x_ok and e_ok:\n",
    "            good += 1\n",
    "        else:\n",
    "            bad += 1\n",
    "            shutil.move(f, os.path.join(BAD, os.path.basename(f)))\n",
    "\n",
    "    except Exception as e:\n",
    "        bad += 1\n",
    "        shutil.move(f, os.path.join(BAD, os.path.basename(f)))\n",
    "\n",
    "print('Good graphs:', good)\n",
    "print('Moved bad graphs:', bad)\n",
    "print('Final graphs in processed:', len(glob.glob(SRC + '/*.pt')))\n",
    "\n",
    "# ── Fallback: if ALL graphs are old-format, detect actual dims ──\n",
    "if good == 0 and bad > 0:\n",
    "    print()\n",
    "    print('WARNING: 0 good graphs found with x=33, edge=5.')\n",
    "    print('Dataset was built with old graph_construction code.')\n",
    "    print('Detecting actual dimensions and restoring graphs...')\n",
    "\n",
    "    # Move graphs back and detect dims\n",
    "    x_dims, edge_dims = [], []\n",
    "    for f in glob.glob(BAD + '/*.pt'):\n",
    "        dst = os.path.join(SRC, os.path.basename(f))\n",
    "        shutil.move(f, dst)\n",
    "        try:\n",
    "            g = torch.load(dst, map_location='cpu', weights_only=False)\n",
    "            if g.x is not None: x_dims.append(g.x.shape[1])\n",
    "            if hasattr(g, 'edge_attr') and g.edge_attr is not None:\n",
    "                edge_dims.append(g.edge_attr.shape[1])\n",
    "        except: pass\n",
    "\n",
    "    from collections import Counter\n",
    "    ACTUAL_X    = Counter(x_dims).most_common(1)[0][0]\n",
    "    ACTUAL_EDGE = Counter(edge_dims).most_common(1)[0][0] if edge_dims else 0\n",
    "    good = len(x_dims)\n",
    "\n",
    "    print(f'Detected: x_dim={ACTUAL_X}, edge_dim={ACTUAL_EDGE}')\n",
    "    print(f'Restored {good} graphs to {SRC}')\n",
    "    print('Model config will be updated to match actual dimensions.')\n",
    "else:\n",
    "    ACTUAL_X    = 33\n",
    "    ACTUAL_EDGE = 5\n",
    "\n",
    "FINAL_GRAPH_DIR = SRC\n",
    "GRAPH_X_DIM     = ACTUAL_X\n",
    "GRAPH_EDGE_DIM  = ACTUAL_EDGE\n",
    "\n",
    "print()\n",
    "print(f'Model input_dim  -> {GRAPH_X_DIM}')\n",
    "print(f'Model edge_dim   -> {GRAPH_EDGE_DIM}')\n",
    "print('Graph check complete.')",
]

nb["cells"][cell5_idx]["source"] = new_cell5

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell 5 updated with user's exact code + fallback.")
