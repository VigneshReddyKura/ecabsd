import json

NB = "Clean_Kaggle_Training.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 5 to REMOVE the fallback that moves bad graphs back
code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
cell5_idx = code_cells[4]

new_cell5 = [
    "# ============================================================\n",
    "# CELL 5: Remove bad graphs — user's exact code\n",
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
    "print('Ready for recovery step.')\n"
]

nb["cells"][cell5_idx]["source"] = new_cell5

# Update Cell 6 (Config) to force x=33 and edge=5, since we recover them to these dimensions
cell6_idx = code_cells[6]  # The config update cell
new_cell6 = [
    "# ============================================================\n",
    "# CELL 6: Update config.yaml with final dimensions + verify\n",
    "# ============================================================\n",
    "import yaml, os, pandas as pd\n",
    "\n",
    "with open('config.yaml', 'r') as f:\n",
    "    cfg = yaml.safe_load(f)\n",
    "\n",
    "cfg['data']['processed_dir']      = 'data/processed'\n",
    "cfg['data']['splits_csv']         = 'data/splits.csv'\n",
    "cfg['model']['esm_dim']           = 33      # match actual node feature dim\n",
    "cfg['model']['edge_feature_dim']  = 5   # match actual edge feature dim\n",
    "cfg['training']['epochs']         = 100\n",
    "cfg['training']['num_workers']    = 0\n",
    "\n",
    "with open('config.yaml', 'w') as f:\n",
    "    yaml.dump(cfg, f, default_flow_style=False)\n",
    "\n",
    "# Print verification\n",
    "final_cfg = yaml.safe_load(open('config.yaml'))\n",
    "print('Config [data] section after update:')\n",
    "print(f\"  processed_dir  : {final_cfg['data']['processed_dir']}\")\n",
    "print(f\"  splits_csv     : {final_cfg['data']['splits_csv']}\")\n",
    "print()\n",
    "print('Config [model] section after update:')\n",
    "print(f\"  esm_dim        : {final_cfg['model'].get('esm_dim')}\")\n",
    "print(f\"  edge_feature_dim: {final_cfg['model'].get('edge_feature_dim')}\")\n",
    "print(f\"  hidden_dim     : {final_cfg['model'].get('hidden_dim')}\")\n",
    "print()\n",
    "\n",
    "df = pd.read_csv(final_cfg['data']['splits_csv'])\n",
    "vc = df['split'].value_counts()\n",
    "print('Usable sample counts:')\n",
    "print(f\"  train : {vc.get('train', 0)}\")\n",
    "print(f\"  val   : {vc.get('val',   0)}\")\n",
    "print(f\"  test  : {vc.get('test',  0)}\")\n",
    "print()\n",
    "\n",
    "t = set(df[df['split']=='train']['pdb_id'])\n",
    "v = set(df[df['split']=='val']['pdb_id'])\n",
    "e = set(df[df['split']=='test']['pdb_id'])\n",
    "print('Leakage check:')\n",
    "print(f'  Train-Val overlap  : {len(t & v)}')\n",
    "print(f'  Train-Test overlap : {len(t & e)}')\n",
    "print(f'  Val-Test overlap   : {len(v & e)}')\n",
    "print()\n",
    "\n",
    "assert vc.get('train', 0) >= 1000, f'Too few train samples: {vc.get(\"train\",0)}'\n",
    "assert vc.get('val',   0) >= 100,  f'Too few val samples: {vc.get(\"val\",0)}'\n",
    "assert len(t & v) == 0, 'LEAKAGE: train/val'\n",
    "assert len(t & e) == 0, 'LEAKAGE: train/test'\n",
    "assert len(v & e) == 0, 'LEAKAGE: val/test'\n",
    "\n",
    "print('All checks passed — safe to train!')",
]

nb["cells"][cell6_idx]["source"] = new_cell6

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook prepared.")
