"""
Patches Cell 4 (Dataset Preparation) in Kaggle_Training_V2.ipynb.
Run once: python patch_notebook.py
"""
import json, pathlib

NB_PATH = pathlib.Path("Kaggle_Training_V2.ipynb")

new_cell4_source = [
    "import os\n",
    "import glob\n",
    "import shutil\n",
    "\n",
    "# ── Normalise folder names so config.yaml can find the data ─────────────────\n",
    "# Your local processing saves to 'data/processed' and 'data/splits.csv'.\n",
    "# The config expects 'data/db5_processed' and 'data/db5_splits.csv'.\n",
    "if os.path.exists('data/processed') and not os.path.exists('data/db5_processed'):\n",
    "    print('Renaming data/processed -> data/db5_processed ...')\n",
    "    os.rename('data/processed', 'data/db5_processed')\n",
    "\n",
    "if os.path.exists('data/splits.csv') and not os.path.exists('data/db5_splits.csv'):\n",
    "    shutil.copy('data/splits.csv', 'data/db5_splits.csv')\n",
    "    print('Copied splits.csv -> db5_splits.csv')\n",
    "\n",
    "PROCESSED_DIR = 'data/db5_processed'\n",
    "pt_files = glob.glob(f'{PROCESSED_DIR}/*.pt')\n",
    "\n",
    "# ── Summary of what we found ────────────────────────────────────────────────\n",
    "print('=' * 55)\n",
    "print(f'Processed graphs found : {len(pt_files)}')\n",
    "if os.path.exists('data/db5_splits.csv'):\n",
    "    import pandas as pd\n",
    "    splits = pd.read_csv('data/db5_splits.csv')\n",
    "    print(f'Split file rows        : {len(splits)}')\n",
    "    if 'split' in splits.columns:\n",
    "        print(splits['split'].value_counts().to_string())\n",
    "print('=' * 55)\n",
    "\n",
    "# ── Only run DB5 fallback if data is genuinely missing ──────────────────────\n",
    "if len(pt_files) > 10:\n",
    "    print('\\u2705 Your full dataset is ready!  Proceeding to training.')\n",
    "else:\n",
    "    print('\\u26a0\\ufe0f No .pt files found. Falling back to DB5 (230 complexes)...')\n",
    "    if not os.path.exists('data/BM5-clean'):\n",
    "        os.system('git clone https://github.com/haddocking/BM5-clean.git data/BM5-clean')\n",
    "    os.system('python scripts/prepare_db5.py '\n",
    "              '--db5-dir data/BM5-clean/HADDOCK-ready '\n",
    "              '--output-dir data/db5_processed '\n",
    "              '--threads 2')\n",
    "    print('\\u2705 DB5 fallback preparation complete.')"
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Find the right cell (Step 4 — the one with "db5_processed")
patched = False
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and "db5_processed" in src:
        nb["cells"][i]["source"] = new_cell4_source
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        patched = True
        print(f"✅ Patched cell index {i}")
        break

if not patched:
    print("❌ Could not find the target cell. Check the notebook manually.")
else:
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("✅ Notebook saved successfully!")
