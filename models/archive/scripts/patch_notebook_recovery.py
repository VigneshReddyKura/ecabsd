import json

NB = "Clean_Kaggle_Training.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 5 and insert a new Cell 6 for graph recovery right after it
code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
cell5_idx = code_cells[4]

new_cell_recover = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# CELL 5.5: Auto-Recover Bad Graphs\n",
        "# Downloads raw PDBs and rebuilds missing/bad graphs with x=33\n",
        "# ============================================================\n",
        "import sys, subprocess\n",
        "print('[RECOVERY] Starting graph recovery script...')\n",
        "subprocess.run([sys.executable, 'scripts/recover_graphs.py'])\n",
        "print('[RECOVERY] Done.')\n"
    ]
}

nb["cells"].insert(cell5_idx + 1, new_cell_recover)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Inserted recovery cell into the notebook.")
