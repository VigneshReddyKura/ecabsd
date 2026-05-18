#!/bin/bash
# ECABSD Kaggle Pipeline Runner
# This script is designed to be executed inside a Kaggle Notebook environment.
# It assumes you have uploaded your raw DIPS PDBs as a Kaggle Dataset.

set -e

echo "=================================================="
echo "    ECABSD Kaggle V3 Training Pipeline"
echo "=================================================="

# 1. Install Dependencies
echo "[1/5] Installing Dependencies..."
pip install -q transformers sentence-transformers
pip install -q torch-geometric
conda install -c conda-forge -c bioconda mmseqs2 -y

# 2. Extract Data (Assuming DB5 is downloaded or uploaded)
echo "[2/5] Preparing DB5 Benchmark..."
python scripts/download_benchmarks.py --output-dir /kaggle/working/db5_raw
python scripts/prepare_dataset.py \
    --pdb-dir /kaggle/working/db5_raw \
    --output-dir /kaggle/working/db5_graphs \
    --cutoff 5.0 \
    --threads 4

# 3. Process DIPS (Assuming DIPS uploaded to /kaggle/input/dips-dataset)
# Note: Change the --pdb-dir below to the actual name of your Kaggle input dataset.
DIPS_RAW_DIR="/kaggle/input/dips-dataset" 

echo "[3/5] Processing DIPS with Rigorous Filters..."
python scripts/prepare_kaggle_dips.py \
    --pdb-dir $DIPS_RAW_DIR \
    --output-dir /kaggle/working/dips_graphs \
    --cutoff 5.0 \
    --threads 4

# 4. Strict Leakage Removal
echo "[4/5] Running MMSeqs2 30% Leakage Filter..."
python scripts/filter_dips_mmseqs.py \
    --dips-splits /kaggle/working/dips_graphs/splits.csv \
    --dips-pdb-dir $DIPS_RAW_DIR \
    --db5-splits /kaggle/working/db5_graphs/splits.csv \
    --db5-pdb-dir /kaggle/working/db5_raw \
    --output-csv /kaggle/working/dips_graphs/splits_cleaned.csv \
    --threshold 0.30

# 5. Train V3 Model
echo "[5/5] Launching GPU Training..."
python train_v3.py \
    --graph-dir /kaggle/working/dips_graphs \
    --splits /kaggle/working/dips_graphs/splits_cleaned.csv \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4

echo "=================================================="
echo "    Pipeline Finished Successfully! "
echo "=================================================="
