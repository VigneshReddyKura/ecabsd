"""
generate_homology_splits.py
===========================
Generates homology-aware train/val/test splits for ECABSD using MMseqs2
sequence clustering at a configurable identity threshold (default: 30%).

This ensures no homologous protein sequences appear across splits, which
is the primary concern for data leakage in structural bioinformatics.

Usage
-----
    # Requires MMseqs2 installed: conda install -c bioconda mmseqs2
    python scripts/generate_homology_splits.py \
        --splits data/splits.csv \
        --pdb-dir data/raw/pdbs \
        --output data/splits_homology.csv \
        --identity 0.30 \
        --coverage 0.80

Output
------
    data/splits_homology.csv  — same format as splits.csv but with
                                homology-aware partition labels.

Publication Note
----------------
    Report metrics on splits_homology.csv for peer-reviewed submissions.
    The --identity 0.30 threshold is the standard for structural biology
    benchmarks (see MaSIF, PIPER, CrossPPI).
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ─────────────────────────────────────────────────────────────────────────────
# PDB → FASTA extraction
# ─────────────────────────────────────────────────────────────────────────────

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
    """Extract one-letter amino acid sequence from a PDB file for a given chain."""
    seq = []
    seen_residues = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[21].strip() == chain_id:
                    res_name = line[17:20].strip()
                    res_seq  = line[22:26].strip()
                    ins_code = line[26].strip()
                    res_key  = (res_seq, ins_code)
                    if res_key not in seen_residues and res_name in THREE_TO_ONE:
                        seen_residues.add(res_key)
                        seq.append(THREE_TO_ONE[res_name])
    except Exception as e:
        print(f"  [WARN] Could not parse {pdb_path} chain {chain_id}: {e}")
    return "".join(seq)


# ─────────────────────────────────────────────────────────────────────────────
# MMseqs2 clustering
# ─────────────────────────────────────────────────────────────────────────────

def check_mmseqs2() -> bool:
    """Return True if mmseqs2 is available in PATH."""
    return shutil.which("mmseqs") is not None


def run_mmseqs2_clustering(
    fasta_path: str,
    identity: float = 0.30,
    coverage: float = 0.80,
    threads: int = 4,
) -> Dict[str, str]:
    """
    Cluster sequences using MMseqs2 easy-cluster.

    Returns
    -------
    dict mapping sequence_id → cluster_representative_id
    """
    tmpdir = tempfile.mkdtemp(prefix="mmseqs2_ecabsd_")
    cluster_prefix = os.path.join(tmpdir, "clusters")
    tmp_prefix     = os.path.join(tmpdir, "tmp")

    try:
        cmd = [
            "mmseqs", "easy-cluster",
            fasta_path,
            cluster_prefix,
            tmp_prefix,
            "--min-seq-id",  str(identity),
            "--coverage",    str(coverage),
            "--cov-mode",    "0",
            "--cluster-mode", "0",
            "--threads",     str(threads),
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)

        # Parse cluster TSV: rep_id \t member_id
        cluster_map: Dict[str, str] = {}
        tsv_path = cluster_prefix + "_cluster.tsv"
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep, member = parts
                    cluster_map[member] = rep

        return cluster_map

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def fallback_random_clustering(
    seq_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Fallback when MMseqs2 is not available.
    Assigns each seq_id a cluster = itself (no homology removal),
    then does a random split. Warns clearly.
    """
    print("""
  [WARNING] MMseqs2 not found — falling back to RANDOM splitting.
  This does NOT remove homologous sequences across splits.
  For publication-quality results, install MMseqs2:

      conda install -c bioconda mmseqs2

  Then re-run this script.
""")
    rng = random.Random(seed)
    ids = list(seq_ids)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    split_map = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    return split_map


# ─────────────────────────────────────────────────────────────────────────────
# Split assignment from clusters
# ─────────────────────────────────────────────────────────────────────────────

def assign_splits_from_clusters(
    cluster_map: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Assign train/val/test splits at the cluster level (not the complex level).
    All members of a cluster go to the same split — this prevents leakage.
    """
    # Get unique cluster representatives
    reps = sorted(set(cluster_map.values()))
    rng  = random.Random(seed)
    rng.shuffle(reps)

    n_train = int(len(reps) * train_ratio)
    n_val   = int(len(reps) * val_ratio)

    rep_to_split: Dict[str, str] = {}
    for i, rep in enumerate(reps):
        if i < n_train:
            rep_to_split[rep] = "train"
        elif i < n_train + n_val:
            rep_to_split[rep] = "val"
        else:
            rep_to_split[rep] = "test"

    # Map each member → split via its representative
    member_to_split: Dict[str, str] = {}
    for member, rep in cluster_map.items():
        member_to_split[member] = rep_to_split.get(rep, "test")

    return member_to_split


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate homology-aware train/val/test splits using MMseqs2."
    )
    parser.add_argument("--splits",    required=True,
                        help="Input splits CSV (data/splits.csv)")
    parser.add_argument("--pdb-dir",   required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--output",    default="data/splits_homology.csv",
                        help="Output homology-aware splits CSV")
    parser.add_argument("--identity",  type=float, default=0.30,
                        help="MMseqs2 sequence identity threshold (default: 0.30)")
    parser.add_argument("--coverage",  type=float, default=0.80,
                        help="MMseqs2 coverage threshold (default: 0.80)")
    parser.add_argument("--threads",   type=int,   default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("ERROR: pandas required. pip install pandas")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ECABSD — Homology-Aware Split Generator")
    print(f"  Identity threshold: {args.identity*100:.0f}%")
    print(f"  Coverage threshold: {args.coverage*100:.0f}%")
    print(f"{'='*60}\n")

    # Load existing splits CSV to get PDB IDs and chains
    df = pd.read_csv(args.splits)
    print(f"  Loaded {len(df)} complexes from {args.splits}")

    # Extract sequences for chain A of each complex
    print(f"\n  Extracting sequences from PDB files...")
    records = []
    skipped = 0
    for _, row in df.iterrows():
        pdb_id   = str(row["pdb_id"]).upper()
        chain_a  = str(row.get("chain_a", "A"))
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_path):
            # Try lowercase
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_id.lower()}.pdb")

        if not os.path.exists(pdb_path):
            skipped += 1
            continue

        seq = extract_sequence_from_pdb(pdb_path, chain_a)
        if len(seq) >= 10:
            records.append((pdb_id, seq))

    print(f"  Extracted {len(records)} sequences ({skipped} PDB files not found)")

    if len(records) == 0:
        print("ERROR: No sequences extracted. Check --pdb-dir path.")
        sys.exit(1)

    # Write FASTA
    tmpdir    = tempfile.mkdtemp(prefix="ecabsd_splits_")
    fasta_out = os.path.join(tmpdir, "sequences.fasta")
    with open(fasta_out, "w") as f:
        for pdb_id, seq in records:
            f.write(f">{pdb_id}\n{seq}\n")
    print(f"  Written {len(records)} sequences to {fasta_out}")

    # Run MMseqs2 or fallback
    seq_ids = [r[0] for r in records]

    if check_mmseqs2():
        print(f"\n  MMseqs2 found. Running clustering...")
        cluster_map = run_mmseqs2_clustering(
            fasta_out,
            identity=args.identity,
            coverage=args.coverage,
            threads=args.threads,
        )
        n_clusters = len(set(cluster_map.values()))
        print(f"  Clustered {len(cluster_map)} sequences into {n_clusters} clusters "
              f"at {args.identity*100:.0f}% identity")

        member_to_split = assign_splits_from_clusters(
            cluster_map,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        member_to_split = fallback_random_clustering(
            seq_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Build output dataframe
    pdb_to_split = {pdb_id: member_to_split.get(pdb_id, "test")
                    for pdb_id, _ in records}

    df_out = df.copy()
    df_out["split"] = df_out["pdb_id"].apply(
        lambda x: pdb_to_split.get(str(x).upper(), "unassigned")
    )

    # Report split sizes
    counts = df_out["split"].value_counts()
    print(f"\n  Split sizes:")
    for split in ["train", "val", "test", "unassigned"]:
        n = counts.get(split, 0)
        print(f"    {split:12s}: {n:5d} ({100*n/len(df_out):.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n  Saved homology-aware splits to: {args.output}")
    print(f"\n  NEXT STEP: Retrain with --splits {args.output}")
    print(f"  Then evaluate and report metrics for peer review.\n")


if __name__ == "__main__":
    main()
