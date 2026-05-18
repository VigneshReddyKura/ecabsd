"""
Filter DIPS Dataset using MMSeqs2 (Strict 30% Sequence Identity Leakage Removal).

This script compares all chains in the newly generated DIPS splits against
the DB5 benchmark splits. Any DIPS complex containing a chain that shares
>30% sequence identity with ANY chain in the DB5 test set is completely removed.

Prerequisites:
- mmseqs must be installed and available in your PATH.
  (e.g., conda install -c conda-forge -c bioconda mmseqs2)
"""

import os
import csv
import subprocess
import argparse
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from tqdm import tqdm

def get_sequence_from_pdb(pdb_path, chain_id):
    """Extract amino acid sequence from PDB chain."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        chain = model[chain_id]
        
        seq = ""
        for r in chain:
            if is_aa(r, standard=True):
                try:
                    seq += three_to_one(r.get_resname())
                except KeyError:
                    seq += "X"
        return seq
    except Exception:
        return ""

def create_fasta(splits_csv, pdb_dir, output_fasta):
    """Extract sequences from a splits CSV and write to a FASTA file."""
    if not os.path.exists(splits_csv):
        print(f"File not found: {splits_csv}")
        return {}

    seq_map = {}
    with open(splits_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Extracting {os.path.basename(splits_csv)}"):
            pdb_id = row["pdb_id"]
            chain_a = row["chain_a"]
            
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_path):
                # Try finding matching PDB (in case of naming variations like _l_u.pdb)
                import glob
                matches = glob.glob(os.path.join(pdb_dir, f"{pdb_id}*.pdb"))
                if matches:
                    pdb_path = matches[0]
                else:
                    continue

            seq = get_sequence_from_pdb(pdb_path, chain_a)
            if seq and len(seq) > 20:
                header = f"{pdb_id}_{chain_a}"
                seq_map[header] = seq

    with open(output_fasta, "w") as f:
        for header, seq in seq_map.items():
            f.write(f">{header}\n{seq}\n")
            
    print(f"Wrote {len(seq_map)} sequences to {output_fasta}")
    return seq_map

def run_mmseqs_filter(dips_splits, dips_pdb_dir, db5_splits, db5_pdb_dir, output_csv, threshold=0.30):
    tmp_dir = "tmp_mmseqs"
    os.makedirs(tmp_dir, exist_ok=True)
    
    dips_fasta = os.path.join(tmp_dir, "dips.fasta")
    db5_fasta = os.path.join(tmp_dir, "db5.fasta")
    
    print("1. Creating FASTA for DB5...")
    create_fasta(db5_splits, db5_pdb_dir, db5_fasta)
    
    print("2. Creating FASTA for DIPS...")
    create_fasta(dips_splits, dips_pdb_dir, dips_fasta)
    
    tsv_out = os.path.join(tmp_dir, "search_results.tsv")
    
    print(f"3. Running MMSeqs2 easy-search (Identity Threshold: {threshold*100}%)...")
    # Columns: query, target, sequence identity, alignment length, mismatches, gap openings, q. start, q. end, t. start, t. end, e-value, bit score
    cmd = [
        "mmseqs", "easy-search",
        dips_fasta, db5_fasta,
        tsv_out, os.path.join(tmp_dir, "mmseqs_tmp"),
        "--min-seq-id", str(threshold),
        "-c", "0.8", # 80% coverage required to count as a leak
        "--cov-mode", "1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] mmseqs command not found. Please install MMSeqs2 first.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MMSeqs2 failed: {e}")
        return

    # Parse leaked chains
    leaked_chains = set()
    if os.path.exists(tsv_out):
        with open(tsv_out, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    dips_id = parts[0] # pdbid_chain
                    leaked_chains.add(dips_id)
                    
    print(f"Found {len(leaked_chains)} individual DIPS chains that leak into DB5.")
    
    # We must drop the ENTIRE complex if ANY chain inside it leaks.
    leaked_pdbs = set([x.split("_")[0] for x in leaked_chains])
    print(f"This contaminates {len(leaked_pdbs)} whole PDB complexes.")

    print(f"4. Generating cleaned splits CSV: {output_csv}")
    kept_rows = []
    dropped_count = 0
    with open(dips_splits, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pdb_id"] in leaked_pdbs:
                dropped_count += 1
            else:
                kept_rows.append(row)
                
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        
    print(f"\n{'='*50}")
    print("  LEAKAGE FILTERING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total DIPS Chains:     {dropped_count + len(kept_rows)}")
    print(f"  Leaked Chains Dropped: {dropped_count}")
    print(f"  Clean Chains Kept:     {len(kept_rows)}")
    print(f"  Output saved to:       {output_csv}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter DIPS dataset against DB5 using MMSeqs2")
    parser.add_argument("--dips-splits", default="data/processed/splits.csv", help="Splits CSV for DIPS")
    parser.add_argument("--dips-pdb-dir", default="data/raw/dips", help="Raw DIPS PDBs")
    parser.add_argument("--db5-splits", default="data/processed_db5/splits.csv", help="Splits CSV for DB5")
    parser.add_argument("--db5-pdb-dir", default="data/raw/db5", help="Raw DB5 PDBs")
    parser.add_argument("--output-csv", default="data/processed/splits_cleaned.csv", help="Output leakage-free splits")
    parser.add_argument("--threshold", type=float, default=0.30, help="Sequence identity threshold (0-1)")
    args = parser.parse_args()

    run_mmseqs_filter(args.dips_splits, args.dips_pdb_dir, args.db5_splits, args.db5_pdb_dir, args.output_csv, args.threshold)
