"""
Prepare Kaggle DIPS Dataset (Rigorous Quality Filters).

Processes raw DIPS PDB files into PyTorch Geometric graph objects, applying
strict structural sanity filters before generating 5.0Å ground truth labels.

Filters:
- Chains < 30 residues or > 512 residues are discarded.
- Chains with missing C-alpha coordinates are discarded.
- Self-pairs (same chain) are ignored.
- Corrupt PDB files are skipped.
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
import torch
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_construction import build_residue_graph, compute_binding_labels
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

def validate_chain(chain):
    """
    Validates a PDB chain based on rigorous criteria:
    1. Only standard Amino Acids.
    2. Length between 30 and 512 residues.
    3. No missing C-alpha coordinates.
    """
    residues = [r for r in chain if is_aa(r, standard=True)]
    
    if len(residues) < 30:
        return False, f"Too short ({len(residues)} res)"
    if len(residues) > 512:
        return False, f"Too long ({len(residues)} res)"
        
    for r in residues:
        if "CA" not in r:
            return False, f"Missing CA atom in residue {r.get_id()}"
            
    return True, residues

def process_single_pdb(pdb_path, output_dir, distance_cutoff=5.0):
    """Worker function to process all valid chains in a single DIPS PDB file."""
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    results = []
    errors = []
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        chains = list(model)
        
        valid_chains = {}
        for chain in chains:
            is_valid, data = validate_chain(chain)
            if is_valid:
                valid_chains[chain.id] = data
            else:
                errors.append(f"{pdb_name}_{chain.id} skipped: {data}")
                
        if len(valid_chains) < 2:
            errors.append(f"{pdb_name} skipped: Less than 2 valid chains for a complex.")
            return results, errors

        # Process pairs
        chain_ids = list(valid_chains.keys())
        for i, chain_a_id in enumerate(chain_ids):
            # We treat the first other valid chain as the partner for simplicity,
            # but DIPS complexes can be multimeric. We'll label against ALL other valid chains.
            partner_chains = [c_id for c_id in chain_ids if c_id != chain_a_id]
            
            # Combine all partner chains into one giant string for tracking
            partner_str = "_".join(partner_chains)
            
            save_path = os.path.join(output_dir, f"{pdb_name}_{chain_a_id}.pt")
            
            if os.path.exists(save_path):
                try:
                    graph = torch.load(save_path, map_location="cpu")
                    results.append({
                        "pdb_id": pdb_name,
                        "chain_a": chain_a_id,
                        "chain_b": partner_str,
                        "num_residues": len(graph.x),
                        "num_binding": int(graph.y.sum()),
                    })
                    continue
                except:
                    pass

            try:
                graph = build_residue_graph(pdb_path, chain_a_id)
                # Compute labels against ALL valid partner chains simultaneously
                labels = compute_binding_labels(pdb_path, chain_a_id, None, distance_cutoff)
                
                # Double check that we actually parsed the right number of labels
                if len(labels) != len(graph.x):
                    raise ValueError(f"Label mismatch: {len(labels)} labels vs {len(graph.x)} nodes.")
                    
                graph.y = torch.tensor(labels, dtype=torch.float)
                torch.save(graph, save_path)

                results.append({
                    "pdb_id": pdb_name,
                    "chain_a": chain_a_id,
                    "chain_b": partner_str,
                    "num_residues": len(labels),
                    "num_binding": sum(labels),
                })
            except Exception as e:
                errors.append(f"{pdb_name}_{chain_a_id} generation failed: {str(e)}")
    except Exception as e:
        errors.append(f"{pdb_name} parsing failed: {str(e)}")
    
    return results, errors

def prepare_dataset(pdb_dir, output_dir, distance_cutoff, train_ratio, val_ratio, seed, threads):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    pdb_files = sorted(list(set(glob.glob(os.path.join(pdb_dir, "*.pdb")) + glob.glob(os.path.join(pdb_dir, "*.PDB")))))
    if not pdb_files:
        print(f"[ERROR] No PDB files found in: {pdb_dir}")
        return

    print(f"[ECABSD] Processing {len(pdb_files)} PDB files using {threads} processes...")
    print(f"[ECABSD] Applying rigorous length (30-512) and quality filters.")

    successful = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_single_pdb, f, output_dir, distance_cutoff): f for f in pdb_files}
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="Processing PDBs"):
            results, errors = future.result()
            successful.extend(results)
            all_errors.extend(errors)

    # Group by PDB ID so all chains of the same complex stay in the same split
    complexes = list(set([s["pdb_id"] for s in successful]))
    random.shuffle(complexes)

    n = len(complexes)
    if n == 0:
        print("[ERROR] No complexes survived filtering.")
        return
        
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_complexes = set(complexes[:n_train])
    val_complexes = set(complexes[n_train:n_train + n_val])

    for s in successful:
        if s["pdb_id"] in train_complexes:
            s["split"] = "train"
        elif s["pdb_id"] in val_complexes:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # Write splits CSV
    splits_path = os.path.join(os.path.dirname(output_dir), "splits.csv")
    fieldnames = ["pdb_id", "chain_a", "chain_b", "split", "num_residues", "num_binding"]
    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful)

    print(f"\n{'='*60}\n  Dataset Preparation Complete (Rigorous)\n{'='*60}")
    print(f"  Total Chains Extracted: {len(successful)}")
    print(f"  Errors / Skipped:       {len(all_errors)}")
    print(f"  Train / Val / Test:     {n_train} / {n_val} / {n - n_train - n_val}")
    print(f"  Splits CSV saved to:    {splits_path}\n{'='*60}")
    
    # Save a quick error log to see what was dropped
    with open(os.path.join(os.path.dirname(output_dir), "filter_log.txt"), "w") as f:
        for err in all_errors:
            f.write(err + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kaggle DIPS Dataset with Strict Filters")
    parser.add_argument("--pdb-dir", default="data/raw/dips", help="Raw DIPS PDB files directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for .pt graphs")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Physical distance cutoff (Å)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8, help="Parallel processes")
    args = parser.parse_args()

    prepare_dataset(args.pdb_dir, args.output_dir, args.cutoff, args.train_ratio, args.val_ratio, args.seed, args.threads)
