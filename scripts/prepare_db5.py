import os
import sys
import csv
import glob
import random
import argparse
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_construction import build_residue_graph, compute_binding_labels
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

def get_chain_ids(pdb_path):
    """Return all chain IDs present in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    models = list(structure.get_models())
    if not models:
        return []
    chains = list(models[0].get_chains())
    return [c.id for c in chains if c.id.strip()]


def process_single_pdb(pdb_path, output_dir, distance_cutoff):
    """Process all chain pairs for a plain PDB file."""
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "").strip()
    results = []
    errors = []

    chain_ids = get_chain_ids(pdb_path)
    if len(chain_ids) < 2:
        errors.append(f"{pdb_name}: only {len(chain_ids)} chain(s), need ≥2")
        return results, errors

    # Process first 2 chains: A→B and B→A
    target_id, partner_id = chain_ids[0], chain_ids[1]

    for (t_id, p_id) in [(target_id, partner_id), (partner_id, target_id)]:
        save_path = os.path.join(output_dir, f"{pdb_name}_{t_id}.pt")

        if os.path.exists(save_path):
            try:
                graph = torch.load(save_path, map_location="cpu")
                results.append({
                    "pdb_id": pdb_name,
                    "chain_a": t_id,
                    "chain_b": p_id,
                    "num_residues": len(graph.y),
                    "num_binding": int(graph.y.sum()),
                })
                continue
            except Exception:
                pass

        try:
            graph = build_residue_graph(pdb_path, t_id)
            labels = compute_binding_labels(pdb_path, t_id, p_id, distance_cutoff)
            if labels is None or len(labels) != graph.x.shape[0]:
                errors.append(f"{pdb_name}_{t_id}: label/node count mismatch")
                continue
            graph.y = torch.tensor(labels, dtype=torch.float)
            torch.save(graph, save_path)

            results.append({
                "pdb_id": pdb_name,
                "chain_a": t_id,
                "chain_b": p_id,
                "num_residues": len(labels),
                "num_binding": sum(labels),
            })
        except Exception as e:
            errors.append(f"{pdb_name}_{t_id}: {str(e)}")

    return results, errors

def prepare_db5(db5_dir, output_dir, distance_cutoff, train_ratio, val_ratio, seed, threads):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Support both flat and nested PDB directories
    pdb_files = sorted(glob.glob(os.path.join(db5_dir, "*.pdb")))
    if not pdb_files:
        pdb_files = sorted(glob.glob(os.path.join(db5_dir, "**", "*.pdb"), recursive=True))
    if not pdb_files:
        print(f"[ERROR] No PDB files found in: {db5_dir}")
        return

    print(f"[ECABSD] Processing {len(pdb_files)} PDB structures sequentially...")

    successful = []
    all_errors = []

    for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
        results, errors = process_single_pdb(pdb_path, output_dir, distance_cutoff)
        successful.extend(results)
        all_errors.extend(errors)

    # Group by the 4-character base PDB ID (e.g., '1AK4' from '1AK4_l_u')
    # so bound and unbound structures of the same complex stay in the same split
    base_ids = list(set([s["pdb_id"][:4] for s in successful]))
    random.shuffle(base_ids)

    n = len(base_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_base = set(base_ids[:n_train])
    val_base = set(base_ids[n_train:n_train + n_val])

    for s in successful:
        base = s["pdb_id"][:4]
        if base in train_base:
            s["split"] = "train"
        elif base in val_base:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # Write splits CSV
    splits_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "db5_splits.csv")
    fieldnames = ["pdb_id", "chain_a", "chain_b", "split", "num_residues", "num_binding"]
    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful)

    print(f"\n{'='*60}\n  Dataset Preparation Complete\n{'='*60}")
    print(f"  PDBs processed:       {len(pdb_files)}")
    print(f"  Chains saved:         {len(successful)}")
    print(f"  Errors:               {len(all_errors)}")
    print(f"  Train/Val/Test split: {len(train_complexes)} / {len(val_complexes)} / {n - len(train_complexes) - len(val_complexes)} complexes")
    print(f"  Splits CSV:           {splits_path}")
    if all_errors:
        print(f"\n  Failed structures:")
        for e in all_errors[:20]:
            print(f"    - {e}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db5-dir", "--input", default="data/raw/pdbs", help="Directory containing PDB files")
    parser.add_argument("--output-dir", "--output", default="data/processed", help="Output directory for .pt graphs")
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    prepare_db5(args.db5_dir, args.output_dir, args.cutoff, args.train_ratio, args.val_ratio, args.seed, args.threads)
