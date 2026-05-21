import os
import csv
import argparse
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

def get_sequence_from_pdb(pdb_path, chain_id):
    """Extract amino acid sequence from PDB chain."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
        model = list(structure.get_models())[0]
        chain = model[chain_id]
        
        seq = ""
        for r in chain:
            if is_aa(r, standard=True):
                resname = r.get_resname()
                try:
                    seq += protein_letters_3to1[resname.capitalize()]
                except KeyError:
                    seq += "X"
        return seq
    except Exception:
        return ""

def check_sequence_leakage(splits_csv, pdb_dir):
    """Check for 100% identical sequences across splits."""
    if not os.path.exists(splits_csv):
        print(f"Splits file not found: {splits_csv}")
        return

    train_seqs = {}
    val_seqs = {}
    test_seqs = {}

    print("Extracting sequences from PDBs...")
    with open(splits_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"]
            chain_a = row["chain_a"]
            split = row["split"]
            
            # Assuming files are named like 1BRS.pdb or 1BRS_l_u.pdb
            # Try to find the file
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_path):
                # Try finding any file that starts with this pdb_id
                import glob
                matches = glob.glob(os.path.join(pdb_dir, f"{pdb_id}*.pdb"))
                if matches:
                    pdb_path = matches[0]
                else:
                    continue

            seq = get_sequence_from_pdb(pdb_path, chain_a)
            if not seq:
                continue
                
            if split == "train":
                train_seqs[f"{pdb_id}_{chain_a}"] = seq
            elif split == "val":
                val_seqs[f"{pdb_id}_{chain_a}"] = seq
            elif split == "test":
                test_seqs[f"{pdb_id}_{chain_a}"] = seq

    print(f"Extracted {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences.")
    
    leakage = False
    
    # Check exact substring matching (handling cases where one is a cropped version of another)
    print("Checking Train vs Test leakage...")
    for tr_id, tr_seq in train_seqs.items():
        for te_id, te_seq in test_seqs.items():
            if tr_seq in te_seq or te_seq in tr_seq:
                if len(tr_seq) > 30 and len(te_seq) > 30: # ignore very short matches
                    print(f"[LEAKAGE] Train {tr_id} and Test {te_id} have identical or overlapping sequences!")
                    leakage = True

    print("Checking Train vs Val leakage...")
    for tr_id, tr_seq in train_seqs.items():
        for val_id, val_seq in val_seqs.items():
            if tr_seq in val_seq or val_seq in tr_seq:
                if len(tr_seq) > 30 and len(val_seq) > 30:
                    print(f"[LEAKAGE] Train {tr_id} and Val {val_id} have identical or overlapping sequences!")
                    leakage = True

    if not leakage:
        print("PASS: No exact sequence leakage detected between splits.")
    else:
        print("FAIL: Sequence leakage detected. You should use MMSeqs2/CD-HIT to cluster your dataset.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/db5_splits.csv")
    parser.add_argument("--pdb-dir", default="data/raw/pdbs")
    args = parser.parse_args()
    
    check_sequence_leakage(args.splits, args.pdb_dir)

if __name__ == "__main__":
    main()
