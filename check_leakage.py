import pandas as pd
import sys

def check_leakage(splits_csv_path):
    try:
        df = pd.read_csv(splits_csv_path)
    except Exception as e:
        print(f"Could not load {splits_csv_path}: {e}")
        return False
        
    train_pdbs = set(df[df['split'] == 'train']['pdb_id'])
    val_pdbs = set(df[df['split'] == 'val']['pdb_id'])
    test_pdbs = set(df[df['split'] == 'test']['pdb_id'])
    
    leakage = False
    if train_pdbs.intersection(val_pdbs):
        print("LEAKAGE DETECTED: Train and Val splits share PDB IDs.")
        leakage = True
    if train_pdbs.intersection(test_pdbs):
        print("LEAKAGE DETECTED: Train and Test splits share PDB IDs.")
        leakage = True
    if val_pdbs.intersection(test_pdbs):
        print("LEAKAGE DETECTED: Val and Test splits share PDB IDs.")
        leakage = True
        
    if leakage:
        print("CRITICAL ERROR: Data leakage detected. Stopping.")
        sys.exit(1)
        
    print("Leakage check passed! No overlapping PDBs between splits.")
    return True

if __name__ == "__main__":
    check_leakage("data/splits.csv")
