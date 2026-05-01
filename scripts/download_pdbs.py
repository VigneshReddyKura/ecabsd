import os
import argparse
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

def download_pdb(pdb_id, output_dir):
    pdb_id = str(pdb_id).upper().strip()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = Path(output_dir) / f"{pdb_id}.pdb"
    
    if output_path.exists():
        return "exists"
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return "success"
    except requests.exceptions.RequestException:
        return "failed"

def main():
    parser = argparse.ArgumentParser(description="Download PDB files from a CSV list.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file containing 'pdb_id' column.")
    parser.add_argument("--out", required=True, help="Directory to save downloaded PDB files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of PDB files to download.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.out)
    
    # Save failed PDBs in the parent directory of the output directory, as requested (data/raw/failed_pdbs.txt)
    # If out is data/raw/pdbs, parent is data/raw
    failed_log_path = output_dir.parent / "failed_pdbs.txt"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    if 'pdb_id' not in df.columns:
        print("Error: CSV must contain a 'pdb_id' column.")
        return

    pdb_ids = df['pdb_id'].dropna().unique()
    
    if args.limit:
        pdb_ids = pdb_ids[:args.limit]
        print(f"Limiting download to the first {args.limit} PDB IDs")
        
    total_pdbs = len(pdb_ids)
    
    print(f"Found {total_pdbs} unique PDB IDs in {csv_path.name}")
    print(f"Downloading to {output_dir}")
    
    success_count = 0
    exists_count = 0
    failed_pdbs = []

    for pdb_id in tqdm(pdb_ids, desc="Downloading PDBs"):
        status = download_pdb(pdb_id, output_dir)
        if status == "success":
            success_count += 1
        elif status == "exists":
            exists_count += 1
        else:
            failed_pdbs.append(pdb_id)

    # Save failed PDBs
    if failed_pdbs:
        with open(failed_log_path, 'w') as f:
            for pdb in failed_pdbs:
                f.write(f"{pdb}\n")

    print("\n--- Download Summary ---")
    print(f"Total PDB IDs: {total_pdbs}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Already existing: {exists_count}")
    print(f"Failed downloads: {len(failed_pdbs)}")
    
    if failed_pdbs:
        print(f"Failed PDB IDs saved to: {failed_log_path}")

if __name__ == "__main__":
    main()
