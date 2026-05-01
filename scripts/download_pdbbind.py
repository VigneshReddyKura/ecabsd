"""
Download PDBbind Dataset for ECABSD.

Downloads and organizes PDB files from the PDBbind database
or fetches structures directly from RCSB PDB.

Usage:
    python scripts/download_pdbbind.py --output-dir data/raw/pdbs
    python scripts/download_pdbbind.py --pdb-ids 1AY7,2XYZ,3ABC
"""

import os
import sys
import argparse
import urllib.request
import gzip
import shutil

from tqdm import tqdm


# A curated list of PDB IDs commonly used for binding site benchmarks
BENCHMARK_PDB_IDS = [
    "1AY7", "1BVN", "1CGI", "1DFJ", "1E6E",
    "1FC2", "1GRN", "1HE1", "1JTG", "1KAC",
    "1MAH", "1NCA", "1PPE", "1QA9", "1R0R",
    "1UDI", "1WEJ", "2BTF", "2HRK", "2I25",
    "2OOB", "2SIC", "3HFL", "3SGB", "4CPA",
]


def download_pdb(pdb_id: str, output_dir: str) -> str:
    """
    Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-letter PDB ID.
    output_dir : str
        Directory to save the PDB file.

    Returns
    -------
    str : Path to the downloaded PDB file.
    """
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path):
        return output_path

    try:
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception:
        # Try .pdb.gz format
        gz_url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
        gz_path = output_path + ".gz"
        try:
            urllib.request.urlretrieve(gz_url, gz_path)
            with gzip.open(gz_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            return output_path
        except Exception as e:
            print(f"  [ERROR] Failed to download {pdb_id}: {e}")
            return None


def download_pdbbind(
    output_dir: str = "data/raw/pdbs",
    pdb_ids: list = None,
    use_benchmark: bool = True,
):
    """
    Download PDB structures for ECABSD training/evaluation.

    Parameters
    ----------
    output_dir : str
        Directory to save PDB files.
    pdb_ids : list, optional
        Specific PDB IDs to download. If None, uses benchmark set.
    use_benchmark : bool
        Whether to use the built-in benchmark PDB list.
    """
    os.makedirs(output_dir, exist_ok=True)

    if pdb_ids:
        ids_to_download = [pid.strip().upper() for pid in pdb_ids]
    elif use_benchmark:
        ids_to_download = BENCHMARK_PDB_IDS
    else:
        print("[ERROR] No PDB IDs specified. Use --pdb-ids or --benchmark.")
        return

    print(f"[ECABSD] Downloading {len(ids_to_download)} PDB structures...")
    print(f"[ECABSD] Output directory: {output_dir}\n")

    successful = []
    failed = []

    for pdb_id in tqdm(ids_to_download, desc="Downloading"):
        result = download_pdb(pdb_id, output_dir)
        if result:
            successful.append(pdb_id)
        else:
            failed.append(pdb_id)

    print(f"\n{'='*50}")
    print(f"  Download Complete")
    print(f"{'='*50}")
    print(f"  Successful:  {len(successful)}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*50}")

    if failed:
        print(f"\n  Failed IDs: {', '.join(failed)}")

    print(f"\n  Next steps:")
    print(f"    python scripts/prepare_dataset.py --pdb-dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDB structures for ECABSD")
    parser.add_argument("--output-dir", default="data/raw/pdbs", help="Output directory")
    parser.add_argument(
        "--pdb-ids",
        default=None,
        help="Comma-separated PDB IDs (e.g., 1AY7,2XYZ)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Download benchmark PDB set",
    )
    args = parser.parse_args()

    pdb_ids = args.pdb_ids.split(",") if args.pdb_ids else None
    download_pdbbind(
        output_dir=args.output_dir,
        pdb_ids=pdb_ids,
        use_benchmark=args.benchmark,
    )
