"""
Fetch 1000+ protein-protein interaction PDB IDs from RCSB
and build a CSV dataset with chain information (Optimized with Multi-threading).

Usage:
    python scripts/build_ppi_dataset.py --count 1000 --out data/ppi_dataset.csv
"""

import argparse
import csv
import json
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def search_ppi_pdb_ids(count=1000):
    """
    Use the RCSB Search API to find PDB structures that contain
    at least 2 protein entities (i.e., protein-protein complexes).
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": count
            },
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        }
    }

    print(f"Querying RCSB for {count} protein-protein complex structures...")
    response = requests.post(url, json=query, timeout=60)
    response.raise_for_status()
    data = response.json()

    total_available = data.get("total_count", 0)
    print(f"Total matching structures in RCSB: {total_available}")

    pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
    print(f"Retrieved {len(pdb_ids)} PDB IDs")
    return pdb_ids


def get_chain_info(pdb_id):
    """
    Use the RCSB GraphQL API to get protein chain IDs for a given PDB entry.
    Returns a list of protein chain IDs.
    """
    entry_url = "https://data.rcsb.org/graphql"
    graphql_query = {
        "query": f"""
        {{
          entry(entry_id: "{pdb_id}") {{
            polymer_entities {{
              entity_poly {{
                type
              }}
              polymer_entity_instances {{
                rcsb_polymer_entity_instance_container_identifiers {{
                  auth_asym_id
                }}
              }}
            }}
          }}
        }}
        """
    }
    try:
        resp = requests.post(entry_url, json=graphql_query, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        protein_chains = []
        if result.get("data") and result["data"].get("entry"):
            for entity in result["data"]["entry"]["polymer_entities"]:
                poly_type = entity["entity_poly"]["type"]
                if poly_type == "polypeptide(L)":
                    for instance in entity["polymer_entity_instances"]:
                        chain_id = instance["rcsb_polymer_entity_instance_container_identifiers"]["auth_asym_id"]
                        protein_chains.append(chain_id)
        
        if len(protein_chains) >= 2:
            return {
                "pdb_id": pdb_id,
                "chain_a": protein_chains[0],
                "chain_b": protein_chains[1],
                "split": "train"
            }
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Build a PPI dataset CSV by querying RCSB (Optimized)."
    )
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of PDB structures to fetch (default: 1000)")
    parser.add_argument("--out", type=str, default="data/ppi_dataset.csv",
                        help="Output CSV file path (default: data/ppi_dataset.csv)")
    parser.add_argument("--threads", type=int, default=10,
                        help="Number of parallel threads (default: 10)")
    args = parser.parse_args()

    pdb_ids = search_ppi_pdb_ids(args.count)

    if not pdb_ids:
        print("No PDB IDs found. Check your internet connection.")
        return

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0

    print(f"\nFetching chain information for {len(pdb_ids)} structures using {args.threads} threads...")
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(get_chain_info, pdb_id): pdb_id for pdb_id in pdb_ids}
        
        for future in tqdm(as_completed(futures), total=len(pdb_ids), desc="Processing PDBs"):
            result = future.result()
            if result:
                rows.append(result)
            else:
                failed += 1

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "chain_a", "chain_b", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Dataset Summary ---")
    print(f"PDB IDs queried: {len(pdb_ids)}")
    print(f"Valid PPI entries: {len(rows)}")
    print(f"Failed/skipped: {failed}")
    print(f"CSV saved to: {output_path}")
    print(f"\nNow run:")
    print(f"  python scripts/download_pdbs.py --csv {output_path} --out data/raw/pdbs")


if __name__ == "__main__":
    main()
