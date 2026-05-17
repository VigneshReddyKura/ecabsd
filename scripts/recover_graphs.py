import os
import glob
import torch
import traceback
from Bio.PDB import PDBList
from models.graph_construction import build_residue_graph

def recover_graphs():
    bad_dir = "data/bad_graphs"
    raw_pdb_dir = "data/raw_pdbs"
    processed_dir = "data/processed"
    
    os.makedirs(raw_pdb_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    pdbl = PDBList()
    
    bad_files = glob.glob(os.path.join(bad_dir, "*.pt"))
    if not bad_files:
        print("[Recovery] No bad graphs found in", bad_dir)
        return
        
    print(f"[Recovery] Found {len(bad_files)} bad graphs to recover.")
    
    success_count = 0
    fail_count = 0
    
    for fpath in bad_files:
        basename = os.path.basename(fpath) # e.g. "1AY7_A.pt"
        name_parts = basename.replace('.pt', '').split('_')
        
        if len(name_parts) >= 2:
            pdb_id = name_parts[0].lower() # PDBList uses lowercase
            chain_id = name_parts[1]
        else:
            print(f"Skipping {basename} - unexpected format")
            fail_count += 1
            continue
            
        pdb_file = os.path.join(raw_pdb_dir, f"pdb{pdb_id}.ent")
        
        # Download if not exists
        if not os.path.exists(pdb_file):
            print(f"[Recovery] Downloading {pdb_id}...")
            try:
                pdbl.retrieve_pdb_file(pdb_id, pdir=raw_pdb_dir, file_format="pdb")
            except Exception as e:
                print(f"[Recovery] Failed to download {pdb_id}: {e}")
                fail_count += 1
                continue
                
        # Rebuild graph
        if os.path.exists(pdb_file):
            print(f"[Recovery] Rebuilding graph for {pdb_id} Chain {chain_id}...")
            try:
                # build_residue_graph returns a PyG Data object
                data = build_residue_graph(pdb_file, chain_id)
                
                # Check dimensions
                x_ok = hasattr(data, "x") and data.x is not None and data.x.shape[1] == 33
                e_ok = hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[1] == 5
                
                if x_ok and e_ok:
                    out_path = os.path.join(processed_dir, basename)
                    torch.save(data, out_path)
                    print(f"[Recovery] Success -> {out_path}")
                    success_count += 1
                else:
                    x_dim = data.x.shape[1] if data.x is not None else "None"
                    e_dim = data.edge_attr.shape[1] if data.edge_attr is not None else "None"
                    print(f"[Recovery] Built graph but wrong dims: x={x_dim}, edge={e_dim}")
                    fail_count += 1
                    
            except Exception as e:
                print(f"[Recovery] Error building {basename}: {e}")
                # traceback.print_exc()
                fail_count += 1
        else:
            print(f"[Recovery] PDB file not found after download attempt: {pdb_file}")
            fail_count += 1
            
    print("-" * 40)
    print(f"Recovery complete. Success: {success_count}, Fail: {fail_count}")
    print("New graphs are saved to data/processed")

if __name__ == "__main__":
    recover_graphs()
