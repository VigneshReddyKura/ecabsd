"""
ECABSD Dataset — PyTorch Geometric Dataset for binding site detection.

Loads preprocessed .pt graph files and returns paired protein graphs
with per-residue binding labels.
"""

import os
import csv
import torch
from torch.utils.data import Dataset


class BindingSiteDataset(Dataset):
    """
    Dataset for protein-protein binding site detection.

    Each sample is a dictionary:
        - data_a: PyG Data for chain A (target)
        - data_b: PyG Data for chain B (partner) or None
        - labels: per-residue binding labels for chain A

    Directory structure expected:
        processed_dir/
            <pdb_id>_<chain_a>.pt    — graph for chain A with .y labels
            <pdb_id>_<chain_b>.pt    — graph for chain B (optional)

    splits_csv format:
        pdb_id,chain_a,chain_b,split
        1AY7,A,B,train
        2XYZ,A,C,val
    """

    def __init__(self, processed_dir: str, splits_csv: str, split: str = "train"):
        """
        Parameters
        ----------
        processed_dir : str
            Directory containing preprocessed .pt files.
        splits_csv : str
            CSV file with columns: pdb_id, chain_a, chain_b, split
        split : str
            One of "train", "val", "test".
        """
        self.processed_dir = processed_dir
        self.split = split
        self.samples = []

        # Load split information
        if os.path.exists(splits_csv):
            with open(splits_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["split"] == split:
                        self.samples.append(
                            {
                                "pdb_id": row["pdb_id"],
                                "chain_a": row["chain_a"],
                                "chain_b": row.get("chain_b", ""),
                            }
                        )

        print(f"[Dataset] Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdb_id = sample["pdb_id"]
        chain_a = sample["chain_a"]
        chain_b = sample["chain_b"]

        # Load chain A graph
        path_a = os.path.join(self.processed_dir, f"{pdb_id}_{chain_a}.pt")
        data_a = torch.load(path_a, weights_only=False)

        # Load chain B graph (if available)
        data_b = None
        if chain_b:
            path_b = os.path.join(self.processed_dir, f"{pdb_id}_{chain_b}.pt")
            if os.path.exists(path_b):
                data_b = torch.load(path_b, weights_only=False)

        # Extract labels from chain A
        labels = data_a.y if hasattr(data_a, "y") and data_a.y is not None else torch.zeros(data_a.num_nodes)

        return {
            "data_a": data_a,
            "data_b": data_b,
            "labels": labels,
            "pdb_id": pdb_id,
        }


def collate_fn(batch):
    """
    Custom collate for variable-size protein graphs.
    Since each protein has different number of residues, we process one at a time.
    """
    # For simplicity, return single items (batch_size=1 effective)
    # For batched training, use torch_geometric's Batch.from_data_list
    return batch[0]
