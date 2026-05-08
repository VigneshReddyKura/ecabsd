import torch
import os
import pandas as pd

folder = 'data/processed'
files = set(os.listdir(folder))

# Fix 1 - Remove zero binding files
print("[1] Removing zero-binding files...")
removed = 0
for f in list(files):
    if f.endswith('.pt'):
        d = torch.load(os.path.join(folder, f), weights_only=False)
        if (d.y == 1).sum().item() == 0:
            os.remove(os.path.join(folder, f))
            files.discard(f)
            removed += 1
print(f"Removed {removed} zero-binding files")

# Fix 2 - Clean splits.csv
print("\n[2] Cleaning splits.csv...")
df = pd.read_csv('data/splits.csv')
print(f"Before: {len(df)} rows")

# Drop rows with NaN in chain columns
df = df.dropna(subset=['chain_a', 'chain_b'])
print(f"After dropping NaN chains: {len(df)} rows")

# Drop rows where .pt files don't exist
def both_exist(row):
    fa = f"{row['pdb_id']}_{row['chain_a']}.pt"
    fb = f"{row['pdb_id']}_{row['chain_b']}.pt"
    return fa in files and fb in files

mask = df.apply(both_exist, axis=1)
df = df[mask]
print(f"After dropping missing files: {len(df)} rows")

# Reassign splits 80/10/10
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
total = len(df)
train_end = int(0.8 * total)
val_end = int(0.9 * total)
df['split'] = 'train'
df.loc[train_end:val_end-1, 'split'] = 'val'
df.loc[val_end:, 'split'] = 'test'

print(f"\nFinal split counts:")
print(df['split'].value_counts())
df.to_csv('data/splits.csv', index=False)
print("Saved clean splits.csv")

print("\n[3] Final check...")
print(f"Total .pt files: {len([f for f in os.listdir(folder) if f.endswith('.pt')])}")
print("Done!")