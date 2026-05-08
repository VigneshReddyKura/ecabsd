import torch
import os
import numpy as np

folder = 'data/processed'
files = [f for f in os.listdir(folder) if f.endswith('.pt')]

print("="*50)
print("DEEP DATA AUDIT")
print("="*50)

# Check 1 - Basic stats
print("\n[1] BASIC STATS")
print(f"Total .pt files: {len(files)}")

sizes = []
binding_ratios = []
zero_binding = []
all_binding = []

for f in files:
    d = torch.load(os.path.join(folder, f), weights_only=False)
    n = d.num_nodes
    b = (d.y == 1).sum().item()
    sizes.append(n)
    if n > 0:
        ratio = b / n
        binding_ratios.append(ratio)
    if b == 0:
        zero_binding.append(f)
    if b == n:
        all_binding.append(f)

print(f"Protein sizes - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {np.mean(sizes):.1f}")
print(f"Binding ratio per protein - Min: {min(binding_ratios):.3f}, Max: {max(binding_ratios):.3f}, Avg: {np.mean(binding_ratios):.3f}")
print(f"Files with ZERO binding residues: {len(zero_binding)}")
print(f"Files with ALL binding residues: {len(all_binding)}")

if zero_binding:
    print("  Zero binding files:", zero_binding[:5])
if all_binding:
    print("  All binding files:", all_binding[:5])

# Check 2 - Feature sanity
print("\n[2] FEATURE SANITY")
d = torch.load(os.path.join(folder, files[0]), weights_only=False)
x = d.x
print(f"Feature shape: {x.shape}")
print(f"Feature min: {x.min().item():.4f}")
print(f"Feature max: {x.max().item():.4f}")
print(f"Any NaN in features: {torch.isnan(x).any().item()}")
print(f"Any Inf in features: {torch.isinf(x).any().item()}")

# Check amino acid one-hot (first 20 cols should sum to 1)
aa_sum = x[:, :20].sum(dim=1)
print(f"AA one-hot sums - Min: {aa_sum.min().item():.2f}, Max: {aa_sum.max().item():.2f} (should be 1.0)")

# Check secondary structure (cols 20-22)
ss_sum = x[:, 20:23].sum(dim=1)
print(f"SS one-hot sums - Min: {ss_sum.min().item():.2f}, Max: {ss_sum.max().item():.2f} (should be 1.0)")

# Check 3 - Edge sanity
print("\n[3] EDGE SANITY")
ea = d.edge_attr
print(f"Edge attr shape: {ea.shape}")
print(f"Distance range: {ea[:, 0].min().item():.2f} to {ea[:, 0].max().item():.2f} Angstroms (should be 0-8)")
print(f"Any NaN in edges: {torch.isnan(ea).any().item()}")

# Check 4 - Label sanity
print("\n[4] LABEL SANITY")
all_labels = []
for f in files:
    d = torch.load(os.path.join(folder, f), weights_only=False)
    all_labels.extend(d.y.tolist())
all_labels = np.array(all_labels)
unique = np.unique(all_labels)
print(f"Unique label values: {unique} (should be only 0.0 and 1.0)")
print(f"Total 0s: {(all_labels==0).sum():,}")
print(f"Total 1s: {(all_labels==1).sum():,}")
print(f"Ratio: {(all_labels==0).sum()/(all_labels==1).sum():.2f}")

# Check 5 - Splits sanity
print("\n[5] SPLITS SANITY")
import pandas as pd
df = pd.read_csv('data/splits.csv')
print(f"Columns: {list(df.columns)}")
print(f"Split counts:\n{df['split'].value_counts()}")
print(f"Total rows: {len(df)}")

# Check if all files in splits actually exist
missing = []
for _, row in df.iterrows():
    fa = f"{row['pdb_id']}_{row['chain_a']}.pt"
    fb = f"{row['pdb_id']}_{row['chain_b']}.pt"
    if not os.path.exists(os.path.join(folder, fa)):
        missing.append(fa)
    if not os.path.exists(os.path.join(folder, fb)):
        missing.append(fb)
print(f"Missing .pt files referenced in splits.csv: {len(missing)}")
if missing:
    print("  Sample missing:", missing[:5])

print("\n" + "="*50)
print("AUDIT COMPLETE")
print("="*50)