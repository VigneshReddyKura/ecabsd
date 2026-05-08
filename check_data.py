import torch, os

folder = 'data/processed'
files = [f for f in os.listdir(folder) if f.endswith('.pt')]

# Check 1 - inspect one file
data = torch.load(os.path.join(folder, files[0]), weights_only=False)
print('=== Single File Inspection ===')
print('File:', files[0])
print('x shape:', data.x.shape)
print('edge_index shape:', data.edge_index.shape)
print('y shape:', data.y.shape)
print('y unique values:', data.y.unique())
print('num_nodes:', data.num_nodes)

# Check 2 - splits
print('\n=== Splits ===')
with open('data/splits.csv') as f:
    lines = f.readlines()
print('Header:', lines[0].strip())
splits = {}
for l in lines[1:]:
    s = l.strip().split(',')[-1]
    splits[s] = splits.get(s, 0) + 1
print('Split counts:', splits)
print('Total pairs:', len(lines)-1)

# Check 3 - corrupted files
print('\n=== Corruption Check ===')
bad = []
for f in files:
    try:
        d = torch.load(os.path.join(folder, f), weights_only=False)
        assert hasattr(d, 'x') and hasattr(d, 'y')
        assert d.x.shape[1] == 23
    except Exception as e:
        bad.append((f, str(e)))
print(f'Checked: {len(files)} files')
print(f'Corrupted: {len(bad)}')
for b in bad:
    print(' -', b)