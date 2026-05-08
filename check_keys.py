import torch, os

files = [f for f in os.listdir('data/processed') if f.endswith('.pt')]
data = torch.load('data/processed/' + files[0], weights_only=False)
print('All attributes:', data.keys())
print('File:', files[0])