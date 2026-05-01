import os
import torch

folder = "data/processed"

total_0 = 0
total_1 = 0

for file in os.listdir(folder):
    if file.endswith(".pt"):
        data = torch.load(os.path.join(folder, file), weights_only=False)

        y = data.y
        zeros = (y == 0).sum().item()
        ones = (y == 1).sum().item()

        total_0 += zeros
        total_1 += ones

print("Total non-binding 0:", total_0)
print("Total binding 1:", total_1)

if total_1 == 0:
    print("PROBLEM: no binding labels found")
else:
    print("Ratio 0/1:", total_0 / total_1)