import json

# Show prediction output
print("=" * 60)
print("  OUTPUT FILE: results/predictions_1AY7_A.json")
print("=" * 60)
d = json.load(open("results/predictions_1AY7_A.json"))
print(f"  pdb_file:              {d['pdb_file']}")
print(f"  chain_a:               {d['chain_a']}")
print(f"  chain_b:               {d['chain_b']}")
print(f"  threshold:             {d['threshold']}")
print(f"  total_residues:        {d['total_residues']}")
print(f"  binding_residues:      {d['binding_residues_count']}")

print("\n  Sample residues (first 10):")
print(f"  {'Idx':>4}  {'Res':>4}  {'ResID':>5}  {'Probability':>11}  {'Binding?':>8}")
print(f"  {'-'*45}")
for r in d["residues"][:10]:
    flag = "YES  <--" if r["is_binding"] else "no"
    print(f"  {r['index']:>4}  {r['resname']:>4}  {r['resid']:>5}  {r['probability']:>11.4f}  {flag:>8}")

# Show metrics output
print("\n" + "=" * 60)
print("  OUTPUT FILE: results/metrics.json")
print("=" * 60)
m = json.load(open("results/metrics.json"))
for k, v in m.items():
    print(f"  {k:<30} {v}")

# Show training history summary
print("\n" + "=" * 60)
print("  OUTPUT FILE: logs/training_history.json (last 5 epochs)")
print("=" * 60)
h = json.load(open("logs/training_history.json"))
print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train F1':>9}  {'Val Loss':>9}  {'Val F1':>7}")
print(f"  {'-'*50}")
for ep in h[-5:]:
    print(f"  {ep['epoch']:>6}  {ep['train']['loss']:>10.4f}  {ep['train']['f1']:>9.4f}  {ep['val']['loss']:>9.4f}  {ep['val']['f1']:>7.4f}")

# Show checkpoint files
print("\n" + "=" * 60)
print("  OUTPUT FILES: checkpoints/")
print("=" * 60)
import os
for f in sorted(os.listdir("checkpoints")):
    if f.endswith(".pt"):
        size = os.path.getsize(f"checkpoints/{f}") / 1024
        print(f"  {f:<25}  {size:.0f} KB")

print("\n  Confusion matrix: results/confusion_matrix.png")
print("=" * 60)
