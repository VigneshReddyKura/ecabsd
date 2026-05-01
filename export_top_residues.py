import json
import csv

json_path = "results/predictions_1AY7_A.json"
csv_path = "results/top_binding_residues_1AY7_A.csv"

with open(json_path, "r") as f:
    data = json.load(f)

residues = data["residues"]

# sort high probability first
residues = sorted(residues, key=lambda x: x["probability"], reverse=True)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "resname", "resid", "chain", "probability", "is_binding"])

    for i, r in enumerate(residues[:20], start=1):
        writer.writerow([
            i,
            r["resname"],
            r["resid"],
            r["chain"],
            round(r["probability"], 4),
            r["is_binding"]
        ])

print("Saved:", csv_path)