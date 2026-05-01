"""
ECABSD — JSON Export with metadata and confidence bands.
"""
import os
import json
from datetime import datetime
from typing import Optional

ECABSD_VERSION = "1.0.0"

def export_json(results_path: str, output_path: Optional[str] = None) -> str:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    if output_path is None:
        output_path = results_path.replace(".json", "_export.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    residues = results.get("residues", [])
    probs = [r["probability"] for r in residues]

    high_conf = [r for r in residues if r["probability"] >= 0.75 and r["is_binding"]]
    med_conf  = [r for r in residues if 0.5 <= r["probability"] < 0.75 and r["is_binding"]]

    export = {
        "ecabsd_version": ECABSD_VERSION,
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "input": {
            "pdb_file": results.get("pdb_file"),
            "chain_a": results.get("chain_a"),
            "chain_b": results.get("chain_b"),
            "threshold": results.get("threshold"),
        },
        "summary": {
            "total_residues": len(residues),
            "binding_residues": results.get("binding_residues_count", 0),
            "mean_probability": round(sum(probs) / len(probs), 4) if probs else 0,
            "max_probability": round(max(probs), 4) if probs else 0,
        },
        "binding_residues": {
            "high_confidence_>=0.75": [{"index": r["index"], "resid": r["resid"], "resname": r["resname"], "probability": round(r["probability"], 4)} for r in high_conf],
            "medium_confidence_0.5-0.75": [{"index": r["index"], "resid": r["resid"], "resname": r["resname"], "probability": round(r["probability"], 4)} for r in med_conf],
        },
        "all_residues": residues,
    }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[Export] JSON saved to: {output_path}")
    return output_path
