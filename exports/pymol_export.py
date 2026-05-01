"""
ECABSD — PyMOL Export.

Generates a PyMOL .pml script that colors residues by binding probability.
Green (low) → Yellow (medium) → Red (high confidence binding).
"""
import os
import json
from typing import Optional


def export_pymol(results_path: str, output_path: Optional[str] = None) -> str:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    if output_path is None:
        output_path = results_path.replace(".json", ".pml")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    pdb_file = results.get("pdb_file", "protein.pdb")
    chain = results.get("chain_a", "A")
    threshold = results.get("threshold", 0.5)
    residues = results.get("residues", [])

    binding = [r for r in residues if r["is_binding"]]
    binding_resids = [str(r["resid"]) for r in binding]

    lines = [
        "# ECABSD Binding Site Visualization",
        f"# Generated for {pdb_file}, chain {chain}",
        "",
        f"load {pdb_file}, protein",
        "hide everything",
        "show cartoon, protein",
        "bg_color white",
        "",
        "# Color all residues grey (non-binding)",
        f"color grey80, chain {chain}",
        "",
        "# Color binding site residues by probability",
    ]

    for r in residues:
        prob = r["probability"]
        resid = r["resid"]
        # RGB: interpolate green→yellow→red
        if prob < 0.5:
            red   = int(prob * 2 * 255)
            green = 255
            blue  = 0
        else:
            red   = 255
            green = int((1 - (prob - 0.5) * 2) * 255)
            blue  = 0
        hex_color = f"0x{red:02X}{green:02X}{blue:02X}"
        lines.append(f"color {hex_color}, chain {chain} and resi {resid}")

    if binding_resids:
        sel_str = "+".join(binding_resids)
        lines += [
            "",
            f"# Select predicted binding site (threshold={threshold})",
            f"select binding_site, chain {chain} and resi {sel_str}",
            "show sticks, binding_site",
            "label binding_site and name CA, \"%s%s\" % (resn, resi)",
        ]

    lines += [
        "",
        "# Final display settings",
        "zoom protein",
        "set label_size, 12",
        "set cartoon_transparency, 0.2",
        "ray 1200, 900",
        "png ecabsd_binding_site.png, dpi=150",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Export] PyMOL script saved to: {output_path}")
    print(f"[Export] Open in PyMOL: pymol {output_path}")
    return output_path
