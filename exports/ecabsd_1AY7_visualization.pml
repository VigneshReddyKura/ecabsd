# ============================================================
# ECABSD V3 — Binding Site Visualization Script
# Structure: 1AY7 (RNase Sa / Barstar complex)
# Chain A: Guanyl-specific Ribonuclease Sa (96 residues)
# Chain B: Barstar inhibitor
# Model: ECABSD V3 (GATv2 × 6 + Cross-Attention)
# Predicted binding residues: 16 | True interface: 15
# Color scheme: white (low prob) → red (high prob binding)
# ============================================================

load 1AY7.pdb, protein
hide everything
show cartoon, protein
bg_color white

# Show both chains clearly
color slate, chain B
set cartoon_transparency, 0.5, chain B

# Base color: all chain A residues in light grey
color grey85, chain A

# Color each residue by predicted binding probability
# White = 0.0 (no binding), Red = 1.0 (high confidence binding)
color 0xFFCDCD, chain A and resi 1
color 0xFFF3F3, chain A and resi 2
color 0xFFE3E3, chain A and resi 3
color 0xFFE7E7, chain A and resi 4
color 0xFFC7C7, chain A and resi 5
color 0xFFCBCB, chain A and resi 6
color 0xFFBEBE, chain A and resi 7
color 0xFFEFEF, chain A and resi 8
color 0xFFDADA, chain A and resi 9
color 0xFFF2F2, chain A and resi 10
color 0xFFE7E7, chain A and resi 11
color 0xFFD5D5, chain A and resi 12
color 0xFFF3F3, chain A and resi 13
color 0xFFE8E8, chain A and resi 14
color 0xFFCDCD, chain A and resi 15
color 0xFFD3D3, chain A and resi 16
color 0xFFE7E7, chain A and resi 17
color 0xFFD0D0, chain A and resi 18
color 0xFFC3C3, chain A and resi 19
color 0xFFF4F4, chain A and resi 20
color 0xFFC3C3, chain A and resi 21
color 0xFFCACA, chain A and resi 22
color 0xFFDFDF, chain A and resi 23
color 0xFFEBEB, chain A and resi 24
color 0xFFBABA, chain A and resi 25
color 0xFFE0E0, chain A and resi 26
color 0xFFEFEF, chain A and resi 27
color 0xFFEEEE, chain A and resi 28
color 0xFFC0C0, chain A and resi 29
color 0xFFCFCF, chain A and resi 30
color 0xFF7474, chain A and resi 31
color 0xFF1E1E, chain A and resi 32
color 0xFFD3D3, chain A and resi 33
color 0xFFB9B9, chain A and resi 34
color 0xFFDDDD, chain A and resi 35
color 0xFF8181, chain A and resi 36
color 0xFF1818, chain A and resi 37
color 0xFF2424, chain A and resi 38
color 0xFF1717, chain A and resi 39
color 0xFF2727, chain A and resi 40
color 0xFF1F1F, chain A and resi 41
color 0xFF9B9B, chain A and resi 42
color 0xFFE6E6, chain A and resi 43
color 0xFFE3E3, chain A and resi 44
color 0xFFEFEF, chain A and resi 45
color 0xFFE6E6, chain A and resi 46
color 0xFFEEEE, chain A and resi 47
color 0xFFE3E3, chain A and resi 48
color 0xFFCDCD, chain A and resi 49
color 0xFFDEDE, chain A and resi 50
color 0xFFDEDE, chain A and resi 51
color 0xFFE7E7, chain A and resi 52
color 0xFFE4E4, chain A and resi 53
color 0xFFBBBB, chain A and resi 54
color 0xFFCDCD, chain A and resi 55
color 0xFFCFCF, chain A and resi 56
color 0xFFEAEA, chain A and resi 57
color 0xFFC8C8, chain A and resi 58
color 0xFFEAEA, chain A and resi 59
color 0xFFDDDD, chain A and resi 60
color 0xFFB8B8, chain A and resi 61
color 0xFFCDCD, chain A and resi 62
color 0xFF8181, chain A and resi 63
color 0xFF2020, chain A and resi 64
color 0xFF1818, chain A and resi 65
color 0xFF1B1B, chain A and resi 66
color 0xFF3A3A, chain A and resi 67
color 0xFF9C9C, chain A and resi 68
color 0xFF3535, chain A and resi 69
color 0xFF9090, chain A and resi 70
color 0xFFE7E7, chain A and resi 71
color 0xFFBBBB, chain A and resi 72
color 0xFFBFBF, chain A and resi 73
color 0xFFE1E1, chain A and resi 74
color 0xFFCCCC, chain A and resi 75
color 0xFFDCDC, chain A and resi 76
color 0xFFBCBC, chain A and resi 77
color 0xFFD8D8, chain A and resi 78
color 0xFFE4E4, chain A and resi 79
color 0xFFE5E5, chain A and resi 80
color 0xFFD2D2, chain A and resi 81
color 0xFFE4E4, chain A and resi 82
color 0xFF8080, chain A and resi 83
color 0xFF1515, chain A and resi 84
color 0xFF3030, chain A and resi 85
color 0xFF3B3B, chain A and resi 86
color 0xFF0F0F, chain A and resi 87
color 0xFF8484, chain A and resi 88
color 0xFFEFEF, chain A and resi 89
color 0xFFF1F1, chain A and resi 90
color 0xFFEEEE, chain A and resi 91
color 0xFFCECE, chain A and resi 92
color 0xFFC4C4, chain A and resi 93
color 0xFFDADA, chain A and resi 94
color 0xFFF0F0, chain A and resi 95
color 0xFFDDDD, chain A and resi 96

# ---- Binding site selection and display ----
select binding_site_A, chain A and resi 31+32+37+38+39+40+41+64+65+66+67+69+84+85+86+87
show sticks, binding_site_A
set stick_radius, 0.15, binding_site_A

# Label Cα of binding residues
label binding_site_A and name CA, "%s%s" % (resn, resi)
set label_size, 11
set label_color, black

# Highlight interface contact zone
select interface_zone, (chain A and resi 31+32+37+38+39+40+41+64+65+66+67+69+84+85+86+87) within 5 of chain B
show surface, interface_zone
set surface_transparency, 0.3, interface_zone
color red, interface_zone

# Final view
zoom protein
orient protein
set ray_shadows, 0
set antialias, 2
ray 1600, 1200
png ecabsd_1AY7_binding_site.png, dpi=300

# ---- Save session ----
save ecabsd_1AY7.pse

# Run in PyMOL: File > Open > select this .pml file
# Or from terminal: pymol ecabsd_1AY7_visualization.pml