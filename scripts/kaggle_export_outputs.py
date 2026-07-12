"""
Cell 13 — Smart Checkpoint Finder & Exporter
Paste this as a new cell in your Kaggle notebook.
Finds the checkpoint and all results regardless of where they were saved.
"""
import shutil, os, glob, zipfile

print("🔍 Searching for checkpoint and results files everywhere...")

def find_file(filename):
    matches = glob.glob(f'/kaggle/working/**/{filename}', recursive=True)
    return matches[0] if matches else None

# ── Find all key outputs ───────────────────────────────────────────────────
TOP = '/kaggle/working'  # Top-level = visible in Kaggle Output panel

targets = {
    'best_model_v3.pt':             find_file('best_model_v3.pt'),
    'test_metrics.json':            find_file('test_metrics.json'),
    'statistical_validation.json':  find_file('statistical_validation.json'),
    'calibration_stats.json':       find_file('calibration_stats.json'),
    'hotspot_correlations.json':    find_file('hotspot_correlations.json'),
    'error_analysis_report.json':   find_file('error_analysis_report.json'),
    'training_history_v3.json':     find_file('training_history_v3.json'),
}

print('\n📦 Copying files to /kaggle/working/ (top-level = easy download from Output panel):')
for dst_name, src in targets.items():
    dst = os.path.join(TOP, dst_name)
    if src and os.path.exists(src):
        shutil.copy2(src, dst)
        size = os.path.getsize(dst) / 1e6
        print(f'  ✅ {dst_name} ({size:.1f} MB)')
    else:
        print(f'  ⚠️  {dst_name} — not found')

# ── Zip all figures ────────────────────────────────────────────────────────
figures = glob.glob('/kaggle/working/**/figures/*.png', recursive=True)
if figures:
    zip_path = os.path.join(TOP, 'ecabsd_figures.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for fig in figures:
            zf.write(fig, os.path.basename(fig))
    print(f'  ✅ ecabsd_figures.zip ({len(figures)} figures zipped)')

# ── Show what's downloadable ───────────────────────────────────────────────
print('\n✅ Done! Look for these files in the Kaggle Output panel (right side):')
downloadable = [f for f in os.listdir(TOP)
                if os.path.isfile(os.path.join(TOP, f))]
for f in sorted(downloadable):
    size = os.path.getsize(os.path.join(TOP, f)) / 1e6
    print(f'   📄 {f} ({size:.1f} MB)')
