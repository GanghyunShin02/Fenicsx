# Code

This code is running Fencisx on colab.
When you want to use Fenicsx, writhe magic world %%fenicsx --np

Behind np, write count of pararell compute.
Behind that, write -- time to print runtime.

# --------------------------------------------------
# 1️⃣ Mount Google Drive (optional, for cache)
# --------------------------------------------------
from google.colab import drive
import os

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
else:
    print("📦 Google Drive already mounted")

# --------------------------------------------------
# 2️⃣ Clone fenicsx-colab repository (idempotent)
# --------------------------------------------------
from pathlib import Path
import subprocess

REPO_URL = "https://github.com/seoultechpse/fenicsx-colab.git"
ROOT = Path("/content")
REPO_DIR = ROOT / "fenicsx-colab"

def run(cmd):
    subprocess.run(cmd, check=True)

if not REPO_DIR.exists():
    print("📥 Cloning fenicsx-colab...")
    run(["git", "clone", REPO_URL, str(REPO_DIR)])
elif not (REPO_DIR / ".git").exists():
    raise RuntimeError("Directory exists but is not a git repository")
else:
    print("📦 Repository already exists — skipping clone")

# --------------------------------------------------
# 3️⃣ Run setup_fenicsx.py IN THIS KERNEL (CRITICAL)
# --------------------------------------------------
print("🚀 Running setup_fenicsx.py in current kernel")

# ⚙️ Configuration
USE_COMPLEX = False  # <--- Set True ONLY if you need complex PETSc
USE_CLEAN = False    # <--- Set True to remove existing environment

# Build options
opts = []
if USE_COMPLEX:
    opts.append("--complex")
if USE_CLEAN:
    opts.append("--clean")

opts_str = " ".join(opts) if opts else ""

get_ipython().run_line_magic(
    "run", f"{REPO_DIR / 'setup_fenicsx.py'} {opts_str}"
)

# --------------------------------------------------
# 4️⃣ Sanity check
# --------------------------------------------------
try:
    get_ipython().run_cell_magic('fenicsx', '--info -np 4', '')
except Exception as e:
    print("⚠️ %%fenicsx magic not found:", e)
