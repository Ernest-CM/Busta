# LEARNT.md — Errors Encountered and Lessons Applied

A living reference of real errors hit during development and deployment of this project.
Each entry includes: what went wrong, why, and the exact fix. Use this in future projects.

---

## 1. Wrong Working Directory

**Error:** Commands failing because the project root was misidentified.

**Context:** The repo was cloned/uploaded as a nested folder — the actual project root was at
`PROJECTS\Busta\Busta\` not `PROJECTS\Busta\`.

**Fix:** Always verify the directory that contains `requirements.txt`, `src/`, `tests/` before running anything:
```bash
ls   # should show src/, scripts/, requirements.txt, etc.
```

---

## 2. `pip install` Without Arguments

**Error:**
```
ERROR: You must give at least one requirement to install
```

**Cause:** Ran `pip install` with no arguments. The `-r` flag and filename are required.

**Fix:**
```bash
pip install -r requirements.txt
```

---

## 3. pip Network Timeout on Large Packages

**Error:**
```
TimeoutError: The read operation timed out
ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

**Cause:** Default pip timeout (15s) too short for large packages like scipy (45 MB) or TensorFlow (600 MB) on a slow connection.

**Fix:**
```bash
pip install --timeout 300 --retries 5 -r requirements.txt
```
Or use conda which has better CDN coverage:
```bash
conda install -c conda-forge <package>
```

---

## 4. pyenv Not Installed on Windows

**Error:**
```
'pyenv' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

**Cause:** pyenv-win requires a separate installer — it is not bundled with Windows.

**Fix (Windows):** Use the built-in Python Launcher instead:
```bash
py --list          # see all installed versions
py -3.11 -m venv .venv   # create venv with specific version
```
Or if Anaconda/Miniconda is installed, use `conda create -n myenv python=3.11`.

---

## 5. TensorFlow Incompatible with Python 3.13+

**Error:** `tensorflow=2.16` has no wheel for Python 3.13 or 3.14.

**Cause:** TensorFlow releases lag behind CPython. TF 2.16.x only supports Python 3.9–3.12.

**Fix:** Always check TF compatibility before creating the environment:
- Use `py -3.11` or `py -3.12` to create the virtualenv
- Or: `conda create -n myenv python=3.11`
- Verify: `python --version` must show 3.9–3.12 before installing TF

---

## 6. `gdk-pixbuf-query-loaders.exe` Entry Point Not Found (Conda on Windows)

**Error (popup dialog):**
```
The procedure entry point libintl_bind_textdomain_codeset could not be located in gdk-pixbuf-2.0.dll
```

**Cause:** Conda-forge's `gdk-pixbuf` package has a DLL version conflict with an existing system library on Windows. Appears during conda transaction execution.

**Fix:** This is **harmless** — click OK on every popup. The conda installation completes successfully. Does not affect Python packages or TensorFlow.

---

## 7. TensorFlow Not Available on conda-forge

**Error:**
```
PackagesNotFoundError: The following packages are not available from current channels:
  - tensorflow=2.16
```

**Cause:** conda-forge does not always carry all TensorFlow versions for all platforms.

**Fix:** Install TensorFlow via pip inside the conda environment:
```bash
conda activate myenv
pip install tensorflow==2.16.1 --timeout 300
```
Conda handles all other scientific packages (numpy, scipy, opencv) faster; pip handles TF.

---

## 8. OpenCV DLL Load Failure on Windows (conda-forge Qt6 variant)

**Error:**
```
ImportError: DLL load failed while importing cv2: The specified procedure could not be found.
Windows fatal exception: code 0xc0000139
```

**Cause:** conda-forge installs `opencv` as the Qt6 GUI variant (`libopencv-4.12.0-qt6_...`) which pulls in dozens of system DLLs that conflict on Windows.

**Fix:** Remove the conda opencv and install the self-contained pip version:
```bash
conda remove opencv py-opencv libopencv --force -y
pip install opencv-python
```
The pip wheel bundles its own DLLs — no system DLL conflicts.

---

## 9. NumPy 2.x Incompatible with TensorFlow 2.16

**Error:**
```
ImportError('numpy.core.umath failed to import')
```

**Cause:** conda-forge installs `numpy 2.4.x` by default, but TensorFlow 2.16.1 was built against NumPy 1.x. In NumPy 2.x, `numpy.core` was restructured/removed.

**Fix:** Downgrade NumPy after installing TF:
```bash
pip install "numpy==1.26.4"
```
**Rule:** When mixing conda + pip for TF projects, always pin numpy to `<2.0`.

---

## 10. OpenCV Requires NumPy ≥ 2 (Version Conflict)

**Error:**
```
opencv-python 4.13.0.92 requires numpy>=2; python_version >= "3.9"
```

**Cause:** After downgrading numpy to 1.26.4, the latest opencv-python (4.13+) refuses to install.

**Fix:** Pin opencv-python to a version that supports numpy 1.x:
```bash
pip install "opencv-python==4.10.0.84"
```
**Rule:** When pinning numpy `<2`, also pin `opencv-python<=4.10.x`.

---

## 11. `ModuleNotFoundError: No module named 'src'`

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** Python doesn't add the current directory to `sys.path` when running scripts directly. `pyproject.toml` sets `pythonpath = ["."]` for pytest only — not for plain `python` calls.

**Fix (per session):**
```bash
set PYTHONPATH=.          # Windows CMD / Anaconda Prompt
$env:PYTHONPATH="."       # PowerShell
export PYTHONPATH=.       # bash/zsh
```
**Fix (permanent for the project):** Add to `.env` and load with `python-dotenv`, or add a `conftest.py` that inserts the root into `sys.path`.

---

## 12. Training Runs Silently — No Progress Output

**Symptoms:** Script runs for minutes with no output. User has no idea if it's working or hung.

**Cause 1:** `model.fit(..., verbose=0)` — Keras silences all epoch output.
**Cause 2:** `preprocess_records()` iterates 17k+ images with no progress indicator.

**Fix:**
- Change `verbose=0` → `verbose=1` in all `model.fit()` calls for interactive runs.
- Wrap the preprocessing loop with `tqdm`:
```python
from tqdm import tqdm
for record in tqdm(record_list, desc="Preprocessing images", unit="img"):
    ...
```
**Rule:** Never ship a long-running loop without a progress indicator. Use `tqdm` everywhere.

---

## 13. Running Evaluation Before Training

**Error:**
```
FileNotFoundError: Model artifact not found: models_artifacts/baseline_cnn.keras
```

**Cause:** `run_evaluation.py` loads the trained model, but it was run before `run_training.py` created the artifact.

**Fix:** Always follow the correct execution order:
```
1. run_training.py       → creates baseline_cnn.keras
2. run_evaluation.py     → loads baseline_cnn.keras
3. run_api.py            → loads baseline_cnn.keras and serves
```

---

## 14. Kaggle CLI Not Installed / No Credentials

**Error:**
```
'kaggle' is not recognized ...
```
or
```
OSError: Could not find kaggle.json
```

**Fix:**
```bash
pip install kaggle
mkdir %USERPROFILE%\.kaggle
# place kaggle.json from kaggle.com → Settings → API → Create New Token
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```
For CI/scripted use, set the env var:
```bash
set KAGGLE_TOKEN=KGAT_xxxxx
```

---

## 15. No `.gitignore` — Large Files Staged by Accident

**Symptoms:** `git status` shows model `.keras` files, 675 MB zip, 27k raw images, `.venv/`, `__pycache__/` all as untracked.

**Fix:** Always create `.gitignore` before the first commit. Minimum entries for a Python ML project:
```gitignore
__pycache__/
*.pyc
.venv/
data/raw/          # never commit raw datasets
data/processed/    # never commit generated arrays
models_artifacts/*.keras
models_artifacts/*.h5
```

---

## Summary Table

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | Wrong directory | Nested project folder | `ls` to verify before running |
| 2 | `pip install` no args | Missing `-r requirements.txt` | Always use `-r` flag |
| 3 | pip timeout | Slow connection + small default timeout | `--timeout 300 --retries 5` or conda |
| 4 | pyenv not found | Not installed on Windows | Use `py -3.11` launcher or conda |
| 5 | TF + Python 3.14 | No wheel for 3.13+ | Use Python 3.9–3.12 |
| 6 | gdk-pixbuf DLL popup | conda-forge Windows DLL conflict | Harmless — click OK |
| 7 | TF missing on conda-forge | Not always packaged | `pip install tensorflow` inside conda env |
| 8 | cv2 DLL load failed | conda opencv Qt6 variant conflicts | `conda remove opencv && pip install opencv-python` |
| 9 | numpy.core.umath import fail | numpy 2.x breaks TF 2.16 | `pip install "numpy==1.26.4"` |
| 10 | opencv requires numpy≥2 | opencv 4.13+ dropped numpy 1.x | `pip install "opencv-python==4.10.0.84"` |
| 11 | No module named 'src' | `sys.path` missing project root | `set PYTHONPATH=.` |
| 12 | Silent training | `verbose=0` + no tqdm | `verbose=1` + tqdm in loops |
| 13 | Eval before train | Wrong execution order | Train → Evaluate → Serve |
| 14 | Kaggle CLI missing | Not installed / no credentials | `pip install kaggle` + kaggle.json |
| 15 | Large files in git | No .gitignore | Create .gitignore before first commit |
