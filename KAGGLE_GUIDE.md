# Kaggle Step-by-Step Guide
## EuroSAT Land-Use Classification

This guide takes you from zero to a fully trained model on Kaggle, step by step.
Every action is explained — no assumed knowledge.

---

## Table of Contents

1. [Kaggle Account & API Setup](#1-kaggle-account--api-setup)
2. [Dataset Setup](#2-dataset-setup)
3. [Upload the Project to Kaggle](#3-upload-the-project-to-kaggle)
4. [Create and Configure a Kaggle Notebook](#4-create-and-configure-a-kaggle-notebook)
5. [Install Dependencies & Run Training](#5-install-dependencies--run-training)
6. [Expected Outputs & Logs](#6-expected-outputs--logs)
7. [Saving & Downloading Results](#7-saving--downloading-results)
8. [Common Errors & Fixes](#8-common-errors--fixes)
9. [Trying Different Models](#9-trying-different-models)

---

## 1. Kaggle Account & API Setup

### 1.1 Create a Kaggle Account

1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Click **Register** and create a free account.
3. Verify your email address.

### 1.2 Generate Your API Token

The Kaggle API token lets you download datasets and interact with Kaggle programmatically.

1. Log in to Kaggle.
2. Click your profile picture (top-right) → **Settings**.
3. Scroll down to the **API** section.
4. Click **Create New Token**.
5. A file called `kaggle.json` will download automatically.

   It looks like this:
   ```json
   {"username": "your_username", "key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
   ```
6. **Keep this file private.** Do not commit it to git or share it publicly.

### 1.3 Enable Phone Verification (Required for GPU)

Kaggle requires phone verification before you can use free GPU compute.

1. Go to **Settings → Phone Verification**.
2. Enter your phone number and verify via SMS.
3. This unlocks **30 GPU hours per week** for free.

### 1.4 Enable GPU in a Notebook

When you open a Kaggle notebook:

1. On the right sidebar, find **Session options** (or click the three-dot menu `⋮`).
2. Click **Accelerator** → select **GPU T4 x2** (or P100 if available).
3. Click **Save**.
4. The notebook will restart with GPU enabled.

> **Why:** The model trains ~50× faster on a GPU. Without it, 30 epochs on EuroSAT
> (~27,000 images) would take several hours instead of ~15 minutes.

---

## 2. Dataset Setup

### 2.1 Find EuroSAT on Kaggle

The EuroSAT RGB dataset is available at:

```
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
```

Alternative mirrors (same data, different slug):
- `https://www.kaggle.com/datasets/madhavmalhotra/eurosat-dataset`
- Search "eurosat" on Kaggle Datasets and look for a dataset with ~90 MB, 10 classes

### 2.2 Add EuroSAT as a Notebook Input

You do **not** need to download the dataset locally.

1. Open your Kaggle notebook (see Section 4).
2. On the right panel, click **+ Add Data**.
3. Search for `eurosat`.
4. Click the dataset → click **Add** (blue button).
5. The dataset will mount at `/kaggle/input/eurosat-dataset/` (or similar).

**Verify the path** by running this cell:
```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for d in dirs:
        print(os.path.join(root, d))
    break  # only top level
```

The EuroSAT folder should contain 10 class subdirectories:
```
/kaggle/input/eurosat-dataset/EuroSAT/AnnualCrop/
/kaggle/input/eurosat-dataset/EuroSAT/Forest/
/kaggle/input/eurosat-dataset/EuroSAT/HerbaceousVegetation/
...
```

> **Note:** If the path differs, update `DATASET_PATH` in notebook cell 4 or pass
> `--override data.dataset_path=<your_path>` to the training script.

---

## 3. Upload the Project to Kaggle

You have two options.

### Option A — Upload as a Kaggle Dataset (Recommended)

This is the cleanest approach.

**Steps:**

1. On your local machine, zip the project folder:
   ```bash
   # On Windows PowerShell:
   Compress-Archive -Path Kaggle_EuroSAT_Classification -DestinationPath Kaggle_EuroSAT_Classification.zip
   # On Linux/Mac:
   zip -r Kaggle_EuroSAT_Classification.zip Kaggle_EuroSAT_Classification/
   ```

2. Go to [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets).
3. Click **New Dataset** (top right).
4. Click **Upload** → drag in `Kaggle_EuroSAT_Classification.zip`.
5. Name it `eurosat-ml-project`.
6. Set visibility to **Private**.
7. Click **Create**.
8. Wait for Kaggle to process it (~1–2 minutes).

**In your notebook:**

1. Click **+ Add Data** on the right panel.
2. Search for `eurosat-ml-project`.
3. Click **Add**.
4. The project will be at `/kaggle/input/eurosat-ml-project/Kaggle_EuroSAT_Classification/`.

In notebook cell 3, the `POSSIBLE_ROOTS` list already includes this path:
```python
"/kaggle/input/eurosat-ml-project/Kaggle_EuroSAT_Classification"
```

### Option B — Copy Files Directly via Cell Magic

If you prefer not to upload a dataset, you can use the Kaggle notebook editor to
create each file manually (tedious for large projects but works for quick tests):

1. In a notebook cell, use `%%writefile` magic:
   ```python
   %%writefile /kaggle/working/src/utils/config.py
   # paste file contents here
   ```

2. Or clone from a public GitHub repo if you've pushed the code:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/eurosat-classification.git /kaggle/working/Kaggle_EuroSAT_Classification
   ```

---

## 4. Create and Configure a Kaggle Notebook

### 4.1 Create a New Notebook

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code).
2. Click **New Notebook**.
3. A blank notebook opens in the browser editor.

### 4.2 Upload the Notebook File

1. Click **File** (top menu) → **Import Notebook**.
2. Upload `notebooks/kaggle_notebook.ipynb` from your local project.

   Or copy the notebook cell-by-cell into the new notebook.

### 4.3 Enable GPU

1. In the right sidebar, click **Session options** (or **Settings**).
2. Set **Accelerator** → **GPU T4 x2**.
3. The session will restart.

### 4.4 Attach the Datasets

In the right sidebar under **Data**:

1. Click **+ Add Data** → search `eurosat-dataset` → click **Add**.
2. Click **+ Add Data** → search `eurosat-ml-project` → click **Add**.

---

## 5. Install Dependencies & Run Training

### 5.1 Run the Notebook Top-to-Bottom

Click **Run All** (double-arrow icon at the top) or run cells one at a time.

**Cell 1** — Environment check:
```
PyTorch version : 2.1.x
CUDA available  : True
GPU             : Tesla T4
VRAM            : 15.0 GB
```
If CUDA is `False`, stop and enable GPU (see Section 1.4).

**Cell 2** — Install dependencies:
```bash
pip install -q timm>=0.9.12 tqdm scikit-learn PyYAML
```
This takes ~20 seconds.

**Cell 3** — Locate project root:
```
Project root: /kaggle/input/eurosat-ml-project/Kaggle_EuroSAT_Classification
✓ src package importable
```

**Cell 4** — Verify dataset:
```
Dataset found at: /kaggle/input/eurosat-dataset/EuroSAT
AnnualCrop                      2000 images
Forest                          3000 images
HerbaceousVegetation            3000 images
Highway                         2500 images
Industrial                      2500 images
Pasture                         2000 images
PermanentCrop                   2500 images
Residential                     3000 images
River                           2500 images
SeaLake                         3000 images
TOTAL                          27000 images
```

**Cell 5** — Configure training (no action needed, auto-configured).

**Cell 6** — Start training (this is the main cell):
```
Running: python scripts/train.py --config configs/config.yaml ...
```

### 5.2 Watch the Training Logs

You will see output like:
```
[2024-01-15 10:23:01] [INFO    ] __main__ — Device: cuda
[2024-01-15 10:23:01] [INFO    ] __main__ — GPU: Tesla T4  (VRAM: 15.0 GB)
[2024-01-15 10:23:05] [INFO    ] src.datasets.eurosat — Discovered 27000 images across 10 classes
[2024-01-15 10:23:05] [INFO    ] __main__ — Split sizes → train: 18900  val: 4050  test: 4050
[2024-01-15 10:23:07] [INFO    ] src.models.model — Building model: efficientnet_b0  pretrained=True  num_classes=10
[2024-01-15 10:23:07] [INFO    ] src.models.model — Parameters — total: 5.33M  trainable: 5.33M
...
[2024-01-15 10:25:12] [INFO    ] __main__ — Epoch [1/30]  train_loss=1.2341  train_acc=0.5823  val_loss=0.7234  val_acc=0.7612  lr=1.00e-04  time=62.3s
[2024-01-15 10:27:18] [INFO    ] __main__ — Epoch [2/30]  train_loss=0.8123  train_acc=0.7234  val_loss=0.5432  val_acc=0.8234  ...
...
[2024-01-15 10:58:45] [INFO    ] __main__ — Epoch [30/30] train_loss=0.2341  train_acc=0.9312  val_loss=0.2891  val_acc=0.9187  ...
[2024-01-15 10:58:47] [INFO    ] __main__ — Training complete.  Best val_acc=0.9234
```

> **Estimated time:** ~15–25 minutes for 30 epochs on a T4 GPU with EfficientNet-B0.

---

## 6. Expected Outputs & Logs

### 6.1 Expected Accuracy

| Model              | Val Accuracy | Test Accuracy | F1 Macro |
|--------------------|-------------|--------------|----------|
| EfficientNet-B0    | ~91–93%     | ~90–92%      | ~0.91    |
| ResNet-50          | ~89–91%     | ~88–90%      | ~0.89    |
| ViT-S/16           | ~90–93%     | ~89–92%      | ~0.90    |

These figures are realistic for 30 epochs with the default config.
Running for 50+ epochs may push accuracy to 94–95%.

### 6.2 Output Files

After training, `/kaggle/working/outputs/` will contain:

```
outputs/
├── train.log                          ← full training log
├── history.json                       ← per-epoch metrics
├── checkpoints/
│   ├── best_model.pt                  ← best val-accuracy checkpoint
│   ├── periodic_epoch005.pt
│   ├── periodic_epoch010.pt           ← periodic snapshots
│   └── ckpt_epoch030_acc0.9187.pt     ← top-k checkpoints
└── evaluation/
    ├── metrics.json                   ← accuracy, F1 scores
    ├── classification_report.txt      ← per-class precision/recall/F1
    ├── training_curves.png            ← loss & accuracy plots
    ├── confusion_matrix.png           ← 10×10 confusion matrix
    └── f1_per_class.png               ← per-class F1 bar chart
```

### 6.3 Confusion Matrix Interpretation

The confusion matrix shows which classes the model confuses most.
Common confusions in EuroSAT:
- **AnnualCrop ↔ PermanentCrop** (visually similar vegetation patterns)
- **HerbaceousVegetation ↔ Pasture** (similar green textures)
- **River ↔ SeaLake** (both water bodies)

---

## 7. Saving & Downloading Results

### 7.1 Download Individual Files

1. On the right panel in Kaggle, click the folder icon (**Output** tab).
2. Navigate to `/kaggle/working/outputs/`.
3. Click the three-dot menu next to any file → **Download**.

### 7.2 Download Everything as a Zip

In the notebook, run:
```python
import shutil
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working/outputs")
print("Created: /kaggle/working/results.zip")
```
Then download `results.zip` from the Output panel.

### 7.3 Save as a Kaggle Dataset Output

When you click **Save & Run All (Commit)** in Kaggle:
1. The notebook runs from scratch in a fresh environment.
2. All files in `/kaggle/working/` become notebook outputs.
3. You can access them later from the notebook's **Output** tab.
4. You can also version and share them as a Kaggle dataset.

### 7.4 Load the Checkpoint Locally

```python
# Load the model checkpoint on your local machine
import torch
from src.models.model import build_model
from src.utils.config import load_config

cfg = load_config("configs/config.yaml", overrides=[
    "data.dataset_path=./data/EuroSAT"
])
model = build_model(cfg)
checkpoint = torch.load("outputs/checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded. Val accuracy was:", checkpoint["metrics"]["val_acc"])
```

---

## 8. Common Errors & Fixes

### Error: `CUDA not available` / No GPU

**Symptom:**
```
Device: cpu
WARNING: No GPU detected. Training will be very slow.
```

**Fix:**
1. In the Kaggle notebook sidebar → **Session options** → **Accelerator** → **GPU T4 x2**.
2. Click **Save** and wait for the session to restart.
3. Re-run all cells.

---

### Error: `FileNotFoundError: Dataset not found`

**Symptom:**
```
FileNotFoundError: Dataset not found at: /kaggle/input/eurosat-dataset/EuroSAT
```

**Fix:**
1. Make sure EuroSAT is added as a notebook input (right panel → **+ Add Data**).
2. Check the actual path:
   ```python
   import os
   for d in os.listdir("/kaggle/input"):
       print(d)
   ```
3. Update `DATASET_PATH` in notebook cell 4 to match the actual path.
4. Or add an override: `--override data.dataset_path=/kaggle/input/YOUR_PATH/EuroSAT`

---

### Error: `ModuleNotFoundError: No module named 'src'`

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Fix:**
1. The project root was not found or not added to `sys.path`.
2. Check cell 3 output — it should show `Project root: /kaggle/input/...`.
3. Manually set: `REPO_ROOT = "/kaggle/input/eurosat-ml-project/Kaggle_EuroSAT_Classification"`
4. Re-run cell 3.

---

### Error: `ModuleNotFoundError: No module named 'timm'`

**Symptom:** `ModuleNotFoundError: No module named 'timm'`

**Fix:**
```python
!pip install timm
```
Then restart the kernel and re-run all cells.

---

### Error: `CUDA out of memory`

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Fix:** Reduce batch size:
```python
OVERRIDES.append("training.batch_size=32")  # default is 64
```
Or switch to a smaller model:
```python
OVERRIDES.append("model.architecture=efficientnet_b0")  # already the default
```

For ViT models, use batch size 16–32:
```python
"training.batch_size=16",
"model.architecture=vit_small_patch16_224",
```

---

### Error: `No images found`

**Symptom:**
```
RuntimeError: No images found under /kaggle/input/.../EuroSAT
```

**Fix:** The dataset directory structure doesn't match.
Expected: `EuroSAT/<ClassName>/<image>.jpg`

Investigate:
```python
from pathlib import Path
root = Path("/kaggle/input/eurosat-dataset")
for item in root.rglob("*"):
    print(item)
    break  # remove to see all
```
Adjust `DATASET_PATH` accordingly.

---

### Error: Training seems stuck / progress bar not moving

**Symptom:** The progress bar shows `0%` for minutes.

**Likely cause:** `num_workers > 0` on Kaggle can cause issues with some Kaggle images.

**Fix:**
```python
OVERRIDES.append("data.num_workers=0")
```

---

### Error: Kaggle session disconnects mid-training

**Symptom:** Notebook output truncated, no final results.

**Fix:**
1. Use **Save & Run All (Commit)** instead of interactive execution.
2. Committed runs continue even if your browser closes.
3. Resume training from the latest checkpoint:
   ```python
   OVERRIDES.append("training.resume_from=/kaggle/working/outputs/checkpoints/periodic_epoch010.pt")
   ```

---

## 9. Trying Different Models

Change the architecture with a single override.  No other code changes needed.

### EfficientNet variants (fast, accurate)

```python
"model.architecture=efficientnet_b0",   # 5.3M params, fastest
"model.architecture=efficientnet_b3",   # 12M params, more accurate
"model.architecture=efficientnet_b4",   # 19M params, slower
```

### ResNet baseline

```python
"model.architecture=resnet50",          # classic baseline
"model.architecture=resnet34",          # lighter
```

### ConvNeXt (modern CNN)

```python
"model.architecture=convnext_tiny",     # 28M params
```

### Vision Transformer (ViT)

```python
"model.architecture=vit_small_patch16_224",   # 22M params
"training.batch_size=32",
"training.learning_rate=3e-4",
```

### Swin Transformer

```python
"model.architecture=swin_tiny_patch4_window7_224",  # 28M params
"training.batch_size=32",
```

> **Tip:** To browse all available models:
> ```python
> import timm
> timm.list_models("efficientnet*")
> ```

---

## Quick Reference — Most Important Commands

```bash
# Train with default config (local):
python scripts/train.py

# Train with overrides:
python scripts/train.py \
    --override training.epochs=50 \
    --override model.architecture=resnet50

# Evaluate best model:
python scripts/evaluate.py

# Evaluate a specific checkpoint:
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --split test

# Resume training:
python scripts/train.py \
    --override training.resume_from=outputs/checkpoints/best_model.pt
```
