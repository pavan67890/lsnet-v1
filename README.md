# SOTA Change Detection (SYSU, WHU, LEVIR) 

## Overview

This repository contains a unified, production-ready codebase for semantic change detection across multiple datasets: SYSU-CD, WHU, and LEVIR. It provides:
- Modular model definition (LSNet and related components)
- Deterministic histogram matching preprocessing and geometric augmentations
- Robust training pipelines (base training, fine-tuning, sharpening)
- Evaluation utilities including Test-Time Augmentation (TTA), ROC/PR curves, and final reporting

The project is organized for clarity and contribution, with clean separation of source code, scripts, configuration, assets (checkpoints, logs), and documentation.

## Project Structure

```
S:\sota\
  ├─ sysu_project/           # SYSU dataset training and evaluation
  │   ├─ models/             # LSNet, backbones, decoders (SYSU variants)
  │   ├─ datasets/           # PatchChangeDetectionDataset and preprocessing
  │   ├─ utils/              # Common utilities
  │   ├─ train_sysu.py       # Base training (no resume for clean runs)
  │   ├─ train_sharpen.py    # Edge-weighted fine-tuning
  │   ├─ eval_tta.py         # 8-way TTA with morphology post-processing
  │   ├─ final_evaluation.py # Comprehensive final report generator
  │   ├─ sysu_config.py      # Configuration for training/eval
  │   └─ (checkpoints/, logs/)   # Git-ignored artifacts
  │
  ├─ levir_project/          # LEVIR training scripts and models
  │   ├─ models/ utils/      # LEVIR-specific components
  │   └─ train_levir.py      # Training entrypoint
  │
  ├─ project/                # WHU dataset scripts and data preparation
  │   ├─ models/ losses/     # Reusable components
  │   ├─ train.py            # WHU training
  │   └─ data/               # Dataset (Git-ignored; not tracked)
  │
  ├─ ffbdnetx_env/           # Local virtual environment (Git-ignored)
  ├─ .gitignore              # Excludes env, data, checkpoints, logs, caches
  ├─ requirements.txt        # Python dependencies
  └─ README.md               # This document
```

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare datasets (SYSU, WHU, LEVIR) according to their official splits. Place them under dataset-specific folders (see each project’s config).

## Usage

### SYSU Training (Clean Run)
```
cd S:\sota\sysu_project
set SYSU_CLEAN_RUN=1
python train_sysu.py
```
- Runs full training without resume
- Saves checkpoints to checkpoints/ (Git-ignored)
- Logs metrics to logs/

### Fine-Tuning (Sharpening)
```
cd S:\sota\sysu_project
python train_sharpen.py
```
- Loads base checkpoint (sysu_best.pth) and performs edge-weighted finetuning
- Outputs sysu_sharp_best.pth on IoU improvement

### TTA Evaluation with Morphology
```
cd S:\sota\sysu_project
python eval_tta.py
```
- 8-way TTA with morphological open/close post-processing
- Prints Precision, Recall, F1, IoU

### Final Evaluation Report
```
cd S:\sota\sysu_project
python final_evaluation.py
```
- Generates a comprehensive report under final_report/
- Saves: confusion_matrix.png, roc_curve.png, pr_curve.png, examples/, results.txt

## Configuration

Edit dataset-specific configs in:
- `sysu_project/sysu_config.py`  
For WHU/LEVIR, use their respective `config.py` files under each project folder.

## Dependencies

Core libraries:
- torch, torchvision
- numpy, tqdm
- scikit-image
- opencv-python
- matplotlib

Optional:
- kornia (for edge masks and image ops)

Install via:
```
pip install -r requirements.txt
```

## Contribution Guidelines

1. Fork the repository and create a feature branch.
2. Follow the existing code style and module layout:
   - Dataset preprocessing in `datasets/`
   - Models in `models/`
   - Utilities in `utils/`
   - Training/evaluation entrypoints as top-level scripts
3. Add tests or evaluation outputs where applicable.
4. Update README with usage instructions if introducing new scripts.
5. Submit a pull request with a clear description and checklist.

## Notes on Data and Checkpoints

- Datasets and checkpoints are not tracked by Git. Use `.gitignore` to exclude:
  - `data/`, `checkpoints/`, `logs/`, local `env/` directories
- Provide paths via config files; do not hardcode absolute paths in code if you plan to share.

## License

Add your project license here (e.g., MIT) before uploading to GitHub.

## Troubleshooting

- If plotting fails in headless environments, `final_evaluation.py` uses the Agg backend.
- For Windows multiprocessing issues, set:
  - `NUM_WORKERS = 0` for maximum stability, or
  - `NUM_WORKERS = 2`, `PIN_MEMORY = True`, `PERSISTENT_WORKERS = True`, `PREFETCH_FACTOR = 2` for performance

"# sota-try" 
