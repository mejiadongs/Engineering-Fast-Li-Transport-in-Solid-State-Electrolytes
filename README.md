# Engineering Fast Li Transport in Solid-State Electrolytes

This repository hosts code, data, and analysis for studying fast lithium-ion transport in solid-state electrolytes. The index below explains the purpose of each top-level folder and how to get started.

## Repository Layout
- [1_DFT_parameter/](#1_dft_parameter) — High-throughput DFT calculation parameters and templates  
- [2_Datasets/](#2_datasets) — Datasets of examples and accompanying documentation  
- [3_CGCNN/](#3_cgcnn) — Crystal Graph Convolutional Network (CGCNN) training and inference  
- [4_XGBoost/](#4_xgboost) — Feature engineering and XGBoost baselines  
- [5_SISSO/](#5_sisso) — SISSO descriptor search and selection  
- [6_Explainability/](#6_explainability) — Model explainability and diagnostics  

---

### 1_DFT_parameter
Inputs and scripts for first-principles (DFT) calculations.  
**Notes**
- All DFT calculations in this project were performed using **VASP 5.4.4**. Please ensure you (or your HPC center) hold a valid VASP license before running or modifying any workflows.  
- Typical contents:
  - `inputs/` parameter templates (e.g., INCAR/INP, KPOINTS, POTCAR guidance)
  - `examples/` benchmark systems and recommended settings
  - `scripts/` submission helpers and parsers
- Recommended citations for VASP methodology (add to your papers/slides):  
  - Kresse & Furthmüller, *Phys. Rev. B* **54**, 11169 (1996), doi: **10.1103/PhysRevB.54.11169**  
  - Kresse & Joubert, *Phys. Rev. B* **59**, 1758 (1999), doi: **10.1103/PhysRevB.59.1758**  

---

### 2_Datasets
Datasets and data preparation artifacts.
- Suggested structure:
  - `raw/` original structures/labels/metadata
  - `processed/` cleaned/featurized data (ready for modeling)
  - `splits/` train/val/test (or CV) indices
  - `metadata.md` field dictionary, provenance, and licensing
- **Large files**: please use Git LFS for versioning and/or GitHub Releases for distributing artifacts.

---

### 3_CGCNN
Training/inference code and configs for **CGCNN** (Crystal Graph Convolutional Neural Networks).
- Paper to cite: Xie & Grossman, *Phys. Rev. Lett.* **120**, 145301 (2018), doi: **10.1103/PhysRevLett.120.145301**.  
- Environment: install dependencies per the upstream CGCNN instructions (PyTorch + Python toolchain).  
- Typical contents:
  - `train.py`, `predict.py` entry points
  - `configs/` hyperparameters and experiment settings
  - `data/` graph conversion utilities or cached graphs
  - `checkpoints/` trained weights and logs

---

### 4_XGBoost
Feature-based baselines using **XGBoost**.
- Paper to cite: Chen & Guestrin, KDD’16, doi: **10.1145/2939672.2939785**.  
- Environment: install the `xgboost` package (via `pip` or `conda`) and the usual Python stack for data handling/plots.
- Suggested structure:
  - `features/` feature generation scripts
  - `train.py` training with CV/early stopping
  - `eval/` metrics, plots, and ablations

---

### 5_SISSO
Sparse descriptor discovery with **SISSO**.
- Paper to cite: Ouyang *et al.*, *Phys. Rev. Materials* **2**, 083802 (2018), doi: **10.1103/PhysRevMaterials.2.083802**.  
- Environment: build the original Fortran SISSO or **SISSO++** (C++ with Python bindings); see the referenced installation guides.
- Typical contents:
  - `prep/` feature space definitions and operators
  - `run/` configs and execution scripts
  - `descriptors/` candidate and final descriptors

---

### 6_Explainability
Model interpretability and diagnostics across methods.
- Common tooling: SHAP (SHapley Additive exPlanations) for global/local importance.  
  - Reference: Lundberg & Lee, NeurIPS 2017 (SHAP).  
- Suggested structure:
  - `shap/` SHAP value computation and plots
  - `feature_importance/` aggregated importances across models
  - `case_studies/` qualitative analyses and error forensics

---

## Quick Start

```bash
# 1) Clone
git clone <REPO_URL>
cd Engineering-Fast-Li-Transport-in-Solid-State-Electrolytes

# 2) Create a clean environment (example with conda)
conda create -n fastli python=3.11 -y
conda activate fastli

# 3) Install common Python deps (edit requirements.txt as your project evolves)
pip install -r requirements.txt

# 4) Per-module setup (examples)
# CGCNN: follow upstream instructions (PyTorch + dependencies), then:
#   python 3_CGCNN/train.py --config configs/base.yaml

# XGBoost:
pip install xgboost
#   or: conda install -c conda-forge xgboost

# SISSO/SISSO++:
#   Build from source per SISSO or SISSO++ docs (CMake + C++14/BLAS/LAPACK/MPI).
