# Engineering Fast Li Transport in Solid-State Electrolytes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Code, data, and analysis for studying fast lithium-ion transport in solid-state electrolytes. This README explains the repository layout and how to get started.

---

## Table of Contents

* [Repository Layout](#repository-layout)
* [1_HT-DFT_parameter](#1_ht-dft_parameter)
* [2_Datasets](#2_datasets)
* [3_CGCNN](#3_cgcnn)
* [4_XGBoost](#4_xgboost)
* [5_SISSO](#5_sisso)
* [6_Explainability](#6_explainability)
* [Quick Start](#quick-start)
* [Reproducibility Checklist](#reproducibility-checklist)
* [Data & Large Files](#data--large-files)
* [Citations](#citations)
* [License](#license)
* [How to Cite This Repository](#how-to-cite-this-repository)

---

## Repository Layout

* [**1_HT-DFT_parameter/**](#1_ht-dft_parameter) — High-throughput DFT workflows (VASP 5.4.4): data collection, surface models, frozen atoms, Li/S single-atom placement, LiTFSI placement, and run parameters.
* [**2_Datasets/**](#2_datasets) — De-duplicated, cleaned datasets; categorized by band gap.
* [**3_CGCNN/**](#3_cgcnn) — Crystal Graph Convolutional Network (CGCNN) training and inference.
* [**4_XGBoost/**](#4_xgboost) — Feature engineering and XGBoost baselines.
* [**5_SISSO/**](#5_sisso) — Parameters and inputs for descriptor discovery (SISSO).
* [**6_Explainability/**](#6_explainability) — Descriptor/model explainability (e.g., SHAP).

---

## 1_HT-DFT_parameter

All first-principles calculations in this project were performed with **VASP 5.4.4** (use under a valid VASP license).

**Contents & workflow (1 → 6):**

1. `1_Data_collection.ipynb` — Collect crystal structures and metadata for target systems.
2. `2_Surfaces_building.ipynb` — Build surface/interface models (slabs, supercells, terminations).
3. `3_Fix_atoms.ipynb` — Define frozen layers/atoms (e.g., bottom slab layers).
4. `4_Surface+Li_or_S_building.ipynb` — Place single **Li** or **S** atoms at candidate adsorption sites.
5. `5_Surface+LiTFSI_building.ipynb` — Place **LiTFSI** molecular groups on surfaces/interfaces.
6. `6_DFT_calculations_parameters.py` — High-throughput VASP runtime settings (INCAR/KPOINTS/POTCAR choices, queue submission helpers).

> **Notes**
>
> * This repo does **not** distribute VASP binaries or POTCARs. Use your institution’s licensed VASP installation and update submission scripts accordingly.
> * Keep per-system settings (ENCUT, k-mesh, smearing, dipole correction, U/functional) version-controlled with summary tables for reproducibility.
> * Recommended methodology citations are listed in [Citations](#citations).

---

## 2_Datasets

De-duplicated and cleaned datasets used across models. Data are **categorized by band gap**, with JSON files corresponding to energy ranges:

* `binary_tm_compounds_0eV.json` — 0 eV (metallic)
* `binary_tm_compounds_0.1-1.5eV.json` — 0.1–1.5 eV
* `binary_tm_compounds_1.5-3eV.json` — 1.5–3 eV
* `binary_tm_compounds_3-6eV.json` — 3–6 eV
* `binary_tm_compounds_6-10eV.json` — 6–10 eV

**Suggested structure**

```
2_Datasets/
├─ raw/            # (optional) original structures/labels
├─ processed/      # featurized or normalized data
├─ splits/         # train/val/test indices or CV folds
└─ metadata.md     # dataset schema, provenance, and licensing
```

> **Large files**: Track big artifacts with **Git LFS** (GitHub blocks pushing files ≥ **100 MB**). For public distribution, publish versioned artifacts via **GitHub Releases**.

---

## 3_CGCNN

Code for training and evaluating **Crystal Graph Convolutional Neural Networks**.

**Files**

* `main_regress_basic.py` — Basic regression training pipeline.
* `main_regress_k_fold.py` — K-fold cross-validation training/evaluation.

**Run (example)**

```bash
# after environment setup
python 3_CGCNN/main_regress_basic.py
# or
python 3_CGCNN/main_regress_k_fold.py
```

Please cite: Xie & Grossman, Physical Review Letters 120, 145301 (2018).

---

## 4_XGBoost

Feature-based baselines implemented with **XGBoost**.

**Files**

* `xgboost.py` — Training/evaluation script (configure feature and data paths inside).

**Run (example)**

```bash
python 4_XGBoost/xgboost.py
```

Please cite: Chen & Guestrin, KDD (2016), “XGBoost: A Scalable Tree Boosting System.”

---

## 5_SISSO

Inputs and parameterization for SISSO descriptor discovery.

**Files**

* `SISSO.in` — Operator set, rung/depth, sparsity, target, and constraints for descriptor search.

**Execute**

```bash
# Use your SISSO/SISSO++ binary; see upstream installation guides.
/path/to/SISSO
```

Please cite: Ouyang et al., Physical Review Materials 2, 083802 (2018).

---

## 6_Explainability

Explainability and diagnostics for descriptors/models (e.g., **SHAP**).

**Files**

* `SHAP_cgcnn.ipynb` — SHAP analysis for CGCNN predictions (global/local effects).
* `SHAP_xgboost.py` — SHAP pipeline for XGBoost features.

**Run (example)**

```bash
# Notebook
jupyter lab  # then open 6_Explainability/SHAP_cgcnn.ipynb

# Script
python 6_Explainability/SHAP_xgboost.py
```

Please cite: Lundberg & Lee, NeurIPS 2017, “A Unified Approach to Interpreting Model Predictions.”

---

## Citations

**DFT / VASP methodology**

* G. Kresse, J. Furthmüller, *Phys. Rev. B* 54, 11169–11186 (1996).
* G. Kresse, D. Joubert, *Phys. Rev. B* 59, 1758–1775 (1999).

**CGCNN**

* T. Xie, J. C. Grossman, *Phys. Rev. Lett.* 120, 145301 (2018).

**SISSO**

* R. Ouyang, E. Ahmetcik, M. Scheffler, L. M. Ghiringhelli, *Phys. Rev. Materials* 2, 083802 (2018).

**XGBoost**

* T. Chen, C. Guestrin, KDD (2016). “XGBoost: A Scalable Tree Boosting System.”

**Explainability (SHAP)**

* S. M. Lundberg, S.-I. Lee, NeurIPS 2017. arXiv:1705.07874.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
