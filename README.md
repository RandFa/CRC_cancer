# 🧬 Colorectal Cancer (CRC) Bulk RNA-seq Atlas Exploration

## 📌 Objective

This project focuses on exploratory data analysis and hypothesis-driven bioinformatic workflows using a colorectal cancer (CRC) bulk RNA-seq dataset. The goal is to implement and evaluate state-of-the-art methods for understanding cohort composition, uncovering biological structure, and generating clinical insights.

---

## 📂 Input Data

- **Gene expression matrix** (bulk RNA-seq)
- **Clinical metadata**
- Batch effects already corrected using `pycombat-seq`

> Data are stored in an external drive and not included in this repo.

---

## ⚙️ Setup Instructions

### 1. Clone this Repository

```bash
git clone https://github.com/RandFa/epigene_test_CRC_cancer
cd epigene_test_CRC_cancer
``` 
### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate crc_env
```

## Respratory structure:
project-root/
├── data/           # Placeholder for local data
├── notebooks/      # EDA and analysis notebooks
├── scripts/        # Modular scripts for reproducibility
├── results/        # Figures and summary tables
├── environment.yml # Conda environment
├── .gitignore
└── README.md
