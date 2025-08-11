# 🧬 Colorectal Cancer (CRC) Bulk RNA-seq Atlas Exploration

## 📌 Objective

This project focuses on exploratory data analysis and hypothesis-driven bioinformatic workflows using a colorectal cancer (CRC) bulk RNA-seq dataset. The goal is to implement and evaluate state-of-the-art methods for understanding cohort composition, uncovering biological structure, and generating clinical insights.

---

## 📂 Input Data

- **Gene expression matrix** (bulk RNA-seq)  
- **Clinical metadata**  
- Batch effects already corrected using pycombat-seq  

> Data are stored on an external drive and **not included** in this repository.

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
conda activate epigene_crc_env
```

---

## 🗂️ Repository Structure

```
project-root/
├── data/           # Placeholder for local data (must include original datasets)
├── notebooks/      # EDA and analysis notebooks
├── scripts/        # Modular scripts for reproducibility and utilities
├── results/        # Figures and summary tables
├── environment.yml # Conda environment specification
├── .gitignore
└── README.md
```

---

## 📝 Contents & Notes

This repository contains solutions to **Task 1** of the test.

### Notebooks (in `notebooks/`):

1. **EDA of cancer vs healthy samples**  
2. **Differential Expression (DE) and GSEA analysis of cancer vs healthy**  
3. **Differential Expression analysis for cancer samples**  
4. **Survival analysis**  

- Possible analysis plans and strategies are outlined in **notebook 1**.  
- Considerations for scaling and pipeline building are added where appropriate.

### Scripts (in `scripts/`):

- Automated pipelines for parts of the analysis, including DE_GSEA and survival analysis.  
- Custom plotting utilities used and tested inside the notebooks.

### Results (in `results/`):

- All figures and summary tables generated from the analysis are saved here.

### Data Requirements:

- The `data/` folder **must** contain the two original datasets from the external drive to enable the analysis.  
- Additionally, the `hallmarks.gmt` file containing C6 GSEA hallmark genes should be placed in the data directory.

