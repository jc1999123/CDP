
# CDP (Causal Disentangled Perturbation)
a virtual cell model, Causal Disentangled Perturbation (CDP), which integrates causal representation with meta-learning to enable zero-shot prediction for unseen cell type populations.
This project implements a causal-based framework for disentangling gene expression perturbations using Python.

## 📘 Overview
CDP aims to disentangle the causal effects of perturbations on gene expression across different cell types. It combines causal inference (do-calculus) and meta-learning to predict cellular responses to unseen perturbations.

## 🧩 Project Structure
```
version2/
├── data.py                              # Datasets and preprocessing scripts
├── transformer_vae_cell_rvcse_new.py    # Model definitions (e.g., CDP, VAE, etc.)
├── train.py                             # Main training script
├── result_rvcse.py                      # Evaluation or inference script
└── README.md                            # Project documentation
```

## 🚀 Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Evaluate or predict:
   ```bash
   python test.py
   ```

## 🧠 Key Concepts
- **Causal Inference**: Separates true causal effects from correlations.
- **Meta-learning**: Enables generalization to unseen perturbations or cell types.
- **Disentanglement**: Decouples gene expression variation caused by perturbations and cell identities.

---
data source
Rhinovirus and cigarette smoke-exposed airway organoid datasets and immune cell dataset are available in (https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25g1). Sciplex3 are available in (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378). Cmap dataset are available in NCBI (GSE70138 and GSE92742). TCGA data can be found in (The Cancer Genome Atlas Program (TCGA) - NCI).

The algorithm process is as follows:
![image](https://github.com/jc1999123/CDP/blob/main/figure1.jpg)
