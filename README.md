a virtual cell model, Causal Disentangled Perturbation (CDP), which integrates causal representation with meta-learning to enable zero-shot prediction for unseen cell type populations.

model training
use the train_rmse_new.py to train the model.

model predicting
use the result_rvcse.py to calculate the results of the newly cellular perturbations.


data source
Rhinovirus and cigarette smoke-exposed airway organoid datasets and immune cell dataset are available in (https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25g1). Sciplex3 are available in (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378). Cmap dataset are available in NCBI (GSE70138 and GSE92742). TCGA data can be found in (The Cancer Genome Atlas Program (TCGA) - NCI).

The algorithm process is as follows:
![image](https://github.com/jc1999123/CDP/blob/main/figure1.jpg)
