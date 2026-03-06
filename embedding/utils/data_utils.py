import scanpy as sc
import anndata as ad
from matplotlib import pyplot as plt
import pandas as pd


def getadata(embs_df, emb_dims, labels, seed=0, outputfile = None, color_label = None): 
    
    # from embs to adata
    only_embs_df = embs_df.iloc[:, :emb_dims]
    only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
    only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(str)
    vars_dict = {"embs": only_embs_df.columns}
    obs_dict = {"cell_id": list(only_embs_df.index)}
    for label in labels:
        obs_dict[label] = list(embs_df[label])
    adata = ad.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
    sc.tl.pca(adata, svd_solver="arpack") # default PCs = 50
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    if (outputfile is not None) & (color_label is not None):
        print(f'save umap fig to {outputfile}')
        p = sc.pl.umap(adata, color = color_label, return_fig=True)
        p.savefig(f"{outputfile}")

    return adata



def homo_convert():
    pass
