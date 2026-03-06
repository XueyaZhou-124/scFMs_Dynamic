import pandas as pd
import scanpy as sc

import numpy as np
import scib
import anndata 
import traceback
from typing import Optional, Dict
from matplotlib import pyplot as plt
import os

def eval_scib_metrics(
    adata: anndata.AnnData,
    label_key: str = "celltype",
    batch_key: str = "state",
    notes: Optional[str] = None,
) -> Dict:
    import scib

    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        label_key=label_key,
        batch_key = batch_key,
        embed="X_emb",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )
    result_dict = results[0].to_dict()
    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


def check_obs(adata, label):
    assert label in adata.obs.columns


def eval(embs_df, emb_dims, labels = ['cell_type', 'dataset', 'state'], umap_keys = ['cell_type'], batch_key = 'dataset', seed = 0):
    
    only_embs_df = embs_df.iloc[:, :emb_dims]
    only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
    only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(
        str
    )
    vars_dict = {"embs": only_embs_df.columns}
    obs_dict = {"cell_id": list(only_embs_df.index)}
    
    if labels is None:
        labels = embs_df.columns[emb_dims:].tolist()

    for label in labels:
        obs_dict[label] = list(embs_df[label])
    adata = anndata.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
    adata.obsm['X_emb'] = adata.X.copy()

    for label in labels+umap_keys:
        check_obs(adata, label)
        
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    figs = []
    for eval_label in umap_keys:
        if eval_label not in adata.obs.columns:
            raise ValueError(
                f"eval_label {eval_label} not in adata.obs columns: {adata.obs.columns}"
            )
        adata.obs[eval_label] = adata.obs[eval_label].astype("category")
        try:
            results = eval_scib_metrics(adata, label_key=eval_label, batch_key = batch_key)
        except Exception as e:
            traceback.print_exc()

        fig = sc.pl.umap(
            adata,
            color=eval_label,
            title=[f"{eval_label}, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        # 设置图大小和布局
        fig.set_size_inches(8, 6)  # 或者你自己调试合适的大小 (width, height)
        fig.tight_layout()         # 自动调整子图布局，避免标题、legend被截断

        figs.append(fig)

    return adata,figs


def run(config):
    
    emb_path = config['evaluate']['emb_path']
    emb_df = pd.read_csv(emb_path, index_col = 0)

    emb_dim = config['evaluate']['emb_dim']
    labels = config['evaluate'].get('labels', None)
    umap_keys = config['evaluate']['umap_keys']
    batch_key = config['evaluate'].get('batch_key')
    seed = config['evaluate'].get('seed', 42)
    output_prefix = config['evaluate']['output_prefix']

    adata,figs = eval(emb_df, emb_dim, labels = labels, umap_keys = umap_keys, batch_key=batch_key, seed = seed)
    for fig,umap_key in zip(figs, umap_keys):
        fig.savefig(f'{output_prefix}{umap_key}_umap.pdf', bbox_inches="tight", dpi=300)
    adata_path = f'{output_prefix}adata_eval.h5ad'
    sc.write(adata_path,adata)



def eval_fromadata(embd_adata, emb_key, umap_keys = ['cell_type'], batch_key = 'dataset', seed = 0):
    if emb_key == 'X':
        adata = anndata.AnnData(embd_adata.X)
    else:
        adata = anndata.AnnData(embd_adata.obsm[emb_key])
    adata.obs = embd_adata.obs
    adata.obsm['X_emb'] = adata.X.copy()
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    figs = []
    for eval_label in umap_keys:
        if eval_label not in adata.obs.columns:
            raise ValueError(
                f"eval_label {eval_label} not in adata.obs columns: {adata.obs.columns}"
            )
        adata.obs[eval_label] = adata.obs[eval_label].astype("category")
        
        try:
            results = eval_scib_metrics(adata, label_key=eval_label, batch_key = batch_key)
        except Exception as e:
            traceback.print_exc()

        fig = sc.pl.umap(
            adata,
            color=eval_label,
            title=[f"{eval_label}, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        # 设置图大小和布局
        fig.set_size_inches(8, 6)  # 或者你自己调试合适的大小 (width, height)
        fig.tight_layout()         # 自动调整子图布局，避免标题、legend被截断

        figs.append(fig)

    return adata, figs


def run_fromadata(config):
    emb_path = config['evaluate']['emb_path']
    adata = sc.read_h5ad(emb_path)

    emb_key = config['evaluate']['emb_key']
    # labels = config['evaluate'].get('labels', None)
    umap_keys = config['evaluate']['umap_keys']
    batch_key = config['evaluate'].get('batch_key')
    seed = config['evaluate'].get('seed', 42)
    output_prefix = config['evaluate']['output_prefix']

    adata,figs = eval_fromadata(adata, emb_key, umap_keys = umap_keys, batch_key=batch_key, seed = seed)
    for fig,umap_key in zip(figs, umap_keys):
        fig.savefig(f'{output_prefix}{umap_key}_umap.pdf', bbox_inches="tight", dpi=300)
    adata_path = f'{output_prefix}adata_eval.h5ad'
    sc.write(adata_path,adata)


if __name__ == '__main__':
    emb_df = pd.read_csv('/personal/scFMs_dynamic/data/outputs/EMT_Zeroshot_gf_emb.csv', index_col = 0)
    emb_df.info()
    emb_dim = 512
    adata,fig = eval(emb_df, emb_dim, labels = ['Phase', 'Time'], umap_keys = ['Phase'], batch_key='Phase', seed = 42)
