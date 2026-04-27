# type: ignore
import sys,os
import argparse
import random,os
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from scipy.sparse import issparse
import scanpy as sc
from scipy import sparse


def main_gene_selection(X_df, gene_list):
        """
        Describe:
            rebuild the input adata to select target genes encode protein 
        Parameters:
            adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
            gene_list->list: wanted target gene 
        Returns:
            adata_new->`~anndata.AnnData` object
            to_fill_columns->list: zero padding gene
        """
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                                columns=to_fill_columns, 
                                index=X_df.index)
        X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                            index=X_df.index, 
                            columns=list(X_df.columns) + list(padding_df.columns))
        X_df = X_df[gene_list]
        
        var = pd.DataFrame(index=X_df.columns)
        var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]

        return X_df, to_fill_columns,var


def run(config):
    
    task_name = config['task_name']
    input_path = config["data"]["input_path"]


    species = config['preprocess']['species']
    gene_key = config['preprocess']['gene_key']
    output_path = config['preprocess']['output_path']

    homo_path = config["external"]["homo_path"]
    gene_list_path = config["external"]["gene_list_path"]

    if species != 'human':
        assert os.path.exists(homo_path), f'species is not human, {homo_path} is not exist'
    
    print(f"Loading raw data from {input_path}")
    adata = sc.read_h5ad(input_path)

    if species != 'human':
        print(f"Loading homologues data from {homo_path}")
        homo_df = pd.read_table(homo_path)  # homologue table

        print('Mapping to human gene symbols')
        dict1 = dict(zip(homo_df['Gene name'], homo_df['Human gene name']))
        if gene_key == 'index':
            adata.var['human gene name'] = [dict1.get(i, np.nan) for i in adata.var.index.tolist()]
        else:
            adata.var['human gene name'] = [dict1.get(i, np.nan) for i in adata.var[gene_key]]

        X_df= pd.DataFrame(sparse.csr_matrix.toarray(adata.X), index=adata.obs.index.tolist(), columns=adata.var['human gene name'])
        
    else:
        if gene_key == 'index':
            X_df= adata.to_df()
        else:
            from scipy.sparse import issparse
            if issparse(adata.X):
                X_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var[gene_key])
            else:
                X_df = pd.DataFrame(np.asarray(adata.X), index=adata.obs_names, columns=adata.var[gene_key])

    gene_list_df = pd.read_csv(gene_list_path, header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
    # Duplicate column names: group and take mean
    if X_df.shape[1] > 19266:
        X_df_mean = X_df.groupby(X_df.columns, axis=1).mean()
        X_df = X_df_mean.copy()
    adata_uni = sc.AnnData(sparse.csr_matrix(X_df))
    adata_uni.obs = adata.obs
    print('Gene symbol unified')
    print(adata_uni)
    print(f"[scFoundation] Writing processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    adata_uni.write_h5ad(output_path)
    

