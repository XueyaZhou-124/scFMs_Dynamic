import anndata as ad
import scanpy as sc
import pandas as pd
import gzip
import os
import numpy as np
import gc
# tokenizeing for GF
import scanpy as sc
import os
from geneformer.tokenizer import TranscriptomeTokenizer
from pathlib import Path
import numpy as np
from geneformer import ENSEMBL_DICTIONARY_FILE
import pickle

def ensure_ensembl_id(adata, gene_key):
    """
    将 adata.var 中指定的 gene_key 列的基因名映射为 Ensembl ID。
    
    参数：
        adata: AnnData 对象
        gene_key: str，表示当前基因名使用的字段（例如 'gene_symbol'）
        dict1: dict，symbol -> ensembl_id 的映射字典

    返回：
        更新后的 adata，包含 adata.var['ensembl_id']
    """
    with open(ENSEMBL_DICTIONARY_FILE, 'rb') as fp:
        dict_id = pickle.load(fp)

    # 如果 index 本身是 Ensembl ID（以 ENSG 开头），直接使用
    first_gene = adata.var.index[0]
    if isinstance(first_gene, str) and first_gene.startswith("ENSG"):
        adata.var['ensembl_id'] = adata.var.index.tolist()
    else:
        # 否则通过 gene_key 和映射字典转换为 Ensembl ID
        if gene_key != 'index':
            adata.var['ensembl_id'] = [
                dict_id.get(gene_symbol, np.nan) for gene_symbol in adata.var[gene_key]
            ]
        else:
            adata.var['ensembl_id'] = [
                dict_id.get(gene_symbol, np.nan) for gene_symbol in adata.var.index.tolist()
            ]

    # 去除无法映射的基因
    adata = adata[:, ~adata.var['ensembl_id'].isna()]

    return adata


def ensure_n_counts(adata):
    # 确保存在n_counts, 主要是seurat和sacany命名不同需要兼容
    # 不存在则生成一个
    if 'n_counts' not in adata.obs.columns:
        adata.obs['n_counts'] = adata.X.sum(axis=1)
    return adata


def run(config):
    task_name = config['task_name']
    input_path = config["data"]["input_path"]

    species = config['preprocess']['species']
    gene_key = config['preprocess']['gene_key']
    output_path = config['preprocess']['output_path']
    custom_attr_name_dict = config['preprocess'].get('custom_attr_name_dict', None)
    target_sum = config['preprocess'].get('target_sum', 10_000)


    homo_path = config.get("external", {}).get("homo_path", None)

    print(f"Loading raw data from {input_path}")
    adata = sc.read_h5ad(input_path)

    if species != 'human':
        print('homologues convert')
        assert os.path.exists(homo_path), f'species is not human, {homo_path} is not exist'

        print(f"Loading homologues data from {homo_path}")
        homo_df = pd.read_table(homo_path) # 人同源基因转换

        # human gene symbol covert
        print('human gene symbol covert')
        dict1 = dict(zip(homo_df['Gene name'], homo_df['Human gene stable ID']))
        if gene_key == 'index':
            adata.var['ensembl_id'] = [dict1.get(i, np.nan) for i in adata.var.index.tolist()]
        else:
            adata.var['ensembl_id'] = [dict1.get(i, np.nan) for i in adata.var[gene_key]]
        
        gene_key = 'ensembl_id'

    if gene_key != 'ensembl_id':
        adata = ensure_ensembl_id(adata, gene_key)
    adata = ensure_n_counts(adata)
    adata_file_path = output_path.replace('.dataset', '_preprocessed.h5ad')
    sc.write(adata_file_path, adata)

    print('save preprocessed adata in', adata_file_path)

    print('toknizing adata....')
    
    if custom_attr_name_dict is None:
        custom_attr_name_dict = {}
        for i in adata.obs.keys():
            custom_attr_name_dict[i] = i
    print('custtom attr name', custom_attr_name_dict)
    tk = TranscriptomeTokenizer(custom_attr_name_dict, nproc=6,)

    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata_file_path = adata_file_path, target_sum = target_sum)

    tokenized_dataset = tk.create_dataset(
            tokenized_cells,
            cell_metadata,
        )
    tokenized_dataset = tokenized_dataset.add_column('cell_id', range(0, len(tokenized_dataset)))
    # output_path = (Path(output_path) / f'{task_name}_geneformer').with_suffix(".dataset")
    tokenized_dataset.save_to_disk(str(output_path))

    print('save tokenized data in', output_path)


def main():
    
    ld_lib = os.getenv('LD_LIBRARY_PATH')
    print('ld:', ld_lib)
    breakpoint()
    pass


if __name__ == '__main__':
    main()
