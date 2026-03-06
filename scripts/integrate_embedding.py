import scanpy as sc
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate embeddings from multiple models into a single AnnData object.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset (used for output file naming).')
    parser.add_argument('--models', type=str, nargs='+', default= 'all', help='List of model names whose embeddings to integrate.')
    parser.add_argument('--cell_type_key', type=str, default=None, help='Cell type key in AnnData.obs for matching samples.')
    parser.add_argument('--time_key', type=str, default='time', help='Time key in AnnData.obs for matching samples if cell_type_key is not provided.')
    parser.add_argument('--time_dict', type=str, default=None, help='Optional dict for mapping time values to integers.')

    args = parser.parse_args()

    data_name = args.data_name
    data_path = '/macroverse/public/zhouxy/scllms/scFMs_dynamic/data/embeddings/' + data_name
    models = ['hvg', 'genecompass','uce', 'scgpt', 'scfoundation', 'geneformer']

    if args.models != 'all':
        models = args.models

    
    cell_type_key = args.cell_type_key
    time_key =  args.time_key
    time_dict = args.time_dict

    allembs = [os.path.join(data_path, i) for i in os.listdir(data_path) if '_eval.h5ad' in i]
    ref_key = 'hvg'
    adata = sc.read_h5ad(os.path.join(data_path, ref_key+'_adata_eval.h5ad')) # 以hvg作为参考
    adata.obsm['X_hvg'] = adata.obsm['X_emb']
    del adata.obsm['X_emb']

    if time_dict is not None:
        adata.obs['time'] = [time_dict[i] for i in adata.obs[time_key]]

    for emb_path in allembs:    
        if ref_key not in emb_path:
            adata_key = sc.read_h5ad(emb_path)
            if cell_type_key is not None:
                assert (adata_key.obs[cell_type_key].values == adata.obs[cell_type_key].values).all() # 确保样本配对
            else:
                assert (adata_key.obs['time'].values == adata.obs['time'].values).all() # 确保样本配对

            key = os.path.basename(emb_path).replace('_adata_eval.h5ad', '')
            print('hidden dim of', key, adata_key.obsm['X_emb'].shape[1])
            adata.obsm[f'X_{key}'] = adata_key.obsm['X_emb']

    adata.write_h5ad(os.path.join(data_path, 'benchmark.h5ad'))
    print('save to', os.path.join(data_path, 'benchmark.h5ad'))
