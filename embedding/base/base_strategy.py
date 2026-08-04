from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def get_embedding(self):
        pass

    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    
class HvgStrategy(BaseStrategy):
    def preprocess(self):
        # HVG strategy does not require preprocessing
        print("HVG strategy: No preprocessing required.")

    def get_embedding(self):
        # HVG strategy: select highly variable genes as embedding
        import scanpy as sc
        import pandas as pd
        import os

        input_path = self.config['data']['input_path']
        output_path = self.config['embedding']['output_path']
        n_top_genes = self.config['embedding'].get('n_top_genes', 2000)
        batch_key = self.config['embedding'].get('batch_key')

        print(f"HVG strategy: Reading data from {input_path}")
        adata = sc.read_h5ad(input_path)
        print('Preprocessing')
        adata.layers["counts"] = adata.X.copy()
        print('normalize...')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if batch_key is not None:
            if batch_key not in adata.obs.columns:
                raise KeyError(f"HVG batch_key '{batch_key}' not found in adata.obs")
            # scanpy highly_variable_genes(batched) expects categorical batches.
            if not pd.api.types.is_categorical_dtype(adata.obs[batch_key]):
                adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        print(f"Selecting top {n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key, subset = True)

        # save embedding
        adata.write_h5ad(output_path)
        print(f"HVG embedding saved to {output_path}")

    def finetune(self):
        # HVG strategy does not support finetuning
        print("HVG strategy: Finetuning not supported.")

    def evaluate(self):
        print(f"Evaluate with config: {self.config['evaluate']}")
        from ..utils.evaluate import run_fromadata as evaluate_run
        self.config['evaluate']['emb_key'] = 'X'
        evaluate_run(self.config)

