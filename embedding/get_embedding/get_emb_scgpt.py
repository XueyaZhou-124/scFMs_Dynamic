# type: ignore
import scanpy as sc
import copy
import gc
import json
import os
import pickle
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# Add project root to path
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator, ClsDecoder
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics


class Config:
    """Configuration class for training parameters"""
    
    def __init__(self, **kwargs):
        # Default hyperparameters
        self.seed = 0
        self.load_model = "/macroverse-nas/zhouxy/pretrained_models/scGPT/scgpt_human"
        self.mask_ratio = 0.0
        self.epochs = 10
        self.n_bins = 51
        self.MVC = False
        self.ecs_thres = 0.0
        self.dab_weight = 0.0
        self.lr = 1e-4
        self.batch_size = 12
        self.layer_size = 128
        self.nlayers = 4
        self.nhead = 4
        self.dropout = 0.2
        self.schedule_ratio = 0.9
        self.save_eval_interval = 5
        self.fast_transformer = True
        self.pre_norm = False
        self.amp = True
        self.include_zero_gene = False
        self.freeze = False
        self.DSBN = False
        self.n_hvg = False
        self.gene_key = 'name'

        
        # Update with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert config to dictionary for wandb"""
        return {k: v for k, v in self.__dict__.items()}


class DataProcessor:
    """Handles data loading, preprocessing, and tokenization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vocab = None
        self.gene_ids = None
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        self.max_seq_len = config.n_hvg + 1 if config.n_hvg else 3001
        self.gene_key = config.gene_key
       
        
    def load_data(self, data_path: str, batch_key = 'Time') -> AnnData:
        """Load and prepare AnnData object"""
        adata = sc.read(data_path)

        adata.obs["str_batch"] = adata.obs[batch_key].astype("category")
            
        # Process batch labels
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        
        return adata
    

    def load_vocab_and_model_config(self, model_dir: str) -> Tuple[GeneVocab, Dict]:
        """Load vocabulary and model configuration from pretrained model"""
        model_dir = Path(model_dir)
        model_config_file = model_dir / "args.json"
        vocab_file = model_dir / "vocab.json"
        
        # Load vocabulary
        vocab = GeneVocab.from_file(vocab_file)
        for s in self.special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        
        # Load model config
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
            
        return vocab, model_configs
    

    def preprocess_data(self, adata: AnnData, vocab: GeneVocab = None, species ='human', homo_path = None) -> AnnData:
        """Preprocess AnnData using scGPT preprocessor"""
        if species != 'human':
            print('homologues convert')
            assert homo_path is not None and os.path.exists(homo_path), (
                "species is not human, but homo_path is missing or invalid: "
                f"{homo_path}"
            )

            print(f"Loading homologues data from {homo_path}")
            homo_df = pd.read_table(homo_path)  # human–other species homolog mapping

            # Map to human gene symbols
            print('Mapping to human gene symbols')
            dict1 = dict(zip(homo_df['Gene name'], homo_df['Human gene name']))
            
            if self.gene_key == 'index':
                adata.var['gene_name'] = [dict1.get(i) for i in adata.var.index.tolist()]
            else:
                adata.var['gene_name'] = [dict1.get(i) for i in adata.var[self.gene_key]]
            # Drop genes with missing mapped symbol
            adata = adata[:, ~pd.isna(adata.var['gene_name'])]
            
        else:
            if self.gene_key == 'index':
                adata.var['gene_name'] = adata.var.index.tolist()
            else:
                adata.var['gene_name'] = adata.var[self.gene_key]

        self.gene_key = 'gene_name'
        # Filter genes by vocabulary if vocab is provided
        if vocab is not None:
            gene_col = self.gene_key
            adata = adata[:,~adata.var[gene_col].isna()]
            adata.var["id_in_vocab"] = [
                1 if gene in vocab else -1 for gene in adata.var[gene_col]
            ]
            gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
            adata = adata[:, adata.var["id_in_vocab"] >= 0]
        
        # Setup preprocessor
        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=self.config.n_hvg,
            hvg_flavor="seurat_v3",
            binning=self.config.n_bins,
            result_binned_key="X_binned",
        )

        preprocessor(adata, batch_key=None)
        return adata  


def run(config):
    config_scgpt = Config()
    config_scgpt.load_model = config['embedding']['model_path']
    config_scgpt.batch_size = config['embedding']['batch_size']
    species = config['preprocess'].get('species', 'human')
    config_scgpt.gene_key = config['preprocess']['gene_key']

    # external section is optional; only needed for non-human homolog mapping.
    homo_path = config.get('external', {}).get('homo_path', None)
    save_path = config['embedding']['output_path']

    batch_key = config['preprocess'].get('batch_key', 'Time')
    data_processor = DataProcessor(config_scgpt)
    

    # Load and preprocess data
    print("Loading data...")
    smaple_data_path = config['data']['input_path']
    

    adata = data_processor.load_data(smaple_data_path, batch_key = batch_key)

    print("Loading pretrained model...")
    vocab, model_configs = data_processor.load_vocab_and_model_config(config_scgpt.load_model)
    adata = data_processor.preprocess_data(adata, vocab, species, homo_path)

    embed_adata = scg.tasks.embed_data(
        adata,
        config_scgpt.load_model,
        gene_col='gene_name',
        batch_size= config_scgpt.batch_size,
    )

    print(f'embedding adata save in {save_path}')
    embed_adata.write(save_path)


def main():
    config = Config()
    data_processor = DataProcessor(config)

    # Load and preprocess data
    print("Loading data...")
    smaple_data_path = '/macroverse-nas/zhouxy/projects/scFMs_dynamic/data/raw/EMT/raw.h5ad'

    adata = data_processor.load_data(smaple_data_path)

    print("Loading pretrained model...")
    vocab, model_configs = data_processor.load_vocab_and_model_config(config.load_model)
    adata = data_processor.preprocess_data(adata, vocab)

    embed_adata = scg.tasks.embed_data(
        adata,
        config.load_model,
        gene_col='name',
        batch_size=64,
    )

    adata = AnnData(embed_adata.obsm['X_scGPT'])
    adata.obs = embed_adata.obs
    sc.tl.pca(adata, svd_solver="arpack", )
    sc.pp.neighbors(adata, random_state=42)
    sc.tl.umap(adata, random_state=42)

    result_dict = eval_scib_metrics(embed_adata, label_key = 'Phase', batch_key = 'Time')

    fig = sc.pl.umap(
            adata,
            color='Phase',
            title=[f"phase, avg_bio = {result_dict.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

    fig.savefig('./temp_umap.pdf', bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    main()

