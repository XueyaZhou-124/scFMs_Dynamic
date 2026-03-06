import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata
import numpy as np
import scanpy as sc
import tqdm
from datasets import Dataset, Features, Sequence as HFSequence, Value


# -----------------------------
# Configuration and Constants
# -----------------------------

DEFAULT_CHUNK_LEN: int = 2048


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""

    input_path: str
    dataset_out_dir: str

    # External resources
    gene_median_dict_path: str
    gene_token_dict_path: str
    gene_id_to_name_path: str

    # Data assumptions
    species: str = "mouse"  # "human" or "mouse"
    ensembl_id_column: str = "ensembl_id_mouse"
    chunk_len: int = DEFAULT_CHUNK_LEN


# -----------------------------
# Utility functions
# -----------------------------

def load_pickle_dict(path: str) -> Dict:
    """Load a Python dictionary from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def invert_dict(dict1):
    dict2 = {}
    for key,value in dict1.items():
        dict2[value] = key
    
    return dict2
# -----------------------------
# Mapping and filtering
# -----------------------------

def map_gene_symbols_to_ids(
    adata: anndata.AnnData,
    gene_id_to_name: Dict[str, str],
    gene_key: str,
) -> anndata.AnnData:
    """Map Ensembl gene IDs in `adata.var[ensembl_id_column]` to gene symbols.

    - Adds `gene_symbols_original` to preserve original `adata.var_names`.
    - Sets `adata.var_names` to mapped symbols or "delete" if unmapped.
    - Removes genes marked as "delete".
    """
    
    if (gene_key != 'index') & (gene_key not in adata.var.columns):
        raise KeyError(
            f"Column '{gene_key}' not found in adata.var. Available: {list(adata.var.columns)}"
        )

    adata.var["gene_symbols_original"] = adata.var.index.to_list()

    mapped_symbols: List[str] = []
    gene_name_to_id = invert_dict(gene_id_to_name)
    if gene_key != 'index':
        for gene in adata.var[gene_key].tolist():
            mapped_symbols.append(gene_name_to_id.get(gene, "delete"))
    else:
        for gene in adata.var.index.tolist():
            mapped_symbols.append(gene_name_to_id.get(gene, "delete"))

    adata.var_names = mapped_symbols

    # Filter out unmapped genes
    adata = adata[:, ~(adata.var_names == "delete")].copy()
    return adata


def filter_genes_by_token(
    adata: anndata.AnnData,
    token_dict: Dict[str, int],
) -> anndata.AnnData:
    """Keep only genes whose Ensembl IDs exist in `token_dict`.

    - Sets `adata.var_names` to Ensembl IDs found in `token_dict`, or "delete" if missing.
    - Removes genes marked as "delete".
    """
    keep_or_delete: List[str] = []
    for gene_id in adata.var.index.tolist():
        keep_or_delete.append(gene_id if gene_id in token_dict else "delete")

    adata.var_names = keep_or_delete
    adata = adata[:, ~(adata.var_names == "delete")].copy()
    return adata


# -----------------------------
# Normalization and transforms
# -----------------------------

def normalize_by_gene_median(
    adata: anndata.AnnData,
    gene_median_dict_path: str,
) -> anndata.AnnData:
    """Normalize expression by gene-wise nonzero medians.

    Expects `adata.var_names` to be Ensembl IDs that match keys in the median dict.
    Replaces `adata.X` with the normalized sparse matrix.
    """
    gene_median: Dict[str, float] = load_pickle_dict(gene_median_dict_path)

    # Build per-gene median vector in the current gene order
    gene_names: List[str] = adata.var_names.tolist()
    per_gene_median: np.ndarray = np.array([gene_median.get(g, 1.0) for g in gene_names], dtype=np.float32)

    # Avoid division by zero
    per_gene_median[per_gene_median == 0] = 1.0

    # Divide each gene (row) by its median; work in CSC for efficient column operations then transpose back
    matrix = adata.X  # sparse expected
    # Convert to CSC if needed
    if not hasattr(matrix, "tocsc"):
        raise TypeError("adata.X must be a scipy sparse matrix")

    # matrix shape: cells x genes
    # We want: (genes x cells) / per_gene_median[:, None] -> back to (cells x genes)
    matrix_t = matrix.T.tocsc()
    matrix_t = matrix_t.multiply(1.0 / per_gene_median[:, None])
    adata.X = matrix_t.T.tocsc()
    return adata


def log1p_base2(adata: anndata.AnnData) -> anndata.AnnData:
    sc.pp.log1p(adata, base=2)
    return adata


# -----------------------------
# Rank encoding
# -----------------------------

def tokenize_cell(
    gene_vector: np.ndarray,
    gene_list: Sequence[str],
    token_dict: Dict[str, int],
) -> Tuple[List[int], List[float]]:
    """Convert a normalized gene expression vector to tokenized rank value encoding.

    Returns token ids and corresponding expression values, sorted by descending expression for nonzero entries.
    """
    nonzero_indices = np.nonzero(gene_vector)[0]
    if nonzero_indices.size == 0:
        return [], []

    # Sort nonzero values descending
    sorted_nonzero = np.argsort(-gene_vector[nonzero_indices])
    sorted_indices = nonzero_indices[sorted_nonzero]

    sorted_genes = np.asarray(gene_list)[sorted_indices]
    token_ids = [token_dict[g] for g in sorted_genes]
    values = gene_vector[sorted_indices].tolist()
    return token_ids, values


def rank_encode(
    adata: anndata.AnnData,
    token_dict: Dict[str, int],
    chunk_len: int = DEFAULT_CHUNK_LEN,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Rank-encode each cell into token ids and values with truncation/padding to `chunk_len`.

    Returns (input_ids, lengths, values), where shapes are:
    - input_ids: (n_cells, chunk_len) int32
    - values: (n_cells, chunk_len) float32
    - lengths: list of actual lengths per cell
    """
    n_cells = adata.n_obs
    input_ids = np.zeros((n_cells, chunk_len), dtype=np.int32)
    values = np.zeros((n_cells, chunk_len), dtype=np.float32)
    lengths: List[int] = []

    gene_ids: List[str] = adata.var_names.tolist()

    # Ensure CSR for efficient row slicing
    matrix = adata.X.tocsr()

    for row_index in tqdm.tqdm(range(n_cells), total=n_cells):
        row_vector = matrix.getrow(row_index).toarray().ravel()
        tokenized, ranked_values = tokenize_cell(row_vector, gene_ids, token_dict)

        actual_len = min(len(tokenized), chunk_len)
        if actual_len > 0:
            input_ids[row_index, :actual_len] = np.asarray(tokenized[:actual_len], dtype=np.int32)
            values[row_index, :actual_len] = np.asarray(ranked_values[:actual_len], dtype=np.float32)
        # Remaining positions stay zeros (already initialized)
        lengths.append(actual_len)

    return input_ids, lengths, values


# -----------------------------
# HF dataset conversion and saving
# -----------------------------

def to_hf_dataset(
    species: str,
    lengths: List[int],
    input_ids: np.ndarray,
    values: np.ndarray,
) -> Dataset:
    """Convert encoded arrays to a HuggingFace Dataset.

    - `species` will be encoded as 0 for human, 1 for mouse.
    - `lengths` and `species` are stored as 1-length lists to mirror prior structure.
    """
    if species not in {"human", "mouse"}:
        raise ValueError("species must be one of {'human', 'mouse'}")

    species_id = 0 if species == "human" else 1

    n_cells = input_ids.shape[0]
    species_col: List[List[int]] = [[species_id] for _ in range(n_cells)]
    lengths_col: List[List[int]] = [[l] for l in lengths]

    data_out = {
        "input_ids": input_ids.tolist(),
        "values": values.tolist(),
        "length": lengths_col,
        "species": species_col,
    }

    features = Features(
        {
            "input_ids": HFSequence(feature=Value(dtype="int32")),
            "values": HFSequence(feature=Value(dtype="float32")),
            "length": HFSequence(feature=Value(dtype="int16")),
            "species": HFSequence(feature=Value(dtype="int16")),
        }
    )

    return Dataset.from_dict(data_out, features=features)


def save_dataset(out_dir: str, dataset: Dataset, lengths: List[int]) -> None:
    dataset.save_to_disk(out_dir)
    sorted_lengths = sorted(lengths)
    out_path = os.path.join(out_dir, "sorted_length.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(sorted_lengths, f)


# -----------------------------
# Orchestrator
# -----------------------------

def run(config) -> None:
    input_path = config['data']['input_path']
    gene_id_to_name_path = config['external']['gene_id_to_name_path']
    gene_token_dict_path = config['external']['gene_token_dict_path']
    gene_median_dict_path = config['external']['gene_median_dict_path']
    gene_key = config['preprocess']['gene_key']
    dataset_out_dir = config['preprocess']['output_path']
    species = config['preprocess']['species']


    """Run the complete preprocessing pipeline."""
    logging.info("Reading input AnnData from %s", input_path)
    adata = sc.read_h5ad(input_path)
    logging.info("Loading dictionaries")
    gene_id_to_name: Dict[str, str] = load_pickle_dict(gene_id_to_name_path) # ensembl id to name
    token_dict: Dict[str, int] = load_pickle_dict(gene_token_dict_path) # ensembl id to token

    logging.info("Mapping gene IDs to symbols and filtering unmapped genes")
    # 如果基因key不是ensembleid进行转换
    if gene_key.lower() not in ["ensemblid", "ensembl_id", "ensembl_id_mouse", "ensembl_id_human"]:
        adata = map_gene_symbols_to_ids(adata, gene_id_to_name, gene_key)
    else:
        adata.var_names = adata.var[gene_key]

    logging.info("Filtering genes by token dictionary and switching to Ensembl var_names")
    adata = filter_genes_by_token(adata, token_dict)
    logging.info("Normalizing by gene nonzero medians")
    adata = normalize_by_gene_median(adata, gene_median_dict_path)

    logging.info("Applying log1p base 2")
    adata = log1p_base2(adata)

    logging.info("Rank encoding cells")
    input_ids, lengths, values = rank_encode(adata, token_dict)

    logging.info("Converting to HuggingFace Dataset")
    dataset = to_hf_dataset(species, lengths, input_ids, values)

    # Add metadata columns from adata.obs
    for col in adata.obs_keys():
        dataset = dataset.add_column(col, adata.obs[col].tolist())

    ensure_output_dir(dataset_out_dir)
    logging.info("Saving dataset to %s", dataset_out_dir)
    save_dataset(dataset_out_dir, dataset, lengths)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored GeneCompass preprocessing pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="/share/LLM_Omics/data_ham/h5ad/input/all_tissue_homo.h5ad",
        help="Path to input .h5ad file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/share/LLM_Omics/data_ham/h5ad/geneCompass_input/",
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--gene_median_dict_path",
        type=str,
        default="/personal/llm_bench/preprocess/gene_median_dictionary_gc95M.pkl",
        help="Path to gene median dictionary pickle",
    )
    parser.add_argument(
        "--gene_token_dict_path",
        type=str,
        default="/root/GeneCompass/prior_knowledge/human_mouse_tokens.pickle",
        help="Path to gene token dictionary pickle",
    )
    parser.add_argument(
        "--gene_id_to_name_path",
        type=str,
        default="/root/GeneCompass/prior_knowledge/gene_list/Gene_id_name_dict_human_mouse.pickle",
        help="Path to gene ID to name dictionary pickle",
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "mouse"],
        default="mouse",
        help="Species label to encode in dataset",
    )
    parser.add_argument(
        "--ensembl_id_column",
        type=str,
        default="ensembl_id_mouse",
        help="Column in adata.var with Ensembl gene IDs",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=DEFAULT_CHUNK_LEN,
        help="Max sequence length (truncation/padding)",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    config = PipelineConfig(
        input_path=args.input,
        dataset_out_dir=args.dataset_path,
        gene_median_dict_path=args.gene_median_dict_path,
        gene_token_dict_path=args.gene_token_dict_path,
        gene_id_to_name_path=args.gene_id_to_name_path,
        species=args.species,
        ensembl_id_column=args.ensembl_id_column,
        chunk_len=args.chunk_len,
    )

    run(config)


if __name__ == "__main__":
    main() 