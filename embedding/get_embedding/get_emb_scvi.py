import os

import numpy as np
import scanpy as sc
from anndata import AnnData


def _validate_select_col(obs, select_col):
    if select_col is None:
        return obs.copy()
    missing = [c for c in select_col if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs for select_col: {missing}")
    return obs.loc[:, select_col].copy()


def run(config):
    try:
        import scvi
    except Exception as exc:
        raise ImportError(
            "scvi-tools is required for model=scvi. Please run with an environment that has scvi-tools installed."
        ) from exc

    input_path = config["data"]["input_path"]
    output_path = config["embedding"]["output_path"]

    if output_path and os.path.exists(output_path):
        print(f"[scVI] Embedding already exists, skip: {output_path}")
        return

    adata = sc.read_h5ad(input_path)

    embedding_cfg = config.get("embedding", {})
    batch_key = embedding_cfg.get("batch_key", None)
    obsm_key = embedding_cfg.get("obsm_key", "X_scvi")
    select_col = embedding_cfg.get("select_col", None)
    count_layer = embedding_cfg.get("count_layer", None)
    n_latent = int(embedding_cfg.get("n_latent", 50))
    n_layers = int(embedding_cfg.get("n_layers", 2))
    n_hidden = int(embedding_cfg.get("n_hidden", 128))
    dropout_rate = float(embedding_cfg.get("dropout_rate", 0.1))
    gene_likelihood = embedding_cfg.get("gene_likelihood", "nb")
    max_epochs = int(embedding_cfg.get("max_epochs", 200))
    batch_size = int(embedding_cfg.get("batch_size", 256))
    seed = int(embedding_cfg.get("seed", 0))

    scvi.settings.seed = seed
    np.random.seed(seed)

    adata_scvi = adata.copy()
    if count_layer is not None:
        if count_layer not in adata_scvi.layers:
            raise KeyError(f"count_layer '{count_layer}' not found in adata.layers")
        adata_scvi.X = adata_scvi.layers[count_layer].copy()

    setup_kwargs = {}
    if batch_key is not None:
        if batch_key not in adata_scvi.obs.columns:
            raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
        setup_kwargs["batch_key"] = batch_key

    scvi.model.SCVI.setup_anndata(adata_scvi, **setup_kwargs)

    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        gene_likelihood=gene_likelihood,
    )
    model.train(max_epochs=max_epochs, batch_size=batch_size)

    latent = model.get_latent_representation().astype(np.float32)
    emb_adata = AnnData(X=latent)
    emb_adata.obs = _validate_select_col(adata.obs, select_col)
    emb_adata.obs_names = adata.obs_names.copy()
    emb_adata.obsm[obsm_key] = latent

    emb_adata.write_h5ad(output_path)
    print(f"[scVI] Embedding saved to {output_path} ({obsm_key})")
