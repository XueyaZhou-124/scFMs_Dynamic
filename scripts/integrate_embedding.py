import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc


def _parse_model_keys(arg_models):
    """Normalize model names: lower/strip, drop empty."""
    if not arg_models:
        return None
    out = [x.strip().lower() for x in arg_models if x and str(x).strip()]
    return out or None


def _resolve_time_col(adata_obj, time_key: str) -> str:
    """Resolve which obs column to use for alignment (default: time, fallback Time)."""
    tk = time_key.strip()
    if tk in adata_obj.obs.columns:
        return tk
    if tk == "time" and "Time" in adata_obj.obs.columns:
        return "Time"
    raise KeyError(
        f"time column {tk!r} not found in obs. "
        f"Available keys include: {list(adata_obj.obs.columns)[:40]}"
    )


def _ensure_time_column(adata, time_key: str) -> None:
    """
    Guarantee obs['time'] exists for downstream benchmark configs.
    If missing, copy from the column indicated by time_key (after resolve), then drop the source column if it was not already 'time'.
    """
    if "time" in adata.obs.columns:
        return
    src = _resolve_time_col(adata, time_key)
    adata.obs["time"] = pd.to_numeric(adata.obs[src], errors="coerce")
    if src != "time":
        adata.obs.drop(columns=[src], inplace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrate per-model *_adata_eval.h5ad into one benchmark.h5ad."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with *_adata_eval.h5ad (e.g. data/embeddings/EMT/)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path for merged AnnData, e.g. data/embeddings/EMT/benchmark.h5ad",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model keys to include (e.g. hvg geneformer). Default: all *_adata_eval.h5ad in input_dir",
    )
    parser.add_argument(
        "--ref_key",
        type=str,
        default="hvg",
        help="Reference model; requires {ref_key}_adata_eval.h5ad and defines cell order",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        default=None,
        help="If set, assert obs[cell_type_key] matches across models",
    )
    parser.add_argument(
        "--time_key",
        type=str,
        default="time",
        help="obs column used to align cells across models; if output has no 'time', this column is copied to 'time' (default tries 'time' then 'Time')",
    )

    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    output_file = os.path.normpath(args.output_file)
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith("_adata_eval.h5ad")
    ]

    def key_from_filename(fn):
        return fn[: -len("_adata_eval.h5ad")].lower()

    file_by_key = {key_from_filename(f): os.path.join(input_dir, f) for f in all_files}
    want = _parse_model_keys(args.models)

    if want is not None:
        missing = [k for k in want if k not in file_by_key]
        if missing:
            raise FileNotFoundError(
                f"Missing model files in {input_dir}: {missing}. Found keys: {sorted(file_by_key)}"
            )
        keys_to_merge = [k for k in want if k in file_by_key]
    else:
        keys_to_merge = sorted(file_by_key.keys())

    ref_key = args.ref_key.strip().lower()
    if ref_key not in file_by_key:
        raise FileNotFoundError(
            f"Reference {ref_key}_adata_eval.h5ad not in {input_dir}. "
            f"Found: {sorted(file_by_key)}"
        )
    if ref_key not in keys_to_merge:
        raise ValueError(
            f"ref_key {ref_key!r} must be included; use --models to list it or omit --models"
        )

    ref_path = file_by_key[ref_key]
    adata = sc.read_h5ad(ref_path)
    adata.obsm[f"X_{ref_key}"] = adata.obsm["X_emb"]
    del adata.obsm["X_emb"]

    time_col = _resolve_time_col(adata, args.time_key)
    ref_time_vec = np.asarray(adata.obs[time_col].to_numpy())

    for key in keys_to_merge:
        if key == ref_key:
            continue
        emb_path = file_by_key[key]
        adata_key = sc.read_h5ad(emb_path)
        if args.cell_type_key is not None:
            assert (
                adata_key.obs[args.cell_type_key].values
                == adata.obs[args.cell_type_key].values
            ).all()
        else:
            other_time_col = _resolve_time_col(adata_key, args.time_key)
            assert (
                adata_key.obs[other_time_col].values == adata.obs[time_col].values
            ).all()

        print("hidden dim of", key, adata_key.obsm["X_emb"].shape[1])
        adata.obsm[f"X_{key}"] = adata_key.obsm["X_emb"]

    _ensure_time_column(adata, args.time_key)

    adata.write_h5ad(output_file)
    print("save to", output_file)
