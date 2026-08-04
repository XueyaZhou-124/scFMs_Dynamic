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
    numeric_time = pd.to_numeric(adata.obs[src], errors="coerce")
    if numeric_time.isna().all():
        # Handle labels like "0d", "8h", "t3" by extracting the numeric token.
        extracted = (
            pd.Series(adata.obs[src], dtype="object")
            .astype(str)
            .str.strip()
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )
        numeric_time = pd.to_numeric(extracted, errors="coerce")
    if numeric_time.isna().all():
        raise ValueError(
            f"Failed to derive numeric 'time' from source column '{src}'. "
            "Please provide a numeric time column or labels containing numeric tokens."
        )
    adata.obs["time"] = numeric_time
    if src != "time":
        adata.obs.drop(columns=[src], inplace=True)


def _align_to_reference_obs(ref_adata, other_adata, model_key: str):
    """
    Align other_adata rows to ref_adata by obs_names.
    Safety policy:
    - same set + different order -> reorder
    - different set -> keep original order and rely on key checks
    """
    ref_names = ref_adata.obs_names.to_numpy()
    other_names = other_adata.obs_names.to_numpy()

    if ref_names.shape[0] != other_names.shape[0]:
        raise AssertionError(
            f"{model_key}: n_obs mismatch. ref={ref_names.shape[0]}, other={other_names.shape[0]}"
        )

    if np.array_equal(ref_names, other_names):
        return other_adata, True

    ref_set = set(ref_names.tolist())
    other_set = set(other_names.tolist())
    if ref_set != other_set:
        missing = sorted(ref_set - other_set)[:5]
        extra = sorted(other_set - ref_set)[:5]
        print(
            f"[align] {model_key}: obs_names namespace differs; "
            f"skip name-based reordering and keep current order. "
            f"example missing={missing}, extra={extra}"
        )
        return other_adata, False
    print(f"[align] {model_key}: obs_names order differs; reordering to reference.")
    aligned = other_adata[ref_adata.obs_names, :].copy()
    assert np.array_equal(aligned.obs_names.to_numpy(), ref_names)
    return aligned, True


def _normalize_time_vec_for_compare(vec: np.ndarray):
    """
    Try to normalize time-like labels (e.g. 3 vs '3d') into numeric values.
    Return None if normalization is not possible for all rows.
    """
    s = pd.Series(vec, dtype="object").astype(str).str.strip()
    # Extract first numeric token from each value (supports '3d', 't3', '3.0', etc.)
    extracted = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
    norm = pd.to_numeric(extracted, errors="coerce")
    if norm.isna().any():
        return None
    return norm.to_numpy()


def _assert_time_compatible(ref_time_vec: np.ndarray, other_time_vec: np.ndarray, model_key: str):
    if np.array_equal(ref_time_vec, other_time_vec):
        return

    ref_norm = _normalize_time_vec_for_compare(ref_time_vec)
    other_norm = _normalize_time_vec_for_compare(other_time_vec)
    if ref_norm is not None and other_norm is not None and np.array_equal(ref_norm, other_norm):
        print(
            f"[align] {model_key}: time labels differ in format but normalized values match."
        )
        return

    neq = np.where(ref_time_vec != other_time_vec)[0]
    i0 = int(neq[0]) if len(neq) else -1
    ref_v = ref_time_vec[i0] if i0 >= 0 else "N/A"
    other_v = other_time_vec[i0] if i0 >= 0 else "N/A"
    raise AssertionError(
        f"{model_key}: time alignment mismatch; first mismatch at idx={i0}, "
        f"ref={ref_v!r}, other={other_v!r}, mismatch_count={len(neq)}"
    )


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
        adata_key, aligned_by_names = _align_to_reference_obs(adata, adata_key, key)

        if args.cell_type_key is not None:
            if args.cell_type_key not in adata.obs.columns:
                raise KeyError(f"cell_type_key {args.cell_type_key!r} not in reference obs")
            if args.cell_type_key not in adata_key.obs.columns:
                raise KeyError(f"cell_type_key {args.cell_type_key!r} not in model {key} obs")
            ref_ct = np.asarray(adata.obs[args.cell_type_key].values)
            other_ct = np.asarray(adata_key.obs[args.cell_type_key].values)
            if not np.array_equal(ref_ct, other_ct):
                neq = np.where(ref_ct != other_ct)[0]
                i0 = int(neq[0]) if len(neq) else -1
                raise AssertionError(
                    f"{key}: cell_type_key mismatch at idx={i0}; "
                    f"ref={ref_ct[i0]!r}, other={other_ct[i0]!r}"
                )
        else:
            other_time_col = _resolve_time_col(adata_key, args.time_key)
            other_time_vec = np.asarray(adata_key.obs[other_time_col].values)
            if aligned_by_names:
                try:
                    _assert_time_compatible(ref_time_vec, other_time_vec, key)
                except AssertionError as exc:
                    print(
                        f"[align] {key}: time check warning after obs-name alignment: {exc}"
                    )
            else:
                _assert_time_compatible(ref_time_vec, other_time_vec, key)

        print("hidden dim of", key, adata_key.obsm["X_emb"].shape[1])
        adata.obsm[f"X_{key}"] = adata_key.obsm["X_emb"]

    _ensure_time_column(adata, args.time_key)

    adata.write_h5ad(output_file)
    print("save to", output_file)
