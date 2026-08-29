from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


FLAT_HOLDOUT_RE = re.compile(r"^(?P<dataset>.+)_holdt(?P<time>\d+)$")
NESTED_HOLDOUT_RE = re.compile(r"^holdt(?P<time>\d+)$")


def parse_tag_fields(tag: str) -> Dict[str, object]:
    model = tag.split("_deepruot_", 1)[0] if "_deepruot_" in tag else tag.split("_")[0]
    out: Dict[str, object] = {
        "model": model,
        "dim": None,
        "otmode": None,
        "alignment": None,
    }

    dim_m = re.search(r"_dim(\d+)_", tag)
    if dim_m:
        out["dim"] = int(dim_m.group(1))

    ot_m = re.search(r"_otmode([^_]+)_", tag)
    if ot_m:
        out["otmode"] = ot_m.group(1)

    if "_gpa_consensus_" in tag:
        out["alignment"] = "gpa"
    elif "_identity_" in tag:
        out["alignment"] = "identity"
    elif "_procrustes_" in tag:
        out["alignment"] = "procrustes"
    elif "_ridge_" in tag:
        out["alignment"] = "ridge"
    return out


def iter_holdout_dirs(
    results_roots: Sequence[Path],
    dataset: Optional[str] = None,
) -> List[Tuple[str, int, Path]]:
    found: List[Tuple[str, int, Path]] = []
    seen = set()
    for root in results_roots:
        root = Path(root)
        if not root.exists():
            print(f"[skip] missing results root: {root}")
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            flat = FLAT_HOLDOUT_RE.match(child.name)
            if flat:
                ds, time = flat.group("dataset"), int(flat.group("time"))
                if dataset is None or ds == dataset:
                    key = (ds, time, child.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append((ds, time, child))
                continue
            if dataset is not None and child.name != dataset:
                continue
            for sub in sorted(p for p in child.iterdir() if p.is_dir()):
                nested = NESTED_HOLDOUT_RE.match(sub.name)
                if not nested:
                    continue
                ds, time = child.name, int(nested.group("time"))
                if dataset is None or ds == dataset:
                    key = (ds, time, sub.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append((ds, time, sub))
    return found


def _attach_common_fields(df: pd.DataFrame, dataset: str, time: int) -> pd.DataFrame:
    parsed = df["tag"].map(parse_tag_fields).apply(pd.Series)
    out = pd.concat([df, parsed], axis=1)
    out["dataset"] = dataset
    out["time"] = time
    return out


def _read_grouped_metric(
    csv_path: Path,
    group_agg: Dict[str, Tuple[str, str]],
    extra_mean_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    res = pd.read_csv(csv_path)
    agg_dict = dict(group_agg)
    if extra_mean_cols:
        for col in extra_mean_cols:
            if col in res.columns:
                agg_dict[f"{col}_Mean"] = (col, "mean")
                agg_dict[f"{col}_Std"] = (col, "std")
    return res.groupby("tag", as_index=False).agg(**agg_dict)


def aggregate_one_metric(
    holdouts: Sequence[Tuple[str, int, Path]],
    filename: str,
    builder,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for dataset, time, holdout_dir in holdouts:
        csv_path = holdout_dir / filename
        if not csv_path.exists():
            print(f"[skip] {csv_path}")
            continue
        grp = builder(csv_path)
        parts.append(_attach_common_fields(grp, dataset, time))
    if not parts:
        raise FileNotFoundError(f"No {filename} found in the given holdout directories.")
    return pd.concat(parts, axis=0, ignore_index=True)


def _build_w1(csv_path: Path) -> pd.DataFrame:
    return _read_grouped_metric(
        csv_path,
        {
            "W1_Mean": ("W1 Distance", "mean"),
            "W1_Std": ("W1 Distance", "std"),
            "TMV_Mean": ("TMV", "mean"),
            "TMV_Std": ("TMV", "std"),
        },
    )


def _build_pseudotime(csv_path: Path) -> pd.DataFrame:
    return _read_grouped_metric(
        csv_path,
        {
            "Spearman_Mean": ("Spearman", "mean"),
            "Spearman_Std": ("Spearman", "std"),
        },
        extra_mean_cols=["Kendall"],
    )


def _build_tcvc(csv_path: Path) -> pd.DataFrame:
    res = pd.read_csv(csv_path)
    if "Run" in res.columns:
        return res.groupby("tag", as_index=False).agg(TC=("TC", "mean"), VC=("VC", "mean"))
    if "TC" not in res.columns or "VC" not in res.columns:
        raise ValueError(f"tcvc.csv missing TC/VC columns: {csv_path}")
    return res[["tag", "TC", "VC"]].copy()


def aggregate_metrics(
    results_roots: Sequence[Path],
    output_dir: Path,
    dataset: Optional[str] = None,
) -> Dict[str, Path]:
    holdouts = iter_holdout_dirs(results_roots, dataset=dataset)
    if not holdouts:
        roots = ", ".join(str(p) for p in results_roots)
        raise FileNotFoundError(
            f"No holdout directories found under {roots}"
            + (f" for dataset={dataset}" if dataset else "")
            + ". Expected results/gpa/EMT_holdt0 or results/gpa/EMT/holdt0."
        )

    print(f"[config] holdouts={[(ds, t, str(p)) for ds, t, p in holdouts]}")
    df_w1 = aggregate_one_metric(holdouts, "w1tmv.csv", _build_w1)
    df_pseudo = aggregate_one_metric(holdouts, "pseudotime.csv", _build_pseudotime)
    df_tcvc = aggregate_one_metric(holdouts, "tcvc.csv", _build_tcvc)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "w1": output_dir / "all_w1.csv",
        "pseudotime": output_dir / "all_pseudotime.csv",
        "tcvc": output_dir / "all_tcvc.csv",
    }
    df_w1.to_csv(paths["w1"], index=False)
    df_pseudo.to_csv(paths["pseudotime"], index=False)
    df_tcvc.to_csv(paths["tcvc"], index=False)
    print(f"[saved] {paths['w1']} shape={df_w1.shape}")
    print(f"[saved] {paths['pseudotime']} shape={df_pseudo.shape}")
    print(f"[saved] {paths['tcvc']} shape={df_tcvc.shape}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Fig. 2 metrics from completed holdout CSVs. "
            "Pass one or more result roots that contain EMT_holdt* folders "
            "(or nested EMT/holdt*). Multiple alignments are concatenated as-is."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        nargs="+",
        default=[Path("results/gpa")],
        help="Directories containing {dataset}_holdt{N} (or {dataset}/holdt{N}) outputs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name filter, e.g. EMT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("main_figures"),
        help="Directory to write all_w1.csv / all_pseudotime.csv / all_tcvc.csv.",
    )
    args = parser.parse_args()
    aggregate_metrics(args.results_root, args.output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
