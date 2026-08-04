import os
import yaml
import glob
import pandas as pd
from itertools import product
import scanpy as sc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.preprocessor import EmbeddingPreprocessor
from benchmark.dynamic import DeepRUOTEngine
from benchmark.evaluate_utils import AlignmnetEvaluator, load_runs_npz
from benchmark.alignment import make_aligner, save_aligner

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse


def load_benchmark_config(cfg_or_path):
    # Accept either a dict or a path to a YAML file
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    with open(cfg_or_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(cfg):
    required_top = ["dataset", "paths", "models", "align_strategies", "metrics", "dynamic_methods"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing top-level config key: {k}")
    ds = cfg["dataset"]
    for k in ["name", "path", "ref_key", "time_key", "train_times", "test_times"]:
        if k not in ds:
            raise ValueError(f"Missing dataset config key: {k}")
    paths = cfg["paths"]
    for k in ["input_dir", "results_dir", "artifacts", "benchmark_results"]:
        if k not in paths:
            raise ValueError(f"Missing paths config key: {k}")

    # Validate dynamic_methods
    if not cfg["dynamic_methods"]:
        raise ValueError("dynamic_methods must not be empty")
    for method in cfg["dynamic_methods"]:
        if "name" not in method:
            raise ValueError("Each entry in dynamic_methods must have a 'name' field")
        if "engine_type" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} is missing 'engine_type'")
        if "params" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} is missing 'params'")
        if "engine_config" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} is missing 'engine_config'")

    # Defaults
    cfg.setdefault("options", {})
    cfg["options"].setdefault("skip_existing", True)
    return cfg


def validate_adata_obsm_for_benchmark(cfg, adata):
    """
    Before running many jobs, check that benchmark.h5ad has obsm for ref and each model.
    Avoids repeating errors like "X_xxx is not in adata.obsm" for every combination.
    """
    ds = cfg["dataset"]
    ref_obsm = f"X_{ds['ref_key']}"
    need = {ref_obsm}
    for m in cfg["models"]:
        need.add(f"X_{m}")
    missing = sorted(k for k in need if k not in adata.obsm)
    if missing:
        available = sorted(
            str(k) for k in adata.obsm.keys() if str(k).startswith("X_")
        )
        raise ValueError(
            "Benchmark AnnData is missing required obsm for this config: "
            f"{missing}\nobsm keys starting with X_: {available}\n"
            "Align cfg['models'] / dataset.ref_key with the integrated h5ad, or rerun integrate_embedding."
        )


def expand_jobs(cfg):
    """
    Return the list of all job combinations, each with model + dynamic_method + method_params.
    """
    jobs = []
    for model in cfg["models"]:
        for method_cfg in cfg["dynamic_methods"]:
            # Build parameter grid from method_cfg["params"]
            param_names = list(method_cfg["params"].keys())
            param_values = list(method_cfg["params"].values())

            # Cartesian product
            for param_combo in product(*param_values):
                job = {
                    "model": model,
                    "method_name": method_cfg["name"],
                    "engine_type": method_cfg["engine_type"],
                    "engine_config": method_cfg["engine_config"],
                    "params": dict(zip(param_names, param_combo))
                }
                jobs.append(job)
    return jobs


def deepruot_config_overrides(params):
    """
    Convert benchmark-grid convenience parameters into nested DeepRUOT config keys.
    The original params stay in tags/timing CSVs for traceability.
    """
    overrides = {}
    if "score_sigma" in params:
        overrides.setdefault("score_train", {})["sigma"] = float(params["score_sigma"])
    return overrides


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def has_existing_runs(res_dir, pattern, expected_min=1):
    files = glob.glob(os.path.join(res_dir, pattern))
    return len(files) >= expected_min


def build_context(cfg, adata=None):
    """
    Build shared context for sequential runs: adata, times_all, optional fields, etc.
    If adata is passed in (e.g. already validated in run_benchmark), do not re-read from disk.
    """
    if adata is None:
        adata = sc.read_h5ad(cfg["dataset"]["path"])
    validate_adata_obsm_for_benchmark(cfg, adata)

    time_key = cfg["dataset"]["time_key"]
    times_all = adata.obs[time_key].to_numpy().astype(float)

    pseudotime_key = cfg["dataset"].get("pseudotime_key", None)
    pseudotime = adata.obs[pseudotime_key].to_numpy() if pseudotime_key else None

    cell_types_key = cfg["dataset"].get("cell_type_key", None)
    cell_types = adata.obs[cell_types_key].to_numpy() if cell_types_key else None
    root_cell_type = cfg["dataset"].get("root_cell_type", None) if cell_types_key else None

    return {
        "adata": adata,
        "times_all": times_all,
        "pseudotime": pseudotime,
        "cell_types": cell_types,
        "root_cell_type": root_cell_type,
    }


def execute_job(cfg, job, context=None):
    """
    Run one job (model + method + params): preprocess -> train/generate -> load -> align/evaluate.
    - context: shared object for sequential path; use None for parallel (each process loads data).
    Returns: rows_by_metric (dict metric -> list[df]), timing_rows (list[df]).
    """
    ds = cfg["dataset"]
    paths = cfg["paths"]
    metrics = cfg["metrics"]
    align_strategies = cfg["align_strategies"]
    aligner_kwargs_cfg = cfg.get("aligner_kwargs", {})

    model = job["model"]
    method_name = job["method_name"]
    engine_type = job["engine_type"]
    engine_config = job["engine_config"]
    params = job["params"]

    t0_total = time.perf_counter()

    # Build data and preprocessor
    if context is None:
        # Parallel or isolated: each process reads its own copy to avoid sharing issues
        adata = sc.read_h5ad(ds["path"])
        times_all = adata.obs[ds["time_key"]].to_numpy().astype(float)
        pseudotime_key = ds.get("pseudotime_key", None)
        pseudotime = adata.obs[pseudotime_key].to_numpy() if pseudotime_key else None
        cell_types_key = ds.get("cell_type_key", None)
        cell_types = adata.obs[cell_types_key].to_numpy() if cell_types_key else None
        root_cell_type = ds.get("root_cell_type", None) if cell_types_key else None

    else:
        # Sequential: reuse shared data
        adata = context["adata"]
        times_all = context["times_all"]
        pseudotime = context["pseudotime"]
        cell_types = context["cell_types"]
        root_cell_type = context["root_cell_type"]


    # General dim parameter
    dim = int(params.get("dim", 50))

    # One EmbeddingPreprocessor per task
    emb_prep = EmbeddingPreprocessor(
        adata,
        ds["time_key"],
        ref_key=f'X_{ds["ref_key"]}',
        train_times=ds["train_times"],
        test_times=ds["test_times"]
    )

    # Preprocess
    t0_prep = time.perf_counter()
    emb_prep.fit_ref(k=dim, store_key='Z_ref')
    emb_prep.fit_embedding(model_key=model, k=dim)
    Z_model_all = emb_prep.get_Z(model)
    Z_model_train = emb_prep.get_Z(model, split='train')
    Z_ref_train = emb_prep.get_Z('ref', split='train')
    prep_sec = time.perf_counter() - t0_prep

    # Job id for paths and naming
    test_times = ds["test_times"]
    ref_key = ds["ref_key"]

    # Parameter string
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    exp_name = f"{model}_holdt{test_times[0]}_{method_name}_{param_str}"

    # Directory layout: results_dir/method_name/model/params/
    out_dir_model = ensure_dir(os.path.join(paths["results_dir"], method_name, model, param_str))
    art_dir_model = ensure_dir(os.path.join(paths["artifacts"], method_name, model, param_str))

    # Init engine from engine_type
    if engine_type == "deepruot":
        engine = DeepRUOTEngine(
            train_script=engine_config["train_script"],
            generate_script=engine_config["generate_script"],
            template_cfg=engine_config["template_cfg"],
            python_exec=engine_config.get("python_exec", "python"),
            file_layout={'data_file_name': f"{model}_holdt{test_times[0]}_D{dim}.csv"}
        )
        run_pattern = engine_config.get("run_pattern", "run_*.npz")
        num_runs = int(engine_config.get("num_runs", 10))
    else:
        raise ValueError(f"Unsupported engine_type: {engine_type}")

    skip_existing = cfg["options"].get("skip_existing", True)

    fit_sec = 0.0
    gen_sec = 0.0

    if not (skip_existing and has_existing_runs(os.path.join(out_dir_model, exp_name), run_pattern, expected_min=num_runs)):
        t0_fit = time.perf_counter()

        # Engine-specific arguments
        if engine_type == "deepruot":
            # DeepRUOT-specific
            otmode = params.get("otmode", "ruot")
            cfg_path = engine.fit(
                data={"Z": Z_model_all, "t": times_all, "hold_out": test_times[0], "data_dir": paths["input_dir"]},
                otmode=otmode,
                config={"exp": {"name": exp_name, "output_dir": out_dir_model}},
                config_overrides=deepruot_config_overrides(params),
            )
        fit_sec = time.perf_counter() - t0_fit

        t0_gen = time.perf_counter()
        res_dir = engine.generate(config_path=cfg_path, num_runs=num_runs)
        gen_sec = time.perf_counter() - t0_gen
    else:
        res_dir = os.path.join(out_dir_model, exp_name)

    # Load results
    t0_io = time.perf_counter()
    runs = load_runs_npz(res_dir, pattern=run_pattern)
    vel_path = os.path.join(res_dir, 'velocity.h5ad')
    velocity_adata = sc.read_h5ad(vel_path) if os.path.exists(vel_path) else None
    io_sec = time.perf_counter() - t0_io

    # Align + evaluate
    rows_by_metric = {m: [] for m in metrics}
    timing_rows = []  # one timing row per aligner

    for align_name in align_strategies:

        # For GPA consensus, build shared space from all models as reference
        aligner_kwargs = aligner_kwargs_cfg.get(align_name, {}) or {}

        if align_name == 'gpa_consensus':
            aligner = make_aligner(align_name, **aligner_kwargs)
            # Match cfg['models'] so a subset of models is not hard-coded
            allmodels = list(cfg["models"])
            for _m in allmodels:
                emb_prep.fit_embedding(model_key=_m, k=dim)
            Z_train_allmodels = {key: emb_prep.get_Z(key, split='train') for key in allmodels}
            aligner.fit(Z_train_allmodels) 
            aligner.save(os.path.join(art_dir_model, 'gpa.joblib'))
            model = job['model']
            aligner.set_default_view(model)  # use this model's view
        else:
            aligner = make_aligner(align_name, **aligner_kwargs)
            aligner.fit(Z_model_train, Z_ref_train)  # fit only on training cells
            save_aligner(aligner, art_dir_model)

        evaluator = AlignmnetEvaluator(
            results=runs,
            alldata=Z_model_all,
            alltimes=times_all,
            test_times=test_times
        )

        t0_eval = time.perf_counter()
        bench = evaluator.evaluate_all_metrics(
            aligner=aligner,
            metrics=metrics,
            velocity_adata=velocity_adata,
            pseudotime=pseudotime,
            cell_types=cell_types,
            root_cell_type=root_cell_type,
        )
        eval_sec = time.perf_counter() - t0_eval

        tag = "_".join([model, method_name, param_str, align_name, f"ref{ref_key}"])
        for m in metrics:
            dfm = bench[m].copy()
            if 'tag' not in dfm.columns:
                dfm.insert(0, 'tag', tag)
            rows_by_metric[m].append(dfm)

        # Timing row with all parameters
        timing_row = {
            "tag": tag,
            "model": model,
            "method": method_name,
            "aligner": align_name,
            "preprocess_sec": prep_sec,
            "fit_sec": fit_sec,
            "generate_sec": gen_sec,
            "io_sec": io_sec,
            "eval_sec": eval_sec,
            "total_sec": (time.perf_counter() - t0_total),
        }
        timing_row.update(params)
        timing_rows.append(pd.DataFrame([timing_row]))

    return rows_by_metric, timing_rows


def raise_worker_failures(failures):
    """Raise one final error after all parallel worker failures are collected."""
    if not failures:
        return
    details = []
    for job, error in failures:
        details.append(
            f"{job['model']} {job['method_name']} {job['params']}: "
            f"{type(error).__name__}: {error}"
        )
    raise RuntimeError(
        f"{len(failures)} benchmark worker(s) failed after collection: "
        + " | ".join(details)
    )


def run_benchmark(benchmark_config, max_workers=None, save_path=None):
    """
    Batch run (sequential or parallel); aggregate metrics and per-task timing.
    """
    cfg = validate_config(load_benchmark_config(benchmark_config))
    jobs = expand_jobs(cfg)

    adata0 = sc.read_h5ad(cfg["dataset"]["path"])
    validate_adata_obsm_for_benchmark(cfg, adata0)

    if max_workers is None:
        max_workers = int(cfg.get("options", {}).get("max_workers", 1))

    if max_workers > 1:
        del adata0

    if save_path is not None:
        cfg["paths"]["benchmark_results"] = save_path

    all_rows = {m: [] for m in cfg["metrics"]}
    all_timing = []
    worker_failures = []

    def merge_rows(rows_by_metric, timing_rows):
        if rows_by_metric:
            for m in cfg["metrics"]:
                all_rows[m].extend(rows_by_metric[m])
        if timing_rows:
            all_timing.extend(timing_rows)

    if max_workers <= 1:
        # Sequential: shared context (reuse validated adata)
        context = build_context(cfg, adata=adata0)
        for job in jobs:
            rows_by_metric, timing_rows = execute_job(cfg, job, context=context)
            merge_rows(rows_by_metric, timing_rows)
    else:
        # Parallel: each process runs independently
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(execute_job, cfg, job, None): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    rows_by_metric, timing_rows = fut.result()
                    merge_rows(rows_by_metric, timing_rows)
                    print(f"done: {job['model']} {job['method_name']} {job['params']}")
                except Exception as e:
                    print(f"failed: {job['model']} {job['method_name']} {job['params']} -> {e}")
                    worker_failures.append((job, e))

    ensure_dir(cfg["paths"]["benchmark_results"])
    for m in cfg["metrics"]:
        if all_rows[m]:
            out_df = pd.concat(all_rows[m], axis=0, ignore_index=True)
            out_file = os.path.join(cfg["paths"]["benchmark_results"], f"{m}.csv")
            out_df.to_csv(out_file, index=False)

    if all_timing:
        timing_df = pd.concat(all_timing, axis=0, ignore_index=True)
        timing_file = os.path.join(cfg["paths"]["benchmark_results"], "timing.csv")
        timing_df.to_csv(timing_file, index=False)

    raise_worker_failures(worker_failures)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scFMs dynamic benchmark')
    parser.add_argument('--config', type=str, help='Path to benchmark config YAML file')
    parser.add_argument('--save_path', type=str, help='Override save path for benchmark results')
    args = parser.parse_args()

    if args.config:
        run_benchmark(args.config, save_path=args.save_path)
    else:
        # Example config (runnable)
        demo_cfg = {
            "dataset": {
                "name": "EMT",
                "path": str(_REPO_ROOT / "data" / "embeddings" / "EMT" / "benchmark.h5ad"),
                "ref_key": "hvg",
                "time_key": "time",
                "pseudotime_key": "Pseudotime",
                "train_times": [0, 1, 2],
                "test_times": [3],
            },
            "paths": {
                "input_dir": str(_REPO_ROOT / "data" / "deepruot_input" / "EMT"),
                "results_dir": str(_REPO_ROOT / "results" / "dynamic_results" / "EMT"),
                "artifacts": str(_REPO_ROOT / "artifacts" / "EMT"),
                "benchmark_results": str(_REPO_ROOT / "results" / "EMT"),
            },
            "models": ["hvg", "genecompass", "uce", "scgpt", "scfoundation", "geneformer"],
            "dynamic_methods": [
                {
                    "name": "deepruot",
                    "engine_type": "deepruot",
                    "params": {
                        "dim": [50, 10, 5, 20],
                        "otmode": ["ruot", "dot", "sb", "uot"]
                    },
                    "engine_config": {
                        "train_script": str(_REPO_ROOT / "DeepRUOTv2" / "train_RUOT.py"),
                        "generate_script": str(_REPO_ROOT / "DeepRUOTv2" / "infer_RUOT.py"),
                        "template_cfg": str(
                            _REPO_ROOT / "DeepRUOTv2" / "config" / "emt_config.yaml"
                        ),
                        "python_exec": "python",
                        "num_runs": 10,
                        "run_pattern": "sde_run_*.npz"  # glob for DeepRUOT output npz files
                    }
                }
                # Future: other dynamics methods, e.g.
                # {
                #     "name": "wot",
                #     "engine_type": "wot",
                #     "params": {
                #         "dim": [30, 50],
                #         "lambda_reg": [1, 10, 50]
                #     },
                #     "engine_config": {...}
                # }
            ],
            "align_strategies": ["identity", "procrustes", "ridge"],
            "metrics": ["w1tmv", "tcvc", "pseudotime"],
            "options": {
                "skip_existing": True,
                'max_workers': 10
            }
        }
        run_benchmark(demo_cfg)
