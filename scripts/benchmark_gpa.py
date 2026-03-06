import os
import yaml
import glob
import pandas as pd
from itertools import product
import scanpy as sc
import sys
sys.path.append('/macroverse/public/zhouxy/scllms/scFMs_dynamic')
from benchmark.preprocessor import EmbeddingPreprocessor
from benchmark.dynamic import DeepRUOTEngine
from benchmark.evaluate_utils import AlignmnetEvaluator, load_runs_npz
from benchmark.alignment import make_aligner, save_aligner

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse


def load_benchmark_config(cfg_or_path):
    # 支持直接传 dict 或 YAML 路径
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    with open(cfg_or_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(cfg):
    required_top = ["dataset", "paths", "models", "align_strategies", "metrics", "dynamic_methods"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"缺少配置项: {k}")
    ds = cfg["dataset"]
    for k in ["name", "path", "ref_key", "time_key", "train_times", "test_times"]:
        if k not in ds:
            raise ValueError(f"dataset 缺少配置项: {k}")
    paths = cfg["paths"]
    for k in ["input_dir", "results_dir", "artifacts", "benchmark_results"]:
        if k not in paths:
            raise ValueError(f"paths 缺少配置项: {k}")

    # 验证 dynamic_methods
    if not cfg["dynamic_methods"]:
        raise ValueError("dynamic_methods 不能为空")
    for method in cfg["dynamic_methods"]:
        if "name" not in method:
            raise ValueError("dynamic_methods 中每个方法必须有 'name' 字段")
        if "engine_type" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} 缺少 'engine_type' 字段")
        if "params" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} 缺少 'params' 字段")
        if "engine_config" not in method:
            raise ValueError(f"dynamic_methods.{method['name']} 缺少 'engine_config' 字段")

    # 默认值
    cfg.setdefault("options", {})
    cfg["options"].setdefault("skip_existing", True)
    return cfg


def expand_jobs(cfg):
    """
    返回所有组合的列表，每个条目包含 model + dynamic_method + method_params
    """
    jobs = []
    for model in cfg["models"]:
        for method_cfg in cfg["dynamic_methods"]:
            # 从 method_cfg["params"] 中提取参数网格
            param_names = list(method_cfg["params"].keys())
            param_values = list(method_cfg["params"].values())

            # 生成笛卡尔积
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


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def has_existing_runs(res_dir, pattern, expected_min=1):
    files = glob.glob(os.path.join(res_dir, pattern))
    return len(files) >= expected_min


def build_context(cfg):
    """
    顺序执行时构建共享上下文：adata、times_all、可选字段等
    """
    data_path = os.path.join(cfg["dataset"]["path"], 'benchmark.h5ad')
    adata = sc.read_h5ad(data_path)

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
    单任务执行（model + method + params）：预处理→训练/生成→读结果→对齐/评估。
    - context 为顺序路径共享对象；并行路径传 None（内部独立加载）
    返回：rows_by_metric（dict: metric->list[df]）、timing_rows（list[df]）
    """
    ds = cfg["dataset"]
    paths = cfg["paths"]
    metrics = cfg["metrics"]
    align_strategies = cfg["align_strategies"]

    model = job["model"]
    method_name = job["method_name"]
    engine_type = job["engine_type"]
    engine_config = job["engine_config"]
    params = job["params"]

    t0_total = time.perf_counter()

    # 构建数据与预处理器
    if context is None:
        # 并行或独立执行：各自读取，避免共享冲突
        adata = sc.read_h5ad(os.path.join(ds["path"], 'benchmark.h5ad'))
        times_all = adata.obs[ds["time_key"]].to_numpy().astype(float)
        pseudotime_key = ds.get("pseudotime_key", None)
        pseudotime = adata.obs[pseudotime_key].to_numpy() if pseudotime_key else None
        cell_types_key = ds.get("cell_type_key", None)
        cell_types = adata.obs[cell_types_key].to_numpy() if cell_types_key else None
        root_cell_type = ds.get("root_cell_type", None) if cell_types_key else None

    else:
        # 顺序执行：复用共享数据
        adata = context["adata"]
        times_all = context["times_all"]
        pseudotime = context["pseudotime"]
        cell_types = context["cell_types"]
        root_cell_type = context["root_cell_type"]


    # 获取 dim 参数（通用）
    dim = int(params.get("dim", 50))

    # 每个任务独立创建 emb_prep
    emb_prep = EmbeddingPreprocessor(
        adata,
        ds["time_key"],
        ref_key=f'X_{ds["ref_key"]}',
        train_times=ds["train_times"],
        test_times=ds["test_times"]
    )

    # 预处理
    t0_prep = time.perf_counter()
    emb_prep.fit_ref(k=dim, store_key='Z_ref')
    emb_prep.fit_embedding(model_key=model, k=dim)
    Z_model_all = emb_prep.get_Z(model)
    Z_model_train = emb_prep.get_Z(model, split='train')
    Z_ref_train = emb_prep.get_Z('ref', split='train')
    prep_sec = time.perf_counter() - t0_prep

    # 构建 job 标识符（用于路径和命名）
    test_times = ds["test_times"]
    ref_key = ds["ref_key"]

    # 生成参数字符串
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    exp_name = f"{model}_holdt{test_times[0]}_{method_name}_{param_str}"

    # 目录分层：results_dir/method_name/model/params/
    out_dir_model = ensure_dir(os.path.join(paths["results_dir"], method_name, model, param_str))
    art_dir_model = ensure_dir(os.path.join(paths["artifacts"], method_name, model, param_str))

    # 根据 engine_type 初始化引擎
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
        raise ValueError(f"不支持的 engine_type: {engine_type}")

    skip_existing = cfg["options"].get("skip_existing", True)

    fit_sec = 0.0
    gen_sec = 0.0

    if not (skip_existing and has_existing_runs(os.path.join(out_dir_model, exp_name), run_pattern, expected_min=num_runs)):
        t0_fit = time.perf_counter()

        # 根据不同引擎传递不同参数
        if engine_type == "deepruot":
            # DeepRUOT 特有参数
            otmode = params.get("otmode", "ruot")
            cfg_path = engine.fit(
                data={"Z": Z_model_all, "t": times_all, "hold_out": test_times[0], "data_dir": paths["input_dir"]},
                otmode=otmode,
                config={"exp": {"name": exp_name, "output_dir": out_dir_model}}
            )
        fit_sec = time.perf_counter() - t0_fit

        t0_gen = time.perf_counter()
        res_dir = engine.generate(config_path=cfg_path, num_runs=num_runs)
        gen_sec = time.perf_counter() - t0_gen
    else:
        res_dir = os.path.join(out_dir_model, exp_name)

    # 读结果
    t0_io = time.perf_counter()
    runs = load_runs_npz(res_dir, pattern=run_pattern)
    vel_path = os.path.join(res_dir, 'velocity.h5ad')
    velocity_adata = sc.read_h5ad(vel_path) if os.path.exists(vel_path) else None
    io_sec = time.perf_counter() - t0_io

    # 对齐 + 评估
    rows_by_metric = {m: [] for m in metrics}
    timing_rows = []  # 每个 aligner 一行 timing

    for align_name in align_strategies:

        # 如果 gpa align，在这里构建所有 model 的共识空间作为参考空间
        if align_name == 'gpa_consensus':
            aligner = make_aligner(align_name)
            # 用所有 model 的 embedding gpa 作为共识空间
            allmodels = ['hvg', 'geneformer', 'scgpt', 'scfoundation', 'uce', 'genecompass']
            # prepare all z
            for model in allmodels:
                emb_prep.fit_embedding(model_key=model, k=dim)
            Z_train_allmodels = {key: emb_prep.get_Z(key, split='train') for key in allmodels}
            aligner.fit(Z_train_allmodels) 
            aligner.save(os.path.join(art_dir_model, 'gpa.joblib'))
            model = job['model']
            aligner.set_default_view(model) # 设置用这个 model 的视角
        else:
            aligner = make_aligner(align_name)
            aligner.fit(Z_model_train, Z_ref_train) #只在训练集拟合aligner
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

        # 构建 timing row，包含所有参数
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
        # 添加所有方法参数
        timing_row.update(params)
        timing_rows.append(pd.DataFrame([timing_row]))

    return rows_by_metric, timing_rows


def run_benchmark(benchmark_config, max_workers=None, save_path=None):
    """
    支持顺序与并行的批量运行；聚合指标与任务用时。
    """
    cfg = validate_config(load_benchmark_config(benchmark_config))
    jobs = expand_jobs(cfg)

    # 如果未显式传入，读取配置中的 options.max_workers
    if max_workers is None:
        max_workers = int(cfg.get("options", {}).get("max_workers", 1))

    # 如果通过参数传入 save_path，则覆盖配置文件中的 benchmark_results
    if save_path is not None:
        cfg["paths"]["benchmark_results"] = save_path

    # 汇总表现
    all_rows = {m: [] for m in cfg["metrics"]}
    all_timing = []

    def merge_rows(rows_by_metric, timing_rows):
        if rows_by_metric:
            for m in cfg["metrics"]:
                all_rows[m].extend(rows_by_metric[m])
        if timing_rows:
            all_timing.extend(timing_rows)

    if max_workers <= 1:
        # 顺序：构建共享上下文，提高效率
        context = build_context(cfg)
        for job in jobs:
            rows_by_metric, timing_rows = execute_job(cfg, job, context=context)
            merge_rows(rows_by_metric, timing_rows)
    else:
        # 并行：每个进程独立执行
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

    # 保存指标 CSV
    ensure_dir(cfg["paths"]["benchmark_results"])
    for m in cfg["metrics"]:
        if all_rows[m]:
            out_df = pd.concat(all_rows[m], axis=0, ignore_index=True)
            out_file = os.path.join(cfg["paths"]["benchmark_results"], f"{m}.csv")
            out_df.to_csv(out_file, index=False)

    # 保存 timing CSV
    if all_timing:
        timing_df = pd.concat(all_timing, axis=0, ignore_index=True)
        timing_file = os.path.join(cfg["paths"]["benchmark_results"], "timing.csv")
        timing_df.to_csv(timing_file, index=False)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scFMs dynamic benchmark')
    parser.add_argument('--config', type=str, help='Path to benchmark config YAML file')
    parser.add_argument('--save_path', type=str, help='Override save path for benchmark results')
    args = parser.parse_args()

    if args.config:
        run_benchmark(args.config, save_path=args.save_path)
    else:
        # 示例配置（可直接运行）
        demo_cfg = {
            "dataset": {
                "name": "EMT",
                "path": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/data/embeddings/EMT",
                "ref_key": "hvg",
                "time_key": "time",
                'pseudotime_key': 'Pseudotime',
                "train_times": [0, 1, 2],
                "test_times": [3]
            },
            "paths": {
                "input_dir": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/data/deepruot_input/EMT",
                "results_dir": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/dynamic_results/EMT",
                "artifacts": "./artifacts/EMT",
                "benchmark_results": "./results/EMT"
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
                        "train_script": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/DeepRUOTv2/train_RUOT.py",
                        "generate_script": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/DeepRUOTv2/infer_RUOT.py",
                        "template_cfg": "/macroverse/public/zhouxy/scllms/scFMs_dynamic/DeepRUOTv2/config/emt_config.yaml",
                        "python_exec": "python",
                        "num_runs": 10,
                        "run_pattern": "sde_run_*.npz" # DeepRUOT 生成的文件匹配
                    }
                }
                # 未来可以添加其他动力学方法：
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
