from __future__ import annotations
import os
import json
import glob
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Sequence, List, Tuple
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None


# ----tools----
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if yaml is None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        with open(path, "w") as f:
            yaml.safe_dump(obj, f)

def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def to_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x

# def otmode_config(config, otmode):



# -------- Runner --------
class BaseRunner:
    def fit(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def generate(
        self,
        state: Dict[str, Any],
        t_eval: np.ndarray,
        config: Dict[str, Any],
        num_runs: int = 1,
        clean_previous: bool = True
    ) -> Dict[str, Any]:
        raise NotImplementedError


# -------- deepRUOT adaptor --------
class DeepRUOTEngine(BaseRunner):
    """
    适配你的 config 架构：
      - 模板键：exp.output_dir, data.file_path, data.dim, data.hold_one_out, data.hold_out, model.in_out_dim 等
      - 数据 CSV 列：['samples', 'x1', ..., 'xd']，samples 是时间
      - 训练/生成共用模板；生成在 exp.output_dir 下产出 run_*.npz，包含 point、traj、lnw、weight、ts
      - 可通过 --num_runs 控制生成次数
    """
    def __init__(
        self,
        train_script: str,
        generate_script: str,
        template_cfg: str,
        python_exec: str = "python",
        timeout_sec: Optional[int] = None,
        file_layout: Optional[Dict[str, Any]] = None
    ):
        self.train_script = train_script
        self.generate_script = generate_script
        self.template_cfg = template_cfg
        self.python_exec = python_exec
        self.timeout_sec = timeout_sec

        # 文件契约
        self.file_layout = {
            "data_file_name": "emt.csv",       # 写入 data.file_path
            "meta_file_name": "meta.json",     # 可选
            "ckpt_glob": ["*.pt", "*.ckpt", "checkpoint*", "**/*.pt", "**/*.ckpt"],
            "run_npz_glob": "run_*.npz",       # 生成的多个 npz
            "traj_key_in_npz": "traj",         # 默认取 'traj' 作为轨迹
        }
        if file_layout:
            deep_update(self.file_layout, file_layout)

    # ----- 训练 -----
    def fit(self, data: Dict[str, Any], config: Dict[str, Any], otmode: str) -> Dict[str, Any]:
        Z = np.asarray(data["Z"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=float)
        assert Z.shape[0] == t.shape[0], "Z 与 t 行数不一致"

        tpl = load_yaml(self.template_cfg) if self.template_cfg else {}
        base = dict(config)

        # exp.output_dir 确定
        out_dir = (base.get("exp") or {}).get("output_dir") or (tpl.get("exp") or {}).get("output_dir")
        if not out_dir:
            out_dir = os.path.join("deepruot_res", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        ensure_dir(out_dir)

        # 数据写入
        data_dir = data.get("data_dir") or ensure_dir(os.path.join(out_dir, "train_data"))
        data_file = os.path.join(data_dir, self.file_layout["data_file_name"])
        self._write_dataset_csv(Z, t, data_file)

        dim = Z.shape[1]
        overrides = {
            "exp": {"output_dir": out_dir},
            "data": {"file_path": data_file, "dim": int(dim)},
            "model": {"in_out_dim": int(dim)},
        }
        # 可选：透传 hold_out 初值（若模板或 base 未定义）
        if "hold_out" in data:
            overrides = deep_update(overrides, {"data": {"hold_out": to_scalar(data["hold_out"]), "hold_one_out": True}})

        cfg = deep_update(deep_update(dict(tpl), dict(base)), overrides)
        cfg = self.update_otconfig(cfg, otmode)
        cfg_path = os.path.join(out_dir, f"{cfg['exp']['name']}.yaml")
        save_yaml(cfg, cfg_path)

        res_dir = os.path.join(cfg['exp']['output_dir'], cfg['exp']['name'])

        if self._exist_checkpoint(res_dir):
            print('model is exist, skip training')
            return cfg_path

        # training
        train_log = os.path.join(res_dir, "train.log")
        self._run_cli(self.train_script, cfg_path, train_log)

        return cfg_path


    # ----- generate -----
    def generate(
        self,
        config_path: Dict[str, Any],
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        对于每个测试时间点 t：
          - 在 state.config_path 的基础上覆盖 data.hold_out = t
          - 将 exp.output_dir 切到子目录 out_dir/gen_t{t}（避免 run 混淆）
          - 调用 generate.py --config <cfg> --num_runs <num_runs>
          - 收集该目录下所有 run_*.npz 并解析出 point/traj/lnw/weight/ts

        返回:
        {
          "times": [t1, t2, ...],
          "by_time": {
            "t1": {
              "out_dir": ".../gen_t1",
              "config_path": ".../config.gen_t1.yaml",
              "runs": [
                {"idx": 0, "path": ".../run_0.npz", "point": np.ndarray, "traj": np.ndarray, "lnw": np.ndarray, "weight": np.ndarray, "ts": np.ndarray},
                ...
              ],
              "stacked": {
                # 若所有 run 形状一致则提供这些快捷堆叠
                "traj": np.ndarray,   # 形如 (num_runs, ...原始形状...)
                "point": np.ndarray,
                "lnw": np.ndarray,
                "weight": np.ndarray,
                "ts": np.ndarray      # 若各 run ts 完全一致，形如 (T,)；否则不提供
              }
            },
            ...
          },
          "num_runs": int
        }
        """
        cfg = load_yaml(config_path)
        res_dir = os.path.join(cfg['exp']['output_dir'], cfg['exp']['name'])
        if not self._exist_checkpoint(res_dir):
            print('fit model first')
            raise

        # 执行 generate.py，透传 --num_runs
        gen_log = os.path.join(res_dir, 'infer.log')
        self._run_cli(self.generate_script, config_path, gen_log, extra_args=["--num_runs", str(int(num_runs))])

        return res_dir
    

    # ----- methods -----
    def _write_dataset_csv(self, Z: np.ndarray, t: np.ndarray, csv_path: str):
        cols = ["samples"] + [f"x{i+1}" for i in range(Z.shape[1])]
        df = pd.DataFrame(np.hstack([t.reshape(-1, 1), Z]), columns=cols)
        ensure_dir(os.path.dirname(csv_path))
        df.to_csv(csv_path, index=False)

    
    def update_otconfig(self, config, ot_mode):
        if ot_mode == 'ruot':
            config['use_mass'] = True
            config['score_train']['sigma'] = 0.1
        elif ot_mode == 'dot':
            config['use_mass'] = False
            config['score_train']['sigma'] = 0
        elif ot_mode == 'uot':
            config['use_mass'] = True
            config['score_train']['sigma'] = 0
        elif ot_mode == 'sb':
            config['use_mass'] = False
            config['score_train']['sigma'] = 0.1

        return config


    def _run_cli(self, script: str, cfg_path: str, log_path: str, extra_args: Optional[Sequence[str]] = None):
        ensure_dir(os.path.dirname(log_path))
        cmd = [self.python_exec, script, "--config", cfg_path]
        if extra_args:
            cmd.extend(extra_args)
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=self.timeout_sec, check=False)
        if proc.returncode != 0:
            tail = self._tail_file(log_path, 60)
            raise RuntimeError(f"CLI 执行失败（{script}）。查看日志: {log_path}\n--- LOG TAIL ---\n{tail}")
        

    def _tail_file(self, path: str, n: int) -> str:
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return f"<无法读取日志 {path}>"


    def _exist_checkpoint(self, out_dir: str) -> Optional[str]:

        if os.path.exists(out_dir):
            ckpt = 'model_final'
            p = ckpt in os.listdir(out_dir)
            if os.path.exists(p):
                return p

        return False


    def _collect_runs(self, out_dir_t: str) -> List[Dict[str, Any]]:
        npz_paths = sorted(glob.glob(os.path.join(out_dir_t, self.file_layout["run_npz_glob"])))
        if not npz_paths:
            raise FileNotFoundError(f"未找到 run_*.npz 于 {out_dir_t}")
        runs = []
        for p in npz_paths:
            with np.load(p) as data:
                run = {
                    "idx": self._infer_run_idx(p),
                    "path": p,
                    "point": np.array(data["point"]),
                    "traj": np.array(data[self.file_layout["traj_key_in_npz"]]),
                    "lnw": np.array(data["lnw"]) if "lnw" in data else None,
                    "weight": np.array(data["weight"]) if "weight" in data else None,
                    "ts": np.array(data["ts"]) if "ts" in data else None,
                }
                runs.append(run)
        return runs

    def _infer_run_idx(self, path: str) -> int:
        # 从 run_123.npz 提取 123
        base = os.path.basename(path)
        try:
            s = os.path.splitext(base)[0]  # run_123
            return int(s.split("_")[-1])
        except Exception:
            return -1

    def _stack_if_possible(self, runs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        stacked: Dict[str, np.ndarray] = {}
        if not runs:
            return stacked
        # 尝试堆叠 traj/point/lnw/weight；要求形状一致
        for key in ["traj", "point", "lnw", "weight"]:
            arrs = [r[key] for r in runs if r.get(key) is not None]
            if len(arrs) == len(runs):
                shapes = {tuple(a.shape) for a in arrs}
                if len(shapes) == 1:
                    stacked[key] = np.stack(arrs, axis=0)  # (num_runs, ...)
        # ts：必须完全一致才堆叠（否则返回空）
        ts_arrs = [r["ts"] for r in runs if r.get("ts") is not None]
        if len(ts_arrs) == len(runs):
            # 全部等长且逐元素相等
            same = all((a.shape == ts_arrs[0].shape and np.allclose(a, ts_arrs[0])) for a in ts_arrs)
            if same:
                stacked["ts"] = ts_arrs[0].copy()
        return stacked


# sf2m adptor 
class sf2mEngine(BaseRunner):
    """
    适配你的 config 架构：
      - 模板键：exp.output_dir, data.file_path, data.dim, data.hold_one_out, data.hold_out, model.in_out_dim 等
      - 数据 CSV 列：['samples', 'x1', ..., 'xd']，samples 是时间
      - 训练/生成共用模板；生成在 exp.output_dir 下产出 run_*.npz，包含 point、traj、lnw、weight、ts
      - 可通过 --num_runs 控制生成次数
    """
    def __init__(
        self,
        train_script: str,
        generate_script: str,
        template_cfg: str,
        python_exec: str = "python",
        timeout_sec: Optional[int] = None,
        file_layout: Optional[Dict[str, Any]] = None
    ):
        self.train_script = train_script
        self.generate_script = generate_script
        self.template_cfg = template_cfg
        self.python_exec = python_exec
        self.timeout_sec = timeout_sec

        # 文件契约
        self.file_layout = {
            "data_file_name": "emt.csv",       # 写入 data.file_path
            "ckpt_glob": ["sf2m_model.pt", "*.ckpt", "checkpoint*", "**/*.pt", "**/*.ckpt"],
            "run_npz_glob": "sf2m_run_*.npz",       # 生成的多个 npz
            "traj_key_in_npz": "traj",         # 默认取 'traj' 作为轨迹
        }
        if file_layout:
            deep_update(self.file_layout, file_layout)

    # ----- 训练 -----
    def fit(self, data: Dict[str, Any], config: Dict[str, Any], otmode: str) -> Dict[str, Any]:
        Z = np.asarray(data["Z"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=float)
        assert Z.shape[0] == t.shape[0], "Z 与 t 行数不一致"

        tpl = load_yaml(self.template_cfg) if self.template_cfg else {}
        base = dict(config)

        # exp.output_dir 确定
        out_dir = (base.get("exp") or {}).get("output_dir") or (tpl.get("exp") or {}).get("output_dir")
        if not out_dir:
            out_dir = os.path.join("deepruot_res", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        ensure_dir(out_dir)

        # 数据写入
        data_dir = data.get("data_dir") or ensure_dir(os.path.join(out_dir, "train_data"))
        data_file = os.path.join(data_dir, self.file_layout["data_file_name"])
        self._write_dataset_csv(Z, t, data_file)

        dim = Z.shape[1]
        overrides = {
            "data": {"exp_dir": out_dir},
            "data": {"file_path": data_file, "dim": int(dim)},
        }
        # 可选：透传 hold_out 初值（若模板或 base 未定义）
        if "hold_out" in data:
            overrides = deep_update(overrides, {"train": {"hold_out": to_scalar(data["hold_out"])}})

        cfg = deep_update(deep_update(dict(tpl), dict(base)), overrides)
        cfg = self.update_otconfig(cfg, otmode)
        cfg_path = os.path.join(out_dir, f"{cfg['exp']['name']}.yaml")
        save_yaml(cfg, cfg_path)

        res_dir = os.path.join(cfg['exp']['output_dir'], cfg['exp']['name'])

        if self._exist_checkpoint(res_dir):
            print('model is exist, skip training')
            return cfg_path

        # training
        train_log = os.path.join(res_dir, "train.log")
        self._run_cli(self.train_script, cfg_path, train_log)

        return cfg_path


    # ----- generate -----
    def generate(
        self,
        config_path: Dict[str, Any],
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        cfg = load_yaml(config_path)
        res_dir = os.path.join(cfg['exp']['output_dir'], cfg['exp']['name'])
        if not self._exist_checkpoint(res_dir):
            print('fit model first')
            raise

        gen_log = os.path.join(res_dir, 'infer.log')
        self._run_cli(self.generate_script, config_path, gen_log, extra_args=["--num_runs", str(int(num_runs))])

        return res_dir
    

    # ----- methods -----
    def _write_dataset_csv(self, Z: np.ndarray, t: np.ndarray, csv_path: str):
        cols = ["samples"] + [f"x{i+1}" for i in range(Z.shape[1])]
        df = pd.DataFrame(np.hstack([t.reshape(-1, 1), Z]), columns=cols)
        ensure_dir(os.path.dirname(csv_path))
        df.to_csv(csv_path, index=False)

    
    def update_otconfig(self, config, ot_mode):
        if ot_mode == 'ruot':
            config['use_mass'] = True
            config['score_train']['sigma'] = 0.1
        elif ot_mode == 'dot':
            config['use_mass'] = False
            config['score_train']['sigma'] = 0
        elif ot_mode == 'uot':
            config['use_mass'] = True
            config['score_train']['sigma'] = 0
        elif ot_mode == 'sb':
            config['use_mass'] = False
            config['score_train']['sigma'] = 0.1

        return config


    def _run_cli(self, script: str, cfg_path: str, log_path: str, extra_args: Optional[Sequence[str]] = None):
        ensure_dir(os.path.dirname(log_path))
        cmd = [self.python_exec, script, "--config", cfg_path]
        if extra_args:
            cmd.extend(extra_args)
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=self.timeout_sec, check=False)
        if proc.returncode != 0:
            tail = self._tail_file(log_path, 60)
            raise RuntimeError(f"CLI 执行失败（{script}）。查看日志: {log_path}\n--- LOG TAIL ---\n{tail}")
        

    def _tail_file(self, path: str, n: int) -> str:
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return f"<无法读取日志 {path}>"


    def _exist_checkpoint(self, out_dir: str) -> Optional[str]:

        if os.path.exists(out_dir):
            ckpt = 'model_final'
            p = ckpt in os.listdir(out_dir)
            if os.path.exists(p):
                return p

        return False


    def _collect_runs(self, out_dir_t: str) -> List[Dict[str, Any]]:
        npz_paths = sorted(glob.glob(os.path.join(out_dir_t, self.file_layout["run_npz_glob"])))
        if not npz_paths:
            raise FileNotFoundError(f"未找到 run_*.npz 于 {out_dir_t}")
        runs = []
        for p in npz_paths:
            with np.load(p) as data:
                run = {
                    "idx": self._infer_run_idx(p),
                    "path": p,
                    "point": np.array(data["point"]),
                    "traj": np.array(data[self.file_layout["traj_key_in_npz"]]),
                    "lnw": np.array(data["lnw"]) if "lnw" in data else None,
                    "weight": np.array(data["weight"]) if "weight" in data else None,
                    "ts": np.array(data["ts"]) if "ts" in data else None,
                }
                runs.append(run)
        return runs

    def _infer_run_idx(self, path: str) -> int:
        # 从 run_123.npz 提取 123
        base = os.path.basename(path)
        try:
            s = os.path.splitext(base)[0]  # run_123
            return int(s.split("_")[-1])
        except Exception:
            return -1

    def _stack_if_possible(self, runs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        stacked: Dict[str, np.ndarray] = {}
        if not runs:
            return stacked
        # 尝试堆叠 traj/point/lnw/weight；要求形状一致
        for key in ["traj", "point", "lnw", "weight"]:
            arrs = [r[key] for r in runs if r.get(key) is not None]
            if len(arrs) == len(runs):
                shapes = {tuple(a.shape) for a in arrs}
                if len(shapes) == 1:
                    stacked[key] = np.stack(arrs, axis=0)  # (num_runs, ...)
        # ts：必须完全一致才堆叠（否则返回空）
        ts_arrs = [r["ts"] for r in runs if r.get("ts") is not None]
        if len(ts_arrs) == len(runs):
            # 全部等长且逐元素相等
            same = all((a.shape == ts_arrs[0].shape and np.allclose(a, ts_arrs[0])) for a in ts_arrs)
            if same:
                stacked["ts"] = ts_arrs[0].copy()
        return stacked
