import os, glob, json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from abc import ABC, abstractmethod
import torch
import ot
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, kendalltau
import joblib

import warnings
warnings.filterwarnings("ignore")


def normalize_data(adata, dimreducted_key = 'X_PCA'):

    original_data = adata.layers['Ms']
    
    M = np.max(original_data, axis=0)  
    m = np.min(original_data, axis=0)  
    
    constant_cols = (M == m)
    M[constant_cols] += 1  
    m[constant_cols] -= 1  
    
    normalized_data = 0.05 + 0.9 * (original_data - m) / (M - m)
    
    adata.layers['Ms_nor'] = normalized_data

    pca_data = adata.obsm[dimreducted_key]
    
    M = np.max(pca_data, axis=0)  
    m = np.min(pca_data, axis=0)  
    
    constant_cols = (M == m)
    M[constant_cols] += 1  
    m[constant_cols] -= 1  
    
    normalized_data = 0.05 + 0.9 * (pca_data - m) / (M - m)
    adata.obsm['X_nor'] = normalized_data

    
    return adata  


def load_runs_npz(out_dir: str, pattern: str = "run_*.npz") -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(out_dir, pattern)))
    runs = []
    for p in paths:
        with np.load(p, allow_pickle= True) as data:
            runs.append({
                "path": p,
                "point": np.array(data["point"]),
                "traj": np.array(data["traj"]),
                "lnw": np.array(data["lnw"]) if "lnw" in data else None,
                "weight": np.array(data["weight"]) if "weight" in data else None,
                "ts": np.array(data["ts"]) if "ts" in data else None,
            })
    return runs


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    @abstractmethod
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate metrics and return results as DataFrame."""
        pass


class W1TMVCalculator(MetricCalculator):
    """Calculator for W1 distance and TMV metrics."""

    @staticmethod
    def _calW1(gt_data: torch.Tensor, model_data: torch.Tensor, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Wasserstein-1 distance."""
        M = torch.cdist(gt_data, model_data, p=2).cpu().numpy()
        if np.isnan(M).any() or np.isinf(M).any():
            return np.nan
        w1 = ot.emd2(a, b, M, numItermax=1e7)

        if isinstance(w1, list):
            w1 = w1[0]
        return w1

    def calculate(
        self,
        df: pd.DataFrame, # 真实数据，cols: ['samples', 'x1', 'x2' ... 'xn']
        all_times: List[Any], # 从0出发需要计算的时间点
        sde_point_array: List[np.ndarray], # 列表，形状为（T, N, G）
        weight: List[np.ndarray], # 列表同上
    ) -> pd.DataFrame:
        """Calculate W1 and TMV metrics."""
        base_time = 0
        gt_base_n = int((df['samples'] == base_time).sum())

        results = []
        for time_point in all_times:
            gt_block = df[df['samples'] == time_point]
            gt_data = gt_block.iloc[:, 1:].values.astype(np.float32)
            gt_n = gt_data.shape[0]

            if gt_n == 0:
                results.append({
                    'Time Point': time_point,
                    'W1 Distance': np.nan,
                    'TMV': np.nan
                })
                continue

            a = np.ones((gt_n,), dtype=np.float64) / gt_n
            gt_mass = gt_n / max(gt_base_n, 1)

            pred_data = np.asarray(sde_point_array[time_point], dtype=np.float32)
            w = np.asarray(weight[time_point], dtype=np.float64).reshape(-1)

            pred_mass = w.sum() / max(np.asarray(weight[0]).sum(), 1e-12)
            b = (w / w.sum())

            gt_data_tensor = torch.Tensor(gt_data)
            pred_data_tensor = torch.Tensor(pred_data)
            # import ipdb; ipdb.set_trace()
            w1 = self._calW1(gt_data_tensor, pred_data_tensor, a=a, b=b)
            tmv = float(np.abs(pred_mass - gt_mass))

            results.append({
                'Time Point': time_point,
                'W1 Distance': w1,
                'TMV': tmv
            })

        return pd.DataFrame(results)


class TCVCCalculator(MetricCalculator):
    def evaluate_adata_metrics_sampled(
        self,
        adata,
        use_key: str = 'X_nor',          # 特征矩阵所在 obsm 键（如 'X_nor' 或 'X_pca'）
        label_key: str | None = None,    # 标签列名；None 时优先 time_categorical，否则 time
        velocity_layer: str = 'velocity',# 速度所在 layer
        k: int = 30,                     # 每个查询点的近邻数
        q: int = 5000,                   # 查询点采样数量
        seed: int = 0,                   # 随机种子
        n_jobs: int = -1,                # sklearn 并行
        eps: float = 1e-8,               # 防零范数
        test_idx: np.ndarray = None,
    ):
        # 取特征
        points = np.asarray(adata.obsm[use_key], dtype=np.float32)
        N = np.arange(0, points.shape[0])
        if test_idx is not None:
            # 只算 test time 的 corr
            test_idx = test_idx.reshape(-1)
            N = N[test_idx]
        
        if len(N) == 0:
            raise ValueError("No cells in points.")
        q = int(min(q, len(N)))

        # 取标签
        if label_key is None:
            if 'time_categorical' in adata.obs.columns:
                label_key = 'time_categorical'
            elif 'time' in adata.obs.columns:
                label_key = 'time'
            else:
                raise KeyError("Cannot find label column. Please provide label_key.")
        labels_series = adata.obs[label_key]
        if hasattr(labels_series, 'cat'):
            labels = labels_series.cat.codes.to_numpy()
        else:
            labels = pd.Categorical(labels_series).codes
        labels = labels.astype(np.int64)

        # 取速度
        if velocity_layer not in adata.layers:
            raise KeyError(f"Layer '{velocity_layer}' not found in adata.layers")
        velocity = np.asarray(adata.layers[velocity_layer], dtype=np.float32)
        if velocity.shape[1] != points.shape[1]:
            raise ValueError(f"velocity dim {velocity.shape[1]} != points dim {points.shape[1]}")

        # 采样查询点
        rng = np.random.default_rng(seed)
        q_idx = rng.choice(N, size=q, replace=False)

        # kNN（在全量 points 上建索引）
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=n_jobs)
        nn.fit(points)
        _, inds = nn.kneighbors(points[q_idx], n_neighbors=k+1, return_distance=True)
        inds = inds[:, 1:]  # 去掉自身

        # 标签一致性
        same = (labels[inds] == labels[q_idx, None]).sum()
        total = inds.size
        label_consistency = float(same / total) if total > 0 else float('nan')

        # 速度一致性（余弦）
        norms = np.linalg.norm(velocity, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        U = velocity / norms
        Ui = U[q_idx][:, None, :]        # (q,1,D)
        Uj = U[inds]                     # (q,k,D)
        cos = (Ui * Uj).sum(axis=2)      # (q,k)
        velocity_consistency = float(np.nanmean(cos))

        return {
            'label_consistency': label_consistency,
            'velocity_consistency': velocity_consistency,
            'k': int(k),
            'q': int(q),
            'n_cells': len(N),
            'use_key': use_key,
            'label_key': label_key,
            'velocity_layer': velocity_layer,
        }


    """Calculator for TC/VC metrics."""
    def calculate(
        self,
        velocity_adata: ad.AnnData,
        use_key : str,
        velocity_layer: str, 
        k: int = 50,
        q: int = 5000,
        test_idx: np.ndarray = None
    ) -> pd.DataFrame:
        """Calculate TC/VC metrics."""

        results = self.evaluate_adata_metrics_sampled(
            velocity_adata, k=k, q=q, use_key=use_key, velocity_layer=velocity_layer, test_idx=test_idx
        )
        # 只保留一些key的结果
        res = {
            'TC': results['label_consistency'], 
            'VC': results['velocity_consistency'],
        }

        df_results = pd.DataFrame([res])

        return df_results


class PseudotimeCalculator(MetricCalculator):
    """Calculator for pseudotime correlation metrics."""

    @staticmethod
    def pseudotime_DPT(df: pd.DataFrame, cell_type: np.ndarray, root_cell_type: str) -> ad.AnnData:
        """Calculate pseudotime using DPT method."""
        adata_real = ad.AnnData(df.iloc[:, 1:])
        adata_real.obs['time'] = df.iloc[:, 0].values
        adata_real.obs['cell_type'] = cell_type
        adata_real.obsm['X_pca'] = adata_real.X

        sc.pp.neighbors(adata_real)
        sc.tl.diffmap(adata_real)
        adata_real.uns["iroot"] = np.flatnonzero(adata_real.obs["cell_type"] == root_cell_type)[0]
        sc.tl.dpt(adata_real)

        adata_real.obs['pseudotime_real'] = adata_real.obs['dpt_pseudotime']
        return adata_real

    def calculate(
        self,
        adata_real: ad.AnnData,
        trajectories: List[Dict[str, np.ndarray]],
        embedding_key: str = "X_pca",
        pseudotime_key: str = "pseudotime_real",
        knn_k: int = 5,
        n_times: Optional[int] = None,
        n_steps: int = 100,
    ) -> pd.DataFrame:
        """Calculate pseudotime correlation metrics."""
        X_real = adata_real.obsm[embedding_key]
        pst_real = adata_real.obs[pseudotime_key].to_numpy()

        traj = trajectories["traj"]
        ts = trajectories.get("ts", None)

        T, N, G = traj.shape

        if ts is None:
            assert n_times is not None, "ts missing; provide n_times to rebuild a linear grid"
            t_grid = np.linspace(0, n_times - 1, n_steps)
        else:
            t_grid = ts

        # t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min())
        # t_flat = np.repeat(t_norm, N)
        t_flat = np.repeat(t_grid, N)
        
        X_sim = traj.reshape(T * N, G)
   
        knn = NearestNeighbors(n_neighbors=knn_k, algorithm="auto")
        knn.fit(X_sim) # 对生成的数据 fit KNN

        # dists, idxs = knn.kneighbors(X_sim, n_neighbors=knn_k)
        dists, idxs = knn.kneighbors(X_real, n_neighbors=knn_k)
        # est_pst = pst_real[idxs].mean(axis=1)
        est_pst = t_flat[idxs].mean(axis=1) # 在生成数据里找真实数据的五个最近邻的 ts 均值作为预测的伪时间 label
        est_pst = (est_pst - est_pst.min()) / (est_pst.max() - est_pst.min())

        rho, _ = spearmanr(pst_real, est_pst)
        tau, _ = kendalltau(pst_real, est_pst)
        results = {
            "Spearman": float(rho),
            "Kendall": float(tau),
            "mean_dist": float(dists.mean()),
            "std_dist": float(dists.std()),
        }

        return pd.DataFrame([results])


class AlignmnetEvaluator:
    """Main Alignmnet evaluator class."""
    def __init__(self, 
                 results,
                 alldata, 
                 alltimes,
                 artifacts_path = './artifacts/EMT/PC10', 
                 test_times = [3], ):

        self.calculators = {
            'w1tmv': W1TMVCalculator(),
            'tcvc': TCVCCalculator(),
            'pseudotime': PseudotimeCalculator()
        }
        self.results = results
        self.artifacts_path = artifacts_path
        self.test_times = test_times
        self.df = self._get_df(alldata, alltimes)
        self.test_idx = np.array([alltimes == test_times[0]]).astype(bool)
    
    def _get_df(self, data, times):
        cols = ["samples"] + [f"x{i+1}" for i in range(data.shape[1])]
        df = pd.DataFrame(np.hstack([times.reshape(-1, 1), data]), columns=cols)
        return df


    def validate_config(self):
        pass

    
    def load_artifacts(self, artifacts_path: str):
        """Load alinger."""
        pass
    

    def evaluate_w1tmv_metrics(
        self,
        aligner,
    ) -> pd.DataFrame:
        """Evaluate W1 and TMV metrics for a model."""
        # Use passed parameters or defaults
        test_times = self.test_times
        df = self.df

        if df.columns[0] != 'samples':
            ValueError 
            # df = df.rename(columns={df.columns[0]: 'samples'})

        all_results = []
        for run_idx in range(len(self.results)):
            res = self.results[run_idx]
            sde_point_array = res['point']
            weight = res['weight']

            align_model = aligner
            sde_point_array_aligned = []

            for sde_points in sde_point_array:
                aligned_points = align_model.transform(sde_points.astype(float))
                sde_point_array_aligned.append(aligned_points)
            sde_point_array = sde_point_array_aligned

            results = self.calculators['w1tmv'].calculate(
                df=df,
                all_times=test_times,
                sde_point_array=sde_point_array,
                weight=weight,
            )
            results['Run'] = run_idx
            all_results.append(results)

        df_results = pd.concat(all_results, ignore_index=True)
        return df_results

    
    def evaluate_tcvc_metrics(
        self,
        aligner, 
        velocity_adata,
        k: int = 50,
        q: int = 5000,
    ) -> pd.DataFrame:
        """Evaluate TC/VC metrics for a model."""
        # if len(self.test_times) != 0:
        #     # 只算test time的corr
        #     velocity_adata = velocity_adata[self.test_idx,:]

        velocity_adata = normalize_data(velocity_adata, dimreducted_key='X_PCA')
        align_model = aligner

        Z = velocity_adata.layers['Ms'] # 特征矩阵
        Z_to_ref = align_model.transform(Z)
        velocity_adata.obsm['Ms_aligned'] = Z_to_ref

        velocity = velocity_adata.layers['velocity']
        velocity_aligned = align_model.transform(velocity)
        velocity_adata.layers['velocity_aligned'] = velocity_aligned

        results = self.calculators['tcvc'].calculate(
            velocity_adata=velocity_adata,
            use_key = 'Ms_aligned', 
            velocity_layer = 'velocity_aligned',
            k=k,
            q=q,
            test_idx = self.test_idx
        )
        return results


    def evaluate_pseudotime_metrics(
        self,
        aligner,
        cell_types = None,
        pseudotime = None,
        root_cell_type: str = 'Epithelial',
    ) -> pd.DataFrame:
        """Evaluate pseudotime correlation metrics for a model."""
        # Use passed parameters or defaults
        df = self.df
        adata_real = ad.AnnData(df.iloc[:, 1:])

        assert (cell_types is None) | (pseudotime is None)

        if pseudotime is not None:
            adata_real.obs['pseudotime_real'] = pseudotime
        else:
            adata_real = self.calculators['pseudotime'].pseudotime_DPT(
                df, cell_type=cell_types, root_cell_type=root_cell_type
            )
        align_model = aligner
        adata_real.obsm['aligned'] = align_model.transform(adata_real.X.astype(float))

        all_results = []
        for run_idx in range(len(self.results)):
            res = self.results[run_idx]
            traj = res['traj']

            T, N, G = traj.shape
            traj_reshaped = traj.reshape(T * N, G)
            traj_aligned = align_model.transform(traj_reshaped.astype(float))
            traj = traj_aligned.reshape(T, N, -1)

            res = {
                "traj": traj,
                "lnw": res.get('lnw', None),
                "ts": res.get('ts', None),
            }
            
            df_corr = self.calculators['pseudotime'].calculate(
                adata_real=adata_real,
                trajectories=res,
                embedding_key='aligned',
                pseudotime_key="pseudotime_real",
                knn_k=5
            )
            df_corr['Run'] = run_idx
            all_results.append(df_corr)

        df_results = pd.concat(all_results)
        return df_results


    def evaluate_all_metrics(self, aligner, metrics, velocity_adata = None, 
                             pseudotime = None, cell_types = None, root_cell_type: str = 'Epithelial'):
        results = {}
        for metric in metrics:
            metric_results = []
            if metric == 'w1tmv':
                result = self.evaluate_w1tmv_metrics(aligner)
            elif (metric == 'tcvc') & (velocity_adata is not None):
                result = self.evaluate_tcvc_metrics(aligner = aligner, velocity_adata = velocity_adata)
            elif metric == 'pseudotime':
                result = self.evaluate_pseudotime_metrics(aligner = aligner, pseudotime = pseudotime, cell_types=cell_types, root_cell_type = root_cell_type)
            metric_results.append(result)
            results[metric] = pd.concat(metric_results, ignore_index=True)
        return results
    
