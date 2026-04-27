from __future__ import annotations
from typing import List, Optional, Dict, Any, Sequence
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scanpy as sc
import os


class EmbeddingPreprocessor:
    """
    1) Optionally fit a reference PCA in expression space (adata.X or a given layer) using
       training time points only;
    2) For each candidate embedding (adata.obsm['X_{model}']), fit standardization
       (+ optional) + PCA(k) on training points only to get Z;
    3) Write Z back to adata.obsm['Z_{model}'].
    """

    def __init__(
        self,
        adata: AnnData,
        time_key: str,
        train_times: Sequence[Any],
        test_times: Optional[Sequence[Any]] = None,
        ref_key: Optional[str] = None,
        random_state: int = 0,
        dtype: str = "float32"
    ):
        if time_key not in adata.obs.columns:
            raise KeyError(f"time_key='{time_key}' is not in adata.obs.")
        self.adata = adata
        self.time_key = time_key
        self.train_times = set(train_times)
        self.test_times = set(test_times) if test_times is not None else None
        self.X_obsm = ref_key
        self.random_state = random_state
        self.dtype = np.float32 if dtype == "float32" else np.float64

        tvals = self.adata.obs[self.time_key].values
        self.train_mask = np.isin(tvals, list(self.train_times))

        if self.test_times is not None:
            self.test_mask = np.isin(tvals, list(self.test_times))
            if np.any(self.train_mask & self.test_mask):
                raise ValueError("train_times and test_times must not overlap.")
        else:
            self.test_mask = ~self.train_mask


        # Reference space (optional)
        self.ref_scaler: Optional[StandardScaler] = None
        self.ref_pca: Optional[PCA] = None
        self.ref_pca_dim: Optional[int] = None

        # Per-model transformers and metadata
        self.models_: Dict[str, Dict[str, Any]] = {}

    def _to_dense_np(self, M) -> np.ndarray:
        if hasattr(M, "A"):
            M = M.A
        elif hasattr(M, "toarray"):
            M = M.toarray()
        elif isinstance(M, pd.DataFrame):
            M = M.values
        return np.asarray(M, dtype=self.dtype)


    def _get_expr_matrix(self) -> np.ndarray:
        if self.X_obsm is not None:
            if self.X_obsm not in self.adata.obsm:
                raise ValueError(f"X_obsm='{self.X_obsm}' is not in adata.obsm.")
            X = self.adata.obsm[self.X_obsm]
        else:
            X = self.adata.X
        return self._to_dense_np(X)


    def fit_ref(self, k: Optional[int] = None, scale: bool = True, whiten: bool = False,
                svd_solver: str = "auto", store_key: Optional[str] = None) -> np.ndarray:
        """
        In expression space, fit scaler + PCA(k) on training cells only; write Z_ref to obsm[store_key].
        """
        X_all = self._get_expr_matrix()
        n_feat = X_all.shape[1]
        n_train = int(self.train_mask.sum())
        k_eff = min(k if k is not None else n_feat, n_feat, n_train)

        scaler = StandardScaler(with_mean=True, with_std=True) if scale else None
        X_scaled = X_all
        if scaler is not None:
            scaler.fit(X_all[self.train_mask])
            X_scaled = scaler.transform(X_all)

        pca = PCA(n_components=k_eff, svd_solver=svd_solver, whiten=whiten, random_state=self.random_state)
        pca.fit(X_scaled[self.train_mask])
        Z_all = pca.transform(X_scaled).astype(self.dtype, copy=False)

        self.ref_scaler = scaler
        self.ref_pca = pca
        self.ref_pca_dim = k_eff
        if store_key is not None:
            self.adata.obsm[store_key] = Z_all
        return Z_all


    def fit_embedding(
        self,
        model_key: str,
        k: int = 50,
        scale: bool = True,
        whiten: bool = False,
        svd_solver: str = "auto"
    ) -> np.ndarray:
        """
        Fit standardization + PCA(k) on training cells only; store Z in obsm['Z_{model_key}'].
        """
        obsm_key = f"X_{model_key}"
        if obsm_key not in self.adata.obsm:
            raise ValueError(f"{obsm_key} is not in adata.obsm.")
        R_all = self._to_dense_np(self.adata.obsm[obsm_key])

        n_feat = R_all.shape[1]
        n_train = int(self.train_mask.sum())
        k_eff = int(min(k, n_feat, n_train))

        scaler = StandardScaler(with_mean=True, with_std=True) if scale else None
        R_scaled = R_all
        if scaler is not None:
            scaler.fit(R_all[self.train_mask])
            R_scaled = scaler.transform(R_all)

        pca = PCA(n_components=k_eff, svd_solver=svd_solver, whiten=whiten, random_state=self.random_state)
        pca.fit(R_scaled[self.train_mask])
        Z_all = pca.transform(R_scaled).astype(self.dtype, copy=False)

        self.adata.obsm[f"Z_{model_key}"] = Z_all
        self.models_[model_key] = {
            "k_req": int(k),
            "k_eff": k_eff,
            "n_feat": int(n_feat),
            "n_train": n_train,
            "scaler": scaler,
            "pca": pca,
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        }
        return Z_all


    def transform_embedding(self, model_key: str, R: np.ndarray) -> np.ndarray:
        """Map from raw embedding space R to Z using the fitted scaler+pca."""
        if model_key not in self.models_:
            raise ValueError(
                f"Model {model_key} is not fitted; call fit_embedding('{model_key}') first."
            )
        m = self.models_[model_key]
        X = np.asarray(R, dtype=self.dtype)
        if m["scaler"] is not None:
            X = m["scaler"].transform(X)
        Z = m["pca"].transform(X)
        return Z.astype(self.dtype, copy=False)


    def inverse_transform_embedding(self, model_key: str, Z: np.ndarray) -> np.ndarray:
        """Map from Z back to the raw embedding space R."""
        if model_key not in self.models_:
            raise ValueError(
                f"Model {model_key} is not fitted; call fit_embedding('{model_key}') first."
            )
        m = self.models_[model_key]
        Z = np.asarray(Z, dtype=self.dtype)
        R_scaled = m["pca"].inverse_transform(Z)
        if m["scaler"] is not None:
            R = m["scaler"].inverse_transform(R_scaled)
        else:
            R = R_scaled
        return R.astype(self.dtype, copy=False)


    def get_Z(self, model_key: str, split: Optional[str] = None) -> np.ndarray:
        key = f"Z_{model_key}"
        if key not in self.adata.obsm:
            raise ValueError(
                f"{key} is missing; call fit_embedding('{model_key}') first."
            )
        Z = self.adata.obsm[key]
        if isinstance(Z, pd.DataFrame):
            Z = Z.values
        Z = np.asarray(Z)
        if split is None:
            return Z
        if split == "train":
            return Z[self.train_mask]
        if split == "test":
            return Z[self.test_mask]
        raise ValueError("split must be None, 'train', or 'test'.")
    

    def get_R(self, model_key: str, split: Optional[str] = None) -> np.ndarray:
        """Debug helper: return raw embedding R (X_{model_key})."""
        key = f"X_{model_key}"
        if key not in self.adata.obsm:
            raise ValueError(f"{key} is missing.")
        R = self._to_dense_np(self.adata.obsm[key])
        if split is None:
            return R
        if split == "train":
            return R[self.train_mask]
        if split == "test":
            return R[self.test_mask]
        raise ValueError("split must be None, 'train', or 'test'.")
    

    def info(self) -> Dict[str, Any]:
        out = {
            "time_key": self.time_key,
            "train_times": sorted(list(self.train_times)),
            "test_times": sorted(list(self.test_times)) if self.test_times is not None else None,
            "models": {}
        }
        for k, v in self.models_.items():
            out["models"][k] = {
                "k_req": v.get("k_req"),
                "k_eff": v.get("k_eff"),
                "n_feat": v.get("n_feat"),
                "n_train": v.get("n_train"),
                "explained_variance_ratio": v.get("explained_variance_ratio"),
            }
        if self.ref_pca is not None:
            out["ref"] = {
                "k_eff": self.ref_pca.n_components_,
                "explained_variance_ratio": self.ref_pca.explained_variance_ratio_.astype(float).tolist(),
            }
        return out
