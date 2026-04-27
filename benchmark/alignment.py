import numpy as np
from scipy.linalg import orthogonal_procrustes
from typing import List, Optional, Dict, Any, Tuple
import joblib
from pathlib import Path

class BaseAligner:
    name = "base"
    def fit(self, Z_src_train: np.ndarray, Z_ref_train: np.ndarray, config: dict | None = None):
        raise NotImplementedError

    def transform(self, Z_src: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name}


class IdentityAligner(BaseAligner):
    # No transform; metrics are computed in each model's own space
    name = "identity"
    def fit(self, Z_src_train, Z_ref_train, config=None):
        return self
    def transform(self, Z_src):
        return np.asarray(Z_src)


class ProcrustesAligner(BaseAligner):

    name = "procrustes"
    
    def __init__(self):
        self.R = None
        self.s = None
        self.mu_s = None
        self.mu_r = None

    def fit(self, Z_src_train, Z_ref_train, config=None):
        Zs = np.asarray(Z_src_train, dtype=np.float64)
        Zr = np.asarray(Z_ref_train, dtype=np.float64)
        assert Zs.shape == Zr.shape, "source and reference must be paired (same shape)"

        mu_s = Zs.mean(axis=0, keepdims=True)
        mu_r = Zr.mean(axis=0, keepdims=True)
        Xs = Zs - mu_s
        Xr = Zr - mu_r

        R, _ = orthogonal_procrustes(Xs, Xr)
        self.R = R
        self.mu_s = mu_s
        self.mu_r = mu_r

        return self
    
    def transform(self, Z_src):
        assert self.R is not None, "aligner is not fitted"
        Zs = np.asarray(Z_src, dtype=np.float64)
        return  (Zs - self.mu_s) @ self.R + self.mu_r
    
    def info(self) -> Dict[str, Any]:
        detR = None if self.R is None else float(np.linalg.det(self.R))
        return {"name": self.name, "detR": detR}



class RidgeAligner(BaseAligner):
    name = "ridge"

    def __init__(self, lam: float = 1e-2, fit_intercept: bool = True, standardize: bool = False):
        """
        lam: L2 penalty (lambda >= 0). Larger values are more conservative (more stable, possibly more bias).
        fit_intercept: whether to fit an intercept (centering before fit, add back after).
        standardize: whether to column-standardize source features (Xs only) for numerical stability.
        """
        self.lam = float(lam)
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)

        self.W = None         # linear map (d_src, d_tgt)
        self.mu_s = None      # source mean (1, d)
        self.mu_r = None      # reference mean (1, d)
        self.std_s = None     # source std (1, d), only if standardize=True

    def fit(self, Z_src_train, Z_ref_train, config=None):
        """
        Z_src_train: (n, d)
        Z_ref_train: (n, d), row-aligned with source
        config: optional dict overriding lam / fit_intercept / standardize
        """
        if config is not None:
            if "lam" in config: self.lam = float(config["lam"])
            if "fit_intercept" in config: self.fit_intercept = bool(config["fit_intercept"])
            if "standardize" in config: self.standardize = bool(config["standardize"])

        Zs = np.asarray(Z_src_train, dtype=np.float64)
        Zr = np.asarray(Z_ref_train, dtype=np.float64)
        assert Zs.shape == Zr.shape, "source and reference must be paired (same shape)"
        n, d = Zs.shape

        # Means (for intercept)
        if self.fit_intercept:
            self.mu_s = Zs.mean(axis=0, keepdims=True)
            self.mu_r = Zr.mean(axis=0, keepdims=True)
        else:
            self.mu_s = np.zeros((1, d), dtype=np.float64)
            self.mu_r = np.zeros((1, d), dtype=np.float64)

        Xs = Zs - self.mu_s if self.fit_intercept else Zs
        Xr = Zr - self.mu_r if self.fit_intercept else Zr

        # Optional: column-standardize source only
        if self.standardize:
            std = Xs.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0  # avoid division by zero
            self.std_s = std
            Xs = Xs / std
        else:
            self.std_s = None

        # Multi-output ridge closed form: W = (Xs^T Xs + lambda I)^(-1) Xs^T Xr
        I = np.eye(d, dtype=np.float64)
        A = Xs.T @ Xs + self.lam * I
        B = Xs.T @ Xr
        # Stable linear solve instead of explicit inverse
        self.W = np.linalg.solve(A, B)

        return self

    def transform(self, Z_src):
        """
        Map arbitrary source-domain points to the reference domain.
        """
        assert self.W is not None, "aligner is not fitted"
        Zs = np.asarray(Z_src, dtype=np.float64)

        Xs = Zs - self.mu_s if self.fit_intercept else Zs
        if self.standardize and self.std_s is not None:
            Xs = Xs / self.std_s

        Yr = Xs @ self.W
        if self.fit_intercept:
            Yr = Yr + self.mu_r
        return Yr

    def info(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "lambda": float(self.lam),
            "fit_intercept": bool(self.fit_intercept),
            "standardize": bool(self.standardize),
        }
        if self.W is not None:
            try:
                out["condW"] = float(np.linalg.cond(self.W))
            except Exception:
                pass
        return out


def _center(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu


def _fro_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, ord="fro"))


def _orthogonal_procrustes_svd(X: np.ndarray,
                               Y: np.ndarray,
                               allow_scale: bool = True,
                               allow_reflection: bool = True) -> Tuple[np.ndarray, float]:
    """
    Similarity Procrustes on centered X, Y: return optimal rotation R and isotropic scale s (if allowed).
    Minimizes || s X R - Y ||_F.
    """
    # SVD of X^T Y
    U, S, VT = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = VT.T @ U.T  # R = V U^T (VT is V^T)
    # Reflection control: ensure det(R) >= 0 when reflections disallowed
    if not allow_reflection:
        if np.linalg.det(R) < 0:
            # Flip last column to remove reflection
            VT[-1, :] *= -1.0
            R = VT.T @ U.T

    if allow_scale:
        s = float(S.sum() / (_fro_norm(X) ** 2 + 1e-12))
    else:
        s = 1.0
    return R, s


class GPAConsensusAligner:
    """
    Align multi-view embeddings to a common consensus coordinate system (zero-mean shape after centering).
    - Input: per-view coordinates on the same training cells (paired), same dimension d.
    - Output: per-view similarity transform (R_k, s_k, mu_k) and consensus shape M.
    - transform: map coordinates from a view into the consensus space.
    """
    name = "gpa_consensus"

    def __init__(self,
                 allow_scale: bool = True,
                 allow_reflection: bool = False,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 weights: Optional[Dict[str, float]] = None,
                 normalize_each_view: bool = True):
        """
        Args:
        - allow_scale: allow isotropic scale s_k per view
        - allow_reflection: allow reflection (det(R_k) can be negative); often False = rotation only
        - max_iter / tol: GPA iteration cap and convergence threshold
        - weights: per-view weights when updating consensus (default equal)
        - normalize_each_view: Frobenius-normalize each aligned view each round for stability
        """
        self.allow_scale = allow_scale
        self.allow_reflection = allow_reflection
        self.max_iter = max_iter
        self.tol = tol
        self.weights = weights or {}
        self.normalize_each_view = normalize_each_view

        # Fitted state
        self.views_: List[str] = []
        self.mu_: Dict[str, np.ndarray] = {}     # per-view training means (for centering)
        self.R_: Dict[str, np.ndarray] = {}      # rotations
        self.s_: Dict[str, float] = {}           # scales
        self.M_: Optional[np.ndarray] = None     # consensus shape (centered)
        self.residuals_: Dict[str, float] = {}   # normalized Procrustes distortion
        self.default_view: Optional[str] = None  # default view for transform()
    

    def set_default_view(self, name: str):
        if not self.views_:
            raise AssertionError("Aligner not fitted yet.")
        if name not in self.views_:
            raise KeyError(f"Unknown view '{name}'. Available: {self.views_}")
        self.default_view = name


    def fit(self, Z_train: Dict[str, np.ndarray]) -> "GPAConsensusAligner":
        """
        Z_train: dict[str, (n_cells_train, d)]. Training coordinates per view for the same cells, paired.
        All views must share the same d and n_cells_train.
        """
        if not Z_train:
            raise ValueError("Z_train is empty.")

        # View names and checks
        self.views_ = list(Z_train.keys())
        n_cells, d = None, None
        for name, Z in Z_train.items():
            Z = np.asarray(Z, dtype=np.float64)
            if Z.ndim != 2:
                raise ValueError(f"{name}: Z must be 2D array.")
            if n_cells is None:
                n_cells = Z.shape[0]
                d = Z.shape[1]
            else:
                if Z.shape != (n_cells, d):
                    raise AssertionError("All views must share (n_cells, d).")
            Zc, mu = _center(Z)
            self.mu_[name] = mu
            Z_train[name] = Zc  # work in zero-mean space

        # Init consensus M: weighted average (optional Frobenius norm)
        M = np.zeros((n_cells, d), dtype=np.float64)
        total_w = 0.0
        for name, Zc in Z_train.items():
            w = float(self.weights.get(name, 1.0))
            M += w * Zc
            total_w += w
        M /= (total_w if total_w > 0 else len(self.views_))

        # GPA iterations
        last_obj = np.inf
        for it in range(self.max_iter):
            aligned_sum = np.zeros_like(M)
            total_w = 0.0

            # Per view: best R_k, s_k s.t. s_k X_k R_k ~ M
            for name, X in Z_train.items():
                R, s = _orthogonal_procrustes_svd(
                    X, M, allow_scale=self.allow_scale, allow_reflection=self.allow_reflection
                )
                self.R_[name] = R
                self.s_[name] = s
                X_aligned = (X @ R) * s  # aligned centered shape
                if self.normalize_each_view:
                    nf = _fro_norm(X_aligned)
                    if nf > 0:
                        X_aligned = X_aligned / nf
                w = float(self.weights.get(name, 1.0))
                aligned_sum += w * X_aligned
                total_w += w

            # Update consensus M
            M_new = aligned_sum / (total_w if total_w > 0 else len(self.views_))

            # Weighted sum of squared Procrustes errors || s X R - M_new ||_F^2
            obj = 0.0
            for name, X in Z_train.items():
                X_aligned = (X @ self.R_[name]) * self.s_[name]
                if self.normalize_each_view:
                    nf = _fro_norm(X_aligned)
                    if nf > 0:
                        X_aligned = X_aligned / nf
                diff = X_aligned - M_new
                w = float(self.weights.get(name, 1.0))
                obj += w * (_fro_norm(diff) ** 2)

            # Convergence: small objective improvement and small M change
            rel_improve = (last_obj - obj) / (abs(last_obj) + 1e-12)
            delta_M = _fro_norm(M_new - M) / ( _fro_norm(M) + 1e-12 )
            M = M_new
            if rel_improve < self.tol and delta_M < self.tol:
                break
            last_obj = obj

        # Final consensus
        self.M_ = M

        # Normalized per-view Procrustes residual: || sX R - M ||_F^2 / ||M||_F^2
        denom = _fro_norm(self.M_) ** 2 + 1e-12
        for name, X in Z_train.items():
            X_aligned = (X @ self.R_[name]) * self.s_[name]
            if self.normalize_each_view:
                nf = _fro_norm(X_aligned)
                if nf > 0:
                    X_aligned = X_aligned / nf
            sse = _fro_norm(X_aligned - self.M_) ** 2
            self.residuals_[name] = float(sse / denom)

        if len(self.views_) == 1 and self.default_view is None:
            self.default_view = self.views_[0]

        return self


    def transform_one(self, name: str, Z: np.ndarray) -> np.ndarray:
        """
        Map samples from one view (train or test) to consensus space (centered shape).
        Output is in the same domain as the consensus (mean 0). Add a global mean outside if needed.
        """
        if name not in self.views_:
            raise KeyError(f"Unknown view '{name}'.")
        if self.M_ is None:
            raise AssertionError("Aligner not fitted.")
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[1] != self.M_.shape[1]:
            raise AssertionError("Dim mismatch: Z has different feature dim than consensus.")
        Zc = Z - self.mu_[name]  # center with training means
        Z_aligned = (Zc @ self.R_[name]) * self.s_[name]
        if self.normalize_each_view:
            nf = _fro_norm(Z_aligned)
            if nf > 0:
                Z_aligned = Z_aligned / nf
        return Z_aligned  # consensus coordinates, mean 0
    
    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        No view name: use the default or the only available view.
        - Single view: use it automatically;
        - Multiple views: call set_default_view(name) first.
        """
        if self.M_ is None:
            raise AssertionError("Aligner not fitted.")
        # Prefer default; else single view; else error
        use_name = self.default_view
        if use_name is None:
            if len(self.views_) == 1:
                use_name = self.views_[0]
            else:
                raise KeyError(
                "Multiple views present and no default view set. "
                "Call set_default_view(name) before transform(Z)."
                )
        return self.transform_one(use_name, Z)



    def transform_all(self, Z_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Align all views; return a dict of coordinates in consensus space.
        """
        out = {}
        for name, Z in Z_dict.items():
            out[name] = self.transform_one(name, Z)
        return out


    def info(self) -> Dict[str, Any]:
        """
        Summary: det(R), s, residuals per view, etc.
        """
        dets = {n: float(np.linalg.det(self.R_[n])) for n in self.views_}
        return {
            "name": self.name,
            "views": self.views_,
            "allow_scale": self.allow_scale,
            "allow_reflection": self.allow_reflection,
            "normalize_each_view": self.normalize_each_view,
            "residuals": self.residuals_,
            "detR": dets,
            "s": self.s_,
            "consensus_fro": None if self.M_ is None else _fro_norm(self.M_)
        }

    def save(self, path: str):
        """
        Save fitted aligner (transforms, means, consensus shape).
        """
        state = {
            "allow_scale": self.allow_scale,
            "allow_reflection": self.allow_reflection,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "weights": self.weights,
            "normalize_each_view": self.normalize_each_view,
            "views_": self.views_,
            "mu_": self.mu_,
            "R_": self.R_,
            "s_": self.s_,
            "M_": self.M_,
            "residuals_": self.residuals_,
        }
        joblib.dump(state, path)


    @staticmethod
    def load(path: str) -> "GPAConsensusAligner":
        state = joblib.load(path)
        obj = GPAConsensusAligner(
            allow_scale=state["allow_scale"],
            allow_reflection=state["allow_reflection"],
            max_iter=state["max_iter"],
            tol=state["tol"],
            weights=state["weights"],
            normalize_each_view=state["normalize_each_view"],
        )
        obj.views_ = state["views_"]
        obj.mu_ = state["mu_"]
        obj.R_ = state["R_"]
        obj.s_ = state["s_"]
        obj.M_ = state["M_"]
        obj.residuals_ = state["residuals_"]
        return obj


# Optional: neighborhood overlap (alignment quality / local topology)
def knn_overlap_rate(X_orig: np.ndarray,
                     X_align: np.ndarray,
                     k: int = 15) -> float:
    """
    Simple kNN overlap: mean Jaccard-like overlap of k-NN sets in original vs aligned space.
    Fast local topology preservation check. Rows of X_orig and X_align must match 1:1.
    """
    from sklearn.neighbors import NearestNeighbors
    nn1 = NearestNeighbors(n_neighbors=k).fit(X_orig)
    nn2 = NearestNeighbors(n_neighbors=k).fit(X_align)
    idx1 = nn1.kneighbors(return_distance=False)
    idx2 = nn2.kneighbors(return_distance=False)
    overlaps = []
    for i in range(X_orig.shape[0]):
        s1 = set(idx1[i].tolist())
        s2 = set(idx2[i].tolist())
        inter = len(s1 & s2)
        overlaps.append(inter / float(k))
    return float(np.mean(overlaps))


# Registry / factory
ALIGNER_REGISTRY = {
    "identity": IdentityAligner,
    "procrustes": ProcrustesAligner,
    'ridge': RidgeAligner,
    'gpa_consensus': GPAConsensusAligner
    # extend here
}


def make_aligner(name: str, **kwargs) -> BaseAligner:
    if name not in ALIGNER_REGISTRY:
        raise ValueError(f"Unknown alignment strategy: {name}")
    return ALIGNER_REGISTRY[name](**kwargs)


def save_aligner(aligner, out: str | Path):
    # out is a directory (may be relative); resolve and create
    p = Path(out).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p.mkdir(parents=True, exist_ok=True)

    name = getattr(aligner, "name", aligner.__class__.__name__.lower())
    file_path = p / f"{name}.joblib"
    joblib.dump(aligner, file_path)
    return str(file_path)


def load_aligner(path: str | Path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return joblib.load(p)
    

