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
    # 不做变换，在自己的空间里计算metric
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
        lam: L2 正则系数（λ>=0）。越大越保守，数值更稳但可能偏差增大。
        fit_intercept: 是否拟合偏置（通过对齐前中心化与对齐后还原实现）。
        standardize: 是否对源特征做列标准化（仅对 Xs 做，提升数值稳定性）。
        """
        self.lam = float(lam)
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)

        self.W = None         # 线性映射矩阵 (d_src, d_tgt)
        self.mu_s = None      # 源均值 (1, d)
        self.mu_r = None      # 参照均值 (1, d)
        self.std_s = None     # 源标准差 (1, d) - 仅在 standardize=True 时使用

    def fit(self, Z_src_train, Z_ref_train, config=None):
        """
        Z_src_train: (n, d)
        Z_ref_train: (n, d) 与源一一对应
        config: 可选字典覆盖 lam/fit_intercept/standardize
        """
        if config is not None:
            if "lam" in config: self.lam = float(config["lam"])
            if "fit_intercept" in config: self.fit_intercept = bool(config["fit_intercept"])
            if "standardize" in config: self.standardize = bool(config["standardize"])

        Zs = np.asarray(Z_src_train, dtype=np.float64)
        Zr = np.asarray(Z_ref_train, dtype=np.float64)
        assert Zs.shape == Zr.shape, "source and reference must be paired (same shape)"
        n, d = Zs.shape

        # 均值（用于拟合偏置）
        if self.fit_intercept:
            self.mu_s = Zs.mean(axis=0, keepdims=True)
            self.mu_r = Zr.mean(axis=0, keepdims=True)
        else:
            self.mu_s = np.zeros((1, d), dtype=np.float64)
            self.mu_r = np.zeros((1, d), dtype=np.float64)

        Xs = Zs - self.mu_s if self.fit_intercept else Zs
        Xr = Zr - self.mu_r if self.fit_intercept else Zr

        # 可选：仅对源做列标准化，提升数值稳定性
        if self.standardize:
            std = Xs.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0  # 防止除零
            self.std_s = std
            Xs = Xs / std
        else:
            self.std_s = None

        # 多输出 ridge 闭式解：W = (Xs^T Xs + λI)^(-1) Xs^T Xr
        I = np.eye(d, dtype=np.float64)
        A = Xs.T @ Xs + self.lam * I
        B = Xs.T @ Xr
        # 数值稳定的线性方程求解，避免显式求逆
        self.W = np.linalg.solve(A, B)

        return self

    def transform(self, Z_src):
        """
        将任意源域样本映射到参照域。
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
    相似 Procrustes：给定中心化后的 X, Y，返回最优旋转 R 以及各向同性缩放 s（若允许）。
    最小化 || s X R − Y ||_F。
    """
    # X^T Y 的 SVD
    U, S, VT = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = VT.T @ U.T  # 注意：R = V U^T（这里 VT 为 V^T）
    # 控制是否允许镜像（反射），保证 det(R) >= 0
    if not allow_reflection:
        if np.linalg.det(R) < 0:
            # 翻转最后一列的符号以消除反射
            VT[-1, :] *= -1.0
            R = VT.T @ U.T

    if allow_scale:
        s = float(S.sum() / (_fro_norm(X) ** 2 + 1e-12))
    else:
        s = 1.0
    return R, s


class GPAConsensusAligner:
    """
    将多视角 embedding 对齐到一个共同的“共识坐标系”（中心化后均值为 0 的形状）。
    - 输入：每个视角在训练细胞上的坐标（必须配对同一批细胞），维度需一致。
    - 输出：每个视角的相似变换参数（R_k, s_k, mu_k），以及共识形状 M。
    - transform：将任意该视角的坐标映射到共识空间。
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
        参数：
        - allow_scale: 是否允许各向同性缩放 s_k
        - allow_reflection: 是否允许镜像（det(R_k) 可为负）；通常 False 只允许旋转
        - max_iter/tol: GPA 迭代的最大步数与收敛阈值
        - weights: 各视角在更新共识时的权重（默认等权）
        - normalize_each_view: 每轮对齐时是否将各视角对齐后的形状做 Frobenius 归一，使数值更稳
        """
        self.allow_scale = allow_scale
        self.allow_reflection = allow_reflection
        self.max_iter = max_iter
        self.tol = tol
        self.weights = weights or {}
        self.normalize_each_view = normalize_each_view

        # 拟合后的参数
        self.views_: List[str] = []
        self.mu_: Dict[str, np.ndarray] = {}     # 每视角训练集均值（中心化用）
        self.R_: Dict[str, np.ndarray] = {}      # 旋转
        self.s_: Dict[str, float] = {}           # 缩放
        self.M_: Optional[np.ndarray] = None     # 共识形状（中心化）
        self.residuals_: Dict[str, float] = {}   # 归一化 Procrustes 残差
        self.default_view: Optional[str] = None  # 设置一个默认视角
    

    def set_default_view(self, name: str):
        if not self.views_:
            raise AssertionError("Aligner not fitted yet.")
        if name not in self.views_:
            raise KeyError(f"Unknown view '{name}'. Available: {self.views_}")
        self.default_view = name


    def fit(self, Z_train: Dict[str, np.ndarray]) -> "GPAConsensusAligner":
        """
        Z_train: dict[name -> (n_cells_train, d)]。每个视角的训练数据，必须是同一批细胞的配对坐标。
        所有视角的维度 d 必须一致，n_cells_train 应一致。
        """
        if not Z_train:
            raise ValueError("Z_train is empty.")

        # 视角名列表与检查
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
            Z_train[name] = Zc  # 替换为中心化后的数据，后续都在零均值空间工作

        # 初始化共识形状 M：简单平均（可选做 Frobenius 归一）
        M = np.zeros((n_cells, d), dtype=np.float64)
        total_w = 0.0
        for name, Zc in Z_train.items():
            w = float(self.weights.get(name, 1.0))
            M += w * Zc
            total_w += w
        M /= (total_w if total_w > 0 else len(self.views_))

        # GPA 迭代
        last_obj = np.inf
        for it in range(self.max_iter):
            aligned_sum = np.zeros_like(M)
            total_w = 0.0

            # 对每个视角，求最优 R_k, s_k，使 s_k X_k R_k ≈ M
            for name, X in Z_train.items():
                R, s = _orthogonal_procrustes_svd(
                    X, M, allow_scale=self.allow_scale, allow_reflection=self.allow_reflection
                )
                self.R_[name] = R
                self.s_[name] = s
                X_aligned = (X @ R) * s  # 中心化后数据的对齐结果（仍是中心化形状）
                if self.normalize_each_view:
                    nf = _fro_norm(X_aligned)
                    if nf > 0:
                        X_aligned = X_aligned / nf
                w = float(self.weights.get(name, 1.0))
                aligned_sum += w * X_aligned
                total_w += w

            # 更新共识 M
            M_new = aligned_sum / (total_w if total_w > 0 else len(self.views_))

            # 目标函数：总残差 || sX R − M_new ||_F^2 的加权和
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

            # 收敛判据：目标函数改善很小或 M 变化很小
            rel_improve = (last_obj - obj) / (abs(last_obj) + 1e-12)
            delta_M = _fro_norm(M_new - M) / ( _fro_norm(M) + 1e-12 )
            M = M_new
            if rel_improve < self.tol and delta_M < self.tol:
                break
            last_obj = obj

        # 固定最终共识形状
        self.M_ = M

        # 计算每视角的归一化 Procrustes 残差（失真）： || sX R − M ||_F^2 / ||M||_F^2
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
        将某个视角的任意样本（训练或测试）映射到共识空间（中心化形状）。
        输出与共识形状同域（均值为 0）。如需加回总体均值，可在外层自行加 0（共识均值即 0）。
        """
        if name not in self.views_:
            raise KeyError(f"Unknown view '{name}'.")
        if self.M_ is None:
            raise AssertionError("Aligner not fitted.")
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[1] != self.M_.shape[1]:
            raise AssertionError("Dim mismatch: Z has different feature dim than consensus.")
        Zc = Z - self.mu_[name]  # 使用训练时的均值进行中心化
        Z_aligned = (Zc @ self.R_[name]) * self.s_[name]
        if self.normalize_each_view:
            nf = _fro_norm(Z_aligned)
            if nf > 0:
                Z_aligned = Z_aligned / nf
        return Z_aligned  # 共识空间坐标，中心化后均值为 0
    
    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        无需传 name；使用默认视角或唯一视角进行转换。
        - 若仅有一个视角，自动用它；
        - 若有多个视角，需先 set_default_view(name)。
        """
        if self.M_ is None:
            raise AssertionError("Aligner not fitted.")
        # 优先使用已有默认视角；否则若仅有一个视角则使用它；否则报错
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
        对所有视角进行对齐，返回映射到共识空间的坐标字典。
        """
        out = {}
        for name, Z in Z_dict.items():
            out[name] = self.transform_one(name, Z)
        return out


    def info(self) -> Dict[str, Any]:
        """
        返回基本信息与每视角的 det(R)、s、残差等。
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
        保存拟合后的对齐器（含变换参数与均值、共识形状）。
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


# 可选：邻域保持/重合率（对齐失真度量之一）
def knn_overlap_rate(X_orig: np.ndarray,
                     X_align: np.ndarray,
                     k: int = 15) -> float:
    """
    计算简单的 kNN 重合率：在原生空间与对齐空间里，每个点的 k 近邻集合重合比例的平均。
    用作快速的局部拓扑保持指标。X_orig 与 X_align 的行对应同一批样本。
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


# 注册表/工厂
ALIGNER_REGISTRY = {
    "identity": IdentityAligner,
    "procrustes": ProcrustesAligner,
    'ridge': RidgeAligner,
    'gpa_consensus': GPAConsensusAligner
    # 可后续扩展
}


def make_aligner(name: str, **kwargs) -> BaseAligner:
    if name not in ALIGNER_REGISTRY:
        raise ValueError(f"未知对齐策略: {name}")
    return ALIGNER_REGISTRY[name](**kwargs)


def save_aligner(aligner, out: str | Path):
    # out 是目录（可相对）；自动转绝对并创建
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
    

