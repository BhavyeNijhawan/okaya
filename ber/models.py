"""Gradient-boosted tree wrapper: XGBoost on CUDA when a GPU is available (Colab T4: minutes instead of
hours on two vCPUs), LightGBM on CPU otherwise. Both permissively licensed (Apache-2.0 / MIT)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class GBDT:
    def __init__(self, backend: Optional[str] = None, leaves: int = 255, lr: float = 0.05, min_child: int = 50,
                 rounds: int = 3000, early_stopping: int = 100, seed: int = 42, threads: int = 2):
        self.backend = backend or ("xgb_cuda" if cuda_available() else "lgbm")
        self.leaves = leaves; self.lr = lr; self.min_child = min_child; self.rounds = rounds
        self.early_stopping = early_stopping; self.seed = seed; self.threads = threads
        self.model = None
        self.features: Optional[list] = None
        self.best_iteration: Optional[int] = None

    # -- training -------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "GBDT":
        self.features = list(X.columns)
        if self.backend == "xgb_cuda":
            import xgboost as xgb
            params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                          grow_policy="lossguide", max_depth=0, max_leaves=self.leaves, learning_rate=self.lr,
                          min_child_weight=self.min_child, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          seed=self.seed, max_bin=256)
            dtr = xgb.QuantileDMatrix(X, y, max_bin=256)
            dva = xgb.QuantileDMatrix(X_val, y_val, ref=dtr)
            self.model = xgb.train(params, dtr, num_boost_round=self.rounds, evals=[(dva, "val")],
                                   early_stopping_rounds=self.early_stopping, verbose_eval=False)
            self.best_iteration = int(self.model.best_iteration) + 1
        else:
            import lightgbm as lgb
            params = dict(objective="binary", learning_rate=self.lr, num_leaves=self.leaves, min_data_in_leaf=self.min_child,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                          num_threads=self.threads, seed=self.seed)
            dtr = lgb.Dataset(X, y); dva = lgb.Dataset(X_val, y_val)
            self.model = lgb.train(params, dtr, num_boost_round=self.rounds, valid_sets=[dva],
                                   callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            self.best_iteration = int(self.model.best_iteration or self.model.num_trees())
        return self

    # -- inference --------------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame, chunk: int = 2_000_000) -> np.ndarray:
        if self.features is not None and list(X.columns) != self.features:
            X = X[self.features]
        out = np.empty(len(X), dtype=np.float32)
        for s in range(0, len(X), chunk):
            xs = X.iloc[s:s + chunk]
            if self.backend == "xgb_cuda":
                import xgboost as xgb
                out[s:s + chunk] = self.model.predict(xgb.DMatrix(xs), iteration_range=(0, self.best_iteration or 0)).astype(np.float32)
            else:
                out[s:s + chunk] = self.model.predict(xs, num_iteration=self.best_iteration).astype(np.float32)
        return out

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        meta = {"backend": self.backend, "features": self.features, "best_iteration": self.best_iteration}
        if self.backend == "xgb_cuda":
            self.model.save_model(str(path.with_suffix(".xgb.json")))
        else:
            self.model.save_model(str(path.with_suffix(".lgb.txt")))
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "GBDT":
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        m = cls(backend=meta["backend"])
        m.features = meta["features"]; m.best_iteration = meta["best_iteration"]
        if meta["backend"] == "xgb_cuda":
            import xgboost as xgb
            m.model = xgb.Booster(); m.model.load_model(str(path.with_suffix(".xgb.json")))
            if not cuda_available():
                m.model.set_param({"device": "cpu"})
        else:
            import lightgbm as lgb
            m.model = lgb.Booster(model_file=str(path.with_suffix(".lgb.txt")))
        return m

    @staticmethod
    def exists(path: Path) -> bool:
        return Path(path).with_suffix(".meta.json").exists()
