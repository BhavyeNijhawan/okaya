"""Official metric: macro-averaged F0.5 per Source-1 entity (singletons included)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def entity_f05(pred: Sequence[str], truth: Sequence[str]) -> float:
    ps, ts = set(pred), set(truth)
    if not ps and not ts:
        return 1.0
    if not ps or not ts:
        return 0.0
    tp = len(ps & ts)
    if tp == 0:
        return 0.0
    p = tp / len(ps); r = tp / len(ts)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: Dict[str, Sequence[str]], truth: Dict[str, Sequence[str]]) -> float:
    return float(np.mean([entity_f05(pred.get(k, ()), v) for k, v in truth.items()]))


def macro_f05_pairs(n_s1: int, pred_i1: np.ndarray, pred_i2: np.ndarray, true_i1: np.ndarray, true_i2: np.ndarray
                    ) -> Tuple[float, Dict[str, float]]:
    """Vectorised metric on integer pair arrays. Returns (macro F0.5, summary dict)."""
    n_true = np.bincount(true_i1, minlength=n_s1).astype(np.float64)
    n_pred = np.bincount(pred_i1, minlength=n_s1).astype(np.float64)
    true_keys = set(zip(true_i1.tolist(), true_i2.tolist()))
    tp_mask = np.fromiter(((a, b) in true_keys for a, b in zip(pred_i1.tolist(), pred_i2.tolist())), dtype=bool, count=len(pred_i1))
    tp = np.bincount(pred_i1[tp_mask], minlength=n_s1).astype(np.float64)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, 1.25 * tp / np.maximum(n_pred + 0.25 * n_true, 1e-9))
    f[(n_true == 0) & (n_pred > 0)] = 0.0
    f[(n_true > 0) & (n_pred == 0)] = 0.0
    fp = int((~tp_mask).sum()); fn = int(len(true_i1) - tp_mask.sum())
    summary = {
        "macro_f05": float(f.mean()),
        "pair_precision": float(tp_mask.mean()) if len(tp_mask) else 1.0,
        "pair_recall": float(tp_mask.sum() / max(1, len(true_i1))),
        "fp": fp, "fn": fn, "n_pred": int(len(pred_i1)), "n_true": int(len(true_i1)),
        "singleton_fp_rate": float(((n_true == 0) & (n_pred > 0)).sum() / max(1, (n_true == 0).sum())),
        "pred_per_s1": float(n_pred.mean()), "true_per_s1": float(n_true.mean()),
        "empty_share": float((n_pred == 0).mean()),
    }
    return float(f.mean()), summary


def per_entity_scores(n_s1: int, pred_i1, pred_i2, true_i1, true_i2) -> np.ndarray:
    n_true = np.bincount(true_i1, minlength=n_s1).astype(np.float64)
    n_pred = np.bincount(pred_i1, minlength=n_s1).astype(np.float64)
    true_keys = set(zip(true_i1.tolist(), true_i2.tolist()))
    tp_mask = np.fromiter(((a, b) in true_keys for a, b in zip(pred_i1.tolist(), pred_i2.tolist())), dtype=bool, count=len(pred_i1))
    tp = np.bincount(pred_i1[tp_mask], minlength=n_s1).astype(np.float64)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, 1.25 * tp / np.maximum(n_pred + 0.25 * n_true, 1e-9))
    f[(n_true == 0) & (n_pred > 0)] = 0.0
    f[(n_true > 0) & (n_pred == 0)] = 0.0
    return f
