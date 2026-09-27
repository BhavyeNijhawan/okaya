"""GPU top-K cosine retrieval for sparse TF-IDF matrices via random projection (Colab T4 path).

The exact sparse product (sparse_dot_topn) is CPU-bound and, at 1.7M x 10M records, takes hours on
two vCPUs. On a GPU we instead project the L2-normalised TF-IDF vectors with a fixed sparse random
sign matrix (each vocabulary gram -> `n_hash` random dimensions of a D-dimensional dense vector),
re-normalise, and run dense fp16 matmuls tile by tile with torch.topk. Cosines are approximated
(std ~ 1/sqrt(D) ~ 0.03 at D=1024); a larger shortlist compensates. Exact cosines are recomputed
afterwards for every candidate pair by `blocking.rowwise_cosine`, so features are exact.

Memory discipline: the index is projected one shard at a time (default 400k rows -> 0.8 GB fp16 on
the device); query rows are projected per tile (2048 rows); score tiles are 2048 x 400k fp16
(1.6 GB) and released immediately; the running top-K per query row lives on the host.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def gpu_available() -> bool:
    return torch is not None and torch.cuda.is_available()


class Projector:
    def __init__(self, V: int, D: int = 1024, n_hash: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        rows = np.repeat(np.arange(V), n_hash)
        cols = rng.integers(0, D, size=V * n_hash)
        signs = rng.choice([-1.0, 1.0], size=V * n_hash).astype(np.float32) / np.sqrt(n_hash)
        self.R = sp.csr_matrix((signs, (rows, cols)), shape=(V, D), dtype=np.float32)
        self.D = D

    def project(self, X: sp.csr_matrix, chunk: int = 100_000) -> np.ndarray:
        """Dense fp16 (n, D) projection, L2-normalised per row, computed in bounded chunks."""
        out = np.empty((X.shape[0], self.D), dtype=np.float16)
        for s in range(0, X.shape[0], chunk):
            Y = (X[s:s + chunk] @ self.R)
            Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
            Y = Y.astype(np.float32, copy=False)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Y /= norms
            out[s:s + chunk] = Y.astype(np.float16)
        return out


def topk_dense(A: sp.csr_matrix, B: sp.csr_matrix, k: int, threshold: float, projector: Projector,
               device: Optional[str] = None, chunk_a: int = 2048, chunk_b: int = 400_000
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of A: up to k columns of B with (approximate) cosine >= threshold."""
    assert torch is not None
    n1, n2 = A.shape[0], B.shape[0]
    if n1 == 0 or n2 == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    k_eff = min(k, n2)
    best_v = np.full((n1, k_eff), -np.inf, dtype=np.float32)
    best_i = np.zeros((n1, k_eff), dtype=np.int64)
    for b0 in range(0, n2, chunk_b):
        Bp = torch.from_numpy(projector.project(B[b0:b0 + chunk_b])).to(device=device, dtype=dtype)
        BT = Bp.T.contiguous(); del Bp
        kk = min(k_eff, BT.shape[1])
        for a0 in range(0, n1, chunk_a):
            a = torch.from_numpy(projector.project(A[a0:a0 + chunk_a])).to(device=device, dtype=dtype)
            sc = a @ BT
            v, i = torch.topk(sc, kk, dim=1)
            del sc, a
            v = v.float().cpu().numpy(); i = i.cpu().numpy() + b0
            # merge with the running best (host side)
            cur_v = best_v[a0:a0 + v.shape[0]]; cur_i = best_i[a0:a0 + v.shape[0]]
            allv = np.concatenate([cur_v, v], axis=1); alli = np.concatenate([cur_i, i], axis=1)
            sel = np.argsort(-allv, axis=1, kind="stable")[:, :k_eff]
            best_v[a0:a0 + v.shape[0]] = np.take_along_axis(allv, sel, axis=1)
            best_i[a0:a0 + v.shape[0]] = np.take_along_axis(alli, sel, axis=1)
        del BT
        if device == "cuda":
            torch.cuda.empty_cache()
    mask = best_v >= threshold
    rows = np.repeat(np.arange(n1), k_eff).reshape(n1, k_eff)[mask]
    return rows.astype(np.int64), best_i[mask].astype(np.int64), best_v[mask].astype(np.float32)
