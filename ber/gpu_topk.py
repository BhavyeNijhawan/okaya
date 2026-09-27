"""GPU top-K cosine retrieval for sparse TF-IDF matrices via random projection (Colab T4 path).

The exact sparse product (sparse_dot_topn) is CPU-bound and, at 1.7M x 10M records, takes hours on
two vCPUs. On a GPU we instead project the L2-normalised TF-IDF vectors with a fixed sparse random
sign matrix (each vocabulary gram -> `n_hash` random dimensions of a D-dimensional dense vector),
re-normalise, and run dense fp16 matmuls chunk by chunk with torch.topk. Cosines are approximated
(std ~ 1/sqrt(D) ~ 0.03 at D=1024); a slightly larger K compensates. Exact cosines are recomputed
afterwards for every candidate pair by `blocking.rowwise_cosine`, so features are exact.
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

    def project(self, X: sp.csr_matrix) -> np.ndarray:
        Y = np.asarray((X @ self.R).todense(), dtype=np.float32) if sp.issparse(X @ self.R) else np.asarray(X @ self.R, dtype=np.float32)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (Y / norms).astype(np.float16)


def topk_dense(A: sp.csr_matrix, B: sp.csr_matrix, k: int, threshold: float, projector: Projector,
               device: Optional[str] = None, chunk_a: int = 4096, chunk_b: int = 1_000_000
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of A: up to k columns of B with (approximate) cosine >= threshold."""
    assert torch is not None
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    Bp = projector.project(B)
    n2 = Bp.shape[0]
    k_eff = min(k, n2)
    rows_out, cols_out, vals_out = [], [], []
    # B in shards on the device (a 1M x 1024 fp16 shard is 2GB)
    shards = []
    for s in range(0, n2, chunk_b):
        shards.append((s, torch.from_numpy(Bp[s:s + chunk_b]).to(device=device, dtype=dtype).T.contiguous()))
    Ap = projector.project(A)
    for a0 in range(0, Ap.shape[0], chunk_a):
        a = torch.from_numpy(Ap[a0:a0 + chunk_a]).to(device=device, dtype=dtype)
        best_v, best_i = None, None
        for s, BT in shards:
            sc = a @ BT
            kk = min(k_eff, sc.shape[1])
            v, i = torch.topk(sc, kk, dim=1)
            i = i + s
            if best_v is None:
                best_v, best_i = v, i
            else:
                v = torch.cat([best_v, v], dim=1); i = torch.cat([best_i, i], dim=1)
                best_v, sel = torch.topk(v, min(k_eff, v.shape[1]), dim=1)
                best_i = torch.gather(i, 1, sel)
        v = best_v.float().cpu().numpy(); i = best_i.cpu().numpy()
        mask = v >= threshold
        r = np.repeat(np.arange(a0, a0 + v.shape[0]), v.shape[1]).reshape(v.shape)[mask]
        rows_out.append(r.astype(np.int64)); cols_out.append(i[mask].astype(np.int64)); vals_out.append(v[mask].astype(np.float32))
    del shards
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(vals_out)
