"""Loss ledger: where does macro-F0.5 go? (validation slice of the training universe)

For each country: oracle after blocking (all retrieved true pairs predicted, nothing else), oracle
after pruning, the model's decisions, plus a breakdown of lost score by cause (never retrieved /
pruned / rejected / wrong merge) and by pool-record type (empty address, Indic script, source).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .decide import decide
from .evaluate import per_entity_scores


def ledger(work: Path, split: str = "train", role: str = "val", decision: Dict | None = None) -> Dict:
    from .pipeline import Calibrator
    dec = decision or json.loads((work / "train" / "decision.json").read_text())
    cal = Calibrator.from_json(dec["calibrator"])
    out: Dict[str, Dict] = {}
    for p in sorted((work / split).glob("p2_*.npy")):
        c = p.stem[3:]
        roles = pd.read_parquet(work / split / f"s1ids_{c}.parquet").role.to_numpy()
        n_s1 = len(roles); val = roles == role
        raw = pd.read_parquet(work / split / f"cand_raw_{c}.parquet", columns=["i1", "i2", "y"])
        C = pd.read_parquet(work / split / f"cand_{c}.parquet", columns=["i1", "i2", "y"])
        T = pd.read_parquet(work / split / f"truth_{c}.parquet")
        p2 = np.load(p); pc = cal(p2)
        ti1, ti2 = T.i1.to_numpy(), T.i2.to_numpy()
        res = {}
        # oracles
        m = raw.y.to_numpy() == 1
        f = per_entity_scores(n_s1, raw.i1.to_numpy()[m], raw.i2.to_numpy()[m], ti1, ti2); res["oracle_after_blocking"] = float(f[val].mean())
        m = C.y.to_numpy() == 1
        f = per_entity_scores(n_s1, C.i1.to_numpy()[m], C.i2.to_numpy()[m], ti1, ti2); res["oracle_after_pruning"] = float(f[val].mean())
        mask = decide(C.i1.to_numpy(), C.i2.to_numpy(), pc, method=dec["method"], thr=dec.get("thr", 0.7), lam_miss=dec.get("lam_miss", 0.02))
        f = per_entity_scores(n_s1, C.i1.to_numpy()[mask], C.i2.to_numpy()[mask], ti1, ti2); res["model"] = float(f[val].mean())
        # remove all FP (keep only true predicted) -> shows the cost of false merges
        m2 = mask & (C.y.to_numpy() == 1)
        f = per_entity_scores(n_s1, C.i1.to_numpy()[m2], C.i2.to_numpy()[m2], ti1, ti2); res["model_without_fp"] = float(f[val].mean())
        # counts on val S1
        vi = val[C.i1.to_numpy()]
        y = C.y.to_numpy()
        res["val_true_pairs"] = int(val[ti1].sum())
        res["retrieved_true"] = int((raw.y.to_numpy() == 1)[val[raw.i1.to_numpy()]].sum())
        res["kept_true"] = int((y == 1)[vi].sum())
        res["fp"] = int((mask & (y == 0) & vi).sum()); res["fn_rejected"] = int((~mask & (y == 1) & vi).sum())
        res["pred_per_s1"] = float(mask[vi].sum() / max(1, val.sum()))
        res["true_per_s1"] = float(val[ti1].sum() / max(1, val.sum()))
        # by pool-record type (needs pool table): empty address / script
        try:
            pool = pd.read_parquet(work / "norm" / f"{split}_pool.parquet", columns=["addr_empty", "raw_name", "country"])
            ids = pd.read_parquet(work / split / f"poolids_{c}.parquet").eid.to_numpy()
            pool = pool[pool.country == c].reset_index(drop=True)
            empty = pool.addr_empty.to_numpy().astype(bool)
            e = empty[C.i2.to_numpy()]
            res["fn_rejected_empty_addr"] = int((~mask & (y == 1) & vi & e).sum())
            res["true_pairs_empty_addr_share"] = float(e[(y == 1) & vi].mean()) if ((y == 1) & vi).any() else 0.0
            res["recall_empty_addr"] = float(mask[(y == 1) & vi & e].mean()) if ((y == 1) & vi & e).any() else 0.0
            res["recall_nonempty_addr"] = float(mask[(y == 1) & vi & ~e].mean()) if ((y == 1) & vi & ~e).any() else 0.0
        except Exception as ex:  # pragma: no cover
            res["type_breakdown_error"] = str(ex)
        out[c] = res
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(ledger(Path(sys.argv[1])), indent=1))
