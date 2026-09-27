"""End-to-end orchestration with per-country checkpoints.

Training universe (faithful to the test distribution): all training S1 entities minus a hidden
fraction (their pool records become orphans, raising the distractor density to the test level).
Within the visible S1s: `tab` slice (rate tables + pruner), `train` slice (matcher), `val` slice
(calibration, decisions, reporting); the rest only provide competition/context.

Stages (each cached under <work>/):
  norm      TSV -> normalized parquet (train and test)
  tables    region alias tables (per country), transliteration dictionary, universe split
  block     candidates per country (raw union) with labels                     -> <split>/cand_raw_{c}.parquet
  prune     pruner model + pruned candidates                                    -> <split>/cand_{c}.parquet
  feat      stage-1 features (parquet, streamed in chunks)                      -> <split>/feat_{c}.parquet
  stage1    LightGBM stage 1 (+ OOF for the train slice), p1 for every pair     -> train/p1_{c}.npy
  stage2    stage-2 features + LightGBM stage 2, p2 for every pair               -> train/p2_{c}.npy
  tune      calibration + decision parameters on the val slice, reports
  predict   test: same chain with the trained models, writes the two TSVs + diagnostics
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import blocking as B
from .data import build_norm_tables, gt_pairs, read_ground_truth
from .decide import decide
from .evaluate import per_entity_scores
from .expand import sibling_expansion
from .features import RateTables, stage1_features, stage2_features
from .geo import RegionTable, add_region_columns, load_tables, save_tables
from .models import GBDT
from .prune import prune_features, prune_mask
from .translit import TranslitDict, retranslit_frame


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Config:
    hide_frac = 0.188          # hidden S1 share -> test-like pool density (5.75 records / S1)
    tab_frac = 0.12            # slice for rate tables + pruner
    train_frac = 0.40          # matcher training slice
    val_frac = 0.12            # calibration / decision / reporting slice
    seed = 42
    n_jobs = 2
    prune_tau = 0.002
    prune_keep_top = 3
    prune_cap = 40
    max_train_pairs = 6_000_000
    lgb_threads = 2
    stage1_rounds = 3000
    stage2_rounds = 2000
    feat_chunk = 1_000_000
    use_gpu = True
    expand_siblings = True   # second retrieval pass through anchored pool records (after pruning)
    block_cfg: Optional[B.BlockConfig] = None


NORM_COLS = ["eid", "country", "raw_name", "raw_addr", "n_name", "n_name_alt", "n_core", "n_core2", "n_sorted2", "n_compact",
             "n_skel", "n_legal", "n_dom", "n_first", "n_ntok", "a_alpha", "a_hn", "a_hn_runs", "a_nums", "a_street", "a_stype",
             "a_loc", "a_unit", "a_has_hn_kw", "a_ncomp", "a_comps", "addr_empty", "src"]
SHORT_STR = ("n_legal", "n_first", "n_dom", "a_hn", "a_stype", "a_unit")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _load_country(work: Path, split: str, country: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """S1 and pool frames of one country. Text columns are Arrow-backed strings (compact)."""
    flt = [("country", "=", country)]
    out = []
    for name in ("s1", "pool"):
        tbl = pq.read_table(work / "norm" / f"{split}_{name}.parquet", filters=flt, columns=NORM_COLS)
        df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
        for c in SHORT_STR + ("eid", "country", "a_comps", "a_loc"):
            df[c] = np.array(["" if x is None else x for x in df[c].tolist()], dtype=object)
        for c in ("n_ntok", "a_has_hn_kw", "a_ncomp", "addr_empty", "src"):
            df[c] = df[c].to_numpy().astype(np.int32)
        out.append(df.reset_index(drop=True))
        del tbl
    return out[0], out[1]


def _labels(gt_map: Dict[str, List[str]], s1_eids: np.ndarray, pool_eids: np.ndarray, i1: np.ndarray, i2: np.ndarray):
    P = gt_pairs(gt_map, s1_eids, pool_eids)
    truth = set(zip(P[:, 0].tolist(), P[:, 1].tolist()))
    y = np.fromiter(((a, b) in truth for a, b in zip(i1.tolist(), i2.tolist())), dtype=np.int8, count=len(i1))
    return y, P


def _fit_gbdt(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: "Config", rounds: int, leaves: int = 255,
              seed_shift: int = 0, min_child: int = 50) -> GBDT:
    """Fit a GBDT with early stopping on a 5% hold-out of S1 groups."""
    rng = np.random.default_rng(cfg.seed + seed_shift)
    ug = np.unique(groups)
    hold = np.isin(groups, rng.choice(ug, max(1, len(ug) // 20), replace=False))
    m = GBDT(leaves=leaves, rounds=rounds, seed=cfg.seed + seed_shift, threads=cfg.lgb_threads, min_child=min_child)
    m.fit(X[~hold], y[~hold], X[hold], y[hold])
    return m


def _oof_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: "Config", rounds: int, n_folds: int = 2,
                 max_fit: int = 4_000_000) -> np.ndarray:
    """Out-of-fold predictions (folds by S1 group) for stage-2 training."""
    rng = np.random.default_rng(cfg.seed + 7)
    ug = np.unique(groups)
    fold_of = pd.Series(rng.integers(0, n_folds, len(ug)), index=ug)
    folds = fold_of.reindex(groups).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(n_folds):
        m = folds == f
        fit_idx = np.flatnonzero(~m)
        if len(fit_idx) > max_fit:
            fit_idx = np.sort(rng.choice(fit_idx, max_fit, replace=False))
        model = _fit_gbdt(X.iloc[fit_idx], y[fit_idx], groups[fit_idx], cfg, rounds, seed_shift=100 + f)
        oof[m] = model.predict(X[m])
    return oof


def read_feature_rows(path: Path, mask: Optional[np.ndarray] = None, columns: Optional[List[str]] = None,
                      batch_size: int = 500_000) -> pd.DataFrame:
    """Read a (possibly huge) feature parquet, keeping only rows where mask is True, batch by batch."""
    pf = pq.ParquetFile(path)
    parts = []
    off = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        df = batch.to_pandas()
        if mask is not None:
            df = df[mask[off:off + n]]
        parts.append(df)
        off += n
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns or [])


def predict_chunked(model: GBDT, path: Path, columns: List[str], extra: Optional[pd.DataFrame] = None,
                    batch_size: int = 500_000) -> np.ndarray:
    """model.predict over a feature parquet read in batches; `extra` (aligned rows) is concatenated per batch."""
    pf = pq.ParquetFile(path)
    out = []
    off = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[c for c in columns if extra is None or c not in extra.columns]):
        n = batch.num_rows
        df = batch.to_pandas()
        if extra is not None:
            df = pd.concat([df.reset_index(drop=True), extra.iloc[off:off + n].reset_index(drop=True)], axis=1)
        out.append(model.predict(df[columns]))
        off += n
    return np.concatenate(out) if out else np.zeros(0, np.float32)


class Calibrator:
    """Isotonic regression on validation probabilities (monotone, piecewise constant)."""

    def __init__(self):
        self.x = None; self.y = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        grid = np.linspace(0, 1, 1001)
        self.x = grid; self.y = ir.predict(grid).astype(np.float32)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.x is None:
            return p
        return np.interp(p, self.x, self.y).astype(np.float32)

    def to_json(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist()} if self.x is not None else {}

    @classmethod
    def from_json(cls, d: dict) -> "Calibrator":
        c = cls()
        if d:
            c.x = np.asarray(d["x"], np.float64); c.y = np.asarray(d["y"], np.float32)
        return c


def _countries(work: Path, split: str, pattern: str) -> List[str]:
    prefix = pattern.split("*")[0]
    return sorted(p.stem[len(prefix):] for p in (work / split).glob(pattern) if not p.stem.startswith("cand_raw") or prefix == "cand_raw_")


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, data_dir: Path, work: Path, cfg: Config = Config()):
        self.data_dir = Path(data_dir); self.work = Path(work); self.cfg = cfg
        (self.work / "train").mkdir(parents=True, exist_ok=True)
        (self.work / "tables").mkdir(parents=True, exist_ok=True)

    # -- stage: norm --------------------------------------------------------------------------
    def norm(self, splits=("train", "test")) -> None:
        for s in splits:
            if (self.data_dir / s).exists():
                build_norm_tables(self.data_dir, self.work, s, n_jobs=self.cfg.n_jobs, log=log)

    # -- stage: tables + universe -------------------------------------------------------------
    def tables(self, force: bool = False) -> None:
        cfg = self.cfg
        uni_path = self.work / "train" / "universe.json"
        if uni_path.exists() and (self.work / "tables" / "regions.json").exists() and not force:
            log("[tables] cached"); return
        gt = read_ground_truth(self.data_dir / "train" / "train_ground_truth.tsv")
        s1_meta = pq.read_table(self.work / "norm" / "train_s1.parquet", columns=["eid", "country", "a_comps"]).to_pandas()
        rng = np.random.default_rng(cfg.seed)
        roles: Dict[str, str] = {}
        for c in sorted(s1_meta.country.unique()):
            eids = s1_meta.eid[s1_meta.country == c].to_numpy()
            perm = rng.permutation(len(eids)); n = len(eids)
            n_hide = int(cfg.hide_frac * n); n_tab = int(cfg.tab_frac * n); n_tr = int(cfg.train_frac * n); n_val = int(cfg.val_frac * n)
            cuts = np.cumsum([n_hide, n_tab, n_tr, n_val])
            for name, a, b in zip(("hidden", "tab", "train", "val"), np.concatenate([[0], cuts[:-1]]), cuts):
                for e in eids[perm[a:b]].tolist():
                    roles[e] = name
            log(f"[universe] {c}: S1 {n:,} hidden {n_hide:,} tab {n_tab:,} train {n_tr:,} val {n_val:,} context {n - cuts[-1]:,}")
        uni_path.write_text(json.dumps(roles))
        # region tables: unsupervised on every record of the country (train + test), supervised on train/tab pairs
        pool_meta = pq.read_table(self.work / "norm" / "train_pool.parquet", columns=["eid", "country", "a_comps"]).to_pandas()
        have_test = (self.work / "norm" / "test_s1.parquet").exists()
        test_s1 = pq.read_table(self.work / "norm" / "test_s1.parquet", columns=["country", "a_comps"]).to_pandas() if have_test else None
        test_pool = pq.read_table(self.work / "norm" / "test_pool.parquet", columns=["country", "a_comps"]).to_pandas() if have_test else None
        countries = sorted(set(s1_meta.country) | (set(test_s1.country) if have_test else set()))
        tabs: Dict[str, RegionTable] = {}
        s1_comps = dict(zip(s1_meta.eid, s1_meta.a_comps)); s1_ctry = dict(zip(s1_meta.eid, s1_meta.country))
        pool_comps = dict(zip(pool_meta.eid, pool_meta.a_comps))
        for c in countries:
            t = RegionTable(c)
            parts = [s1_meta.a_comps[s1_meta.country == c], pool_meta.a_comps[pool_meta.country == c]]
            if have_test:
                parts += [test_s1.a_comps[test_s1.country == c], test_pool.a_comps[test_pool.country == c]]
            n_un = t.learn_unsupervised(pd.concat(parts).tolist())
            n_sup = 0
            if c in set(s1_meta.country):
                a_list, b_list = [], []
                for e, lst in gt.items():
                    if roles.get(e) in ("train", "tab") and s1_ctry.get(e) == c:
                        for x in lst:
                            pc_ = pool_comps.get(x)
                            if pc_ is not None:
                                a_list.append(s1_comps[e]); b_list.append(pc_)
                n_sup = t.learn_supervised(a_list, b_list)
            log(f"[regions] {c}: learned {n_un} unsupervised + {n_sup} supervised aliases; merges {t.merges}")
            tabs[c] = t
        save_tables(tabs, self.work / "tables" / "regions.json")
        del pool_comps, s1_comps
        # transliteration dictionary from train/tab pairs
        s1_names = pq.read_table(self.work / "norm" / "train_s1.parquet", columns=["eid", "n_name"]).to_pandas()
        pool_names = pq.read_table(self.work / "norm" / "train_pool.parquet", columns=["eid", "raw_name", "n_name"]).to_pandas()
        pn = dict(zip(pool_names.eid, zip(pool_names.raw_name, pool_names.n_name)))
        sn = dict(zip(s1_names.eid, s1_names.n_name))
        raws, pns, sns = [], [], []
        for e, lst in gt.items():
            if roles.get(e) in ("train", "tab"):
                for x in lst:
                    r = pn.get(x)
                    if r is not None:
                        raws.append(r[0]); pns.append(r[1]); sns.append(sn[e])
        td = TranslitDict(); n = td.learn(raws, pns, sns)
        td.save(self.work / "tables" / "translit.json")
        log(f"[translit] {n} token mappings learned")

    # -- shared: load + enrich a country ------------------------------------------------------
    def _prepare_country(self, split: str, country: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
        s1, pool = _load_country(self.work, split, country)
        roles = None
        if split == "train":
            roles_map = json.loads((self.work / "train" / "universe.json").read_text())
            role = np.array([roles_map.get(e, "context") for e in s1.eid.tolist()], dtype=object)
            keep = role != "hidden"
            s1 = s1[keep].reset_index(drop=True); roles = role[keep]
        tabs = load_tables(self.work / "tables" / "regions.json")
        t = tabs.get(country, RegionTable(country))
        add_region_columns(s1, t); add_region_columns(pool, t)
        td = TranslitDict.load(self.work / "tables" / "translit.json")
        retranslit_frame(pool, td); retranslit_frame(s1, td)
        return s1, pool, roles

    # -- stage: block -------------------------------------------------------------------------
    def block(self, split: str, countries: Optional[List[str]] = None, force: bool = False) -> None:
        cfg = self.cfg
        if countries is None:
            countries = sorted(pq.read_table(self.work / "norm" / f"{split}_s1.parquet", columns=["country"]).to_pandas().country.unique())
        gt = read_ground_truth(self.data_dir / "train" / "train_ground_truth.tsv") if split == "train" else None
        for c in countries:
            out = self.work / split / f"cand_raw_{c}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not force:
                log(f"[block] {split}/{c}: cached"); continue
            t0 = time.time()
            s1, pool, roles = self._prepare_country(split, c)
            log(f"[block] {split}/{c}: S1 {len(s1):,} pool {len(pool):,}")
            bc = cfg.block_cfg or B.BlockConfig()
            bc.use_gpu = cfg.use_gpu; bc.n_threads = cfg.n_jobs
            C, _ = B.block_country(s1, pool, bc, log=lambda m: log(f"   {m}"))
            if gt is not None:
                y, P = _labels(gt, s1.eid.to_numpy(), pool.eid.to_numpy(), C.i1.to_numpy(), C.i2.to_numpy())
                C["y"] = y
                log(f"[block] {split}/{c}: pair recall {int(y.sum()) / max(1, len(P)):.5f} ({int(y.sum()):,}/{len(P):,}); "
                    f"{len(C) / len(s1):.1f} cand/S1 ({time.time() - t0:.0f}s)")
                pd.DataFrame(P, columns=["i1", "i2"]).to_parquet(self.work / split / f"truth_{c}.parquet", index=False)
            C.to_parquet(out, index=False)
            pd.DataFrame({"eid": s1.eid.to_numpy(), "role": roles if roles is not None else np.array(["test"] * len(s1), dtype=object)}
                         ).to_parquet(self.work / split / f"s1ids_{c}.parquet", index=False)
            pd.DataFrame({"eid": pool.eid.to_numpy()}).to_parquet(self.work / split / f"poolids_{c}.parquet", index=False)
            del s1, pool, C; gc.collect()

    # -- stage: prune -------------------------------------------------------------------------
    def prune(self, split: str, countries: Optional[List[str]] = None, force: bool = False) -> None:
        cfg = self.cfg
        if countries is None:
            countries = _countries(self.work, split, "cand_raw_*.parquet")
        model_path = self.work / "train" / "pruner"
        if not GBDT.exists(model_path):
            assert split == "train", "the pruner must be trained first (run the train split)"
            Xs, ys, gs = [], [], []
            for c in countries:
                s1, pool, roles = self._prepare_country(split, c)
                C = pd.read_parquet(self.work / split / f"cand_raw_{c}.parquet")
                m = roles[C.i1.to_numpy()] == "tab"
                Cm = C[m].reset_index(drop=True)
                Xs.append(prune_features(s1, pool, Cm)); ys.append(Cm.y.to_numpy()); gs.append(Cm.i1.to_numpy() + 10_000_000 * len(Xs))
                del s1, pool, C, Cm; gc.collect()
            X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); g = np.concatenate(gs)
            log(f"[prune] fitting pruner on {len(X):,} tab-slice candidates ({int(y.sum()):,} positives)")
            pruner = _fit_gbdt(X, y, g, cfg, rounds=600, leaves=63, seed_shift=3, min_child=100)
            pruner.save(model_path)
            del X, Xs; gc.collect()
        pruner = GBDT.load(model_path)
        for c in countries:
            out = self.work / split / f"cand_{c}.parquet"
            if out.exists() and not force:
                log(f"[prune] {split}/{c}: cached"); continue
            t0 = time.time()
            s1, pool, roles = self._prepare_country(split, c)
            C = pd.read_parquet(self.work / split / f"cand_raw_{c}.parquet")
            X = prune_features(s1, pool, C)
            pp = pruner.predict(X); del X
            keep = prune_mask(C.i1.to_numpy(), pp, tau=cfg.prune_tau, keep_top=cfg.prune_keep_top, cap=cfg.prune_cap)
            C["pp"] = pp
            Ck = C[keep].reset_index(drop=True)
            msg = f"[prune] {split}/{c}: {len(C):,} -> {len(Ck):,} ({len(Ck) / max(1, len(s1)):.2f}/S1)"
            if "y" in C:
                msg += f"; true kept {int(Ck.y.sum()):,}/{int(C.y.sum()):,} ({Ck.y.sum() / max(1, C.y.sum()):.5f} of retrieved)"
            log(msg + f" ({time.time() - t0:.0f}s)")
            if cfg.expand_siblings:
                E = sibling_expansion(s1, pool, Ck, n_threads=cfg.n_jobs, log=log, use_gpu=cfg.use_gpu)
                if len(E):
                    E = E.drop(columns=["sib_cos"])
                    Xe = prune_features(s1, pool, E)
                    E["pp"] = pruner.predict(Xe); del Xe
                    if "y" in Ck:
                        gt_ = read_ground_truth(self.data_dir / "train" / "train_ground_truth.tsv")
                        ye, _ = _labels(gt_, s1.eid.to_numpy(), pool.eid.to_numpy(), E.i1.to_numpy(), E.i2.to_numpy())
                        E["y"] = ye
                        log(f"[expand] {split}/{c}: {len(E):,} sibling candidates, {int(ye.sum()):,} true")
                    Ck = pd.concat([Ck, E[Ck.columns]], ignore_index=True)
                    Ck = Ck.sort_values(["i1", "i2"]).reset_index(drop=True)
            Ck.to_parquet(out, index=False)
            del s1, pool, C, Ck; gc.collect()

    # -- stage: rate tables + stage-1 features ------------------------------------------------
    def feat(self, split: str, countries: Optional[List[str]] = None, force: bool = False) -> None:
        cfg = self.cfg
        if countries is None:
            countries = _countries(self.work, split, "cand_*.parquet")
        rt_path = self.work / "tables" / "rates.json"
        if rt_path.exists():
            tables = RateTables.load(rt_path)
        else:
            assert split == "train"
            tables = RateTables()
            allS, allQ, ii1, ii2, yy = [], [], [], [], []
            off1 = off2 = 0
            for c in countries:
                s1, pool, roles = self._prepare_country(split, c)
                C = pd.read_parquet(self.work / split / f"cand_{c}.parquet", columns=["i1", "i2", "y"])
                m = roles[C.i1.to_numpy()] == "tab"
                ii1.append(C.i1.to_numpy()[m] + off1); ii2.append(C.i2.to_numpy()[m] + off2); yy.append(C.y.to_numpy()[m])
                allS.append(pd.DataFrame({"n_core2": s1["n_core2"].astype(object).tolist(), "n_legal": s1["n_legal"].tolist()}))
                allQ.append(pd.DataFrame({"n_core2": pool["n_core2"].astype(object).tolist(), "n_legal": pool["n_legal"].tolist()}))
                off1 += len(s1); off2 += len(pool)
                del s1, pool, C; gc.collect()
            tables.fit(pd.concat(allS, ignore_index=True), pd.concat(allQ, ignore_index=True), np.concatenate(ii1), np.concatenate(ii2), np.concatenate(yy))
            tables.save(rt_path)
            log(f"[rates] extra {len(tables.extra):,} missing {len(tables.missing):,} legal {len(tables.legal):,} keys")
            del allS, allQ; gc.collect()
        for c in countries:
            out = self.work / split / f"feat_{c}.parquet"
            if out.exists() and not force:
                log(f"[feat] {split}/{c}: cached"); continue
            t0 = time.time()
            s1, pool, roles = self._prepare_country(split, c)
            C = pd.read_parquet(self.work / split / f"cand_{c}.parquet")
            state = {"writer": None, "n": 0, "cols": 0}

            def writer(df: pd.DataFrame):
                tbl = pa.Table.from_pandas(df, preserve_index=False)
                if state["writer"] is None:
                    state["writer"] = pq.ParquetWriter(out, tbl.schema, compression="zstd")
                    state["cols"] = df.shape[1]
                state["writer"].write_table(tbl); state["n"] += len(df)

            stage1_features(s1, pool, C, tables, chunk=cfg.feat_chunk, writer=writer, log=lambda m: log(f"   {m}"))
            if state["writer"] is not None:
                state["writer"].close()
            log(f"[feat] {split}/{c}: {state['n']:,} pairs x {state['cols']} features ({time.time() - t0:.0f}s)")
            del s1, pool, C; gc.collect()

    # -- stage: stage 1 -----------------------------------------------------------------------
    def stage1(self, countries: Optional[List[str]] = None, force: bool = False) -> None:
        cfg = self.cfg; split = "train"
        if countries is None:
            countries = _countries(self.work, split, "feat_*.parquet")
        model_path = self.work / "train" / "stage1"
        Xs, ys, gs, sizes = [], [], [], []
        for k, c in enumerate(countries):
            roles = pd.read_parquet(self.work / split / f"s1ids_{c}.parquet").role.to_numpy()
            C = pd.read_parquet(self.work / split / f"cand_{c}.parquet", columns=["i1", "y"])
            m = roles[C.i1.to_numpy()] == "train"
            Xs.append(read_feature_rows(self.work / split / f"feat_{c}.parquet", m)); ys.append(C.y.to_numpy()[m])
            gs.append(C.i1.to_numpy()[m] + 10_000_000 * k)
            sizes.append(int(m.sum()))
            del C; gc.collect()
        X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); g = np.concatenate(gs)
        del Xs; gc.collect()
        self.feature_names = list(X.columns)
        (self.work / "train" / "features1.json").write_text(json.dumps(self.feature_names))
        fit_idx = np.arange(len(X))
        if len(X) > cfg.max_train_pairs:
            rng = np.random.default_rng(cfg.seed)
            fit_idx = np.sort(rng.choice(len(X), cfg.max_train_pairs, replace=False))
        log(f"[stage1] {len(X):,} train-slice pairs ({int(y.sum()):,} positives) x {X.shape[1]} features; fitting on {len(fit_idx):,}")
        if not GBDT.exists(model_path) or force:
            bst = _fit_gbdt(X.iloc[fit_idx], y[fit_idx], g[fit_idx], cfg, cfg.stage1_rounds)
            bst.save(model_path)
            log(f"[stage1] model ({bst.backend}): {bst.best_iteration} trees")
        bst = GBDT.load(model_path)
        oof_paths = [self.work / split / f"stage1_oof_{c}.npy" for c in countries]
        if not all(p.exists() for p in oof_paths) or force:
            oof = _oof_predict(X, y, g, cfg, cfg.stage1_rounds, max_fit=cfg.max_train_pairs // 2)
            off = 0
            for c, n, p in zip(countries, sizes, oof_paths):
                np.save(p, oof[off:off + n]); off += n
            log("[stage1] OOF predictions written")
        del X; gc.collect()
        for c in countries:
            out = self.work / split / f"p1_{c}.npy"
            if out.exists() and not force:
                continue
            np.save(out, predict_chunked(bst, self.work / split / f"feat_{c}.parquet", self.feature_names)); gc.collect()
        log("[stage1] p1 written")

    # -- stage: stage 2 -----------------------------------------------------------------------
    def stage2(self, countries: Optional[List[str]] = None, force: bool = False) -> None:
        cfg = self.cfg; split = "train"
        if countries is None:
            countries = _countries(self.work, split, "feat_*.parquet")
        feats1 = json.loads((self.work / "train" / "features1.json").read_text())
        model_path = self.work / "train" / "stage2"
        Xs, ys, gs = [], [], []
        for k, c in enumerate(countries):
            s1, pool, roles = self._prepare_country(split, c)
            C = pd.read_parquet(self.work / split / f"cand_{c}.parquet", columns=["i1", "i2", "y"])
            p1 = np.load(self.work / split / f"p1_{c}.npy")
            m_train = roles[C.i1.to_numpy()] == "train"
            p1_used = p1.copy()
            oof_path = self.work / split / f"stage1_oof_{c}.npy"
            if oof_path.exists():
                p1_used[m_train] = np.load(oof_path)
            F2 = stage2_features(s1, pool, C, p1_used)
            F2.to_parquet(self.work / split / f"feat2_{c}.parquet", index=False)
            F1 = read_feature_rows(self.work / split / f"feat_{c}.parquet", m_train, columns=feats1)
            X = pd.concat([F1, F2[m_train].reset_index(drop=True)], axis=1)
            Xs.append(X); ys.append(C.y.to_numpy()[m_train]); gs.append(C.i1.to_numpy()[m_train] + 10_000_000 * k)
            del F1, F2, s1, pool, C; gc.collect()
        X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); g = np.concatenate(gs)
        del Xs; gc.collect()
        self.feature_names2 = list(X.columns)
        (self.work / "train" / "features2.json").write_text(json.dumps(self.feature_names2))
        fit_idx = np.arange(len(X))
        if len(X) > cfg.max_train_pairs:
            rng = np.random.default_rng(cfg.seed + 1)
            fit_idx = np.sort(rng.choice(len(X), cfg.max_train_pairs, replace=False))
        log(f"[stage2] training on {len(fit_idx):,} pairs x {X.shape[1]} features")
        if not GBDT.exists(model_path) or force:
            bst = _fit_gbdt(X.iloc[fit_idx], y[fit_idx], g[fit_idx], cfg, cfg.stage2_rounds, seed_shift=5)
            bst.save(model_path)
            log(f"[stage2] model ({bst.backend}): {bst.best_iteration} trees")
        del X; gc.collect()
        bst = GBDT.load(model_path)
        for c in countries:
            F2 = pd.read_parquet(self.work / split / f"feat2_{c}.parquet")
            np.save(self.work / split / f"p2_{c}.npy", predict_chunked(bst, self.work / split / f"feat_{c}.parquet", self.feature_names2, extra=F2))
            del F2; gc.collect()
        log("[stage2] p2 written")

    # -- stage: tune (calibration + decision) on the val slice --------------------------------
    def tune(self, countries: Optional[List[str]] = None) -> dict:
        cfg = self.cfg; split = "train"
        if countries is None:
            countries = _countries(self.work, split, "p2_*.npy")
        rows = []
        for c in countries:
            roles = pd.read_parquet(self.work / split / f"s1ids_{c}.parquet").role.to_numpy()
            C = pd.read_parquet(self.work / split / f"cand_{c}.parquet", columns=["i1", "i2", "y"])
            p2 = np.load(self.work / split / f"p2_{c}.npy")
            T = pd.read_parquet(self.work / split / f"truth_{c}.parquet")
            rows.append((c, roles, len(roles), C, p2, T))
        pv = np.concatenate([r[4][r[1][r[3].i1.to_numpy()] == "val"] for r in rows])
        yv = np.concatenate([r[3].y.to_numpy()[r[1][r[3].i1.to_numpy()] == "val"] for r in rows])
        cal = Calibrator().fit(pv, yv)
        report: Dict[str, object] = {"calibration_points": int(len(pv)), "variants": {}}
        best = None

        def evaluate(method: str, **kw) -> Tuple[float, Dict[str, float]]:
            scores = []
            for (c, roles, n_s1, C, p2, T) in rows:
                pc_ = cal(p2)
                mask = decide(C.i1.to_numpy(), C.i2.to_numpy(), pc_, method=method, seed=cfg.seed, **kw)
                val_s1 = np.flatnonzero(roles == "val")
                f = per_entity_scores(n_s1, C.i1.to_numpy()[mask], C.i2.to_numpy()[mask], T.i1.to_numpy(), T.i2.to_numpy())
                scores.append((c, float(f[val_s1].mean()), len(val_s1)))
            total = sum(s * n for _, s, n in scores) / sum(n for _, _, n in scores)
            return total, {c: s for c, s, _ in scores}

        for lam in (0.0, 0.01, 0.02, 0.04):
            total, by_c = evaluate("ef", lam_miss=lam, n_samples=128)
            report["variants"][f"ef_lam_{lam}"] = {"macro_f05": total, "by_country": by_c}
            log(f"[tune] EF lam={lam}: {total:.5f} " + " ".join(f"{c}={s:.5f}" for c, s in by_c.items()))
            if best is None or total > best[1]:
                best = (("ef", lam), total)
        for thr in (0.6, 0.65, 0.7, 0.75, 0.8):
            total, by_c = evaluate("thr", thr=thr)
            report["variants"][f"thr_{thr}"] = {"macro_f05": total, "by_country": by_c}
            log(f"[tune] THR {thr}: {total:.5f} " + " ".join(f"{c}={s:.5f}" for c, s in by_c.items()))
            if total > best[1]:
                best = (("thr", thr), total)
        method, param = best[0]
        decision = {"method": method, "lam_miss": param if method == "ef" else 0.02, "thr": param if method == "thr" else 0.7,
                    "val_macro_f05": best[1], "calibrator": cal.to_json()}
        (self.work / "train" / "decision.json").write_text(json.dumps(decision))
        (self.work / "train" / "tune_report.json").write_text(json.dumps(report, indent=1))
        log(f"[tune] best: {method} {param} -> {best[1]:.5f}")
        return report

    def train_all(self) -> None:
        self.norm(("train",))
        self.tables()
        self.block("train"); self.prune("train"); self.feat("train")
        self.stage1(); self.stage2(); self.tune()


# ---------------------------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------------------------
class Predictor(Trainer):
    def predict(self, countries: Optional[List[str]] = None, out_dir: Optional[Path] = None, force: bool = False) -> dict:
        split = "test"
        out_dir = Path(out_dir) if out_dir else self.work / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        if countries is None:
            countries = sorted(pq.read_table(self.work / "norm" / "test_s1.parquet", columns=["country"]).to_pandas().country.unique())
        self.block(split, countries, force=force)
        self.prune(split, countries, force=force)
        self.feat(split, countries, force=force)
        feats1 = json.loads((self.work / "train" / "features1.json").read_text())
        feats2 = json.loads((self.work / "train" / "features2.json").read_text())
        bst1 = GBDT.load(self.work / "train" / "stage1")
        bst2 = GBDT.load(self.work / "train" / "stage2")
        dec = json.loads((self.work / "train" / "decision.json").read_text())
        cal = Calibrator.from_json(dec["calibrator"])
        diag: Dict[str, dict] = {}
        all_s1 = pq.read_table(self.work / "norm" / "test_s1.parquet", columns=["eid"]).to_pandas().eid.to_numpy()
        pos_all = {e: i for i, e in enumerate(all_s1.tolist())}
        cand_lists: Dict[int, List[str]] = {}; match_lists: Dict[int, List[str]] = {}
        for c in countries:
            t0 = time.time()
            s1, pool, _ = self._prepare_country(split, c)
            C = pd.read_parquet(self.work / split / f"cand_{c}.parquet")
            fpath = self.work / split / f"feat_{c}.parquet"
            p1 = predict_chunked(bst1, fpath, feats1)
            F2 = stage2_features(s1, pool, C, p1)
            p2 = predict_chunked(bst2, fpath, feats2, extra=F2)
            del F2; gc.collect()
            pc_ = cal(p2)
            mask = decide(C.i1.to_numpy(), C.i2.to_numpy(), pc_, method=dec["method"], thr=dec["thr"], lam_miss=dec["lam_miss"], seed=self.cfg.seed)
            pd.DataFrame({"i1": C.i1, "i2": C.i2, "p1": p1, "p2": p2, "pc": pc_, "match": mask}).to_parquet(self.work / split / f"scored_{c}.parquet", index=False)
            s1_eids = s1.eid.to_numpy(); pool_eids = pool.eid.to_numpy()
            i1 = C.i1.to_numpy(); i2 = C.i2.to_numpy()
            for a, b in zip(i1.tolist(), i2.tolist()):
                cand_lists.setdefault(pos_all[s1_eids[a]], []).append(pool_eids[b])
            for a, b in zip(i1[mask].tolist(), i2[mask].tolist()):
                match_lists.setdefault(pos_all[s1_eids[a]], []).append(pool_eids[b])
            n_pred = np.bincount(i1[mask], minlength=len(s1))
            src = pool.src.to_numpy()[i2]
            diag[c] = {"s1": int(len(s1)), "pool": int(len(pool)), "candidates": int(len(C)), "cand_per_s1": float(len(C) / len(s1)),
                       "pred_per_s1": float(n_pred.mean()), "empty_share": float((n_pred == 0).mean()),
                       "pred_s2_per_s1": float(mask[src == 2].sum() / len(s1)), "pred_s3_per_s1": float(mask[src == 3].sum() / len(s1)),
                       "mean_calibrated_p_of_matches": float(pc_[mask].mean()) if mask.any() else 0.0,
                       "share_matches_p_below_0.8": float((pc_[mask] < 0.8).mean()) if mask.any() else 0.0,
                       "seconds": round(time.time() - t0, 1)}
            log(f"[predict] {c}: {diag[c]}")
            del s1, pool, C; gc.collect()
        with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc, \
             open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fm:
            fc.write("source1_entity_id\tcandidate_entity_ids\n"); fm.write("source1_entity_id\tmatched_entity_ids\n")
            for k, e in enumerate(all_s1.tolist()):
                fc.write(f"{e}\t{','.join(cand_lists.get(k, []))}\n")
                fm.write(f"{e}\t{','.join(match_lists.get(k, []))}\n")
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=1))
        log(f"[predict] wrote {out_dir / 'matching_results.tsv'} and candidate_pairs.tsv")
        return diag
