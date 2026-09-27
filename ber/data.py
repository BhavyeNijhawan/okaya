"""Loading the challenge TSVs and producing normalized parquet tables (checkpointed).

Layout produced under <work>/norm/:
    {split}_s1.parquet      one row per Source-1 record   (row id = position, 0..n1-1)
    {split}_pool.parquet    one row per Source-2/3 record (row id = position, S2 rows first, then S3)
Both carry the raw columns plus the normalized views from `normalize.py` (lists are space-joined).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .normalize import ADDR_COLS, NAME_COLS, normalize_country, normalize_records

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a challenge TSV (tab separated, no quoting) into a pandas frame of Python strings."""
    tbl = pacsv.read_csv(
        str(path),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in RAW_COLS}, strings_can_be_null=False),
        read_options=pacsv.ReadOptions(block_size=1 << 24),
    )
    df = tbl.to_pandas()
    for c in RAW_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df[RAW_COLS]


def read_ground_truth(path: Path) -> Dict[str, List[str]]:
    gt: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\r\n").partition("\t")
            gt[a] = [x for x in b.split(",") if x]
    return gt


def _norm_chunk_to_parquet(args) -> str:
    """Worker: normalize one chunk and write it as parquet (avoids shipping millions of strings back)."""
    names, addrs, eids, countries, out_path = args
    rn, ra = normalize_records(names, addrs)
    tbl = {"eid": pa.array(eids, type=pa.string()),
           "country": pa.array([normalize_country(c) for c in countries], type=pa.string()),
           "raw_name": pa.array(names, type=pa.string()),
           "raw_addr": pa.array(addrs, type=pa.string())}
    for k in NAME_COLS:
        if k == "ntok":
            tbl["n_" + k] = pa.array(rn[k], type=pa.int16())
        else:
            tbl["n_" + k] = pa.array(rn[k], type=pa.string())
    for k in ADDR_COLS:
        if k == "comps":
            tbl["a_comps"] = pa.array(["|".join(x) for x in ra[k]], type=pa.string())
        elif k in ("hn_runs", "nums"):
            tbl["a_" + k] = pa.array([" ".join(x) for x in ra[k]], type=pa.string())
        elif k in ("has_hn_kw", "ncomp"):
            tbl["a_" + k] = pa.array(ra[k], type=pa.int16())
        else:
            tbl["a_" + k] = pa.array(ra[k], type=pa.string())
    tbl["addr_empty"] = pa.array([1 if not x else 0 for x in ra["comps"]], type=pa.int8())
    pq.write_table(pa.table(tbl), out_path, compression="zstd")
    return out_path


def normalize_to_parquet(df: pd.DataFrame, out_path: Path, n_jobs: int = 1, chunk: int = 100_000, log=print,
                         extra_cols: Dict[str, np.ndarray] | None = None) -> None:
    """Normalize a raw frame in parallel chunks; workers write parquet parts, merged into out_path."""
    t0 = time.time()
    n = len(df)
    tmp_dir = out_path.parent / (out_path.stem + "_parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    names = df["business_name"].tolist(); addrs = df["business_address"].tolist()
    eids = df["entity_id"].tolist(); ctry = df["country"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], eids[i:i + chunk], ctry[i:i + chunk],
             str(tmp_dir / f"part_{i // chunk:05d}.parquet")) for i in range(0, n, chunk)]
    del names, addrs, eids, ctry
    paths: List[str] = []
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for j, pth in enumerate(ex.map(_norm_chunk_to_parquet, jobs, chunksize=1)):
                paths.append(pth)
                if (j + 1) % 10 == 0:
                    log(f"    normalized {min((j + 1) * chunk, n):,}/{n:,} ({time.time() - t0:.0f}s)")
    else:
        for j, job in enumerate(jobs):
            paths.append(_norm_chunk_to_parquet(job))
            if (j + 1) % 10 == 0:
                log(f"    normalized {min((j + 1) * chunk, n):,}/{n:,} ({time.time() - t0:.0f}s)")
    tables = [pq.read_table(pth) for pth in paths]
    tbl = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if extra_cols:
        for k, v in extra_cols.items():
            tbl = tbl.append_column(k, pa.array(v))
    pq.write_table(tbl, out_path, compression="zstd", row_group_size=200_000)
    for pth in paths:
        os.remove(pth)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    log(f"    wrote {out_path.name}: {n:,} rows ({time.time() - t0:.0f}s)")


def build_norm_tables(data_dir: Path, work: Path, split: str, n_jobs: int = 1, log=print, force: bool = False) -> Tuple[Path, Path]:
    """Normalize S1 and pool (S2+S3) of a split into parquet files. Skips work already done."""
    out_dir = work / "norm"
    out_dir.mkdir(parents=True, exist_ok=True)
    p1, pp = out_dir / f"{split}_s1.parquet", out_dir / f"{split}_pool.parquet"
    if p1.exists() and pp.exists() and not force:
        log(f"[norm] {split}: cached")
        return p1, pp
    t0 = time.time()
    if not p1.exists() or force:
        s1 = read_tsv(data_dir / split / f"{split}_source1.tsv")
        log(f"[norm] {split} S1: {len(s1):,} rows read ({time.time() - t0:.0f}s)")
        normalize_to_parquet(s1, p1, n_jobs=n_jobs, log=log, extra_cols={"src": np.full(len(s1), 1, dtype=np.int8)})
        del s1
    s2 = read_tsv(data_dir / split / f"{split}_source2.tsv")
    s3 = read_tsv(data_dir / split / f"{split}_source3.tsv")
    n_s2 = len(s2)
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    log(f"[norm] {split} pool: {len(pool):,} rows read ({time.time() - t0:.0f}s)")
    src = np.where(np.arange(len(pool)) < n_s2, 2, 3).astype(np.int8)
    normalize_to_parquet(pool, pp, n_jobs=n_jobs, log=log, extra_cols={"src": src})
    log(f"[norm] {split}: done ({time.time() - t0:.0f}s)")
    return p1, pp


def load_norm(work: Path, split: str, columns: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_parquet(work / "norm" / f"{split}_s1.parquet", columns=columns)
    pool = pd.read_parquet(work / "norm" / f"{split}_pool.parquet", columns=columns)
    return s1, pool


def gt_pairs(gt: Dict[str, List[str]], s1_eids: np.ndarray, pool_eids: np.ndarray) -> np.ndarray:
    """Ground truth as an int array of (s1_row, pool_row) pairs (ids missing from the tables are dropped)."""
    s1_pos = {e: i for i, e in enumerate(s1_eids.tolist())}
    pool_pos = {e: i for i, e in enumerate(pool_eids.tolist())}
    out = []
    for a, lst in gt.items():
        ia = s1_pos.get(a)
        if ia is None:
            continue
        for b in lst:
            ib = pool_pos.get(b)
            if ib is not None:
                out.append((ia, ib))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)
