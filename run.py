"""Command-line entry point.

Examples
--------
    # full training on the training split, then test prediction (Colab: GPU used for retrieval)
    python run.py train   --data /content/data --work /content/work --jobs 2
    python run.py predict --data /content/data --work /content/work --out /content/output

    # individual stages (all are cached; re-running skips finished work)
    python run.py stage --data ... --work ... --stages norm,tables,block,prune,feat,stage1,stage2,tune

    # validate the outputs with the official checker
    python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber import blocking as B  # noqa: E402
from ber.pipeline import Config, Predictor, Trainer, log  # noqa: E402


def make_cfg(a: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.n_jobs = a.jobs
    cfg.lgb_threads = a.jobs
    cfg.use_gpu = not a.no_gpu
    cfg.hide_frac = a.hide_frac
    cfg.train_frac = a.train_frac
    cfg.max_train_pairs = a.max_train_pairs
    bc = B.BlockConfig()
    bc.max_df_name = a.max_df; bc.max_df_addr = a.max_df; bc.ngram = a.ngram
    cfg.block_cfg = bc
    cfg.prune_tau = a.prune_tau
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    ap.add_argument("command", choices=["train", "predict", "stage"])
    ap.add_argument("--data", required=True, help="folder containing train/ and test/ (the challenge dataset folder)")
    ap.add_argument("--work", required=True, help="working directory for caches and models")
    ap.add_argument("--out", default=None, help="output folder for the two TSVs (predict)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--hide-frac", type=float, default=0.188)
    ap.add_argument("--train-frac", type=float, default=0.40)
    ap.add_argument("--max-train-pairs", type=int, default=6_000_000)
    ap.add_argument("--max-df", type=float, default=0.03)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--prune-tau", type=float, default=0.003)
    ap.add_argument("--stages", default="norm,tables,block,prune,feat,stage1,stage2,tune")
    ap.add_argument("--countries", default=None, help="comma-separated subset (default: all)")
    a = ap.parse_args()
    cfg = make_cfg(a)
    data, work = Path(a.data), Path(a.work)
    countries = a.countries.split(",") if a.countries else None
    if a.command == "train":
        tr = Trainer(data, work, cfg)
        tr.norm(("train", "test")); tr.tables()
        tr.block("train", countries); tr.prune("train", countries); tr.feat("train", countries)
        tr.stage1(countries); tr.stage2(countries); tr.tune(countries)
    elif a.command == "predict":
        pr = Predictor(data, work, cfg)
        pr.norm(("test",))
        pr.predict(countries, out_dir=Path(a.out) if a.out else None)
    else:
        tr = Trainer(data, work, cfg)
        for st in a.stages.split(","):
            log(f"=== stage {st} ===")
            if st == "norm": tr.norm(("train", "test"))
            elif st == "tables": tr.tables()
            elif st == "block": tr.block("train", countries)
            elif st == "prune": tr.prune("train", countries)
            elif st == "feat": tr.feat("train", countries)
            elif st == "stage1": tr.stage1(countries)
            elif st == "stage2": tr.stage2(countries)
            elif st == "tune": tr.tune(countries)
            elif st == "predict": Predictor(data, work, cfg).predict(countries, out_dir=Path(a.out) if a.out else None)
            else: raise SystemExit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
