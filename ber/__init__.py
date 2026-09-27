"""Business Entity Resolution pipeline (Amazon ML Challenge 2026).

Modules
-------
normalize   record-level text normalization and light address parsing
geo         learned region/state alias tables and blocking partitions
blocking    candidate generation (exact keys + char n-gram TF-IDF top-K), pruner
features    pairwise + context + sibling features
train       training universe construction, model fitting, calibration, decision tuning
decide      per-entity expected-F0.5 subset selection with pool exclusivity
evaluate    official macro F0.5
predict     end-to-end inference writing candidate_pairs.tsv / matching_results.tsv
"""
__version__ = "2.0.0"
