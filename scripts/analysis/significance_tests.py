#!/usr/bin/env python3
"""
Statistical significance for the pooling comparison (DeepLoc + Meltome),
at a single MATCHED projection dimension d_c (constant across cov/hybrid).

For each task we run the appropriate PER-SEED test, then combine the three
per-seed p-values across seeds with FISHER'S METHOD.

Why Fisher and not pooling the raw predictions: the three seeds share the same
fixed test set, so pooling each protein's predictions across seeds treats the
same proteins as independent observations, inflating the effective sample size
and making the aggregated p-value anti-conservative. Each seed is one
independent experiment, so combine the three independent per-seed p-values.
Fisher: chi2 = -2 sum ln(p), df = 2k.

Why a constant d_c: holding the bottleneck size fixed isolates the pooling
STRUCTURE (mean vs cov vs hybrid) from capacity. Note that mean has no d_c;
cov and hybrid are both taken at the chosen --dc.

Tests:
    DeepLoc (classification): McNemar's exact test on per-protein correctness.
    Meltome (regression):     Wilcoxon signed-rank on |error|, and Williams'
                              test on Spearman R (dependent correlations).

Usage:
    python scripts/significance_tests.py --runs_root <dir> --dc 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.stats as stats


SEEDS = [657, 921, 969]
RESULTS_FILE = "results_array_test_set_after_train.npy"


def find_run(runs_root: Path, task: str, method: str, dc: int, seed: int) -> Path:
    """Locate the run folder by glob, ignoring the timestamp suffix.
    mean has no d_c; cov/hybrid are matched at the given dc."""
    if method == 'mean':
        pattern = f"PoolingFFN_{task}_mean_seed{seed}_*"
    else:
        pattern = f"PoolingFFN_{task}_{method}_dc{dc}_seed{seed}_*"
    matches = [m for m in sorted(runs_root.rglob(pattern)) if (m / RESULTS_FILE).exists()]
    if not matches:
        avail = sorted(p.name for p in runs_root.rglob(f"PoolingFFN_{task}_{method}_*seed{seed}_*"))
        raise FileNotFoundError(
            f"No run for task={task} method={method} dc={dc} seed={seed} "
            f"(pattern {pattern!r}).\n  Available for this method/seed: {avail}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple runs match {pattern!r}: {[m.name for m in matches]}")
    return matches[0]


def load_results(runs_root: Path, task: str, method: str, dc: int, seed: int) -> np.ndarray:
    """Columns are [prediction, target]."""
    return np.load(find_run(runs_root, task, method, dc, seed) / RESULTS_FILE)


def mcnemar_exact(a_correct, b_correct):
    """Exact (binomial) McNemar test on paired correct/incorrect booleans."""
    a_c = np.asarray(a_correct, dtype=bool)
    b_c = np.asarray(b_correct, dtype=bool)
    n10 = int(np.sum(a_c & ~b_c))
    n01 = int(np.sum(~a_c & b_c))
    total = n10 + n01
    if total == 0:
        return n10, n01, 1.0
    p = min(1.0, 2.0 * stats.binom.cdf(min(n10, n01), total, 0.5))
    return n10, n01, p


def williams_test(r12, r13, r23, n):
    """Williams' test comparing two dependent correlations sharing a variable.
    r12 = corr(A, truth), r13 = corr(B, truth), r23 = corr(A, B)."""
    if n <= 3:
        return 0.0, 1.0
    num = (r12 - r13) * np.sqrt(n - 3) * np.sqrt(1 + r23)
    det = 2 * (1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23)
    if det <= 0:
        return 0.0, 1.0
    t = num / np.sqrt(det)
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 3))
    return t, p


def fisher_combine(pvals):
    clipped = [min(max(float(p), 1e-300), 1.0) for p in pvals]
    stat, p = stats.combine_pvalues(clipped, method='fisher')
    return stat, p


def report(title, per_seed_p, labels):
    print(f"\n{title}")
    header = f"{'comparison':<18}" + "".join(f"seed {s:<11}" for s in SEEDS) + "Fisher (all seeds)"
    print(header)
    print("-" * len(header))
    for label in labels:
        ps = per_seed_p[label]
        _, p_agg = fisher_combine(ps)
        cells = "".join(f"{p:<16.3e}" for p in ps)
        sig = "" if p_agg < 0.05 else "  (NS)"
        print(f"{label:<18}{cells}{p_agg:.3e}{sig}")


COMPS = [('Cov vs Mean', 'cov', 'mean'),
         ('Hybrid vs Mean', 'hybrid', 'mean'),
         ('Hybrid vs Cov', 'hybrid', 'cov')]


def run_localization(runs_root: Path, dc: int):
    correct = {}
    for m in ('mean', 'cov', 'hybrid'):
        correct[m] = []
        for seed in SEEDS:
            res = load_results(runs_root, 'loc', m, dc, seed)
            correct[m].append(res[:, 0] == res[:, 1])

    per_seed_p = {label: [] for label, _, _ in COMPS}
    for label, a, b in COMPS:
        for i in range(len(SEEDS)):
            _, _, p = mcnemar_exact(correct[a][i], correct[b][i])
            per_seed_p[label].append(p)

    report(f"=== DeepLoc (McNemar's exact, matched d_c={dc}, Fisher across seeds) ===",
           per_seed_p, [c[0] for c in COMPS])


def run_meltome(runs_root: Path, dc: int):
    p_wilcoxon = {label: [] for label, _, _ in COMPS}
    p_williams = {label: [] for label, _, _ in COMPS}

    for i, seed in enumerate(SEEDS):
        pred, true = {}, None
        for m in ('mean', 'cov', 'hybrid'):
            res = load_results(runs_root, 'meltome', m, dc, seed)
            pred[m] = res[:, 0]
            true = res[:, 1]

        ae = {m: np.abs(pred[m] - true) for m in pred}
        r = {m: stats.spearmanr(pred[m], true)[0] for m in pred}
        r_between = {
            ('cov', 'mean'):    stats.spearmanr(pred['cov'], pred['mean'])[0],
            ('hybrid', 'mean'): stats.spearmanr(pred['hybrid'], pred['mean'])[0],
            ('hybrid', 'cov'):  stats.spearmanr(pred['hybrid'], pred['cov'])[0],
        }
        n = len(true)

        for label, a, b in COMPS:
            _, p_w = stats.wilcoxon(ae[a], ae[b])
            p_wilcoxon[label].append(p_w)
            _, p_wms = williams_test(r[a], r[b], r_between[(a, b)], n)
            p_williams[label].append(p_wms)

    report(f"=== Meltome — Wilcoxon on |error| (matched d_c={dc}, Fisher across seeds) ===",
           p_wilcoxon, [c[0] for c in COMPS])
    report(f"=== Meltome — Williams' on Spearman R (matched d_c={dc}, Fisher across seeds) ===",
           p_williams, [c[0] for c in COMPS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', type=Path, default=Path('runs'),
                        help='Directory containing the PoolingFFN_* run folders.')
    parser.add_argument('--dc', type=int, default=48,
                        help='Matched projection dim for cov and hybrid (mean ignores it).')
    parser.add_argument('--tasks', nargs='+', default=['loc', 'meltome'],
                        choices=['loc', 'meltome'])
    args = parser.parse_args()

    if 'loc' in args.tasks:
        run_localization(args.runs_root, args.dc)
    if 'meltome' in args.tasks:
        run_meltome(args.runs_root, args.dc)


if __name__ == '__main__':
    main()
