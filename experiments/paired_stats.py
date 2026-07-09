"""Paired statistical comparison of the two frameworks over the 100 shared
questions, for cost, running time, and accuracy (RAGAS average).

scored_runs.csv already has one row per (question_id, framework), with each
metric's 5 repeats averaged into a single *_avg column (cost_thb_avg,
latency_s_avg, ragas_average_avg — see run_experiment.py / ragas_eval.py).
This script pairs each question's langgraph row with its skillsmd row and
runs a paired test across the 100 pairs, because the same question answered
by both frameworks is naturally paired data (Wilcoxon signed-rank, not an
unpaired t-test) — see README, "Hypothesis" for why a paired test is the
right choice here.

Usage:
    python -m experiments.paired_stats

Requires: experiments/results/scored_runs.csv from experiments/ragas_eval.py
Output:   experiments/results/paired_stats_summary.csv + printed report
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shared.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "paired_stats.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

SCORED_CSV = RESULTS_DIR / "scored_runs.csv"
SUMMARY_CSV = RESULTS_DIR / "paired_stats_summary.csv"

METRICS = {
    "cost_thb_avg": "Cost (THB)",
    "latency_s_avg": "Running time (s)",
    "ragas_average_avg": "RAGAS average (accuracy)",
}

ALPHA = 0.05
N_BOOTSTRAP = 10_000
RNG_SEED = 42


def bootstrap_ci_of_mean_diff(diffs: np.ndarray, n_boot=N_BOOTSTRAP, seed=RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def matched_pairs_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for Wilcoxon signed-rank: r = W_stat-derived rank-biserial
    correlation, robust to the same zero-diff handling as scipy's test."""
    diffs = x - y
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = pd.Series(np.abs(nz)).rank().values
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total > 0 else 0.0


def main():
    if not SCORED_CSV.exists():
        raise FileNotFoundError(f"{SCORED_CSV} not found — run experiments/ragas_eval.py first")

    df = pd.read_csv(SCORED_CSV)

    # One row per (question_id, framework) already; just pivot langgraph vs
    # skillsmd onto the same question_id so each question is a matched pair.
    pivot = {m: df.pivot(index="question_id", columns="framework", values=m) for m in METRICS}

    summary_rows = []
    log.info("=" * 78)
    log.info("PAIRED COMPARISON — 100 questions, langgraph vs skillsmd")
    log.info("=" * 78)

    for metric_col, label in METRICS.items():
        table = pivot[metric_col].dropna()
        lg = table["langgraph"].values
        sk = table["skillsmd"].values
        n = len(table)

        diffs = sk - lg  # skillsmd minus langgraph
        mean_lg, mean_sk = lg.mean(), sk.mean()
        mean_diff = diffs.mean()
        pct_diff = (mean_diff / mean_lg * 100) if mean_lg != 0 else float("nan")

        try:
            stat, p_value = wilcoxon(sk, lg, zero_method="wilcox")
        except ValueError:
            stat, p_value = float("nan"), float("nan")

        effect = matched_pairs_rank_biserial(sk, lg)
        ci_lo, ci_hi = bootstrap_ci_of_mean_diff(diffs)
        significant = p_value < ALPHA if not np.isnan(p_value) else False

        log.info("")
        log.info("%s  (n=%d paired questions)", label, n)
        log.info("  langgraph mean = %.5f   skillsmd mean = %.5f", mean_lg, mean_sk)
        log.info("  mean diff (skillsmd - langgraph) = %+.5f  (%+.2f%%)", mean_diff, pct_diff)
        log.info("  95%% bootstrap CI of mean diff    = [%+.5f, %+.5f]", ci_lo, ci_hi)
        log.info("  Wilcoxon signed-rank: stat=%.2f, p=%.5f", stat, p_value)
        log.info("  matched-pairs rank-biserial effect size r = %+.3f", effect)
        verdict = "SIGNIFICANT (p<0.05)" if significant else "NOT significant (p>=0.05)"
        log.info("  -> %s", verdict)

        summary_rows.append({
            "metric": label, "n_pairs": n, "langgraph_mean": mean_lg, "skillsmd_mean": mean_sk,
            "mean_diff_skillsmd_minus_langgraph": mean_diff, "pct_diff": pct_diff,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi, "wilcoxon_stat": stat, "p_value": p_value,
            "rank_biserial_effect_size": effect, "significant_at_0.05": significant,
        })

    log.info("")
    log.info("=" * 78)
    summary_df = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log.info("Summary written to %s", SUMMARY_CSV)
    log.info("Log written to %s", LOG_FILE)


if __name__ == "__main__":
    main()
