"""Score every row in raw_runs.csv with RAGAS (faithfulness + answer
relevancy), using gpt-5-mini as the judge.

The judge is deliberately a different model family than the answer model
(gemini-3.1-flash-lite generates every answer; gpt-5-mini only ever judges,
never answers) — this avoids the self-preference bias of a model judging
its own family's output. See README, "Limitations".

raw_runs.csv is wide: one row per (question_id, framework), with each of the
N_REPEATS_PER_QUESTION answer-generation repeats in its own numbered column
(answer_1..answer_5, context_1..context_5, ...). Every answer that has
context gets judged once per pass; RAGAS is run N_RAGAS_REPEATS passes over
ALL answers at once (not one answer at a time) so its internal concurrency
(RunConfig.max_workers) parallelizes across questions instead of this script
waiting on one HTTP round-trip at a time. The N_RAGAS_REPEATS judged scores
for a given answer are then averaged — because the judge itself is an LLM
call and fluctuates run to run, independent of any fluctuation in the answer
being judged. That gives one judged score per answer repeat
(faithfulness_1..faithfulness_5, etc.); the final *_avg column is the average
across those 5 already-judge-averaged values, i.e. every score is a mean of
N_RAGAS_REPEATS (3) judge calls x 5 answer repeats = 15 RAGAS calls per metric
per question per framework.

Usage:
    python -m experiments.ragas_eval

Resumable: after every judge pass, progress is checkpointed to
experiments/results/ragas_eval_checkpoint.json (raw per-unit judge scores +
how many of the N_RAGAS_REPEATS passes are done) and scored_runs.csv is
rewritten from that checkpoint. If the run is stopped (Ctrl-C, `kill`, a
crash, closing the machine) before finishing, re-running this exact command
resumes from the next unfinished pass instead of re-scoring — and re-paying
for — passes already completed. The checkpoint is invalidated automatically
if raw_runs.csv changes shape (e.g. a re-run of experiments/run_experiment.py
with different questions).

Requires: raw_runs.csv from experiments/run_experiment.py already exists.
Output: experiments/results/scored_runs.csv
  Same rows as raw_runs.csv, plus for each metric (faithfulness,
  answer_relevancy, ragas_average):
    {metric}_1..5          one already-judge-averaged score per answer repeat
    {metric}_judge_sd_1..5 how much the N_RAGAS_REPEATS judge calls disagreed
                            with each other on that SAME answer (judging noise)
    {metric}_avg            mean across the 5 answer repeats — the final
                            per-question, per-framework number paired_stats.py uses
    {metric}_sd              how much the answer itself varied across the 5
                            generation repeats (LLM-output noise)
"""
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared.config import (
    OPENAI_API_KEY, JUDGE_MODEL, JUDGE_EMBED_MODEL, N_REPEATS_PER_QUESTION,
    N_RAGAS_REPEATS, REPO_ROOT,
)

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "ragas_eval.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

RAW_CSV = RESULTS_DIR / "raw_runs.csv"
SCORED_CSV = RESULTS_DIR / "scored_runs.csv"
CHECKPOINT_JSON = RESULTS_DIR / "ragas_eval_checkpoint.json"

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your .env file before running RAGAS scoring "
        "(gpt-5-mini judge)."
    )


class _ReasoningModelChatOpenAI(ChatOpenAI):
    """gpt-5-mini (a reasoning-family model) only accepts the default
    temperature and rejects any explicit value, including 0. RAGAS always
    sets `.temperature` on the langchain LLM it's given (it's how RAGAS
    controls sampling for its self-consistency repeats), so a plain
    ChatOpenAI would send an unsupported temperature on every call. This
    subclass drops temperature from the outgoing request instead."""

    def _get_request_payload(self, *args, **kwargs) -> dict:
        payload = super()._get_request_payload(*args, **kwargs)
        payload.pop("temperature", None)
        return payload


_judge_llm = _ReasoningModelChatOpenAI(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)
_embeddings = OpenAIEmbeddings(model=JUDGE_EMBED_MODEL, api_key=OPENAI_API_KEY)
_run_config = RunConfig(max_workers=16)


def collect_scoring_units(df: pd.DataFrame) -> list[dict]:
    """One unit per (row, answer repeat) that has both an answer and context.
    Smalltalk/off_topic/no_data/errored repeats have no context and are
    skipped — faithfulness is undefined with no context, so those become
    N/A rather than 0."""
    units = []
    for idx, row in df.iterrows():
        for rep in range(1, N_REPEATS_PER_QUESTION + 1):
            answer = row.get(f"answer_{rep}", "")
            context = row.get(f"context_{rep}", "")
            has_context = isinstance(context, str) and context.strip() != ""
            has_answer = isinstance(answer, str) and answer.strip() != ""
            if has_context and has_answer:
                units.append({"row_idx": idx, "rep": rep, "question": row["question"],
                              "answer": answer, "context": context})
    return units


def run_judge_pass(units: list[dict], judge_rep: int) -> list[dict]:
    """One RAGAS pass over every scoring unit at once. Returns a list aligned
    with `units`, each either {"faithfulness":.., "answer_relevancy":..} or
    None if that unit's evaluation failed."""
    ds = Dataset.from_dict({
        "question": [u["question"] for u in units],
        "answer": [u["answer"] for u in units],
        "contexts": [[u["context"]] for u in units],
    })
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=_judge_llm,
                       embeddings=_embeddings, run_config=_run_config, raise_exceptions=False)
    log.info("Judge pass %d/%d: scored %d answers", judge_rep, N_RAGAS_REPEATS, len(units))
    result_df = result.to_pandas()

    scored = []
    for _, r in result_df.iterrows():
        f, rel = r["faithfulness"], r["answer_relevancy"]
        if pd.isna(f) or pd.isna(rel):
            scored.append(None)
        else:
            scored.append({"faithfulness": float(f), "answer_relevancy": float(rel)})
    return scored


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Sample standard deviation (ddof=1); 0.0 if fewer than 2 values."""
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def units_signature(units: list[dict]) -> list[list[int]]:
    """Identity of the scoring units, independent of judge results — used to
    confirm a checkpoint still matches the current raw_runs.csv before resuming."""
    return [[u["row_idx"], u["rep"]] for u in units]


def save_checkpoint(units: list[dict], unit_scores: list[dict], completed_passes: int) -> None:
    CHECKPOINT_JSON.write_text(json.dumps({
        "units_signature": units_signature(units),
        "completed_passes": completed_passes,
        "n_ragas_repeats": N_RAGAS_REPEATS,
        "unit_scores": unit_scores,
    }))


def load_checkpoint(units: list[dict]) -> tuple[list[dict], int]:
    """Resume support: if a checkpoint exists and still matches the current
    units (same rows/repeats from raw_runs.csv, same N_RAGAS_REPEATS target),
    return its (unit_scores, completed_passes) so main() can skip judge passes
    that were already paid for and run. Otherwise start fresh — e.g. if
    raw_runs.csv changed since the checkpoint was written."""
    empty = [{"faithfulness": [], "answer_relevancy": []} for _ in units]
    if not CHECKPOINT_JSON.exists():
        return empty, 0

    try:
        state = json.loads(CHECKPOINT_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("Checkpoint file unreadable, starting fresh")
        return empty, 0

    if state.get("units_signature") != units_signature(units) or state.get("n_ragas_repeats") != N_RAGAS_REPEATS:
        log.warning("Checkpoint doesn't match current raw_runs.csv/config, starting fresh")
        return empty, 0

    log.info("Resuming from checkpoint: %d/%d judge passes already done",
              state["completed_passes"], N_RAGAS_REPEATS)
    return state["unit_scores"], state["completed_passes"]


def assemble_scored_df(base_df: pd.DataFrame, units: list[dict], unit_scores: list[dict]) -> pd.DataFrame:
    """Build the scored dataframe from whatever judge passes have completed so far.
    Safe to call after every pass (checkpointing) as well as at the end (final)."""
    df = base_df.copy()
    metric_names = ["faithfulness", "answer_relevancy", "ragas_average"]
    per_repeat_cols = {m: [[None] * N_REPEATS_PER_QUESTION for _ in range(len(df))] for m in metric_names}
    per_repeat_judge_sd = {m: [[None] * N_REPEATS_PER_QUESTION for _ in range(len(df))] for m in metric_names}

    for unit, scores in zip(units, unit_scores):
        f_scores, r_scores = scores["faithfulness"], scores["answer_relevancy"]
        if not f_scores:
            continue
        fm, f_sd = mean_sd(f_scores)
        rm, r_sd = mean_sd(r_scores)
        ragas_per_call = [(f + r) / 2 for f, r in zip(f_scores, r_scores)]
        ragas_m, ragas_sd = mean_sd(ragas_per_call)

        col_idx = unit["rep"] - 1
        row_idx = unit["row_idx"]
        per_repeat_cols["faithfulness"][row_idx][col_idx] = round(fm, 4)
        per_repeat_cols["answer_relevancy"][row_idx][col_idx] = round(rm, 4)
        per_repeat_cols["ragas_average"][row_idx][col_idx] = round(ragas_m, 4)
        per_repeat_judge_sd["faithfulness"][row_idx][col_idx] = round(f_sd, 4)
        per_repeat_judge_sd["answer_relevancy"][row_idx][col_idx] = round(r_sd, 4)
        per_repeat_judge_sd["ragas_average"][row_idx][col_idx] = round(ragas_sd, 4)

    for m in metric_names:
        for rep in range(1, N_REPEATS_PER_QUESTION + 1):
            df[f"{m}_{rep}"] = [vals[rep - 1] for vals in per_repeat_cols[m]]
            df[f"{m}_judge_sd_{rep}"] = [vals[rep - 1] for vals in per_repeat_judge_sd[m]]

        avgs, sds = [], []
        for vals in per_repeat_cols[m]:
            present = [v for v in vals if v is not None]
            if present:
                a, s = mean_sd(present)
                avgs.append(round(a, 4))
                sds.append(round(s, 4))
            else:
                avgs.append(None)
                sds.append(None)
        df[f"{m}_avg"] = avgs
        df[f"{m}_sd"] = sds

    return df


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found — run experiments/run_experiment.py first")

    df = pd.read_csv(RAW_CSV)
    units = collect_scoring_units(df)
    log.info("Loaded %d rows (%d scoreable answers out of up to %d), %d judge passes",
              len(df), len(units), len(df) * N_REPEATS_PER_QUESTION, N_RAGAS_REPEATS)

    SCORED_CSV.parent.mkdir(parents=True, exist_ok=True)

    # unit_scores[i] accumulates the N_RAGAS_REPEATS judge scores for units[i].
    # Resumable: if a previous run was stopped partway, pick up from the next
    # judge pass instead of re-scoring (and re-paying for) passes already done.
    unit_scores, completed_passes = load_checkpoint(units)
    if completed_passes >= N_RAGAS_REPEATS:
        log.info("Checkpoint shows all %d judge passes already done — nothing left to score.", N_RAGAS_REPEATS)
    elif units:
        for judge_rep in range(completed_passes + 1, N_RAGAS_REPEATS + 1):
            pass_scores = run_judge_pass(units, judge_rep)
            for i, s in enumerate(pass_scores):
                if s is not None:
                    unit_scores[i]["faithfulness"].append(s["faithfulness"])
                    unit_scores[i]["answer_relevancy"].append(s["answer_relevancy"])

            # Checkpoint after every judge pass so stopping the run (Ctrl-C, a
            # crash, or `kill`) doesn't lose already-paid-for judging — both the
            # resumable JSON state and the human-readable scored_runs.csv are
            # updated after every single pass, not just at the very end.
            save_checkpoint(units, unit_scores, judge_rep)
            checkpoint_df = assemble_scored_df(df, units, unit_scores)
            checkpoint_df.to_csv(SCORED_CSV, index=False)
            log.info("Checkpoint saved after judge pass %d/%d -> %s", judge_rep, N_RAGAS_REPEATS, SCORED_CSV)

    final_df = assemble_scored_df(df, units, unit_scores)
    final_df.to_csv(SCORED_CSV, index=False)
    log.info("Done. Scored results written to %s", SCORED_CSV)
    log.info("Log written to %s", LOG_FILE)


if __name__ == "__main__":
    main()
