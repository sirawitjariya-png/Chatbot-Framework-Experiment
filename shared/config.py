"""Central configuration shared by both framework implementations.

Both frameworks_impl/* modules import from here — never redefine a model
name, timeout, or temperature locally. This is what makes "same LLM" an
enforced fact rather than a claim.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Gemini (OpenAI-compatible endpoint) ------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Add it to your .env file before running experiments."
    )
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- OpenAI (RAGAS judge only — never used to generate answers) -------------
# The judge is deliberately a different model family than the answer model
# (gemini-3.1-flash-lite) to avoid self-preference bias: a judge scoring its
# own family's answers tends to rate them more favorably. See README,
# "Limitations".
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-mini")
JUDGE_EMBED_MODEL = os.getenv("JUDGE_EMBED_MODEL", "text-embedding-3-small")

# --- Model + decoding config -------------------------------------------------
# Single knob for both roles, matched to the graph-based framework's original
# production choice. Both frameworks use the SAME model for the SAME role
# (router vs answer).
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-3.1-flash-lite")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.1-flash-lite")

# Fixed at 0 for the whole experiment. Note temperature=0 on most hosted APIs
# reduces but does NOT eliminate run-to-run variance (MoE routing jitter,
# infra nondeterminism) — this is exactly why the harness runs each question
# 5x per framework rather than trusting a single call. See README, "Why 5x5".
TEMPERATURE = float(os.getenv("EXPERIMENT_TEMPERATURE", "0"))

LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "60"))

# --- Data ---------------------------------------------------------------------
RAW_DIR = os.getenv("RAW_DIR", str(REPO_ROOT / "data" / "raw"))

# --- Gemini pricing (THB per 1M tokens) — edit to match your current billing --
# Used only for cost estimation in experiments/run_experiment.py.
PRICE_INPUT_THB_PER_1M = float(os.getenv("PRICE_INPUT_THB_PER_1M", "3.5"))
PRICE_OUTPUT_THB_PER_1M = float(os.getenv("PRICE_OUTPUT_THB_PER_1M", "14.0"))
PRICE_CACHED_INPUT_THB_PER_1M = float(os.getenv("PRICE_CACHED_INPUT_THB_PER_1M", "0.875"))

# --- Experiment run parameters -------------------------------------------------
N_REPEATS_PER_QUESTION = int(os.getenv("N_REPEATS_PER_QUESTION", "5"))   # answer-generation repeats
N_RAGAS_REPEATS = int(os.getenv("N_RAGAS_REPEATS", "3"))                  # judge repeats per answer
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
