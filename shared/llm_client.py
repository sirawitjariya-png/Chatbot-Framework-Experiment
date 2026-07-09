"""Single LLM call path used by BOTH frameworks.

Both frameworks import `call_llm` from here rather than instantiating their
own OpenAI client. This guarantees identical model, identical temperature,
identical timeout, and one place to log token usage for cost accounting.
"""
import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import (
    GEMINI_API_KEY, GEMINI_BASE_URL, ROUTER_MODEL, ANSWER_MODEL,
    TEMPERATURE, LLM_TIMEOUT_S,
)

log = logging.getLogger(__name__)
_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


@dataclass
class LLMResult:
    content: str | None
    tool_calls: list | None
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_s: float
    model: str
    raw_message: object


def call_llm(messages: list, *, use_router: bool = False, **kw) -> LLMResult:
    """Chat completion call at TEMPERATURE (fixed, shared). Raises on failure.

    Pass use_router=True to use ROUTER_MODEL (the cheap/fast role); otherwise
    ANSWER_MODEL is used. Both are set once in shared/config.py so the two
    frameworks can never silently diverge on model choice.
    """
    model = ROUTER_MODEL if use_router else ANSWER_MODEL
    t0 = time.perf_counter()
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            timeout=LLM_TIMEOUT_S,
            **kw,
        )
    except Exception as e:
        log.error("LLM call failed (model=%s): %s", model, e)
        raise
    latency = time.perf_counter() - t0

    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    cached = 0
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0

    msg = response.choices[0].message
    return LLMResult(
        content=msg.content,
        tool_calls=getattr(msg, "tool_calls", None),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_tokens=cached,
        latency_s=latency,
        model=model,
        raw_message=msg,
    )
