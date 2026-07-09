"""Single-agent (no graph) implementation of the same dental FAQ workflow.

The FULL skills.md (built from shared/prompts.py — see skills_builder.py) is
sent as the system prompt on EVERY LLM call in this module, whether the step
needs the routing rules, the answering rules, or neither. That is the
independent variable under test: same instruction text as the graph, but
delivered as one large always-on prompt instead of small per-step prompts.

    1 LLM call  — smalltalk / off_topic (agent answers directly, no tool call)
    2 LLM calls — treatment / general (agent calls load_treatment_files, then
                  writes the final answer from the returned CONTEXT)
    0 LLM calls — no_data (mechanical: tool returned no usable content)
"""
import json
import logging

from shared.llm_client import call_llm
from shared.file_loader import load_files, build_context_text, get_index
from shared.prompts import FILE_CATALOG, GENERAL_INFO_FILE_NUMBER, NO_DATA_TH, NO_DATA_EN, is_thai
from .skills_builder import build_skills_md

log = logging.getLogger(__name__)

LOAD_TREATMENT_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "load_treatment_files",
        "description": (
            "Load reference documents about hospital dental treatments, procedures, prices, "
            "or general hospital info. Call this before answering any question that needs "
            "factual information from hospital records. Never call this for greetings, "
            "smalltalk, or off-topic questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "File numbers (1-12) relevant to the question — use your judgement on "
                        "how many (usually 1-2, only include more if the question genuinely "
                        "spans several treatments). Empty array is fine for general hospital "
                        "info — the pricing/general-info file is always included automatically."
                    ),
                },
            },
            "required": ["file_numbers"],
        },
    },
}


def _catalog_text() -> str:
    """Live catalog string — recomputed each call so a new file in data/raw/
    is selectable immediately, matching the original skills-prompt-based
    framework's behavior."""
    index = get_index()
    return "\n".join(
        f"{num}. {FILE_CATALOG.get(num, f'File {num}')}"
        for num in sorted(index.keys())
        if num != GENERAL_INFO_FILE_NUMBER
    )


def _tool_call_to_dict(call) -> dict:
    if hasattr(call, "model_dump"):
        return call.model_dump()
    return {"id": call.id, "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments}}


def ask(question: str, history: list | None = None) -> dict:
    """Single-turn ask. Returns {answer, trace, metrics}."""
    history = history or []
    trace: list = []
    metrics: list = []
    is_first = len(history) == 0

    system_content = build_skills_md(_catalog_text())
    user_turn = f"(meta: is_first_message={is_first})\n{question}"
    messages = [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": user_turn},
    ]

    result = call_llm(messages, use_router=True, tools=[LOAD_TREATMENT_FILES_TOOL],
                       tool_choice="auto", max_tokens=1024)
    metrics.append({"node": "route_or_direct", "model": result.model, "in_tok": result.input_tokens,
                     "out_tok": result.output_tokens, "cached_tok": result.cached_tokens,
                     "latency_s": result.latency_s})

    tool_calls = result.tool_calls
    if not tool_calls:
        answer = result.content or ""
        trace.append({"node": "direct_reply"})
        return {"answer": answer, "trace": trace, "metrics": metrics, "route": None, "files": [], "context": ""}

    call = tool_calls[0]
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    file_numbers = [n for n in (args.get("file_numbers") or []) if isinstance(n, int)]

    load_result = load_files(file_numbers)
    trace.append(load_result["trace_entry"])

    thai = is_thai(question)
    if not load_result["has_data"]:
        trace.append({"node": "no_data"})
        return {"answer": NO_DATA_TH if thai else NO_DATA_EN, "trace": trace, "metrics": metrics,
                "route": "treatment_or_general", "files": file_numbers, "context": ""}

    context_text = build_context_text(load_result["content"])
    followup_messages = messages + [
        {"role": "assistant", "content": result.content or "",
         "tool_calls": [_tool_call_to_dict(call)]},
        {"role": "tool", "tool_call_id": call.id, "content": f"CONTEXT:\n{context_text}"},
    ]

    final = call_llm(followup_messages, use_router=False, max_tokens=2048)
    metrics.append({"node": "answer", "model": final.model, "in_tok": final.input_tokens,
                     "out_tok": final.output_tokens, "cached_tok": final.cached_tokens,
                     "latency_s": final.latency_s})
    answer = final.content or (NO_DATA_TH if thai else NO_DATA_EN)
    trace.append({"node": "answer"})
    return {"answer": answer, "trace": trace, "metrics": metrics,
            "route": "treatment_or_general", "files": file_numbers, "context": context_text}
