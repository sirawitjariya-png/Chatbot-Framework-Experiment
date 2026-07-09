"""LangGraph implementation of the dental FAQ workflow.

Every node's system prompt is imported verbatim from shared/prompts.py.
No prompt text is authored in this file. Each LLM-calling node receives
ONLY the instructions relevant to that step:

  classify node  -> AGENT_A_SYSTEM only
  smalltalk node -> SMALLTALK_TH/EN only
  answer node    -> AGENT_B_SYSTEM_TH/EN + CONTEXT only

This "small, per-step prompt" delivery is the independent variable being
tested against frameworks/skillsmd_impl, which delivers the identical text
as one large prompt on every call. See README.md.

    START -> classify
        |- off_topic  -> fixed reply -> END
        |- smalltalk  -> LLM reply   -> END
        `- treatment / general
               -> load_files -> check_info
                      |- no data  -> fixed reply -> END
                      `- has data -> format_answer -> END
"""
import json
import logging
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from shared.llm_client import call_llm
from shared.file_loader import load_files, build_context_text
from shared.prompts import (
    AGENT_A_SYSTEM, AGENT_B_SYSTEM_TH, AGENT_B_SYSTEM_EN,
    SMALLTALK_TH, SMALLTALK_EN, FIRST_MSG_TH, FIRST_MSG_EN,
    OFF_TOPIC_TH, OFF_TOPIC_EN, NO_DATA_TH, NO_DATA_EN,
    is_thai,
)

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class State(TypedDict, total=False):
    question: str
    history: list
    route: str
    files: list
    content: dict
    has_data: bool
    answer: str
    trace: Annotated[list, operator.add]
    metrics: Annotated[list, operator.add]  # per-call token/latency records


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


# ---- nodes ------------------------------------------------------------------

def _classify_node(state: State) -> dict:
    """Agent A. System prompt = AGENT_A_SYSTEM only (~1 file catalog, ~600 tokens)."""
    messages = [
        {"role": "system", "content": AGENT_A_SYSTEM},
        *state.get("history", []),
        {"role": "user", "content": state["question"]},
    ]
    try:
        result = call_llm(messages, use_router=True)
        data = json.loads(_strip_fences(result.content))
        route = str(data.get("route", "general")).lower().strip()
        if route not in ("off_topic", "smalltalk", "treatment", "general"):
            route = "general"
        raw_files = data.get("files") or []
        files = []
        for f in raw_files:
            try:
                n = int(f)
                if 1 <= n <= 12:
                    files.append(n)
            except (ValueError, TypeError):
                pass
        metric = {"node": "classify", "model": result.model, "in_tok": result.input_tokens,
                  "out_tok": result.output_tokens, "cached_tok": result.cached_tokens,
                  "latency_s": result.latency_s}
        return {"route": route, "files": files, "trace": [{"node": "classify", "route": route, "files": files}],
                "metrics": [metric]}
    except Exception as e:
        log.error("classify_node failed: %s", e)
        return {"route": "general", "files": [], "trace": [{"node": "classify", "error": str(e)}]}


def _smalltalk_node(state: State) -> dict:
    thai = is_thai(state["question"])
    system = SMALLTALK_TH if thai else SMALLTALK_EN
    fallback = "สวัสดีค่ะ มีอะไรให้ช่วยบ้างคะ" if thai else "Hello! How may I assist you?"
    messages = [
        {"role": "system", "content": system},
        *state.get("history", []),
        {"role": "user", "content": state["question"]},
    ]
    try:
        result = call_llm(messages, use_router=False, max_tokens=2048)
        metric = {"node": "smalltalk", "model": result.model, "in_tok": result.input_tokens,
                  "out_tok": result.output_tokens, "cached_tok": result.cached_tokens,
                  "latency_s": result.latency_s}
        return {"answer": result.content or fallback, "trace": [{"node": "smalltalk"}], "metrics": [metric]}
    except Exception as e:
        log.error("smalltalk_node failed: %s", e)
        return {"answer": fallback, "trace": [{"node": "smalltalk", "error": str(e)}]}


def _off_topic_node(state: State) -> dict:
    thai = is_thai(state["question"])
    return {"answer": OFF_TOPIC_TH if thai else OFF_TOPIC_EN, "trace": [{"node": "off_topic"}]}


def _load_files_node(state: State) -> dict:
    result = load_files(state.get("files", []))
    return {"content": result["content"], "has_data": result["has_data"], "trace": [result["trace_entry"]]}


def _no_data_node(state: State) -> dict:
    thai = is_thai(state["question"])
    return {"answer": NO_DATA_TH if thai else NO_DATA_EN, "trace": [{"node": "no_data"}]}


def _format_answer_node(state: State) -> dict:
    """Agent B. System prompt = AGENT_B_SYSTEM_* + CONTEXT only — no router
    instructions leak into this call."""
    thai = is_thai(state["question"])
    context_text = build_context_text(state.get("content", {}))
    is_first = len(state.get("history", [])) == 0

    base = AGENT_B_SYSTEM_TH if thai else AGENT_B_SYSTEM_EN
    first_msg = (FIRST_MSG_TH if thai else FIRST_MSG_EN) + "\n\n" if is_first else ""
    system = f"{first_msg}{base}\n\nCONTEXT:\n{context_text}"

    error_reply = NO_DATA_TH if thai else NO_DATA_EN
    messages = [
        {"role": "system", "content": system},
        *state.get("history", []),
        {"role": "user", "content": state["question"]},
    ]
    try:
        result = call_llm(messages, use_router=False, max_tokens=2048)
        metric = {"node": "format_answer", "model": result.model, "in_tok": result.input_tokens,
                  "out_tok": result.output_tokens, "cached_tok": result.cached_tokens,
                  "latency_s": result.latency_s}
        return {"answer": result.content or error_reply, "trace": [{"node": "answer"}], "metrics": [metric]}
    except Exception as e:
        log.error("format_answer_node failed: %s", e)
        return {"answer": error_reply, "trace": [{"node": "answer", "error": str(e)}]}


# ---- routing ------------------------------------------------------------------

def _route_after_classify(state: State) -> str:
    return state.get("route", "general")


def _route_after_load(state: State) -> str:
    return "ok" if state.get("has_data") else "no_data"


def _build_graph():
    g = StateGraph(State)
    g.add_node("classify", _classify_node)
    g.add_node("smalltalk", _smalltalk_node)
    g.add_node("off_topic", _off_topic_node)
    g.add_node("load_files", _load_files_node)
    g.add_node("no_data", _no_data_node)
    g.add_node("format_answer", _format_answer_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_after_classify, {
        "off_topic": "off_topic", "smalltalk": "smalltalk",
        "treatment": "load_files", "general": "load_files",
    })
    g.add_conditional_edges("load_files", _route_after_load, {"ok": "format_answer", "no_data": "no_data"})
    g.add_edge("smalltalk", END)
    g.add_edge("off_topic", END)
    g.add_edge("no_data", END)
    g.add_edge("format_answer", END)
    return g.compile()


_graph = _build_graph()


def ask(question: str, history: list | None = None) -> dict:
    """Single-turn ask. Returns {answer, trace, metrics} — no session/state
    persistence here; the experiment harness controls history explicitly
    per repeat so runs are independent and comparable."""
    result = _graph.invoke({"question": question, "history": history or [], "trace": [], "metrics": []})
    context_text = build_context_text(result.get("content", {})) if result.get("content") else ""
    return {
        "answer": result.get("answer", ""),
        "trace": result.get("trace", []),
        "metrics": result.get("metrics", []),
        "route": result.get("route"),
        "files": result.get("files", []),
        "context": context_text,
    }
