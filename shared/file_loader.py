"""DOCX file loading — identical logic for both frameworks so that a
retrieval-content difference can never explain an accuracy difference.

Both frameworks call load_files(file_numbers) and get back the same
{content, has_data, trace_entry} shape.
"""
import logging
from pathlib import Path

try:
    from docx import Document
except ImportError as _e:
    raise ImportError("python-docx required: pip install python-docx") from _e

from .config import RAW_DIR
from .prompts import FILE_CATALOG, GENERAL_INFO_FILE_NUMBER, MAX_CONTEXT_CHARS

log = logging.getLogger(__name__)

_docx_index: dict[int, Path] | None = None


def _build_index() -> dict[int, Path]:
    index: dict[int, Path] = {}
    try:
        for p in Path(RAW_DIR).iterdir():
            if p.suffix != ".docx" or p.name.startswith("~$"):
                continue
            prefix = p.name.split(".")[0]
            try:
                index[int(prefix)] = p
            except ValueError:
                pass
    except Exception as e:
        log.warning("Failed to build docx index from RAW_DIR: %s", e)
    return index


def get_index() -> dict[int, Path]:
    global _docx_index
    if _docx_index is None:
        _docx_index = _build_index()
    return _docx_index


def _read_docx(path: Path) -> str:
    try:
        doc = Document(str(path))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines)
    except Exception as e:
        log.warning("Read %s failed: %s", path, e)
        return ""


def load_files(file_numbers: list[int]) -> dict:
    """Read requested treatment files. The general-info file is always included.

    Returns: {content: dict[int, str], has_data: bool, trace_entry: dict}
    """
    index = get_index()
    numbers = sorted(set(file_numbers) | {GENERAL_INFO_FILE_NUMBER})
    content: dict[int, str] = {}
    file_trace: list[dict] = []

    for num in numbers:
        path = index.get(num)
        if path is None:
            file_trace.append({"file": num, "found": False})
            continue
        text = _read_docx(path)
        entry: dict = {"file": num, "found": True, "chars": len(text), "name": path.name}
        if text:
            content[num] = text
        else:
            entry["readable"] = False
        file_trace.append(entry)

    has_data = bool(content) and any(len(v) > 50 for v in content.values())
    trace_entry = {"node": "read_files", "files": file_trace, "has_data": has_data}
    return {"content": content, "has_data": has_data, "trace_entry": trace_entry}


def build_context_text(content: dict[int, str]) -> str:
    """Render loaded content into the CONTEXT block, truncated to the shared char budget."""
    sections: list[str] = []
    total_chars = 0
    for num in sorted(content.keys()):
        label = FILE_CATALOG.get(num, f"File {num}")
        text = content[num]
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        chunk = text[:remaining]
        sections.append(f"--- {label} ---\n{chunk}")
        total_chars += len(chunk)
    return "\n\n".join(sections)
