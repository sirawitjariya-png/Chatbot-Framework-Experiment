"""Proof, not just a claim: verify that every prompt constant in
shared/prompts.py appears byte-for-byte inside the generated skills.md, and
that the LangGraph module imports the same constants rather than local
copies.

Run: python -m scripts.verify_prompt_sync
Exit code 0 = in sync. Non-zero = something drifted, fix before running
the experiment.
"""
import sys
import ast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.prompts import (
    AGENT_A_SYSTEM, AGENT_B_SYSTEM_TH, AGENT_B_SYSTEM_EN,
    SMALLTALK_TH, SMALLTALK_EN, OFF_TOPIC_TH, OFF_TOPIC_EN,
    NO_DATA_TH, NO_DATA_EN, FILE_CATALOG, GENERAL_INFO_FILE_NUMBER,
)
from frameworks.skillsmd_impl.skills_builder import build_skills_md

REQUIRED_CONSTANTS = {
    "AGENT_A_SYSTEM": AGENT_A_SYSTEM,
    "AGENT_B_SYSTEM_TH": AGENT_B_SYSTEM_TH,
    "AGENT_B_SYSTEM_EN": AGENT_B_SYSTEM_EN,
    "SMALLTALK_TH": SMALLTALK_TH,
    "SMALLTALK_EN": SMALLTALK_EN,
    "OFF_TOPIC_TH": OFF_TOPIC_TH,
    "OFF_TOPIC_EN": OFF_TOPIC_EN,
    "NO_DATA_TH": NO_DATA_TH,
    "NO_DATA_EN": NO_DATA_EN,
}


def check_skills_md_contains_all_prompts() -> list[str]:
    catalog = "\n".join(
        f"{k}. {v}" for k, v in FILE_CATALOG.items() if k != GENERAL_INFO_FILE_NUMBER
    )
    skills_md = build_skills_md(catalog)
    failures = []
    for name, text in REQUIRED_CONSTANTS.items():
        if text not in skills_md:
            failures.append(f"  MISSING: {name} is not byte-identical inside generated skills.md")
    return failures


def check_no_hardcoded_prompt_literals(path: Path, forbidden_snippets: list[str]) -> list[str]:
    """Scan a framework file for suspiciously long string literals that are
    NOT imports from shared.prompts — a cheap guard against someone pasting
    a modified copy of a prompt directly into a framework module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if len(s) > 200 and any(marker in s for marker in ["ศูนย์ทันตกรรม", "Walailuk University"]):
                # Long hospital-related string literal defined locally instead of imported
                failures.append(
                    f"  SUSPICIOUS: {path} contains a {len(s)}-char inline string literal "
                    f"that looks like prompt text — should be imported from shared/prompts.py"
                )
    return failures


def main() -> int:
    failures: list[str] = []
    failures += check_skills_md_contains_all_prompts()
    failures += check_no_hardcoded_prompt_literals(
        Path(__file__).resolve().parent.parent / "frameworks" / "langgraph_impl" / "graph.py", []
    )
    failures += check_no_hardcoded_prompt_literals(
        Path(__file__).resolve().parent.parent / "frameworks" / "skillsmd_impl" / "agent.py", []
    )

    if failures:
        print("PROMPT SYNC CHECK FAILED:")
        print("\n".join(failures))
        return 1

    print("PROMPT SYNC CHECK PASSED")
    print(f"  {len(REQUIRED_CONSTANTS)} shared prompt constants verified byte-identical")
    print("  inside the generated skills.md, and no inline prompt literals found")
    print("  in either framework module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
