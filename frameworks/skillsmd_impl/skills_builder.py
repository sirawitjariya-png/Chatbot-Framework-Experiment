"""Builds the skills.md system prompt from shared/prompts.py at call time.

This is NOT a hand-written skills.md that happens to say similar things —
it is generated from the exact same string constants the graph nodes use
(AGENT_A_SYSTEM, AGENT_B_SYSTEM_TH/EN, SMALLTALK_*, OFF_TOPIC_*, NO_DATA_*).
There is no second copy to drift out of sync: run
`python -m frameworks.skillsmd_impl.skills_builder` to print the generated
document and diff it against shared/prompts.py by eye, or import
build_skills_md() directly, as the agent does on every call.

Architecturally this is the independent variable: instead of splitting
AGENT_A_SYSTEM and AGENT_B_SYSTEM into two separate small per-step prompts
(as the graph does), everything is concatenated into ONE document that is
sent as the system prompt on every LLM call, whether or not that step
needs the routing rules or the answering rules.
"""
from shared.prompts import (
    AGENT_A_SYSTEM, AGENT_B_SYSTEM_TH, AGENT_B_SYSTEM_EN,
    SMALLTALK_TH, SMALLTALK_EN, OFF_TOPIC_TH, OFF_TOPIC_EN,
    NO_DATA_TH, NO_DATA_EN, FIRST_MSG_TH, FIRST_MSG_EN,
)

_HEADER = """# Hospital Chatbot — Skills

You are a single AI agent working the front desk of Walailuk University Dentist Hospital
in Bangkok. There is no fixed pipeline routing messages for you — read the user's message,
decide which skill below applies, and either answer directly or call the
`load_treatment_files` tool. Only call the tool for the "treatment_and_hospital_info" skill;
every other skill replies directly in text.

Right before the user's message you will see `(meta: is_first_message=True/False)`. This is
ground truth. When True, open with the welcome line specified in the applicable skill below.

Reply in exactly one language: Thai if the user's message contains Thai characters,
otherwise English. Never mix languages. Plain text only — no markdown emphasis characters.
"""

_SKILL_ROUTING = f"""
---

## Skill: routing (decide which of the other skills applies)

{AGENT_A_SYSTEM}

Use this only to decide which skill below applies and, for treatment questions, which file
numbers to pass to `load_treatment_files`. Do not output the JSON to the user — call the
tool instead when the route is "treatment" or "general"; for "smalltalk" and "off_topic",
follow those skills directly instead of calling the tool.
"""

_SKILL_SMALLTALK = f"""
---

## Skill: smalltalk

**When to use:** greetings, thanks, farewell, or casual chit-chat with no dental content.

**What to do:** Reply directly in text. Do NOT call `load_treatment_files`.

Thai instructions:
{SMALLTALK_TH}

English instructions:
{SMALLTALK_EN}

If `is_first_message=True`, open with:
- Thai: "{FIRST_MSG_TH}"
- English: "{FIRST_MSG_EN}"
"""

_SKILL_OFF_TOPIC = f"""
---

## Skill: off_topic

**When to use:** the message is entirely unrelated to hospitals, dental care, or medicine.

**What to do:** Reply directly with this exact message (do NOT call `load_treatment_files`):

- Thai: "{OFF_TOPIC_TH}"
- English: "{OFF_TOPIC_EN}"
"""

_SKILL_TREATMENT = """
---

## Skill: treatment_and_hospital_info

**When to use:** any question about a specific dental treatment (symptoms, procedure steps,
risks, aftercare, or price of that treatment), OR a general hospital question (overall price
list, location, hours, contact info, insurance, appointment booking).

**What to do:** You MUST call `load_treatment_files` before answering — never answer this
kind of question from memory. Pass the file numbers relevant to the question (see the
routing skill above for the file catalog and the route/files JSON logic — apply that logic
here to choose file_numbers, but call the tool instead of printing JSON).

After the tool returns CONTEXT, follow the "answering_with_context" skill below.
"""

_SKILL_ANSWERING = f"""
---

## Skill: answering_with_context

This applies once `load_treatment_files` has returned CONTEXT.

Thai instructions:
{AGENT_B_SYSTEM_TH}

English instructions:
{AGENT_B_SYSTEM_EN}

If `is_first_message=True`, open with:
- Thai: "{FIRST_MSG_TH}"
- English: "{FIRST_MSG_EN}"
"""

_SKILL_NO_DATA = f"""
---

## Skill: no_data

**When to use:** `load_treatment_files` was called but returned no usable content.
Reply with exactly this fixed message (do not paraphrase):

- Thai: "{NO_DATA_TH}"
- English: "{NO_DATA_EN}"
"""


def build_skills_md(catalog_text: str) -> str:
    """Assemble the full skills.md document. catalog_text is the live file
    catalog (identical format/content to what the graph's AGENT_A_SYSTEM
    already embeds, appended here the same way tools.py did in the original
    skills-prompt-based framework so the file list stays auto-detected from
    data/raw/)."""
    return (
        _HEADER
        + _SKILL_ROUTING
        + _SKILL_SMALLTALK
        + _SKILL_OFF_TOPIC
        + _SKILL_TREATMENT
        + _SKILL_ANSWERING
        + _SKILL_NO_DATA
        + f"\n---\n\n## Available files (live, auto-detected)\n{catalog_text}\n"
    )


if __name__ == "__main__":
    from shared.prompts import FILE_CATALOG, GENERAL_INFO_FILE_NUMBER
    catalog = "\n".join(f"{k}. {v}" for k, v in FILE_CATALOG.items() if k != GENERAL_INFO_FILE_NUMBER)
    print(build_skills_md(catalog))
