"""
Continuity Director Agent.

Ensures that adapted dialogue matches each character's established
voice, personality, and speech patterns stored in ChromaDB.
"""

from dataclasses import dataclass
from typing import Any, Dict

import json
import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq

from groq import Groq

# Make the project root importable so we can access local packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import FAST_MODEL, QUALITY_MODEL, get_models  # type: ignore
from memory.vector_store import (
    query_character_profile,
    query_character_profile_dict,
    query_similar_approved_lines,
)  # type: ignore


@dataclass
class ContinuityDirectorConfig:
    """
    Configuration for the Continuity Director Agent.

    This can be extended later with:
    - Similarity thresholds for determining when a line is "off-voice".
    - LLM parameters and prompt templates.
    """

    # Placeholder for future configuration fields.
    pass


class ContinuityDirectorAgent:
    """
    Thin wrapper class kept for future CrewAI integration.
    """

    def __init__(self, config: ContinuityDirectorConfig) -> None:
        self.config = config


# NOTE: Original spec requested `llama3-70b-8192`, but that model has been
# decommissioned. We use the current Groq 70B Llama 3 model instead.
MODEL_NAME = QUALITY_MODEL


def _build_character_prompt_snippet(
    profile_json_or_dict: str | Dict[str, Any],
    character_name: str,
) -> str:
    """
    Build a human-readable snippet describing the character from their JSON profile.
    """
    if isinstance(profile_json_or_dict, dict):
        data = profile_json_or_dict
    else:
        try:
            data = json.loads(profile_json_or_dict)
        except json.JSONDecodeError:
            return (
                f"The character is named {character_name}. "
                "No structured profile could be parsed."
            )

    role = data.get("role") or "character"
    speech_style = data.get("speech_style") or ""
    forbidden_phrases = data.get("forbidden_phrases") or []

    lines = [f"The character is {character_name}, a {role}."]
    if speech_style:
        lines.append(f"Their typical speech style is: {speech_style}.")

    if forbidden_phrases:
        phrases = ", ".join(str(p) for p in forbidden_phrases)
        lines.append(
            f"Forbidden phrases that MUST NOT appear in the dialogue: {phrases}."
        )

    return " ".join(lines)


def run_continuity_director(
    adapted_text: str,
    character_profile: Dict[str, Any],
    quality_model: ChatGroq,  # <-- LangChain wrapper
) -> str:
    """
    Adjust adapted dialogue to better match a character's established voice.
    """
    character_name = (
        character_profile.get("name")
        or character_profile.get("character")
        or ""
    ).strip()
    manga_id = str(character_profile.get("manga_id") or "default").strip() or "default"
    
    if not character_name:
        return adapted_text

    # Pull recent approved lines
    # TRUE RAG: Pull semantically similar approved lines based on the current text
    approved_lines = query_similar_approved_lines(
        character_name=character_name,
        manga_id=manga_id,
        query_text=adapted_text,
        limit=5,
    )
    if approved_lines:
        approved_lines_formatted = "\n".join(
            f"- [{row.get('panel_id')}] {row.get('final_output')}"
            for row in approved_lines
        )
    else:
        approved_lines_formatted = "None available yet."

    # Build voice/constraint snippet
    character_snippet = _build_character_prompt_snippet(character_profile, character_name)

    system_prompt = (
        "You are a continuity director for a manga/webtoon localization team.\n"
        f"{character_snippet}\n\n"
        "Your task:\n"
        "- Read the given English dialogue, which has already been culturally adapted.\n"
        "- Rewrite it only as much as necessary so that it perfectly matches the\n"
        "  character's established voice, personality, and speech patterns.\n"
        "- Avoid ALL forbidden phrases listed in the character profile.\n"
        "- Preserve the same meaning, emotional content, and cultural adaptation\n"
        "  already present in the input.\n\n"
        "Recent approved lines for this character (context):\n"
        f"{approved_lines_formatted}\n\n"
        "Output rules:\n"
        "- Output ONLY the final rewritten dialogue.\n"
        "- Do NOT add explanations, comments, or any extra text.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    
    chain = prompt | quality_model | StrOutputParser()
    
    continuity_text = chain.invoke({"text": adapted_text})
    return continuity_text.strip()


def _run_continuity_director_legacy(
    adapted_text: str,
    character_name: str,
    client: Groq,
) -> str:
    """
    Backward-compatible path (used nowhere in the updated pipeline).
    """
    profile_str = query_character_profile(character_name, manga_id="default")
    if profile_str is None:
        print(
            f"[ContinuityDirector] No profile found in ChromaDB for '{character_name}'. "
            "Returning text unchanged."
        )
        return adapted_text
    character_profile = (
        query_character_profile_dict(character_name, manga_id="default")
        or {"name": character_name, "manga_id": "default"}
    )
    return run_continuity_director(adapted_text, character_profile, client)


def grade_continuity_output(
    adapted_text: str,
    continuity_text: str,
    character_name: str,
    fast_model: ChatGroq,  # <-- LangChain wrapper
    manga_id: str = "default",
) -> Dict[str, Any]:
    """
    Grade how well the continuity-directed line matches the character and input.
    """
    profile_str = query_character_profile(character_name, manga_id=manga_id)
    profile_snippet = ""
    if profile_str is not None:
        profile_snippet = _build_character_prompt_snippet(profile_str, character_name)

    system_prompt = (
        "You are an expert continuity editor for localized manga/webtoon dialogue.\n"
        "You will receive:\n"
        "- The ADAPTED line (already culturally adapted by a previous agent).\n"
        "- The CONTINUITY line (the same line after you attempted to align it to the\n"
        "  character's voice).\n"
        "- Optionally, a short description of the character and any forbidden phrases.\n\n"
        "You must grade the CONTINUITY line with three scores from 1 to 10:\n"
        "- voice_consistency: How well it matches the described character voice and\n"
        "  feels like something they would actually say.\n"
        "- forbidden_phrase_compliance: How well it avoids using any forbidden phrases.\n"
        "- meaning_preservation: How well it preserves the meaning and cultural\n"
        "  adaptation present in the ADAPTED line.\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n"
        "- The final `pass` value must be true ONLY if all three scores are >= 7.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  voice_consistency, forbidden_phrase_compliance, meaning_preservation, pass\n"
        "- Example:\n"
        '  {{"voice_consistency": 8, "forbidden_phrase_compliance": 9, "meaning_preservation": 8, "pass": true}}\n'
        "- Do NOT include any explanations, comments, or extra text.\n"
    )

    user_parts = [
        f"CHARACTER NAME:\n{character_name}\n",
        "ADAPTED (input to continuity director):\n{adapted}\n",
        "CONTINUITY (output from continuity director):\n{continuity}\n",
    ]
    if profile_snippet:
        user_parts.append(f"CHARACTER PROFILE SUMMARY:\n{profile_snippet}\n")

    user_content = "\n".join(user_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_content)
    ])
    
    grading_chain = prompt | fast_model | JsonOutputParser()

    try:
        parsed = grading_chain.invoke({
            "adapted": adapted_text,
            "continuity": continuity_text
        })

        voice_consistency = int(parsed.get("voice_consistency", 0))
        forbidden_phrase_compliance = int(parsed.get("forbidden_phrase_compliance", 0))
        meaning_preservation = int(parsed.get("meaning_preservation", 0))

        passed = (
            voice_consistency >= 7
            and forbidden_phrase_compliance >= 7
            and meaning_preservation >= 7
        )

        return {
            "voice_consistency": voice_consistency,
            "forbidden_phrase_compliance": forbidden_phrase_compliance,
            "meaning_preservation": meaning_preservation,
            "pass": passed,
        }

    except Exception as exc:
        print(f"[Continuity Grading Error] Failed to parse JSON: {exc}")
        return {
            "voice_consistency": 0,
            "forbidden_phrase_compliance": 0,
            "meaning_preservation": 0,
            "pass": False,
        }

def test_continuity_agent() -> None:
    """
    Simple manual test for the Continuity Director Agent.
    """
    try:
        quality_model, fast_model = get_models()
    except Exception as exc:
        print(f"[ContinuityAgent Test] Failed to initialize LangChain models: {exc}")
        return

    character_name = "Kira"
    adapted_text = "I won't back down, no matter what."

    print("[ContinuityAgent Test] Character:", character_name)
    print("[ContinuityAgent Test] Adapted input:")
    print(adapted_text)

    try:
        from memory.vector_store import query_character_profile_dict
        character_profile = query_character_profile_dict(
            character_name,
            manga_id="default",
        ) or {
            "name": character_name,
            "manga_id": "default",
            "role": "character",
            "speech_style": "",
            "forbidden_phrases": [],
            "speech_rules": [],
        }
        continuity_text = run_continuity_director(
            adapted_text,
            character_profile,
            quality_model=quality_model,
        )
    except Exception as exc:
        print(f"[ContinuityAgent Test] Error during continuity adjustment: {exc}")
        return

    print("\n[ContinuityAgent Test] Continuity-adjusted output:")
    print(continuity_text)

    try:
        scores = grade_continuity_output(
            adapted_text=adapted_text,
            continuity_text=continuity_text,
            character_name=character_name,
            fast_model=fast_model,
        )
    except Exception as exc:
        print(f"[ContinuityAgent Test] Error during grading: {exc}")
        return

    print("\n[ContinuityAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[ContinuityAgent Test] Result: {status}")

