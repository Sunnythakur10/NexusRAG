"""
Profile extractor for Lumina Pipeline.
Uses LangChain LCEL to infer and cleanly merge character profile fields.
"""

from typing import Any, Dict, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

def extract_profiles(chapter_data: dict, quality_model: ChatGroq) -> list:
    """
    Infer character profiles from a chapter JSON using LangChain.
    """
    panels = chapter_data.get("panels") or []

    grouped: Dict[str, List[str]] = {}
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        character = str(panel.get("character") or "").strip()
        text = str(panel.get("text") or "").strip()
        if not character or not text:
            continue
        grouped.setdefault(character, []).append(text)

    system_prompt = (
        "You are an expert manga/webtoon character profiling assistant.\n"
        "You will be given multiple English dialogue lines for ONE character.\n\n"
        "Infer the character's:\n"
        "- personality (short description of demeanor)\n"
        "- speech_style (how they generally speak)\n"
        "- speech_rules (a list of concrete rules/constraints about their speech)\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a single valid JSON object.\n"
        "- No markdown. No explanations.\n"
        "- Use EXACT keys: personality, speech_style, speech_rules\n"
        "- Example: {{\"personality\": \"cheerful\", \"speech_style\": \"casual\", \"speech_rules\": [\"Uses slang\"]}}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "CHARACTER NAME: {character_name}\n\nDIALOGUE LINES:\n{joined_lines}")
    ])
    
    chain = prompt | quality_model | JsonOutputParser()

    profiles: List[Dict[str, Any]] = []
    for character_name, lines in grouped.items():
        joined_lines = "\n".join(lines)
        
        try:
            data = chain.invoke({
                "character_name": character_name,
                "joined_lines": joined_lines
            })
        except Exception as exc:
            print(f"[Profile Extractor] JSON parsing failed: {exc}")
            data = {
                "personality": "",
                "speech_style": "",
                "speech_rules": [],
            }

        data["name"] = character_name
        profiles.append(data)

    return profiles

def _intelligent_merge(existing: dict, incoming: dict, fast_model: ChatGroq) -> dict:
    """
    Uses the fast model to cleanly synthesize old and new profile data
    without string-concatenation bloat.
    """
    system_prompt = (
        "You are an expert character consistency manager.\n"
        "You have an EXISTING character profile and NEW observations from a recent chapter.\n"
        "Cleanly merge them into a single, concise profile.\n"
        "- Synthesize the personality and speech_style into unified, short descriptions.\n"
        "- Combine the speech_rules arrays, removing duplicates or conflicting rules.\n\n"
        "Output ONLY a single valid JSON object.\n"
        "Use EXACT keys: personality, speech_style, speech_rules\n"
        "Example: {{\"personality\": \"cheerful but focused\", \"speech_style\": \"casual\", \"speech_rules\": [\"Uses slang\"]}}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "EXISTING PROFILE:\n{existing}\n\nNEW OBSERVATIONS:\n{incoming}")
    ])

    chain = prompt | fast_model | JsonOutputParser()

    try:
        merged = chain.invoke({
            "existing": json.dumps(existing),
            "incoming": json.dumps(incoming)
        })
    except Exception as exc:
        print(f"[Profile Extractor] Merge failed, falling back to incoming: {exc}")
        return incoming

    return merged

def update_or_create_profile(
    character_name: str,
    new_profile: dict,
    vector_store,
    manga_id: str,
    fast_model: ChatGroq  # <-- Now receives the fast model for merging
) -> None:
    """
    Update an existing character profile in ChromaDB, or create it if missing.
    """
    existing = vector_store.query_character_profile_dict(
        character_name,
        manga_id=manga_id,
    )

    if existing is None:
        profile_to_store = dict(new_profile or {})
        profile_to_store["name"] = character_name
        vector_store.add_character_profile(
            character_name,
            profile_to_store,
            manga_id=manga_id,
        )
        return

    merged = _intelligent_merge(existing, new_profile or {}, fast_model)
    merged["name"] = character_name
    
    vector_store.add_character_profile(
        character_name,
        merged,
        manga_id=manga_id,
    )