"""
Translation Agent (Step 0).

Takes raw Japanese manga/webtoon script text and produces a contextual
English translation that preserves emotion, tone, and dramatic weight.
This output feeds into Agent 1 (Cultural Adaptor).
"""

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq


import json
import os
import sys

from groq import Groq

# Ensure the project root is on sys.path so we can import local utilities.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import FAST_MODEL, QUALITY_MODEL, get_models  # type: ignore


@dataclass
class TranslationAgentConfig:
    """
    Placeholder configuration for the Translation Agent.
    """

    # Extendable later (e.g., model name, temperature).
    pass


# NOTE: Original spec requested `llama3-70b-8192`, but that model has been
# decommissioned by Groq. We use the current Groq 70B Llama 3 model instead.
MODEL_NAME = QUALITY_MODEL


def detect_language(text: str) -> str:
    """
    Very simple Unicode-based language detection.

    Returns:
        "japanese", "korean", or "unknown"
    """
    has_japanese = False
    has_korean = False

    for ch in text:
        code = ord(ch)

        # Japanese ranges: Hiragana, Katakana, Kanji/CJK
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (
            0x4E00 <= code <= 0x9FFF
        ):
            has_japanese = True

        # Korean range: Hangul syllables
        if 0xAC00 <= code <= 0xD7A3:
            has_korean = True

    # Prioritize Japanese when mixed.
    if has_japanese:
        return "japanese"
    if has_korean:
        return "korean"
    return "unknown"


def translate_page(page_panels: list, client: Groq) -> dict:
    """
    Translate one page worth of panel lines in a single LLM call.

    Args:
        page_panels: List of panel dicts for a single page.
        client: Initialized Groq client.

    Returns:
        Dict mapping panel_id -> translated text.
    """
    items: list[tuple[str, str, str]] = []
    page_label = "Unknown"
    for idx, panel in enumerate(page_panels):
        if not isinstance(panel, dict):
            continue
        panel_id = panel.get("panel_id") or panel.get("id") or panel.get("index")
        panel_id_str = str(panel_id if panel_id is not None else idx)
        if page_label == "Unknown" and "-" in panel_id_str:
            page_label = panel_id_str.split("-", 1)[0]
        character = str(panel.get("character") or "UNKNOWN").strip() or "UNKNOWN"
        text = str(panel.get("text") or "").strip()
        if not text:
            continue
        items.append((panel_id_str, character, text))

    if not items:
        return {}

    system_prompt = (
        "You are a professional manga translator specializing in Japanese to English.\n"
        "Translate each line independently and return strict JSON only.\n"
        "Preserve emotional state and who is speaking to whom.\n"
        "JSON keys must be panel_id strings and values must be translated English lines.\n"
        "\n--- CRITICAL HARD RULES ---\n"
        "- Preserve verb tense exactly. Present tense in Japanese must be present tense in English. ('しなきゃいけない' = 'I have to' NOT 'I had to').\n"
        "- Preserve speaker perspective. First-person statements must stay first-person. ('うん、ある' said by Haruto = 'Yeah, I do' NOT 'It does').\n"
        "- Preserve emotional intensity. Angry lines must sound angry. ('黙ってろ' = 'Shut up!' NOT 'Stop that').\n"
        "- Preserve self-doubt and hesitation exactly. ('夢だったんじゃないか' = 'Was that all a dream...?' NOT a neutral statement).\n"
        "- Never change character names mid-chapter. Maintain strict name consistency.\n"
        "- Do NOT include any romaji, Japanese script, explanations, translation notes, or comments.\n"
        "- Output ONLY the translated English dialogue text, with no extra lines.\n"
    )

    user_lines = [f"Page {page_label} — translate each line independently:"]
    for panel_id_str, character, text in items:
        user_lines.append(f"[{panel_id_str}] {character}: {text}")
    user_content = "\n".join(user_lines)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = (completion.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None:
            continue
        result[str(k)] = str(v).strip()
    return result


def translate_literal_metadata(text: str, client: Groq) -> str:
    """
    Translate Japanese metadata text literally (dates/locations/proper nouns preserved).
    """
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate this Japanese metadata text literally, preserving all dates, "
                    "locations, and proper nouns. Output only the translated text."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def run_translation_agent(
    raw_text: str,
    quality_model: ChatGroq,  # You MUST pass the LangChain wrapper here, not the old Groq client
    character_name: str = "",
    sliding_context: str = "",
) -> str:
    """
    Run the Translation Agent on a raw Japanese line using LangChain.
    """
    language = detect_language(raw_text)

    # 1. Define the rules as a clean string variable
    rules = (
        "- Preserve emotion, tone, and dramatic weight; do NOT do word-for-word translation.\n"
        "- Preserve Japanese speech levels:\n"
        "  * Keigo (formal/polite) → formal English.\n"
        "  * Casual/plain form → casual English.\n"
        "  * Rough/masculine speech (e.g., ore, ore-sama) → aggressive English.\n"
        "  * Feminine speech (e.g., watashi, kashira) → soft, feminine English.\n"
        "- Preserve the FEELING of Japanese honorifics (like -san, -kun, -sama) as respect levels,\n"
        "  but do NOT include the honorific words themselves in the output.\n"
        "- Before translating, identify the speaker's emotional state: panicking / happy / sad /\n"
        "  angry / teasing / casual farewell / ironic.\n"
        "- Let emotional state drive word choice:\n"
        "  * Panic phrases → desperate English equivalents\n"
        "  * Ironic narrator lines → preserve the irony with phrases like\n"
        "    'or so I thought' / 'at least that was the plan'\n"
        "  * Casual farewells → 'later' / 'see ya' / 'catch you later'\n"
        "  * Offers and giving phrases → preserve who is giving and who is receiving,\n"
        "    never reverse them\n"
        "  * Definitive statements → translate as confident, not hesitant\n"
        "    (never add 'I guess' / 'I suppose' to a character who spoke with conviction)\n"
        "- Never soften strong emotions or add uncertainty to confident statements.\n"
        "\n--- CRITICAL HARD RULES ---\n"
        "- Preserve verb tense exactly. Present tense in Japanese must be present tense in English. ('しなきゃいけない' = 'I have to' NOT 'I had to').\n"
        "- Preserve speaker perspective. First-person statements must stay first-person. ('うん、ある' said by Haruto = 'Yeah, I do' NOT 'It does').\n"
        "- Preserve emotional intensity. Angry lines must sound angry. ('黙ってろ' = 'Shut up!' NOT 'Stop that').\n"
        "- Preserve self-doubt and hesitation exactly. ('夢だったんじゃないか' = 'Was that all a dream...?' NOT a neutral statement).\n"
        "- Never change character names mid-chapter. Maintain strict name consistency.\n"
        "- Do NOT include any romaji, Japanese script, explanations, translation notes, or comments.\n"
        "- Output ONLY the translated English dialogue text, with no extra lines.\n"
    )

    # 2. Setup the LangChain Prompt Template
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional manga translator specializing in {language} to English translation.\nFollow these rules strictly:\n{rules}"),
        ("human", "SPEAKER: {character}\nDIALOGUE: {raw_text}\nCHAPTER CONTEXT: {context}")
    ])

    # 3. Build the LangChain Pipeline
    translation_chain = translation_prompt | quality_model | StrOutputParser()

    # 4. Execute the Chain
    translated = translation_chain.invoke({
        "language": language,
        "rules": rules,
        "character": character_name if character_name else "UNKNOWN",
        "raw_text": raw_text,
        "context": sliding_context if sliding_context else "None"
    })

    return translated.strip()

def grade_translation_output(
    original: str,
    translated: str,
    fast_model: ChatGroq,  # Now accepts the LangChain fast_model
) -> Dict[str, Any]:
    """
    Grade the translation quality using LangChain and JsonOutputParser.
    """
    system_prompt = (
        "You are an expert bilingual Japanese ↔ English manga translator.\n"
        "You will receive the ORIGINAL line (Japanese) and the TRANSLATED\n"
        "English line. Grade the English line on three axes from 1 to 10:\n"
        "- contextual_accuracy: How well it captures the intended meaning and nuance.\n"
        "- tone_preservation: How well it preserves emotion, formality level,\n"
        "  and dramatic weight.\n"
        "- naturalness: How natural and fluent it reads as English dialogue.\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n"
        "- The final `pass` value must be true ONLY if all three scores are >= 7.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  contextual_accuracy, tone_preservation, naturalness, pass\n"
        "- Example:\n"
        '  {{"contextual_accuracy": 8, "tone_preservation": 9, "naturalness": 8, "pass": true}}\n'
    )

    # 1. Setup the Template
    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "ORIGINAL (Japanese):\n{original}\n\nTRANSLATED (English):\n{translated}")
    ])

    # 2. Build the Grading Chain (Template -> Fast Model -> JSON Parser)
    grading_chain = grading_prompt | fast_model | JsonOutputParser()

    # 3. Execute the Chain
    try:
        parsed = grading_chain.invoke({
            "original": original,
            "translated": translated
        })
        
        # Hard-code backend verification just in case the LLM hallucinates the math
        contextual_accuracy = int(parsed.get("contextual_accuracy", 0))
        tone_preservation = int(parsed.get("tone_preservation", 0))
        naturalness = int(parsed.get("naturalness", 0))

        passed = (
            contextual_accuracy >= 7
            and tone_preservation >= 7
            and naturalness >= 7
        )
        
        parsed["pass"] = passed
        return parsed

    except Exception as exc:
        print(f"[Grading Error] Failed to parse JSON: {exc}")
        return {
            "contextual_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
        }


def test_translation_agent() -> None:
    """
    Simple manual test for the Translation Agent.
    """
    try:
        quality_model, fast_model = get_models()
    except Exception as exc:
        print(f"[TranslationAgent Test] Failed to initialize LangChain models: {exc}")
        return

    raw_text = "たとえ天が崩れ落ちようとも、私は一歩も退かない。"
    language = detect_language(raw_text)

    print("[TranslationAgent Test] Detected language:", language)

    try:
        translated = run_translation_agent(raw_text, quality_model=quality_model)
    except Exception as exc:
        print(f"[TranslationAgent Test] Error during translation: {exc}")
        return

    print("\n[TranslationAgent Test] Translated text:")
    print(translated)

    try:
        scores = grade_translation_output(raw_text, translated, fast_model=fast_model)
    except Exception as exc:
        print(f"[TranslationAgent Test] Error during grading: {exc}")
        return

    print("\n[TranslationAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[TranslationAgent Test] Result: {status}")