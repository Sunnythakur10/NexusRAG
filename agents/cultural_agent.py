"""
Cultural Adaptor Agent.

Responsible for:
- Interpreting Japanese/Korean idioms, jokes, and culture-specific references.
- Rewriting them into natural, culturally appropriate English equivalents.
- Preserving tone, intent, and character voice guidelines.
"""

from dataclasses import dataclass
from typing import Any, Dict

import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq

# Ensure the project root is on sys.path so we can import local packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import get_models

@dataclass
class CulturalAdaptorConfig:
    pass

class CulturalAdaptorAgent:
    def __init__(self, config: CulturalAdaptorConfig) -> None:
        self.config = config

    def adapt_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Cultural adaptation logic is not implemented yet.")


def run_cultural_adaptor(raw_text: str, quality_model: ChatGroq) -> str:
    """
    Run the Cultural Adaptor using LangChain LCEL.
    """
    system_prompt = (
        "You are a cultural localization editor for manga and webtoon dialogue.\n"
        "You receive English text that may be a literal translation from Japanese\n"
        "or Korean.\n\n"
        "Follow these rules exactly:\n"
        "1) Identify Japanese/Korean idioms, cultural references, honorific-driven tone,\n"
        "   or unnatural literal phrasing.\n"
        "2) Rewrite into natural, culturally appropriate English equivalents.\n"
        "3) Preserve original meaning, emotion, and tone.\n"
        "4) Keep output length similar to input length.\n"
        "5) Output only the rewritten dialogue.\n"
        "6) Do not add explanations, notes, labels, or extra formatting.\n"
        "8) If the input already reads as natural correct English with clear\n"
        "   meaning and correct speaker attribution — preserve it. Only rewrite\n"
        "   when there is a genuine cultural equivalence improvement to make.\n"
        "   Never change who is speaking or who is receiving an action.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    
    chain = prompt | quality_model | StrOutputParser()
    
    adapted = chain.invoke({
        "text": raw_text
    })

    return adapted.strip()


def grade_cultural_output(original: str, adapted: str, fast_model: ChatGroq) -> Dict[str, Any]:
    """
    Grade the cultural adaptation quality using LangChain LCEL.
    """
    system_prompt = (
        "You are a strict localization quality grader.\n"
        "You will grade an adapted English dialogue line against the original line.\n\n"
        "Score each metric from 1 to 10:\n"
        "- cultural_accuracy: faithful handling of idioms/cultural references.\n"
        "- tone_preservation: preserves emotion, attitude, and voice.\n"
        "- naturalness: reads as fluent, natural English dialogue.\n\n"
        "Set pass=true only when ALL three scores are >= 7.\n\n"
        "Output ONLY valid JSON with exactly these keys:\n"
        "cultural_accuracy, tone_preservation, naturalness, pass\n"
        "No prose. No markdown. No extra keys.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "ORIGINAL:\n{original}\n\nADAPTED:\n{adapted}")
    ])
    
    grading_chain = prompt | fast_model | JsonOutputParser()
    
    try:
        parsed = grading_chain.invoke({
            "original": original,
            "adapted": adapted
        })
        
        # Enforce pass logic on backend
        cultural_accuracy = int(parsed.get("cultural_accuracy", 0))
        tone_preservation = int(parsed.get("tone_preservation", 0))
        naturalness = int(parsed.get("naturalness", 0))
        
        passed = (
            cultural_accuracy >= 7
            and tone_preservation >= 7
            and naturalness >= 7
        )
        
        parsed["pass"] = passed
        return parsed

    except Exception as exc:
        print(f"[Cultural Grading Error] Failed to parse JSON: {exc}")
        return {
            "cultural_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
        }

def test_cultural_agent() -> None:
    """
    Simple manual test for the Cultural Adaptor agent.
    """
    try:
        quality_model, fast_model = get_models()
    except Exception as exc:
        print(f"[CulturalAgent Test] Failed to initialize LangChain models: {exc}")
        return

    raw_text = "Even if the heavens fall, I will not retreat a single step."
    print("[CulturalAgent Test] Original:")
    print(raw_text)

    try:
        adapted = run_cultural_adaptor(raw_text, quality_model=quality_model)
    except Exception as exc:
        print(f"[CulturalAgent Test] Error during adaptation: {exc}")
        return

    print("\n[CulturalAgent Test] Adapted:")
    print(adapted)

    try:
        scores = grade_cultural_output(raw_text, adapted, fast_model=fast_model)
    except Exception as exc:
        print(f"[CulturalAgent Test] Error during grading: {exc}")
        return

    print("\n[CulturalAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[CulturalAgent Test] Result: {status}")