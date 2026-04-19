"""
Typesetting Editor Agent.

Takes continuity-approved dialogue and ensures it physically fits inside
the speech bubble by trimming or rewording, without changing meaning or
character voice.
"""

from dataclasses import dataclass
from typing import Any, Dict

import json
import os
import sys

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq

# Make the project root importable so we can access local packages and data files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import get_models

@dataclass
class TypesettingEditorConfig:
    pass

class TypesettingEditorAgent:
    def __init__(self, config: TypesettingEditorConfig) -> None:
        self.config = config

def load_bubble_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_max_chars_for_bubble(bubble_type: str, config: Dict[str, Any]) -> int:
    default_max = int(config.get("default_max_chars", 80))
    bubbles = config.get("bubbles", {}) or {}
    bubble_max = bubbles.get(bubble_type)
    return int(bubble_max) if bubble_max is not None else default_max

def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip()

@tool
def check_bubble_fit(proposed_text: str, limit: int) -> str:
    """
    Always use this tool to check if your proposed dialogue fits the character limit.
    Pass in the exact text you want to use, and the maximum character limit.
    """
    length = len(proposed_text)
    if length <= limit:
        return f"Success! Length is {length}/{limit}. This fits perfectly."
    else:
        return f"Failed! Length is {length}/{limit}. You are over by {length - limit} characters. Try again."

def run_typesetting_editor(
    text: str,
    bubble_type: str,
    quality_model: ChatGroq,
    *,
    bubble_char_limit: int | None = None,
) -> str:
    """
    Ensure the dialogue fits using a native LangGraph Tool-Calling Agent.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "data", "bubble_config.json")
    
    try:
        config = load_bubble_config(config_path)
    except FileNotFoundError:
        config = {"default_max_chars": 80}
        
    max_chars = (
        int(bubble_char_limit)
        if bubble_char_limit is not None
        else _get_max_chars_for_bubble(bubble_type, config)
    )

    if len(text) <= max_chars:
        return text

    # 1. Define the tools array
    tools = [check_bubble_fit]

    # 2. The system prompt becomes the state_modifier
    system_prompt = (
        "You are a typesetting editor for localized manga/webtoon dialogue.\n"
        "Your job is to make sure the text physically fits inside a speech bubble.\n\n"
        f"Bubble type: {bubble_type}\n"
        f"Maximum characters allowed: {max_chars}\n\n"
        "Rules:\n"
        "- Preserve the character's voice and intent.\n"
        "- You MUST use the `check_bubble_fit` tool to verify your text length BEFORE providing your final answer.\n"
        "- If the tool says Failed, you must shorten the text and use the tool again.\n"
        "- Once the tool says 'Success!', output ONLY the final dialogue text, with no extra commentary."
    )

    # 3. Create the modern LangGraph React Agent
    # 3. Create the agent with NO version-specific modifier arguments
    agent_executor = create_react_agent(quality_model, tools)

    try:
        # Pass the system prompt directly as the first message in the execution state
        result = agent_executor.invoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text)
            ]
        })
        
        # The final output is the content of the very last message in the chain
        rewritten = result["messages"][-1].content.strip()
    except Exception as exc:
        print(f"[Typesetting Agent] Tool execution failed: {exc}")
        rewritten = _truncate_at_word_boundary(text, max_chars)
        
    # Hard programmatic fallback just in case
    if len(rewritten) > max_chars:
        rewritten = _truncate_at_word_boundary(rewritten, max_chars)

    return rewritten

def grade_typesetting_output(
    original: str,
    final: str,
    bubble_type: str,
    fast_model: ChatGroq,  # <-- LangChain wrapper
    *,
    bubble_char_limit: int | None = None,
) -> Dict[str, Any]:
    """
    Grade the typesetting output quality using LangChain LCEL.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "data", "bubble_config.json")
    
    try:
        config = load_bubble_config(config_path)
    except FileNotFoundError:
        config = {"default_max_chars": 80}

    max_chars = (
        int(bubble_char_limit)
        if bubble_char_limit is not None
        else _get_max_chars_for_bubble(bubble_type, config)
    )

    final_len = len(final)
    fits_constraint = 10 if final_len <= max_chars else 1

    system_prompt = (
        "You are an expert localization editor focusing on meaning preservation.\n"
        "You will receive the ORIGINAL continuity-approved dialogue and the FINAL\n"
        "typeset-safe dialogue.\n\n"
        "Your job is to grade meaning_preserved from 1 to 10:\n"
        "- meaning_preserved: How well the meaning of the ORIGINAL is preserved in the FINAL.\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  meaning_preserved\n"
        "- Example:\n"
        '  {{"meaning_preserved": 9}}\n'
        "- Do NOT include any explanations, comments, or extra text.\n"
    )

    user_content = (
        "ORIGINAL (continuity-approved):\n"
        f"{original}\n\n"
        "FINAL (typeset-safe):\n"
        f"{final}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{content}")
    ])
    
    grading_chain = prompt | fast_model | JsonOutputParser()

    try:
        parsed = grading_chain.invoke({"content": user_content})
        
        meaning_preserved = int(parsed.get("meaning_preserved", 0))
        fits = fits_constraint >= 7
        passed = fits and meaning_preserved >= 7

        return {
            "fits_constraint": fits_constraint,
            "meaning_preserved": meaning_preserved,
            "voice_maintained": None,
            "pass": passed,
        }

    except Exception as exc:
        print(f"[Typesetting Grading Error] Failed to parse JSON: {exc}")
        return {
            "fits_constraint": fits_constraint,
            "meaning_preserved": 0,
            "voice_maintained": None,
            "pass": False,
        }

def test_typesetting_agent() -> None:
    """
    Simple manual test for the Typesetting Editor Agent.
    """
    try:
        quality_model, fast_model = get_models()
    except Exception as exc:
        print(f"[TypesettingAgent Test] Failed to initialize LangChain models: {exc}")
        return

    bubble_type = "small"
    original = "I'm not backing down, no way!"

    print("[TypesettingAgent Test] Bubble type:", bubble_type)
    print("[TypesettingAgent Test] Original text:", original)
    print("[TypesettingAgent Test] Original length:", len(original))

    try:
        final = run_typesetting_editor(original, bubble_type, quality_model=quality_model)
    except Exception as exc:
        print(f"[TypesettingAgent Test] Error during typesetting: {exc}")
        return

    print("\n[TypesettingAgent Test] Final text:", final)
    print("[TypesettingAgent Test] Final length:", len(final))

    try:
        scores = grade_typesetting_output(original, final, bubble_type, fast_model=fast_model)
    except Exception as exc:
        print(f"[TypesettingAgent Test] Error during grading: {exc}")
        return

    print("\n[TypesettingAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[TypesettingAgent Test] Result: {status}")

