import os
from typing import Any, Dict
from groq import Groq
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_groq import ChatGroq  # NEW IMPORT
from utils.groq_client import get_models , load_environment
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict
# Initialize caching
set_llm_cache(SQLiteCache(database_path="lumina_cache.db"))

class PanelState(TypedDict):
    """The unified state object passed between LangGraph nodes."""
    # --- Inputs ---
    raw_text: str
    character_name: str
    bubble_type: str
    bubble_char_limit: int | None
    character_profile: Dict[str, Any]
    skip_agents: bool
    quality_model: Any
    fast_model: Any
    
    # --- Intermediary Outputs ---
    detected_language: str
    translated_output: str
    cultural_output: str
    continuity_output: str
    final_output: str
    
    # --- Scores ---
    translation_scores: Dict[str, Any]
    cultural_scores: Dict[str, Any]
    continuity_scores: Dict[str, Any]
    typesetting_scores: Dict[str, Any]
    
    # --- Retry Counters ---
    # We track retries per node so we can break out of infinite loops
    translation_retries: int
    cultural_retries: int
    continuity_retries: int
    typesetting_retries: int



from agents.profile_extractor import extract_profiles, update_or_create_profile
from agents.cultural_agent import grade_cultural_output, run_cultural_adaptor
from agents.continuity_agent import (
    grade_continuity_output,
    run_continuity_director,
)
from agents.translation_agent import (
    detect_language,
    grade_translation_output,
    translate_literal_metadata,
    translate_page,
    run_translation_agent,
)
from agents.typesetting_agent import (
    grade_typesetting_output,
    run_typesetting_editor,
)
from memory import vector_store
from memory.vector_store import (
    add_approved_line,
    query_character_profile_dict,
)

from utils.project_manager import mark_chapter_complete

# TODO: Import CrewAI orchestration primitives when implementing multi-agent Crew.
# from crewai import Crew


def test_groq_connection() -> None:
    """Test the LangChain connection."""
    try:
        quality_model, _ = get_models()
        response = quality_model.invoke("Hello, respond in one sentence.")
        print("[Groq Test] Connection successful. Model response:")
        print(response.content)
    except Exception as exc:
        print(f"[Groq Test] Error: {exc}")

def build_crew():
    """
    Placeholder for future CrewAI-based orchestration.
    """
    raise NotImplementedError("Crew construction is not implemented yet.")


# ==========================================
# LANGGRAPH NODES (The Actors & Critics)
# ==========================================

def translation_node(state: PanelState) -> PanelState:
    print(f"[LangGraph] Node: Translation (Attempt {state['translation_retries'] + 1}/3)")
    

    output = run_translation_agent(
        state["raw_text"], 
        quality_model=state["quality_model"], 
        character_name=state["character_name"]
    )
    scores = grade_translation_output(
        original=state["raw_text"], 
        translated=output, 
        fast_model=state["fast_model"]
    )
    
    state["translated_output"] = output
    state["translation_scores"] = scores
    state["translation_retries"] += 1
    return state

def cultural_node(state: PanelState) -> PanelState:
    print(f"[LangGraph] Node: Cultural (Attempt {state['cultural_retries'] + 1}/3)")
    
    if state["skip_agents"]:
        state["cultural_output"] = state["translated_output"]
        state["cultural_scores"] = {"pass": True, "skipped": True}
        return state

    output = run_cultural_adaptor(
        state["translated_output"], 
        quality_model=state["quality_model"]
    )
    scores = grade_cultural_output(
        original=state["translated_output"], 
        adapted=output, 
        fast_model=state["fast_model"]
    )
    
    state["cultural_output"] = output
    state["cultural_scores"] = scores
    state["cultural_retries"] += 1
    return state

def continuity_node(state: PanelState) -> PanelState:
    print(f"[LangGraph] Node: Continuity (Attempt {state['continuity_retries'] + 1}/3)")
    
    if state["skip_agents"]:
        state["continuity_output"] = state["cultural_output"]
        state["continuity_scores"] = {"pass": True, "skipped": True}
        return state

    output = run_continuity_director(
        adapted_text=state["cultural_output"],
        character_profile=state["character_profile"],
        quality_model=state["quality_model"]
    )
    scores = grade_continuity_output(
        adapted_text=state["cultural_output"],
        continuity_text=output,
        character_name=state["character_name"],
        fast_model=state["fast_model"]
    )
    
    state["continuity_output"] = output
    state["continuity_scores"] = scores
    state["continuity_retries"] += 1
    return state

def typesetting_node(state: PanelState) -> PanelState:
    print(f"[LangGraph] Node: Typesetting (Attempt {state['typesetting_retries'] + 1}/3)")
    
    output = run_typesetting_editor(
        state["continuity_output"],
        bubble_type=state["bubble_type"],
        quality_model=state["quality_model"],
        bubble_char_limit=state["bubble_char_limit"]
    )
    scores = grade_typesetting_output(
        original=state["continuity_output"],
        final=output,
        bubble_type=state["bubble_type"],
        fast_model=state["fast_model"],
        bubble_char_limit=state["bubble_char_limit"]
    )
    
    state["final_output"] = output
    state["typesetting_scores"] = scores
    state["typesetting_retries"] += 1
    return state

# ==========================================
# LANGGRAPH ROUTING (The Conditional Edges)
# ==========================================

def route_translation(state: PanelState) -> str:
    if state["translation_scores"].get("pass") or state["translation_retries"] >= 3:
        return "cultural_node"
    return "translation_node"

def route_cultural(state: PanelState) -> str:
    if state["cultural_scores"].get("pass") or state["cultural_retries"] >= 3:
        return "continuity_node"
    return "cultural_node"

def route_continuity(state: PanelState) -> str:
    if state["continuity_scores"].get("pass") or state["continuity_retries"] >= 3:
        return "typesetting_node"
    return "continuity_node"

def route_typesetting(state: PanelState) -> str:
    if state["typesetting_scores"].get("pass") or state["typesetting_retries"] >= 3:
        return END
    return "typesetting_node"

# ==========================================
# GRAPH COMPILATION
# ==========================================

# 1. Initialize the Graph
workflow = StateGraph(PanelState)

# 2. Add the Nodes
workflow.add_node("translation_node", translation_node)
workflow.add_node("cultural_node", cultural_node)
workflow.add_node("continuity_node", continuity_node)
workflow.add_node("typesetting_node", typesetting_node)

# 3. Set the Entry Point
workflow.set_entry_point("translation_node")

# 4. Add the Conditional Edges (The Retry Loops)
workflow.add_conditional_edges("translation_node", route_translation)
workflow.add_conditional_edges("cultural_node", route_cultural)
workflow.add_conditional_edges("continuity_node", route_continuity)
workflow.add_conditional_edges("typesetting_node", route_typesetting)

# 5. Compile the Engine
lumina_engine = workflow.compile()


def _flagged_from_scores(scores: Dict[str, Any]) -> bool:
    """
    Flag True if any non-skipped/non-batch numeric score is < 6.
    """
    for agent_name, agent_scores in scores.items():
        if not isinstance(agent_scores, dict):
            continue
        if agent_scores.get("skipped"):
            continue
        if agent_scores.get("batch_translated"):
            continue
        for key, val in agent_scores.items():
            if key == "pass":
                continue
            if isinstance(val, bool):
                continue
            if val is None:
                continue
            if isinstance(val, (int, float)) and val < 6:
                return True
    return False


def process_chapter(chapter_data: Dict[str, Any], client: Groq) -> list[Dict[str, Any]]:
    """
    Process an uploaded chapter JSON through the 4-step pipeline per panel.

    Returns a list of panel results:
      - panel_id
      - original
      - final_output
      - scores
      - flagged
    """
    quality_model, fast_model = get_models()
    manga_id = str(chapter_data.get("manga_id") or "unknown").strip() or "unknown"
    chapter = int(chapter_data.get("chapter") or 0)

    panels = chapter_data.get("panels") or []

    # Collect panel jobs first so we can run Step 0 translation across ALL panels
    # before extracting/updating character profiles.
    panel_jobs: list[Dict[str, Any]] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue

        panel_id = panel.get("panel_id") or panel.get("id") or panel.get("index")
        panel_id = str(panel_id) if panel_id is not None else ""

        character_name = str(panel.get("character") or "").strip()
        original_japanese = str(panel.get("text") or "")
        if not character_name or not original_japanese:
            continue

        skip_agents = character_name in (
            "NARRATION",
            "TEACHER",
            "STUDENT",
            "TITLE_CARD",
            "SFX",
            "NARRATOR",
            "CAPTION",
            "SOUND_EFFECT",
        )

        bubble_char_limit = panel.get("bubble_char_limit")
        bubble_char_limit = (
            int(bubble_char_limit) if bubble_char_limit is not None else None
        )

        bubble_type = str(panel.get("bubble_type") or "medium").strip()

        panel_jobs.append(
            {
                "panel_id": panel_id,
                "character_name": character_name,
                "original_japanese": original_japanese,
                "skip_agents": skip_agents,
                "bubble_type": bubble_type,
                "bubble_char_limit": bubble_char_limit,
            }
        )

    # STEP 0 — Translate one page at a time.
    translated_outputs: list[str] = []
    translation_scores_by_panel: list[Dict[str, Any]] = []
    translated_lines_by_character: Dict[str, list[str]] = {}

    # Group panel jobs by page number from panel_id prefix (e.g., "1-2" -> page "1").
    page_groups: Dict[str, list[Dict[str, Any]]] = {}
    page_order: list[str] = []
    for job in panel_jobs:
        panel_id = str(job["panel_id"])
        page_key = panel_id.split("-", 1)[0] if "-" in panel_id else "0"
        job["page_key"] = page_key
        if page_key not in page_groups:
            page_groups[page_key] = []
            page_order.append(page_key)
        page_groups[page_key].append(job)

    translated_by_panel_id: Dict[str, str] = {}
    score_by_panel_id: Dict[str, Dict[str, Any]] = {}
    page_translation_score_by_page: Dict[str, Dict[str, Any]] = {}

    for page_key in page_order:
        page_jobs = page_groups[page_key]

        # TITLE_CARD panels are translated separately with literal metadata behavior.
        normal_page_panels: list[Dict[str, Any]] = []
        for job in page_jobs:
            panel_id = job["panel_id"]
            character_name = job["character_name"]
            original_japanese = job["original_japanese"]

            if character_name == "TITLE_CARD":
                try:
                    translated = translate_literal_metadata(original_japanese, client)
                    if not translated:
                        translated = original_japanese
                    score = grade_translation_output(
                        original=original_japanese,
                        translated=translated,
                        fast_model=fast_model,
                    )
                except Exception as exc:
                    print(
                        f"[ProcessChapter] TITLE_CARD literal translation failed for panel_id='{panel_id}': {exc}"
                    )
                    translated = original_japanese
                    score = {
                        "contextual_accuracy": 0,
                        "tone_preservation": 0,
                        "naturalness": 0,
                        "pass": False,
                    }

                translated_by_panel_id[panel_id] = translated
                score_by_panel_id[panel_id] = score
            else:
                normal_page_panels.append(
                    {
                        "panel_id": panel_id,
                        "character": character_name,
                        "text": original_japanese,
                    }
                )

        if normal_page_panels:
            try:
                page_translation_map = translate_page(normal_page_panels, client)
            except Exception as exc:
                print(f"[ProcessChapter] Page translation failed for page='{page_key}': {exc}")
                page_translation_map = {}

            for panel in normal_page_panels:
                panel_id = str(panel["panel_id"])
                original_japanese = str(panel["text"])
                character_name = str(panel["character"])
                
                translated = page_translation_map.get(panel_id)
                
                # FALLBACK: If the batch translator missed this panel or returned Japanese
                

                try:
                    score = grade_translation_output(
                        original=original_japanese,
                        translated=translated,
                        fast_model=fast_model,
                    )
                except Exception:
                    score = {
                        "contextual_accuracy": 0,
                        "tone_preservation": 0,
                        "naturalness": 0,
                        "pass": False,
                    }
                translated_by_panel_id[panel_id] = translated
                score_by_panel_id[panel_id] = score

        # Real page-level translation quality signal:
        # grade the first non-NARRATION panel on this page.
        sample_job = next(
            (
                pj
                for pj in page_jobs
                if str(pj["character_name"]).strip() != "NARRATION"
            ),
            None,
        )
        if sample_job is not None:
            sample_panel_id = sample_job["panel_id"]
            sample_original = sample_job["original_japanese"]
            sample_translated = translated_by_panel_id.get(sample_panel_id) or sample_original
            try:
                page_translation_score_by_page[page_key] = grade_translation_output(
                    original=sample_original,
                    translated=sample_translated,
                    fast_model=fast_model,
                )
            except Exception:
                page_translation_score_by_page[page_key] = {
                    "contextual_accuracy": 0,
                    "tone_preservation": 0,
                    "naturalness": 0,
                    "pass": False,
                }
        else:
            page_translation_score_by_page[page_key] = {
                "contextual_accuracy": 0,
                "tone_preservation": 0,
                "naturalness": 0,
                "pass": False,
            }

    for job in panel_jobs:
        panel_id = job["panel_id"]
        character_name = job["character_name"]
        original_japanese = job["original_japanese"]
        translated_output = translated_by_panel_id.get(panel_id) or original_japanese
        translation_scores = score_by_panel_id.get(panel_id) or {
            "contextual_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
        }
        translated_outputs.append(translated_output)
        translation_scores_by_panel.append(translation_scores)
        translated_lines_by_character.setdefault(character_name, []).append(
            translated_output
        )

    # Extract character profiles from translated English dialogue.
    chapter_for_profiles = {"panels": []}
    for character_name, lines in translated_lines_by_character.items():
        for line in lines:
            chapter_for_profiles["panels"].append(
                {"character": character_name, "text": line}
            )

    try:
        inferred_profiles = extract_profiles(chapter_for_profiles, quality_model)
        for profile in inferred_profiles:
            inferred_character_name = str(profile.get("name") or "").strip()
            if not inferred_character_name:
                continue
            update_or_create_profile(
                character_name=inferred_character_name,
                new_profile=profile,
                vector_store=vector_store,
                manga_id=manga_id,
                fast_model = fast_model
            )

            # Seed continuity memory from profile sample dialogue on chapter 1 cold start.
            sample_dialogue = profile.get("sample_dialogue") or []
            if isinstance(sample_dialogue, list):
                for sample_line in sample_dialogue:
                    sample_line_str = str(sample_line).strip()
                    if not sample_line_str:
                        continue
                    add_approved_line(
                        panel_id="profile_seed",
                        character_name=inferred_character_name,
                        manga_id=manga_id,
                        original_japanese="",
                        final_output=sample_line_str,
                        scores={"profile_seed": True, "pass": True},
                        chapter=0,
                    )
    except Exception as exc:
        print(f"[ProcessChapter] Profile extraction failed (continuing anyway): {exc}")

    # Run Steps A/B/C per panel using the already-translated English.
    results: list[Dict[str, Any]] = []
    for i, job in enumerate(panel_jobs):
        panel_id = job["panel_id"]
        character_name = job["character_name"]
        original_japanese = job["original_japanese"]
        skip_agents = job["skip_agents"]
        bubble_type = job["bubble_type"]
        bubble_char_limit = job["bubble_char_limit"]

        translated_output = translated_outputs[i]
        translation_scores = translation_scores_by_panel[i]

        character_profile = query_character_profile_dict(
            character_name,
            manga_id=manga_id,
        ) or {
            "name": character_name,
            "role": "character",
            "speech_style": "",
            "forbidden_phrases": [],
            "speech_rules": [],
        }
        character_profile["manga_id"] = manga_id

        # Initialize the LangGraph State
        initial_state: PanelState = {
            "raw_text": original_japanese,
            "character_name": character_name,
            "bubble_type": bubble_type,
            "bubble_char_limit": bubble_char_limit,
            "character_profile": character_profile,
            "skip_agents": skip_agents,
            "quality_model": quality_model,
            "fast_model": fast_model,
            
            "detected_language": "japanese", # Passed down if needed
            "translated_output": "",
            "cultural_output": "",
            "continuity_output": "",
            "final_output": "",
            
            "translation_scores": {},
            "cultural_scores": {},
            "continuity_scores": {},
            "typesetting_scores": {},
            
            "translation_retries": 0,
            "cultural_retries": 0,
            "continuity_retries": 0,
            "typesetting_retries": 0,
        }

        # Execute the compiled LangGraph
        pipeline_result = lumina_engine.invoke(initial_state)

        final_output = pipeline_result.get("final_output", "")
        scores = {
            "translation": translation_scores,
            "cultural": pipeline_result.get("cultural_scores"),
            "continuity": pipeline_result.get("continuity_scores"),
            "typesetting": pipeline_result.get("typesetting_scores"),
        }
        flagged = _flagged_from_scores(scores)

        add_approved_line(
            panel_id=panel_id,
            character_name=character_name,
            manga_id=manga_id,
            original_japanese=original_japanese,
            final_output=final_output,
            scores=scores,
            flagged=flagged,
            chapter=chapter,
        )

        results.append(
            {
                "panel_id": panel_id,
                "original": original_japanese,
                "final_output": final_output,
                "scores": scores,
                "page_translation_score": page_translation_score_by_page.get(
                    str(job.get("page_key", "0"))
                ),
                "flagged": flagged,
            }
        )

    mark_chapter_complete(manga_id, chapter)
    return results


def test_full_pipeline() -> None:
    """
    Manual end-to-end test of the Lumina Pipeline using the LangGraph engine.
    """
    try:
        quality_model, fast_model = get_models()
    except Exception as exc:
        print(f"[Pipeline Test] Failed to initialize clients: {exc}")
        return

    raw_text = "たとえ天が崩れ落ちようとも、私は一歩も退かない。"
    character_name = "Kira"
    bubble_type = "medium"

    # 1. Build the initial LangGraph state for the test
    initial_state: PanelState = {
        "raw_text": raw_text,
        "character_name": character_name,
        "bubble_type": bubble_type,
        "bubble_char_limit": None,
        "character_profile": {
            "name": character_name,
            "role": "character",
            "speech_style": "",
            "forbidden_phrases": [],
            "speech_rules": [],
            "manga_id": "default",
        },
        "skip_agents": False,
        "quality_model": quality_model,
        "fast_model": fast_model,
        
        "detected_language": detect_language(raw_text),
        "translated_output": "",
        "cultural_output": "",
        "continuity_output": "",
        "final_output": "",
        
        "translation_scores": {},
        "cultural_scores": {},
        "continuity_scores": {},
        "typesetting_scores": {},
        
        "translation_retries": 0,
        "cultural_retries": 0,
        "continuity_retries": 0,
        "typesetting_retries": 0,
    }

    print("\n[Pipeline Test] Executing LangGraph...")
    
    # 2. Invoke the compiled graph
    result = lumina_engine.invoke(initial_state)

    # 3. Print the results directly from the final state dictionary
    print("\n[Pipeline Test] Cultural output:")
    print(result["cultural_output"])

    print("\n[Pipeline Test] Continuity output:")
    print(result["continuity_output"])

    print("\n[Pipeline Test] Final typeset output:")
    print(result["final_output"])

    print("\n[Pipeline Test] Cultural scores:")
    print(result["cultural_scores"])

    print("\n[Pipeline Test] Continuity scores:")
    print(result["continuity_scores"])

    print("\n[Pipeline Test] Typesetting scores:")
    print(result["typesetting_scores"])


def main() -> None:
    """
    CLI-style entrypoint for running the pipeline.

    For now, this runs a simple end-to-end test of the full pipeline.
    """
    test_full_pipeline()


if __name__ == "__main__":
    main()

