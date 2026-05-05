"""
Streamlit frontend for the NexusRAG Pipeline — Enterprise Edition.

Dark mode redesign: terminal-minimal aesthetic with cyan accent system.
Split-pane layout:
  Left  (60%) — Project sidebar + chat-style input
  Right (40%) — Live pipeline context: retrieval + agent steps
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv()
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import process_chapter  # type: ignore
from memory.vector_store import (  # type: ignore
    delete_chapter_data,
    delete_manga_data,
    load_characters_from_json,
    query_approved_lines_for_chapter,
)
from utils.project_manager import (  # type: ignore
    create_project,
    delete_project,
    load_projects,
    remove_chapter,
)


# ─────────────────────────────── helpers ────────────────────────────────────

@st.cache_resource
def get_client():
    from utils.groq_client import load_environment
    return load_environment()


def _load_bubble_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "data" / "bubble_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bubble_limit(bubble_type: str, config: Dict[str, Any]) -> int:
    default_max = int(config.get("default_max_chars", 80))
    bubbles = config.get("bubbles", {}) or {}
    val = bubbles.get(bubble_type)
    return int(val) if val is not None else default_max


def _build_panel_character_map(chapter_data: Dict[str, Any]) -> Dict[str, str]:
    panel_id_to_character: dict[str, str] = {}
    for panel in (chapter_data.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        pid = panel.get("panel_id") or panel.get("id") or panel.get("index")
        pid_str = str(pid) if pid is not None else ""
        character = str(panel.get("character") or "").strip()
        if pid_str:
            panel_id_to_character[pid_str] = character
    return panel_id_to_character


@st.cache_resource
def _load_characters_for_manga(manga_id: str) -> None:
    characters_folder = PROJECT_ROOT / "data" / "characters"
    load_characters_from_json(characters_folder, manga_id=manga_id)


def _esc(text: str) -> str:
    import html
    return html.escape(str(text or ""))


# ───────────────────────────── CSS injection ────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

        /* ── CSS variables ── */
        :root {
            --bg-base:       #0d0f12;
            --bg-surface:    #13161b;
            --bg-elevated:   #1a1e26;
            --bg-hover:      #1f2430;
            --border:        #252a35;
            --border-bright: #2e3547;
            --accent:        #00e5c8;
            --accent-dim:    #00b89e;
            --accent-glow:   rgba(0, 229, 200, 0.12);
            --accent-glow2:  rgba(0, 229, 200, 0.06);
            --text-primary:  #e8eaf0;
            --text-secondary:#8b92a8;
            --text-muted:    #4a5168;
            --text-accent:   #00e5c8;
            --red:           #ff5f6d;
            --red-bg:        rgba(255, 95, 109, 0.1);
            --green:         #00c896;
            --green-bg:      rgba(0, 200, 150, 0.1);
            --yellow:        #f5c842;
            --yellow-bg:     rgba(245, 200, 66, 0.1);
            --font-mono:     'IBM Plex Mono', 'Courier New', monospace;
            --font-sans:     'IBM Plex Sans', sans-serif;
        }

        /* ── global reset ── */
        html, body, [class*="css"] {
            font-family: var(--font-sans) !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }

        /* hide default streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }

        .block-container {
            padding: 1.5rem 2rem 2rem 2rem !important;
            max-width: 100% !important;
            background: var(--bg-base) !important;
        }

        /* ── sidebar ── */
        section[data-testid="stSidebar"] {
            background: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1.5rem 1rem !important;
            background: var(--bg-surface) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        /* ── selectbox + text inputs ── */
        .stSelectbox > div > div,
        .stTextInput > div > div > input {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            color: var(--text-primary) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.78rem !important;
        }
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-glow) !important;
        }
        /* dropdown list items */
        [data-baseweb="popover"] li,
        [data-baseweb="select"] li {
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
        }
        [data-baseweb="popover"] li:hover {
            background: var(--bg-hover) !important;
            color: var(--accent) !important;
        }

        /* ── text area ── */
        .stTextArea textarea {
            font-family: var(--font-mono) !important;
            font-size: 0.78rem !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-glow) !important;
        }

        /* ── file uploader ── */
        [data-testid="stFileUploader"] {
            background: var(--bg-elevated) !important;
            border: 1px dashed var(--border-bright) !important;
            border-radius: 6px !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent) !important;
            background: var(--accent-glow2) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] p,
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--text-secondary) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.75rem !important;
        }

        /* ── buttons ── */
        .stButton button {
            font-family: var(--font-mono) !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            border-radius: 4px !important;
            letter-spacing: 0.05em !important;
            transition: all 0.15s ease !important;
        }
        /* primary */
        .stButton button[kind="primary"],
        .stButton button[data-testid*="primary"] {
            background: var(--accent) !important;
            color: #0d0f12 !important;
            border: none !important;
            font-weight: 600 !important;
        }
        .stButton button[kind="primary"]:hover {
            background: var(--accent-dim) !important;
            box-shadow: 0 0 16px var(--accent-glow) !important;
        }
        /* secondary/default */
        .stButton button:not([kind="primary"]) {
            background: var(--bg-elevated) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border) !important;
        }
        .stButton button:not([kind="primary"]):hover {
            background: var(--bg-hover) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-bright) !important;
        }

        /* ── download button ── */
        .stDownloadButton button {
            background: var(--bg-elevated) !important;
            color: var(--accent) !important;
            border: 1px solid var(--border-bright) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.72rem !important;
            border-radius: 4px !important;
            letter-spacing: 0.05em !important;
        }
        .stDownloadButton button:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 0 10px var(--accent-glow) !important;
        }

        /* ── spinner ── */
        .stSpinner > div {
            border-top-color: var(--accent) !important;
        }
        .stSpinner p {
            color: var(--text-secondary) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.75rem !important;
        }

        /* ── alerts / warnings ── */
        .stAlert {
            background: var(--bg-elevated) !important;
            border-radius: 4px !important;
            font-size: 0.8rem !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stAlert"] {
            color: var(--text-primary) !important;
        }

        /* ── expander ── */
        .streamlit-expanderHeader {
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            font-family: var(--font-mono) !important;
            font-size: 0.75rem !important;
        }
        .streamlit-expanderContent {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
        }

        /* ── divider ── */
        hr {
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 1.25rem 0 !important;
        }

        /* ── labels (st.selectbox label etc) ── */
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label,
        .stFileUploader label {
            color: var(--text-secondary) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.65rem !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
        }

        /* ══════════════════════════════
           Custom component classes
        ══════════════════════════════ */

        /* ── wordmark ── */
        .nx-wordmark {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 0 0 1.25rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }
        .nx-logo-mark {
            width: 32px;
            height: 32px;
            border: 1.5px solid var(--accent);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--accent-glow);
            flex-shrink: 0;
        }
        .nx-logo-inner {
            width: 14px;
            height: 14px;
            background: var(--accent);
            clip-path: polygon(0 0, 100% 0, 100% 60%, 60% 100%, 0 100%);
        }
        .nx-wordmark-title {
            font-family: var(--font-mono);
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--text-primary);
        }
        .nx-wordmark-sub {
            font-family: var(--font-sans);
            font-size: 0.68rem;
            font-weight: 300;
            color: var(--text-muted);
            letter-spacing: 0.02em;
            margin-top: 2px;
        }
        .nx-version-pill {
            margin-left: auto;
            font-family: var(--font-mono);
            font-size: 0.6rem;
            background: var(--accent-glow);
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.08em;
        }

        /* ── pane label ── */
        .nx-pane-label {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .nx-pane-label::before {
            content: '';
            width: 3px;
            height: 10px;
            background: var(--accent);
            border-radius: 2px;
            display: inline-block;
        }

        /* ── sidebar section header ── */
        .nx-sidebar-section {
            font-family: var(--font-mono);
            font-size: 0.58rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin: 1.2rem 0 0.5rem 0;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border);
        }

        /* ── project row ── */
        .nx-proj-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            margin-bottom: 5px;
            background: var(--bg-elevated);
            font-size: 0.78rem;
            transition: border-color 0.15s, background 0.15s;
        }
        .nx-proj-row:hover {
            border-color: var(--border-bright);
            background: var(--bg-hover);
        }
        .nx-proj-active {
            border-color: var(--accent) !important;
            background: var(--accent-glow2) !important;
        }
        .nx-proj-dot {
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--accent);
            flex-shrink: 0;
        }
        .nx-proj-name {
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-primary);
            flex: 1;
        }
        .nx-proj-meta {
            font-family: var(--font-mono);
            font-size: 0.58rem;
            color: var(--text-muted);
        }

        /* ── agent info ── */
        .nx-agent-row {
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }
        .nx-agent-num {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--accent);
            letter-spacing: 0.12em;
            margin-bottom: 2px;
        }
        .nx-agent-name {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        .nx-agent-desc {
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 1px;
        }

        /* ── chat bubbles ── */
        .nx-chat-wrap {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 1rem;
        }
        .nx-msg {
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            line-height: 1.6;
            max-width: 92%;
        }
        .nx-msg-user {
            background: var(--accent-glow);
            border: 1px solid var(--accent);
            color: var(--text-primary);
            align-self: flex-end;
            border-radius: 6px 6px 2px 6px;
        }
        .nx-msg-system {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            align-self: flex-start;
            border-radius: 6px 6px 6px 2px;
        }
        .nx-msg-label {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .nx-msg-user .nx-msg-label { color: var(--accent); }
        .nx-msg-meta {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--text-muted);
            margin-top: 6px;
            text-align: right;
        }

        /* ── cards ── */
        .nx-card {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 10px;
            transition: border-color 0.15s;
        }
        .nx-card:hover { border-color: var(--border-bright); }

        .nx-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .nx-card-title {
            font-family: var(--font-mono);
            font-size: 0.62rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        /* ── badges ── */
        .nx-badge {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            padding: 3px 9px;
            border-radius: 20px;
            letter-spacing: 0.08em;
            font-weight: 600;
            text-transform: uppercase;
        }
        .nx-badge-pass {
            background: var(--green-bg);
            color: var(--green);
            border: 1px solid rgba(0,200,150,0.3);
        }
        .nx-badge-fail {
            background: var(--red-bg);
            color: var(--red);
            border: 1px solid rgba(255,95,109,0.3);
        }
        .nx-badge-neutral {
            background: rgba(139,146,168,0.1);
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }

        /* ── score chips ── */
        .nx-score-row {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }
        .nx-score-chip {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            background: var(--bg-surface);
            border: 1px solid var(--border);
            padding: 3px 8px;
            border-radius: 3px;
            color: var(--text-secondary);
        }

        /* ── step connector ── */
        .nx-step-connector {
            width: 1px;
            height: 16px;
            background: linear-gradient(to bottom, var(--accent-dim), var(--border));
            margin: 0 auto 0 18px;
            opacity: 0.5;
        }

        /* ── diff view ── */
        .nx-diff {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        .nx-diff-cell {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 10px 12px;
        }
        .nx-diff-cell-label {
            font-family: var(--font-mono);
            font-size: 0.52rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 5px;
        }
        .nx-diff-cell-value {
            font-size: 0.78rem;
            color: var(--text-primary);
            line-height: 1.55;
        }

        /* ── panel result row ── */
        .nx-panel-row {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 5px;
            padding: 12px 14px;
            margin-bottom: 8px;
            font-size: 0.78rem;
            transition: border-color 0.15s;
        }
        .nx-panel-row:hover { border-color: var(--border-bright); }
        .nx-panel-row-flagged {
            border-left: 3px solid var(--red);
        }
        .nx-panel-row-ok {
            border-left: 3px solid var(--green);
        }
        .nx-panel-id {
            font-family: var(--font-mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ── history item ── */
        .nx-history-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 4px;
            margin-bottom: 6px;
            background: var(--bg-elevated);
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .nx-history-item:hover {
            border-color: var(--accent);
            background: var(--accent-glow2);
        }
        .nx-history-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent);
            flex-shrink: 0;
        }

        /* ── project header ── */
        .nx-proj-header {
            padding: 12px 16px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .nx-proj-header-name {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        .nx-proj-header-meta {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: var(--text-muted);
            margin-top: 3px;
            letter-spacing: 0.05em;
        }

        /* ── source block ── */
        .nx-source-text {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.65;
            background: var(--bg-surface);
            padding: 10px 12px;
            border-radius: 4px;
            border: 1px solid var(--border);
        }

        /* ── constraint row ── */
        .nx-constraint-row {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        .nx-constraint-row strong {
            color: var(--text-primary);
        }

        /* ── run header ── */
        .nx-run-header {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 2px;
        }
        .nx-run-meta {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        /* ── empty state ── */
        .nx-empty {
            text-align: center;
            padding: 4rem 1rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            line-height: 1.8;
            border: 1px dashed var(--border);
            border-radius: 6px;
            background: var(--bg-surface);
        }
        .nx-empty-icon {
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
            opacity: 0.3;
        }

        /* ── section title ── */
        .nx-section-label {
            font-family: var(--font-mono);
            font-size: 0.58rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin: 1.2rem 0 0.6rem 0;
        }

        /* ── preview message ── */
        .nx-preview-msg {
            background: var(--accent-glow);
            border: 1px solid var(--accent);
            border-radius: 6px 6px 2px 6px;
            padding: 10px 14px;
            margin-bottom: 0.75rem;
            font-size: 0.8rem;
            color: var(--text-primary);
            align-self: flex-end;
            max-width: 92%;
            display: inline-block;
        }
        .nx-preview-label {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            letter-spacing: 0.12em;
            color: var(--accent);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        /* ── drop hint ── */
        .nx-drop-hint {
            font-size: 0.73rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            padding: 0.4rem 0;
        }

        /* ── warning box ── */
        .nx-warning {
            background: var(--yellow-bg);
            border: 1px solid rgba(245, 200, 66, 0.3);
            border-radius: 6px;
            padding: 12px 14px;
            font-size: 0.78rem;
            color: var(--yellow);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────── render helpers ──────────────────────────────────

def _wordmark() -> None:
    st.markdown(
        """
        <div class="nx-wordmark">
            <div class="nx-logo-mark">
                <div class="nx-logo-inner"></div>
            </div>
            <div>
                <div class="nx-wordmark-title">NexusRAG</div>
                <div class="nx-wordmark-sub">Autonomous Manga Localization Engine</div>
            </div>
            <span class="nx-version-pill">v0.1.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pane_label(text: str) -> None:
    st.markdown(f'<div class="nx-pane-label">{text}</div>', unsafe_allow_html=True)


def _badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="nx-badge nx-badge-{kind}">{text}</span>'


def _score_chips(scores: Dict[str, Any]) -> str:
    """Return score chip HTML, or empty string with NO trailing tags if nothing to show."""
    if not scores:
        return ""
    chips = ""
    for k, v in scores.items():
        if k == "pass":
            continue
        label = k.replace("_", " ")
        chips += f'<span class="nx-score-chip">{label}: {v if v is not None else "—"}</span>'
    # Only wrap in a div if there are actual chips to show
    if not chips:
        return ""
    return f'<div class="nx-score-row">{chips}</div>'


def _render_agent_card(
    title: str,
    step_num: int,
    input_text: str,
    output_text: str,
    scores: Dict[str, Any],
) -> None:
    passed = scores.get("pass", None)
    if passed is True:
        badge = _badge("PASS", "pass")
    elif passed is False:
        badge = _badge("FAIL", "fail")
    else:
        badge = _badge("—", "neutral")

    score_html = _score_chips(scores)

    # Build the card HTML as a list of parts to avoid injecting empty strings
    # into an f-string that could confuse Streamlit's HTML renderer
    card_parts = [
        '<div class="nx-card">',
        '<div class="nx-card-header">',
        f'<span class="nx-card-title"><span style="color:var(--accent);margin-right:6px;">#{step_num}</span>{_esc(title)}</span>',
        badge,
        '</div>',
        '<div class="nx-diff">',
        '<div class="nx-diff-cell">',
        '<div class="nx-diff-cell-label">↳ Input</div>',
        f'<div class="nx-diff-cell-value">{_esc(input_text)}</div>',
        '</div>',
        '<div class="nx-diff-cell">',
        '<div class="nx-diff-cell-label">⇒ Output</div>',
        f'<div class="nx-diff-cell-value">{_esc(output_text)}</div>',
        '</div>',
        '</div>',
    ]
    if score_html:
        card_parts.append(score_html)
    card_parts.append('</div>')

    st.markdown("".join(card_parts), unsafe_allow_html=True)


def _render_panel_result_row(r: Dict[str, Any], character: str) -> None:
    panel_id = str(r.get("panel_id", ""))
    original = str(r.get("original", ""))
    final_output = str(r.get("final_output", ""))
    flagged = bool(r.get("flagged"))
    flag_class = "nx-panel-row-flagged" if flagged else "nx-panel-row-ok"
    flag_label = _badge("FLAGGED", "fail") if flagged else _badge("OK", "pass")
    char_label = f'<span style="color:var(--accent);font-weight:500;">{_esc(character)}</span>' if character else ""

    st.markdown(
        f"""
        <div class="nx-panel-row {flag_class}">
            <div class="nx-panel-id">
                <span>panel {panel_id}</span>
                {char_label}
                {flag_label}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px;">
                <div>
                    <div class="nx-diff-cell-label" style="font-family:var(--font-mono);font-size:.52rem;color:var(--text-muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px;">Japanese</div>
                    <div style="font-size:.78rem;color:var(--text-secondary);line-height:1.55;">{_esc(original)}</div>
                </div>
                <div>
                    <div class="nx-diff-cell-label" style="font-family:var(--font-mono);font-size:.52rem;color:var(--text-muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px;">Final</div>
                    <div style="font-size:.78rem;color:var(--text-primary);font-weight:500;line-height:1.55;">{_esc(final_output)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_right_pane_for_result(result: Dict[str, Any], raw_text: str, bubble_type: str) -> None:
    _pane_label("Retrieval Context")

    detected_language = result.get("detected_language", "unknown")
    final_output = result.get("final_output", "")
    bubble_cfg = _load_bubble_config()
    max_chars = _bubble_limit(bubble_type, bubble_cfg)
    char_count = len(final_output)
    within = char_count <= max_chars

    st.markdown(
        f"""
        <div class="nx-card">
            <div class="nx-card-header">
                <span class="nx-card-title">Source Document</span>
                {_badge(detected_language.upper(), "neutral")}
            </div>
            <div class="nx-source-text">{_esc(raw_text)}</div>
        </div>
        <div class="nx-card">
            <div class="nx-card-header">
                <span class="nx-card-title">Typeset Constraints</span>
                {_badge(f"{char_count}/{max_chars} chars", "pass" if within else "fail")}
            </div>
            <div class="nx-constraint-row">
                Bubble type: <strong>{bubble_type}</strong> &nbsp;·&nbsp; Max chars: <strong>{max_chars}</strong>
                &nbsp;·&nbsp; Status: <strong style="color:{'var(--green)' if within else 'var(--red)'}">{'within limit' if within else 'OVERFLOW'}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    _pane_label("Agent Processing Steps")

    # Build pipeline steps dynamically — only render a step if its output key
    # is present and non-empty. The first active step always receives raw_text
    # as its input. Each subsequent step chains from the previous output.
    # The last step rendered always displays final_output as its output value.
    all_steps = [
        {"title": "Translation",        "num": 0, "output_key": "translated_output", "scores_key": "translation_scores"},
        {"title": "Cultural Adaptation","num": 1, "output_key": "cultural_output",   "scores_key": "cultural_scores"},
        {"title": "Continuity Check",   "num": 2, "output_key": "continuity_output", "scores_key": "continuity_scores"},
        {"title": "Typesetting",        "num": 3, "output_key": "final_output",      "scores_key": "typesetting_scores"},
    ]

    active_steps = [s for s in all_steps if str(result.get(s["output_key"]) or "").strip()]

    if not active_steps:
        st.markdown(
            '<div class="nx-empty"><div class="nx-empty-icon">◈</div>'
            'No agent output data available for this panel.</div>',
            unsafe_allow_html=True,
        )
    else:
        prev_output = raw_text
        for i, step in enumerate(active_steps):
            is_last = i == len(active_steps) - 1
            output_val = str(result.get(step["output_key"]) or "")
            # Always use final_output on the last rendered card so the terminal
            # node reflects the true end result regardless of which step it is.
            display_output = final_output if is_last else output_val
            _render_agent_card(
                title=step["title"],
                step_num=step["num"],
                input_text=prev_output,
                output_text=display_output,
                scores=result.get(step["scores_key"]) or {},
            )
            if not is_last:
                st.markdown('<div class="nx-step-connector"></div>', unsafe_allow_html=True)
            prev_output = output_val


def _render_right_pane_empty() -> None:
    st.markdown(
        """
        <div class="nx-empty">
            <div class="nx-empty-icon">◈</div>
            No pipeline run yet.<br/>
            Upload a chapter JSON and run the pipeline<br/>to see live agent steps here.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_right_pane_history(history_item: Dict[str, Any], bubble_type: str) -> None:
    saved_results = history_item.get("saved_results", [])
    saved_panel_map = history_item.get("saved_panel_map", {})

    _pane_label(f"Saved Translations — Chapter {history_item.get('chapter', '?')}")

    if not saved_results:
        st.markdown(
            '<div class="nx-empty"><div class="nx-empty-icon">◈</div>No saved translations found for this chapter.</div>',
            unsafe_allow_html=True,
        )
        return

    for r in saved_results:
        panel_id = str(r.get("panel_id", ""))
        char = saved_panel_map.get(panel_id, "")
        _render_panel_result_row(r, char)

    st.download_button(
        label="↓ Download Chapter JSON",
        data=json.dumps(saved_results, ensure_ascii=False, indent=2),
        file_name=f"chapter_{history_item.get('chapter', 'saved')}_results.json",
        mime="application/json",
        key=f"dl_saved_{history_item.get('chapter', 'x')}",
    )


# ──────────────────────────── sidebar ───────────────────────────────────────

def _sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:.8rem;font-weight:600;letter-spacing:.18em;color:var(--text-primary);margin-bottom:1.2rem;padding-bottom:.8rem;border-bottom:1px solid var(--border);">NEXUS</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nx-sidebar-section">Projects</div>', unsafe_allow_html=True)

        projects = load_projects()
        selected_manga_id = st.session_state.get("selected_project_manga_id")

        if projects:
            for proj in projects:
                manga_id = str(proj.get("manga_id") or "").strip()
                display_name = str(proj.get("display_name") or "")
                language = str(proj.get("language") or "")
                chapters_count = len(proj.get("chapters_completed") or [])
                is_active = selected_manga_id == manga_id
                active_class = "nx-proj-active" if is_active else ""

                proj_col, del_col = st.columns([0.85, 0.15], gap="small")
                with proj_col:
                    dot = '<div class="nx-proj-dot"></div>' if is_active else ''
                    st.markdown(
                        f'<div class="nx-proj-row {active_class}">'
                        f'{dot}'
                        f'<span class="nx-proj-name">{_esc(display_name)}</span>'
                        f'<span class="nx-proj-meta">{language} · {chapters_count}ch</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Select", key=f"proj_btn_{manga_id}", use_container_width=True):
                        st.session_state["selected_project_manga_id"] = manga_id
                        st.session_state.pop("right_pane_mode", None)
                        st.rerun()
                with del_col:
                    if st.button("✕", key=f"proj_delete_btn_{manga_id}", help="Delete project"):
                        st.session_state["pending_delete_manga_id"] = manga_id
                        st.session_state["pending_delete_name"] = display_name

        if st.button("+ New Project", key="new_proj_btn"):
            st.session_state["show_new_project_form"] = True

        if st.session_state.get("show_new_project_form"):
            with st.form("new_project_form", clear_on_submit=True):
                display_name = st.text_input("Display Name")
                language = st.selectbox("Language", options=["Japanese", "Korean"])
                submitted = st.form_submit_button("Create")
                if submitted:
                    manga_id = str(display_name).lower().replace(" ", "_").strip() or ""
                    try:
                        create_project(
                            manga_id=manga_id,
                            display_name=str(display_name).strip(),
                            language=language,
                        )
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
                    else:
                        st.session_state["selected_project_manga_id"] = manga_id
                        st.session_state["show_new_project_form"] = False
                        st.rerun()

        st.markdown('<div class="nx-sidebar-section" style="margin-top:1.5rem;">Settings</div>', unsafe_allow_html=True)
        character_name = st.selectbox("Character", options=["Kira", "Unknown"], index=0)
        bubble_type = st.selectbox("Bubble Type", options=["small", "medium", "large", "thought"], index=1)

        st.markdown('<div class="nx-sidebar-section" style="margin-top:1.5rem;">Pipeline Agents</div>', unsafe_allow_html=True)
        agents = [
            ("00", "Translation", "Detects language, translates contextually"),
            ("01", "Cultural Adaptation", "Adapts idioms to natural English"),
            ("02", "Continuity", "Matches character voice from memory"),
            ("03", "Typesetting", "Fits text into speech bubble limits"),
        ]
        for num, name, desc in agents:
            st.markdown(
                f'<div class="nx-agent-row">'
                f'<div class="nx-agent-num">AGENT {num}</div>'
                f'<div class="nx-agent-name">{name}</div>'
                f'<div class="nx-agent-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    return {"character_name": character_name, "bubble_type": bubble_type}


# ──────────────────────────────── main ──────────────────────────────────────

def main() -> None:
    st.set_page_config(layout="wide", page_title="NexusRAG", page_icon=None)
    _inject_css()
    _wordmark()

    settings = _sidebar()
    bubble_type = settings["bubble_type"]

    selected_manga_id = st.session_state.get("selected_project_manga_id")

    # ── pending delete: project ──────────────────────────────────────────────
    pending_delete_manga_id = st.session_state.get("pending_delete_manga_id")
    if pending_delete_manga_id:
        pending_delete_name = st.session_state.get("pending_delete_name", "")
        st.markdown(
            f'<div class="nx-warning">⚠ Delete <strong>{_esc(pending_delete_name)}</strong>? '
            f'This will permanently wipe ALL character profiles, approved lines, and localization memory from ChromaDB. This cannot be undone.</div>',
            unsafe_allow_html=True,
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Yes, delete everything", type="primary", key="confirm_delete_project"):
                delete_manga_data(str(pending_delete_manga_id))
                delete_project(str(pending_delete_manga_id))
                st.session_state.pop("pending_delete_manga_id", None)
                st.session_state.pop("pending_delete_name", None)
                if st.session_state.get("selected_project_manga_id") == pending_delete_manga_id:
                    st.session_state.pop("selected_project_manga_id", None)
                st.rerun()
        with cancel_col:
            if st.button("Cancel", key="cancel_delete_project"):
                st.session_state.pop("pending_delete_manga_id", None)
                st.session_state.pop("pending_delete_name", None)
                st.rerun()
        return

    # ── no project selected ──────────────────────────────────────────────────
    if not selected_manga_id:
        st.markdown(
            '<div class="nx-empty" style="padding:6rem 2rem;margin-top:2rem;">'
            '<div class="nx-empty-icon">◈</div>'
            'Select or create a project from the sidebar to begin.<br/>'
            '<span style="color:var(--text-muted);font-size:.65rem;">NexusRAG · Autonomous Manga Localization Engine</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── load characters ──────────────────────────────────────────────────────
    try:
        _load_characters_for_manga(str(selected_manga_id))
    except Exception as exc:
        st.error(f"Failed to load character profiles: {exc}")
        return

    # ── find selected project ────────────────────────────────────────────────
    projects = load_projects()
    selected_project = next(
        (p for p in projects if str(p.get("manga_id") or "").strip() == str(selected_manga_id)),
        None,
    )
    chapters_completed = selected_project.get("chapters_completed") if selected_project else []
    chapters_int = sorted(
        {int(c) for c in (chapters_completed or []) if str(c).strip().lstrip("-").isdigit()}
    )

    # ── split layout ─────────────────────────────────────────────────────────
    left_col, right_col = st.columns([6, 4], gap="large")

    # ════════════════════════════════════════════════════════════════════════
    # LEFT PANE
    # ════════════════════════════════════════════════════════════════════════
    with left_col:
        _pane_label("Input / Chat")

        project_display_name = selected_project.get("display_name", selected_manga_id) if selected_project else selected_manga_id
        project_language = selected_project.get("language", "") if selected_project else ""

        st.markdown(
            f'<div class="nx-proj-header">'
            f'<div class="nx-proj-header-name">{_esc(project_display_name)}</div>'
            f'<div class="nx-proj-header-meta">{_esc(selected_manga_id)} · {_esc(project_language)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── chapter history strip ────────────────────────────────────────────
        if chapters_int:
            st.markdown('<div class="nx-section-label">Chapter History</div>', unsafe_allow_html=True)
            for chapter in chapters_int:
                ch_col, del_col = st.columns([0.85, 0.15], gap="small")
                with ch_col:
                    if st.button(
                        f"Chapter {chapter}",
                        key=f"open_chapter_btn_{selected_manga_id}_{chapter}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_history_chapter"] = int(chapter)
                        st.session_state["selected_history_chapter_manga"] = str(selected_manga_id)
                        st.session_state["right_pane_mode"] = "history"
                        st.rerun()
                with del_col:
                    if st.button("✕", key=f"delete_chapter_btn_{selected_manga_id}_{chapter}", help=f"Delete chapter {chapter} memory"):
                        st.session_state["pending_delete_chapter"] = int(chapter)
                        st.session_state["pending_delete_chapter_manga"] = str(selected_manga_id)

            pending_del_ch = st.session_state.get("pending_delete_chapter")
            pending_del_ch_manga = st.session_state.get("pending_delete_chapter_manga")
            if pending_del_ch is not None and str(pending_del_ch_manga) == str(selected_manga_id):
                st.markdown(
                    f'<div class="nx-warning">⚠ Delete Chapter {pending_del_ch} memory? Approved lines will be removed.</div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, delete", key="confirm_delete_chapter"):
                        delete_chapter_data(str(selected_manga_id), int(pending_del_ch))
                        remove_chapter(str(selected_manga_id), int(pending_del_ch))
                        st.session_state.pop("pending_delete_chapter", None)
                        st.session_state.pop("pending_delete_chapter_manga", None)
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="cancel_delete_chapter"):
                        st.session_state.pop("pending_delete_chapter", None)
                        st.session_state.pop("pending_delete_chapter_manga", None)
                        st.rerun()

            st.markdown("<hr/>", unsafe_allow_html=True)

        # ── chat message history ─────────────────────────────────────────────
        chat_history = st.session_state.get("chat_messages", [])
        if chat_history:
            st.markdown('<div class="nx-chat-wrap">', unsafe_allow_html=True)
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                ts = msg.get("ts", "")
                bubble_cls = "nx-msg-user" if role == "user" else "nx-msg-system"
                role_label = "YOU" if role == "user" else "NEXUS"
                st.markdown(
                    f'<div class="nx-msg {bubble_cls}">'
                    f'<div class="nx-msg-label">{role_label}</div>'
                    f'{_esc(content)}'
                    f'<div class="nx-msg-meta">{ts}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── translation runs ─────────────────────────────────────────────────
        if "chapter_results_history" in st.session_state and st.session_state["chapter_results_history"]:
            history = st.session_state["chapter_results_history"]
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('<div class="nx-section-label">Translation Runs</div>', unsafe_allow_html=True)

            if st.button("Clear History", key="clear_history_btn"):
                st.session_state.pop("chapter_results_history", None)
                st.session_state.pop("chapter_results", None)
                st.session_state.pop("chat_messages", None)
                st.session_state.pop("right_pane_mode", None)
                st.rerun()

            for idx, item in enumerate(history):
                run_title = str(item.get("run_title") or f"Run {idx + 1}")
                run_at = str(item.get("run_at") or "")
                item_results = item.get("results") or []
                item_panel_map = item.get("panel_id_to_character") or {}
                is_newest = idx == 0

                with st.expander(f"{run_title}  ·  {run_at}" if run_at else run_title, expanded=is_newest):
                    for r in item_results:
                        pid = str(r.get("panel_id", ""))
                        char = item_panel_map.get(pid, "")
                        _render_panel_result_row(r, char)

                    st.download_button(
                        label="↓ Download JSON",
                        data=json.dumps(item_results, ensure_ascii=False, indent=2),
                        file_name=f"results_{idx + 1}.json",
                        mime="application/json",
                        key=f"dl_results_{idx}",
                    )
                    if item_results and st.button("Inspect in Context Pane", key=f"inspect_run_{idx}"):
                        st.session_state["right_pane_mode"] = "run"
                        st.session_state["right_pane_run_idx"] = idx
                        st.rerun()

        # ── sticky input area ────────────────────────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="nx-section-label">New Chapter Run</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload chapter.json", type=["json"], label_visibility="collapsed")
        run_clicked = False

        if uploaded_file is not None:
            try:
                chapter_data = json.load(uploaded_file)
                panel_count = len(chapter_data.get("panels") or [])
                chapter_number = chapter_data.get("chapter") or chapter_data.get("chapter_number")

                preview_text = (
                    f"chapter.json · {panel_count} panels"
                    + (f" · chapter {chapter_number}" if chapter_number not in (None, "") else "")
                )
                st.markdown(
                    f'<div class="nx-preview-msg">'
                    f'<div class="nx-preview-label">YOU</div>'
                    f'{_esc(preview_text)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")
                chapter_data = None
        else:
            chapter_data = None
            st.markdown('<div class="nx-drop-hint">Drop a chapter JSON file to begin.</div>', unsafe_allow_html=True)

        # ── execute pipeline ─────────────────────────────────────────────────
        if run_clicked and chapter_data is not None:
            try:
                client = get_client()
            except Exception as exc:
                st.error(f"Failed to initialize Groq client: {exc}")
                return

            chapter_data["manga_id"] = str(selected_manga_id)
            ts_now = datetime.now().strftime("%H:%M:%S")
            chat_messages = st.session_state.setdefault("chat_messages", [])
            chat_messages.append({
                "role": "user",
                "content": preview_text if 'preview_text' in dir() else uploaded_file.name,
                "ts": ts_now,
            })

            with st.spinner("Pipeline running..."):
                try:
                    results = process_chapter(chapter_data)
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    return

            panel_id_to_character = _build_panel_character_map(chapter_data)
            chapter_number = chapter_data.get("chapter") or chapter_data.get("chapter_number")
            run_title = uploaded_file.name
            if chapter_number not in (None, ""):
                run_title = f"Chapter {chapter_number} — {uploaded_file.name}"

            history = st.session_state.setdefault("chapter_results_history", [])
            history.insert(0, {
                "run_title": run_title,
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "panel_id_to_character": panel_id_to_character,
            })
            st.session_state["chapter_results"] = results

            flagged_count = sum(1 for r in results if r.get("flagged"))
            chat_messages.append({
                "role": "system",
                "content": f"Pipeline complete. {len(results)} panels processed, {flagged_count} flagged.",
                "ts": datetime.now().strftime("%H:%M:%S"),
            })

            st.session_state["right_pane_mode"] = "run"
            st.session_state["right_pane_run_idx"] = 0
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT PANE
    # ════════════════════════════════════════════════════════════════════════
    with right_col:
        right_pane_mode = st.session_state.get("right_pane_mode")

        if right_pane_mode == "history":
            sel_hist_ch = st.session_state.get("selected_history_chapter")
            sel_hist_manga = st.session_state.get("selected_history_chapter_manga")
            if sel_hist_ch is not None and str(sel_hist_manga) == str(selected_manga_id):
                saved_rows = query_approved_lines_for_chapter(
                    manga_id=str(selected_manga_id),
                    chapter=int(sel_hist_ch),
                )
                saved_results: list[Dict[str, Any]] = []
                saved_panel_map: dict[str, str] = {}
                for row in (saved_rows or []):
                    pid = str(row.get("panel_id") or "")
                    saved_panel_map[pid] = str(row.get("character_name") or "")
                    saved_results.append({
                        "panel_id": pid,
                        "original": str(row.get("original_japanese") or ""),
                        "final_output": str(row.get("final_output") or ""),
                        "scores": row.get("scores") or {},
                        "flagged": bool(row.get("flagged")),
                    })

                _render_right_pane_history(
                    {"chapter": sel_hist_ch, "saved_results": saved_results, "saved_panel_map": saved_panel_map},
                    bubble_type,
                )

                if st.button("✕ Close", key="close_hist_view"):
                    st.session_state.pop("selected_history_chapter", None)
                    st.session_state.pop("selected_history_chapter_manga", None)
                    st.session_state.pop("right_pane_mode", None)
                    st.rerun()
            else:
                _render_right_pane_empty()

        elif right_pane_mode == "run":
            run_idx = st.session_state.get("right_pane_run_idx", 0)
            history = st.session_state.get("chapter_results_history", [])
            if history and run_idx < len(history):
                item = history[run_idx]
                item_results = item.get("results") or []
                if item_results:
                    st.markdown(
                        f'<div class="nx-run-header">{_esc(item.get("run_title",""))}</div>'
                        f'<div class="nx-run-meta">{item.get("run_at","")}</div>',
                        unsafe_allow_html=True,
                    )
                    panel_ids = [str(r.get("panel_id", f"panel_{i}")) for i, r in enumerate(item_results)]
                    selected_panel_idx = st.selectbox(
                        "Inspect Panel",
                        options=range(len(item_results)),
                        format_func=lambda i: f"Panel {panel_ids[i]}",
                        key=f"panel_selector_{run_idx}",
                        label_visibility="collapsed",
                    )
                    selected_result = item_results[selected_panel_idx]
                    raw_text = str(selected_result.get("original", ""))
                    _render_right_pane_for_result(selected_result, raw_text, bubble_type)
                else:
                    _render_right_pane_empty()
            else:
                _render_right_pane_empty()

        else:
            _render_right_pane_empty()


if __name__ == "__main__":
    main()