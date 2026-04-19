from pathlib import Path
from typing import Tuple
import os

from groq import Groq


from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAST_MODEL = "llama-3.1-8b-instant"
QUALITY_MODEL = "llama-3.3-70b-versatile"

def get_models() -> Tuple[ChatGroq, ChatGroq]:
    """
    Load environment variables from the .env file and initialize the LangChain Groq clients.

    This ensures that secrets like GROQ_API_KEY are available
    and returns configured LangChain models ready for caching and chains.

    Returns:
        Tuple[ChatGroq, ChatGroq]: A tuple containing (quality_model, fast_model).

    Raises:
        RuntimeError: If the GROQ_API_KEY is missing.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please set it in your .env file."
        )

    # Initialize the Quality Model (for generation)
    quality_model = ChatGroq(
        model=QUALITY_MODEL,
        temperature=0.3,
        api_key=api_key
    )

    # Initialize the Fast Model (for grading/reflexion)
    fast_model = ChatGroq(
        model=FAST_MODEL,
        temperature=0.1,
        api_key=api_key
    )

    return quality_model, fast_model



def load_environment() -> Groq:
    """Legacy UI bridge. Do not remove until app.py is migrated."""
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)