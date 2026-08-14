import sys
from pathlib import Path

# Add NexusRAG project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.vector_store import add_approved_line


add_approved_line(
    panel_id="rpc-test-1",
    character_name="Kira",
    manga_id="default",
    original_japanese="たとえ天が崩れ落ちようとも、私は一歩も退かない。",
    final_output="I will never back down, no matter what happens.",
    scores={
        "rpc_test": True,
        "pass": True
    },
    flagged=False,
    chapter=1,
)

print("Test approved line added successfully.")