from retrieval_client import RetrievalClient


client = RetrievalClient()

print("[Client] Sending retrieval request...")

results = client.retrieve_similar_lines(
    character_name="Kira",
    manga_id="default",
    query_text="I will never give up.",
    limit=5,
)

print(f"[Client] Received {len(results)} results")

for i, row in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(f"Panel   : {row['panel_id']}")
    print(f"Original: {row['original_japanese']}")
    print(f"Output  : {row['final_output']}")