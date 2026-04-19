import chromadb
import json
import os

def inspect_chroma():
    # Matches the persist_directory from your vector_store.py
    db_path = "./data/chroma_db" 
    
    if not os.path.exists(db_path):
        print(f"Directory not found: {db_path}")
        print("Make sure you are running this script from the root 'lumina' folder.")
        return

    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    collections = client.list_collections()
    if not collections:
        print("Your database is completely empty.")
        return
        
    print("=== DATABASE OVERVIEW ===")
    for c in collections:
        # Handle newer chromadb versions where c might be an object or string
        c_name = c.name if hasattr(c, 'name') else c
        col = client.get_collection(name=c_name)
        print(f"- Collection: '{c_name}' | Total Documents stored: {col.count()}")
        
    print("\n=== X-RAY: APPROVED LINES (The Continuity RAG Memory) ===")
    try:
        approved_col = client.get_collection(name="approved_lines")
        if approved_col.count() > 0:
            results = approved_col.peek(limit=30)
            for i in range(len(results['ids'])):
                print(f"Entry {i+1}:")
                print(f"  ID: {results['ids'][i]}")
                print(f"  Document (English Output): {results['documents'][i]}")
                meta = results['metadatas'][i]
                print(f"  Metadata: {json.dumps(meta, indent=2)}")
                print("-" * 40)
        else:
            print("  (Collection is empty)")
    except Exception as e:
        print(f"  Could not load approved_lines: {e}")

    print("\n=== X-RAY: CHARACTER PROFILES ===")
    try:
        char_col = client.get_collection(name="character_profiles")
        if char_col.count() > 0:
            # Just peek at 1 so we don't flood the terminal with giant JSON profiles
            results = char_col.peek(limit=30) 
            for i in range(len(results['ids'])):
                print(f"  ID: {results['ids'][i]}")
                print(f"  Document: {results['documents'][i][:200]}... [TRUNCATED]")
                print("-" * 40)
        else:
            print("  (Collection is empty)")
    except Exception as e:
        print(f"  Could not load character_profiles: {e}")

if __name__ == "__main__":
    inspect_chroma()