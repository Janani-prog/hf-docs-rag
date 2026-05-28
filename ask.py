import sys
from src.retrieval.generator import query

if len(sys.argv) < 2:
    print("Usage: python ask.py \"your question here\"")
    sys.exit(1)

question = " ".join(sys.argv[1:])
print(f"\nQuestion: {question}\n")
print("-" * 60)

result = query(question)

print(result["answer"])
print("\n" + "-" * 60)
print(f"Citations valid: {result['citation_valid']}")
print(f"Tokens used: {result['tokens_used']}")
print(f"Prompt version: {result['prompt_version']}")
print(f"\nSources:")
for c in result["citations"]:
    if c["valid"]:
        print(f"  [{c['number']}] {c['title']} — {c['source_url']}")