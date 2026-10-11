import json
from pathlib import Path
from retrieval import evaluate, load_index

root = Path(__file__).parent / "examples"
index = load_index(root / "docs")
cases = json.loads((root / "queries.json").read_text())
results = [evaluate(index, cases, k=2, method=m) for m in ("overlap", "bm25")]
assert all(r["recall_at_k"] == 1.0 for r in results)
assert index.evidence("volcano magma")["abstained"]
print(json.dumps({"comparison": results, "evidence": index.evidence("idempotency payment", 1)}, indent=2))
