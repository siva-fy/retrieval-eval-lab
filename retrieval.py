"""An inspectable BM25 retrieval baseline with source citations and evaluation."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = set("a an the is are was were be to of for in on and or with how what when do does can i we".split())


def tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


class Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0,1]")
        self.k1, self.b = k1, b
        self.documents = dict(documents)
        self.chunks = []
        for source, text in sorted(self.documents.items()):
            for position, paragraph in enumerate(re.split(r"\n\s*\n", text.strip())):
                counts = Counter(tokens(paragraph))
                if counts:
                    self.chunks.append({"id": f"{source}#{position}", "source": source,
                                        "text": paragraph, "terms": counts,
                                        "length": sum(counts.values())})
        self.df = Counter(term for c in self.chunks for term in c["terms"])
        self.avgdl = sum(c["length"] for c in self.chunks) / max(1, len(self.chunks))
        self.fingerprint = hashlib.sha256(json.dumps(self.documents, sort_keys=True).encode()).hexdigest()

    def search(self, query, k=3, method="bm25"):
        if k < 1:
            raise ValueError("k must be positive")
        if method not in ("bm25", "overlap"):
            raise ValueError("unknown retrieval method")
        query_terms, results = set(tokens(query)), []
        for chunk in self.chunks:
            score = 0.0
            for term in query_terms:
                tf = chunk["terms"].get(term, 0)
                if not tf:
                    continue
                if method == "overlap":
                    score += 1
                else:
                    df = self.df[term]
                    idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                    norm = tf + self.k1 * (1 - self.b + self.b * chunk["length"] / self.avgdl)
                    score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                results.append({key: chunk[key] for key in ("id", "source", "text")} | {"score": score})
        return sorted(results, key=lambda r: (-r["score"], r["id"]))[:k]

    def evidence(self, query, k=3):
        hits = self.search(query, k)
        return {"query": query, "abstained": not hits,
                "excerpts": [{"citation": h["id"], "quote": h["text"]} for h in hits]}


def evaluate(index, cases, k=3, method="bm25"):
    rows = []
    for case in cases:
        relevant = set(case["relevant_sources"])
        if relevant - index.documents.keys():
            raise ValueError("evaluation refers to an unknown source")
        hits = index.search(case["query"], k, method)
        sources = [h["source"] for h in hits]
        reciprocal_rank = next((1 / rank for rank, source in enumerate(sources, 1)
                                if source in relevant), 0.0)
        rows.append({"query": case["query"], "answerable": bool(relevant), "sources": sources,
                     "recall": len(set(sources) & relevant) / len(relevant) if relevant else None,
                     "reciprocal_rank": reciprocal_rank if relevant else None,
                     "abstention_correct": not hits if not relevant else None})
    answerable = [r for r in rows if r["answerable"]]
    unknown = [r for r in rows if not r["answerable"]]
    return {"method": method, "k": k, "corpus_sha256": index.fingerprint,
            "case_count": len(rows), "answerable_count": len(answerable),
            "recall_at_k": sum(r["recall"] for r in answerable) / len(answerable) if answerable else None,
            "mrr_at_k": sum(r["reciprocal_rank"] for r in answerable) / len(answerable) if answerable else None,
            "unanswerable_abstention_rate": sum(r["abstention_correct"] for r in unknown) / len(unknown) if unknown else None,
            "cases": rows}


def load_index(directory):
    return Index({p.name: p.read_text(encoding="utf-8") for p in sorted(Path(directory).glob("*.md"))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--eval", type=Path)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    if bool(args.query) == bool(args.eval):
        parser.error("provide exactly one of --query or --eval")
    index = load_index(args.documents)
    result = (evaluate(index, json.loads(args.eval.read_text()), args.k) if args.eval
              else index.evidence(args.query, args.k))
    print(json.dumps(result, indent=2))
