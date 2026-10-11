# Retrieval Eval Lab

**Inspect what your retrieval layer finds before putting a language model on top.**

[![CI](https://github.com/siva-fy/retrieval-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/siva-fy/retrieval-eval-lab/actions/workflows/ci.yml)

A dependency-free AI engineering baseline: paragraph chunking, BM25 ranking,
verbatim evidence with source IDs, and a repeatable comparison with keyword
overlap. It measures retrieval quality independently from generation.

## Quick start

Python 3.11+; no embedding downloads, model access or API keys.

~~~bash
git clone https://github.com/siva-fy/retrieval-eval-lab.git
cd retrieval-eval-lab
python3 demo.py
python3 -m unittest discover -s tests -v
~~~

The demo compares both retrievers on **five answerable queries and one
unanswerable query** against four original synthetic documents. Both score
1.0 recall@2 on this deliberately small fixture. This verifies the harness;
it does not demonstrate superiority or real-world retrieval quality.

## Retrieve and evaluate

~~~bash
python3 retrieval.py examples/docs --query "idempotency payment" -k 2
python3 retrieval.py examples/docs --eval examples/queries.json -k 2
~~~

~~~mermaid
flowchart LR
  A[Markdown corpus] --> B[Paragraph chunks]
  B --> C[Term counts and document frequency]
  Q[Query] --> D[BM25 ranking]
  C --> D
  D --> E[Verbatim excerpts and citations]
  D --> F[Recall and reciprocal rank]
  G[Labeled queries] --> F
~~~

## What the outputs mean

- **Citations:** filename plus paragraph position, such as `replay.md#1`.
  Returned quotes are verbatim source paragraphs.
- **Recall@k:** fraction of relevant source documents represented in the top
  k chunks. Multiple chunks from one source count once toward recall.
- **MRR@k:** mean reciprocal rank of the first relevant chunk across answerable
  queries. A miss contributes zero.
- **Unanswerable abstention rate:** fraction of labeled unanswerable queries
  producing no lexical match.
- **Corpus fingerprint:** SHA-256 of the sorted document mapping, to identify
  which text was evaluated.

No-answer cases are excluded from recall and MRR and measured separately.
An empty evaluation set returns null aggregate metrics rather than an invented
perfect score. Ties use chunk IDs for deterministic ordering.

## Boundaries

This is lexical retrieval and an evidence API, not an LLM chatbot or semantic
search engine. No generative model is called. Zero-match abstention is a simple
baseline: irrelevant documents can still share keywords. Headings are indexed
as paragraphs; there is no stemming, reranking, token-budget chunking or
multilingual evaluation. Paragraph citations can change when the corpus changes,
so preserve the corpus fingerprint with reports.

The index is held in memory and scans every chunk for each query. A production
extension would use an inverted index, representative held-out queries,
embedding retrieval, stronger abstention calibration and a separate groundedness
evaluation for generated answers.

## Verification

Tests cover ranking, empty queries/corpora, verbatim evidence, duplicate-source
recall, deterministic ties, invalid parameters and missing ground-truth sources.
CI runs the tests and demo on Python 3.11–3.13.
[Actual demo output](examples/demo-output.txt).

Original AI-assisted portfolio project with a synthetic corpus and query set.
[Provenance](PROVENANCE.md) · [MIT license](LICENSE)
