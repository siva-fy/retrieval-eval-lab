import unittest
from retrieval import Index, evaluate


class RetrievalTests(unittest.TestCase):
    def test_expected_source_ranks_first(self):
        index = Index({"a": "payments replay retry", "b": "weather rain"})
        self.assertEqual(index.search("payments retry")[0]["source"], "a")

    def test_abstains_for_unknown_and_stopword_only_queries(self):
        index = Index({"a": "payments replay"})
        for query in ("volcano", "the and is", ""):
            self.assertTrue(index.evidence(query)["abstained"])

    def test_citations_are_verbatim_source_chunks(self):
        index = Index({"a": "first paragraph\n\nsecond payment paragraph"})
        excerpt = index.evidence("payment")["excerpts"][0]
        self.assertEqual(excerpt, {"citation": "a#1", "quote": "second payment paragraph"})

    def test_metrics_count_unique_sources(self):
        index = Index({"a": "payment\n\npayment payment", "b": "retry payment"})
        result = evaluate(index, [{"query": "payment", "relevant_sources": ["a", "b"]}], 3)
        self.assertEqual(result["recall_at_k"], 1)

    def test_empty_corpus_and_empty_eval(self):
        self.assertEqual(Index({}).search("anything"), [])
        self.assertIsNone(evaluate(Index({}), [])["recall_at_k"])

    def test_fingerprint_and_ties_are_deterministic(self):
        left, right = Index({"b": "pay", "a": "pay"}), Index({"a": "pay", "b": "pay"})
        self.assertEqual(left.fingerprint, right.fingerprint)
        self.assertEqual(left.search("pay")[0]["source"], "a")

    def test_invalid_k_and_unknown_reference(self):
        index = Index({"a": "test"})
        with self.assertRaises(ValueError):
            index.search("test", 0)
        with self.assertRaises(ValueError):
            evaluate(index, [{"query": "test", "relevant_sources": ["missing"]}])


if __name__ == "__main__":
    unittest.main()
