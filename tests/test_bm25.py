from rag_eval.bm25 import BM25


def test_exact_term_match_ranks_first():
    docs = [["cat", "sat"], ["dog", "barked", "loudly"], ["cat", "cat", "cat", "purred"]]
    bm = BM25(docs)
    ranked = [i for i, _ in bm.search(["cat"], 3)]
    assert ranked == [2, 0]  # repeated term wins, the doc without the term is absent


def test_rare_terms_weigh_more_than_common_ones():
    docs = [["common", "rare"], ["common"], ["common"], ["common"]]
    bm = BM25(docs)
    assert bm.idf["rare"] > bm.idf["common"]
    assert bm.search(["common", "rare"], 1)[0][0] == 0


def test_unknown_terms_and_empty_query_return_nothing():
    bm = BM25([["a"], ["b"]])
    assert bm.search(["zzz"], 5) == [] and bm.search([], 5) == []


def test_length_normalisation_prefers_shorter_document():
    docs = [["x"] + ["pad"] * 50, ["x"]]
    assert BM25(docs).search(["x"], 2)[0][0] == 1
