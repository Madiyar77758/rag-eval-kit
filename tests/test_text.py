from rag_eval.text import stem, tokenize


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("How do I run the Transactions?") == ["run", "transaction"]


def test_tokenize_keeps_identifiers_whole():
    assert "select_related" in tokenize("use select_related here")


def test_stem_leaves_short_words_alone():
    assert stem("is") == "is" and stem("uses") == "use" and stem("saving") == "sav" and stem("class") == "class"


def test_flags_disable_processing():
    assert tokenize("The Cats", stop=False, do_stem=False) == ["the", "cats"]
