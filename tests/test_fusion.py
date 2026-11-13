from rag_eval.fusion import rrf


def test_item_ranked_well_in_both_lists_wins():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]


def test_top_limits_output_and_ties_are_deterministic():
    assert rrf([[5], [7]], top=1) == [5]


def test_single_list_keeps_order():
    assert rrf([[9, 8, 7]]) == [9, 8, 7]
