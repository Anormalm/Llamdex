from src.multimodal.task_matrix.metrics import keyword_consistency_score


def test_keyword_consistency_score_handles_inflection_and_punctuation():
    text = "Rationale: The sample looks braided, striped, and textured."
    score = keyword_consistency_score(text, ["braid", "striped"])
    assert score == 1.0


def test_keyword_consistency_score_supports_partial_coverage():
    text = "Rationale: The sample has crackled details."
    score = keyword_consistency_score(text, ["crackle", "banded"])
    assert score == 0.5


def test_keyword_consistency_score_empty_keywords_is_zero():
    assert keyword_consistency_score("any rationale", []) == 0.0
