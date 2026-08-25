from collections import Counter

from scripts.evaluation.evaluate_rag import load_dataset


def test_evaluation_dataset_has_exactly_100_curated_cases():
    cases = load_dataset()
    assert len(cases) == 100
    assert len({case["id"] for case in cases}) == 100


def test_evaluation_category_distribution():
    counts = Counter(case["category"] for case in load_dataset())
    assert counts == {
        "answerable": 30,
        "paraphrased": 15,
        "unsupported": 15,
        "wrong_book": 10,
        "prompt_injection": 10,
        "adversarial": 10,
        "system": 10,
    }
