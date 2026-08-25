import json
from pathlib import Path


def load_dataset(path: str | Path = "scripts/evaluation/dataset.json") -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def run_ragas(dataset_path: str = "scripts/evaluation/dataset.json"):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows = [row for row in load_dataset(dataset_path) if row.get("answer") and row.get("contexts")]
    if not rows:
        raise ValueError("Populate answer and contexts by running the evaluation set against the live pipeline first")
    dataset = Dataset.from_list(rows)
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])


if __name__ == "__main__":
    result = run_ragas()
    output = Path("scripts/evaluation/results/latest.json")
    output.write_text(json.dumps(result.to_pandas().to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {output}")
