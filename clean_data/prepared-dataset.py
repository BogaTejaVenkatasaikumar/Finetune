
import json
import random
import re
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("data/cleaned_dataset.jsonl")
OUTPUT_DIR = Path("data/splits")

SEED = 42
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10


def normalize_question(text):
    """Normalize a question for grouping."""
    text = text.casefold().strip()
    return re.sub(r"\s+", " ", text)


def load_dataset(path):
    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                )

            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"Invalid messages on line {line_number}"
                )

            if (
                messages[0].get("role") != "user"
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"Invalid roles on line {line_number}"
                )

            question = messages[0].get("content", "")
            answer = messages[-1].get("content", "")

            if not question.strip() or not answer.strip():
                raise ValueError(
                    f"Empty question or answer on line {line_number}"
                )

            records.append(record)

    return records


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}"
        )

    records = load_dataset(INPUT_FILE)

    if len(records) < 10:
        raise ValueError(
            "Too few examples for this split. "
            "Review the dataset before continuing."
        )

    # Group records with the same normalized user question.
    groups = defaultdict(list)

    for record in records:
        question = record["messages"][0]["content"]
        key = normalize_question(question)
        groups[key].append(record)

    group_list = list(groups.values())
    random.Random(SEED).shuffle(group_list)

    # Split groups, not individual records.
    n_groups = len(group_list)
    n_train = int(n_groups * TRAIN_RATIO)
    n_validation = int(n_groups * VALIDATION_RATIO)

    train_groups = group_list[:n_train]
    validation_groups = group_list[
        n_train:n_train + n_validation
    ]
    test_groups = group_list[
        n_train + n_validation:
    ]

    def flatten(group_list):
        return [
            record
            for group in group_list
            for record in group
        ]

    train = flatten(train_groups)
    validation = flatten(validation_groups)
    test = flatten(test_groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(validation, OUTPUT_DIR / "validation.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print("Dataset splitting completed!")
    print(f"Total records: {len(records)}")
    print(f"Unique question groups: {n_groups}")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")
    print(f"Test records: {len(test)}")
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
