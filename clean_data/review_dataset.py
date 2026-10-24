
import json
import re
import csv
from pathlib import Path
from collections import Counter

INPUT_FILE = "data/cleaned_dataset.jsonl"
OUTPUT_FILE = "data/dataset_review.csv"

# Patterns to flag for manual inspection
SENSITIVE_PATTERNS = {
    "possible_api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|"
        r"hf_[A-Za-z0-9]{15,}|"
        r"AIza[0-9A-Za-z_-]{25,})\b"
    ),
    "possible_email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE
    ),
    "possible_phone": re.compile(
        r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)"
    ),
}

GENERIC_PHRASES = [
    "i don't know",
    "it depends",
    "that's a great question",
    "hope this helps",
    "i cannot help with that",
]


def normalize(text):
    """Normalize text for simple duplicate comparisons."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def review_dataset():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    records = []
    issues = Counter()
    seen_questions = {}
    seen_answers = {}

    with input_path.open(
        "r", encoding="utf-8-sig"
    ) as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                records.append({
                    "line": line_no,
                    "user": "",
                    "assistant": "",
                    "flags": "invalid_json",
                })
                continue

            messages = record.get("messages", [])
            if not isinstance(messages, list):
                issues["invalid_schema"] += 1
                continue

            users = [
                m.get("content", "")
                for m in messages
                if isinstance(m, dict)
                and m.get("role") == "user"
            ]
            assistants = [
                m.get("content", "")
                for m in messages
                if isinstance(m, dict)
                and m.get("role") == "assistant"
            ]

            if not users or not assistants:
                issues["invalid_schema"] += 1
                continue

            question = users[0].strip()
            answer = assistants[-1].strip()
            flags = []

            # 1. Check answer length
            if len(answer) < 25:
                flags.append("very_short_answer")

            if len(answer) > 4000:
                flags.append("very_long_answer")

            # 2. Check empty content
            if not question:
                flags.append("empty_question")

            if not answer:
                flags.append("empty_answer")

            # 3. Check repeated questions
            q_key = normalize(question)
            if q_key in seen_questions:
                flags.append("repeated_question")
            else:
                seen_questions[q_key] = line_no

            # 4. Check repeated answers
            a_key = normalize(answer)
            if a_key in seen_answers:
                flags.append("repeated_answer")
            else:
                seen_answers[a_key] = line_no

            # 5. Flag generic phrases for review
            answer_lower = answer.casefold()
            if any(
                phrase in answer_lower
                for phrase in GENERIC_PHRASES
            ):
                flags.append("check_generic_response")

            # 6. Check possible sensitive information
            combined = question + "\n" + answer

            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(combined):
                    flags.append(label)

            # 7. Save review record
            if flags:
                issues.update(flags)

            records.append({
                "line": line_no,
                "user": question,
                "assistant": answer,
                "flags": "; ".join(flags) if flags else "PASS",
            })

    # Save the review report
    with open(
        OUTPUT_FILE, "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "line", "user", "assistant", "flags"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    flagged = sum(
        1 for record in records
        if record["flags"] != "PASS"
    )

    print("Dataset review completed.")
    print(f"Records inspected: {len(records)}")
    print(f"Records flagged: {flagged}")
    print(f"Records without heuristic flags: "
          f"{len(records) - flagged}")
    print("\nIssue counts:")

    for issue, count in issues.most_common():
        print(f"{issue}: {count}")

    print(f"\nReview report: {OUTPUT_FILE}")
    print("Review flagged rows manually before editing the dataset.")


if __name__ == "__main__":
    review_dataset()
