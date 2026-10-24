
import json
import re
from pathlib import Path

INPUT_FILE = "data/bro_personalized_slm_dataset.txt"
OUTPUT_FILE = "data/cleaned_dataset.jsonl"
REPORT_FILE = "data/cleaning_report.txt"


def clean_text(text):
    # Remove unnecessary leading/trailing whitespace
    text = text.strip()

    # Replace repeated spaces or tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    return text


def clean_dataset():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    cleaned_records = []
    seen_records = set()
    report = []

    stats = {
        "total": 0,
        "invalid_json": 0,
        "invalid_schema": 0,
        "empty_messages": 0,
        "duplicates": 0,
        "kept": 0,
    }

    with input_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            stats["total"] += 1

            # 1. Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                report.append(
                    f"Line {line_number}: Invalid JSON"
                )
                continue

            # 2. Validate message structure
            messages = record.get("messages") if isinstance(record, dict) else None

            if not isinstance(messages, list) or len(messages) < 2:
                stats["invalid_schema"] += 1
                report.append(
                    f"Line {line_number}: Invalid messages structure"
                )
                continue

            if (
                not isinstance(messages[0], dict)
                or not isinstance(messages[-1], dict)
                or messages[0].get("role") != "user"
                or messages[-1].get("role") != "assistant"
            ):
                stats["invalid_schema"] += 1
                report.append(
                    f"Line {line_number}: Invalid message roles"
                )
                continue

            # 3. Clean message content
            valid = True

            for message in messages:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in {
                        "user", "assistant", "system"
                    }
                    or not isinstance(message.get("content"), str)
                ):
                    valid = False
                    break

                message["content"] = clean_text(
                    message["content"]
                )

                if not message["content"]:
                    valid = False
                    break

            if not valid:
                stats["empty_messages"] += 1
                report.append(
                    f"Line {line_number}: Empty or invalid message"
                )
                continue

            # 4. Remove exact duplicate conversations
            normalized = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True
            )

            if normalized in seen_records:
                stats["duplicates"] += 1
                report.append(
                    f"Line {line_number}: Duplicate removed"
                )
                continue

            seen_records.add(normalized)

            # 5. Keep valid conversation
            cleaned_records.append(record)
            stats["kept"] += 1

    # 6. Save cleaned JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in cleaned_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # 7. Save cleaning report
    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        for item in report:
            file.write(item + "\n")

        file.write("\n--- SUMMARY ---\n")

        for key, value in stats.items():
            file.write(f"{key}: {value}\n")

    print("Dataset cleaning completed!")
    print(f"Total records: {stats['total']}")
    print(f"Invalid JSON: {stats['invalid_json']}")
    print(f"Invalid schema: {stats['invalid_schema']}")
    print(f"Empty messages: {stats['empty_messages']}")
    print(f"Duplicates removed: {stats['duplicates']}")
    print(f"Clean records: {stats['kept']}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    clean_dataset()
