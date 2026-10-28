"""Validate teacher traces, remove duplicates, and create deterministic splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


MAX_FILE_BYTES = 2_000_000_000
MAX_RECORDS = 2_000_000
MAX_FIELD_CHARS = 50_000
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: str, *, field: str, line_number: int) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = CONTROL_CHARS.sub("", value)
    value = " ".join(value.split())
    if not value:
        raise ValueError(f"line {line_number}: {field} is empty after normalization")
    if len(value) > MAX_FIELD_CHARS:
        raise ValueError(f"line {line_number}: {field} exceeds {MAX_FIELD_CHARS} characters")
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Input JSONL exceeds 2 GB")
    rows: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        if len(rows) >= MAX_RECORDS:
            raise ValueError(f"dataset exceeds {MAX_RECORDS} records")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: malformed JSON at column {exc.colno}") from None
        if not isinstance(raw, dict) or not all(isinstance(raw.get(key), str) for key in ("prompt", "reasoning", "answer")):
            raise ValueError(f"line {line_number}: expected string fields prompt, reasoning, and answer")
        prompt = clean_text(raw["prompt"], field="prompt", line_number=line_number)
        reasoning = clean_text(raw["reasoning"], field="reasoning", line_number=line_number)
        answer = clean_text(raw["answer"], field="answer", line_number=line_number)
        prompt_hash = hashlib.sha256(prompt.casefold().encode("utf-8")).hexdigest()
        if prompt_hash in seen_prompts:
            continue
        seen_prompts.add(prompt_hash)
        completion = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        rows.append({"prompt": prompt, "completion": completion})
    if len(rows) < 2:
        raise ValueError("at least two unique prompt examples are required for a train/validation split")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Authorized teacher-trace JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.01 <= args.validation_fraction <= 0.4:
        parser.error("--validation-fraction must be in [0.01, 0.4]")

    rows = load_rows(args.input)
    random.Random(args.seed).shuffle(rows)
    validation_count = max(1, round(len(rows) * args.validation_fraction))
    if validation_count >= len(rows):
        raise SystemExit("split left no training rows; provide more examples or reduce validation fraction")
    validation, train = rows[:validation_count], rows[validation_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (args.output_dir / name).open("w", encoding="utf-8") as target:
            for row in subset:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source_file": args.input.name,
        "source_sha256": file_sha256(args.input),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unique_records": len(rows),
        "train_records": len(train),
        "validation_records": len(validation),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "preprocessing": "NFKC normalize, remove control characters, collapse whitespace, deduplicate exact prompts",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(train)} train and {len(validation)} validation examples at {args.output_dir}")


if __name__ == "__main__":
    main()
