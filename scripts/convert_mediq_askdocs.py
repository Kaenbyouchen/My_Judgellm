#!/usr/bin/env python3
"""
Convert MediQ AskDocs preference data to project pairwise JSONL.

Outputs:
1) Raw exported JSONL under data/raw_data/
2) Evaluation JSONL under data/MediQ_AskDocs/
3) Optional evaluation JSON (array) for easier inspection
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def _extract_text(value: Any) -> str:
    """Best-effort conversion for nested preference/chat style fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "value", "answer", "response"):
            if key in value and value[key] is not None:
                text = _extract_text(value[key])
                if text:
                    return text
        return ""
    if isinstance(value, (list, tuple)):
        chunks = []
        for item in value:
            piece = _extract_text(item)
            if piece:
                chunks.append(piece)
        return "\n".join(chunks).strip()
    return str(value).strip()


def _pick_first(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _infer_preferred(
    preferred_raw: Any,
    answer_1: str,
    answer_2: str,
    chosen_text: str,
    rejected_text: str,
) -> Optional[str]:
    if preferred_raw is None:
        if chosen_text and rejected_text:
            if answer_1 == chosen_text and answer_2 == rejected_text:
                return "1"
            if answer_2 == chosen_text and answer_1 == rejected_text:
                return "2"
        return None

    if isinstance(preferred_raw, int):
        if preferred_raw == 1:
            return "1"
        if preferred_raw == 2:
            return "2"

    preferred_str = str(preferred_raw).strip().lower()
    if preferred_str in {"1", "answer_1", "answer1", "a", "left", "chosen"}:
        return "1"
    if preferred_str in {"2", "answer_2", "answer2", "b", "right", "rejected"}:
        return "2"

    if preferred_raw == answer_1:
        return "1"
    if preferred_raw == answer_2:
        return "2"

    if chosen_text and rejected_text:
        if answer_1 == chosen_text and answer_2 == rejected_text:
            return "1"
        if answer_2 == chosen_text and answer_1 == rejected_text:
            return "2"

    return None


def _to_pairwise(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    question = _extract_text(
        _pick_first(
            record,
            (
                "question",
                "prompt",
                "instruction",
                "input",
                "query",
                "post",
                "context",
                "history",
                "conversation",
            ),
        )
    )
    chosen = _extract_text(
        _pick_first(
            record,
            (
                "chosen",
                "response_chosen",
                "answer_chosen",
                "preferred_answer",
                "better_answer",
            ),
        )
    )
    rejected = _extract_text(
        _pick_first(
            record,
            (
                "rejected",
                "response_rejected",
                "answer_rejected",
                "worse_answer",
            ),
        )
    )

    answer_1 = _extract_text(_pick_first(record, ("answer_1", "answer1", "a1", "response_1")))
    answer_2 = _extract_text(_pick_first(record, ("answer_2", "answer2", "a2", "response_2")))

    if not answer_1 and chosen:
        answer_1 = chosen
    if not answer_2 and rejected:
        answer_2 = rejected

    if not question or not answer_1 or not answer_2:
        return None

    preferred_raw = _pick_first(
        record,
        ("preferred", "preference", "label", "winner", "GT", "gt"),
    )
    preferred = _infer_preferred(preferred_raw, answer_1, answer_2, chosen, rejected)
    if preferred is None and chosen and rejected:
        # For preference datasets like MediQ: chosen is preferred over rejected.
        preferred = "1"

    sample_id = _pick_first(record, ("id", "sample_id", "uid", "uuid"))
    if sample_id is None or str(sample_id).strip() == "":
        sample_id = f"mediq_askdocs_{idx:06d}"

    return {
        "id": str(sample_id),
        "question": question,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "preferred": preferred,
        "meta": {
            "source_dataset": "stellalisy/MediQ_AskDocs_preference",
            "raw_index": idx,
            "source_fields": sorted(record.keys()),
        },
    }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert stellalisy/MediQ_AskDocs_preference to project pairwise format."
    )
    parser.add_argument(
        "--dataset-id",
        default="stellalisy/MediQ_AskDocs_preference",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--input-raw-jsonl",
        default=None,
        help="Use existing raw JSONL file instead of downloading from HF.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to convert.")
    parser.add_argument(
        "--raw-output",
        default="data/raw_data/MediQ_AskDocs_preference.train.raw.jsonl",
        help="Path to save raw JSONL export.",
    )
    parser.add_argument(
        "--eval-output",
        default="data/MediQ_AskDocs/mediq_askdocs_pairwise.jsonl",
        help="Path to save converted pairwise JSONL for evaluation.",
    )
    parser.add_argument(
        "--eval-json-output",
        default="data/MediQ_AskDocs/mediq_askdocs_pairwise.json",
        help="Path to save converted pairwise JSON array.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HF token for gated dataset (defaults to HF_TOKEN env var).",
    )
    args = parser.parse_args()

    raw_output = Path(args.raw_output)
    eval_output = Path(args.eval_output)
    eval_json_output = Path(args.eval_json_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    eval_json_output.parent.mkdir(parents=True, exist_ok=True)

    dataset_iter: Iterator[Dict[str, Any]]
    should_export_raw = False
    if args.input_raw_jsonl:
        input_raw = Path(args.input_raw_jsonl)
        if not input_raw.exists():
            raise SystemExit(f"Input raw JSONL not found: {input_raw}")
        dataset_iter = _iter_jsonl(input_raw)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: datasets. Please run `pip install datasets` first."
            ) from exc
        token = args.hf_token or None
        dataset_iter = iter(load_dataset(args.dataset_id, split=args.split, token=token))
        should_export_raw = True

    skipped = 0
    converted_count = 0

    raw_fh = raw_output.open("w", encoding="utf-8") if should_export_raw else None
    with eval_output.open("w", encoding="utf-8") as eval_f, eval_json_output.open(
        "w", encoding="utf-8"
    ) as json_f:
        json_f.write("[\n")
        first_json_item = True
        for idx, row in enumerate(dataset_iter):
            if args.max_samples is not None and idx >= args.max_samples:
                break

            if raw_fh is not None:
                raw_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            pairwise = _to_pairwise(row, idx)
            if pairwise is None:
                skipped += 1
                continue

            converted_count += 1
            eval_f.write(json.dumps(pairwise, ensure_ascii=False) + "\n")
            if not first_json_item:
                json_f.write(",\n")
            json_f.write(json.dumps(pairwise, ensure_ascii=False, indent=2))
            first_json_item = False
        json_f.write("\n]\n")

    if raw_fh is not None:
        raw_fh.close()

    print(f"Done. total={converted_count + skipped}, converted={converted_count}, skipped={skipped}")
    if should_export_raw:
        print(f"Raw file: {raw_output}")
    print(f"Eval JSONL: {eval_output}")
    print(f"Eval JSON: {eval_json_output}")


if __name__ == "__main__":
    main()
