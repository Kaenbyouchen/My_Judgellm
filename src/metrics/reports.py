"""
Report generation for evaluation results.
"""
import json
import os
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger


def save_metrics_json(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metrics saved to {output_path}")


def save_metrics_csv(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics summary to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten metrics for CSV
    rows = []
    
    # Accuracy metrics
    if "accuracy_original" in metrics:
        acc = metrics["accuracy_original"]
        rows.append({
            "metric": "accuracy_original",
            "value": acc.get("accuracy", 0.0),
            "proxy": acc.get("proxy", False),
            "has_gt": acc.get("has_gt", False),
            "total": acc.get("total", 0),
            "note": acc.get("metadata", {}).get("note", "")
        })
    
    # RR metrics
    if "rr" in metrics:
        rr = metrics["rr"]
        rows.append({
            "metric": "rr",
            "value": rr.get("rr", 0.0),
            "gt_wins": rr.get("gt_wins", 0),
            "total": rr.get("total", 0),
            "has_gt_labels": rr.get("has_gt_labels", False),
            "note": rr.get("metadata", {}).get("note", "")
        })

    # Biased accuracy metrics
    if "accuracy_biased" in metrics:
        acc_biased = metrics["accuracy_biased"]
        rows.append({
            "metric": "accuracy_biased",
            "value": acc_biased.get("accuracy_biased", 0.0),
            "has_gt_labels": acc_biased.get("has_gt_labels", False),
            "total": acc_biased.get("total", 0),
            "note": acc_biased.get("metadata", {}).get("note", "")
        })
    
    # CR metrics
    if "cr" in metrics:
        cr = metrics["cr"]
        rows.append({
            "metric": "cr",
            "value": cr.get("cr", 0.0),
            "method": cr.get("method", "unknown"),
            "total": cr.get("total", 0),
            "note": cr.get("metadata", {}).get("note", "")
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.info(f"Metrics summary saved to {output_path}")


def save_judgment_jsonl_single(judgment: Dict[str, Any], output_path: str):
    """
    Append a single judgment to JSONL file (for incremental saving during evaluation).
    
    Args:
        judgment: Single judgment dictionary
        output_path: Path to save JSONL file
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_path), 'a', encoding='utf-8') as f:
            formatted_json = json.dumps(judgment, indent=2, ensure_ascii=False)
            f.write(formatted_json)
            f.write('\n')
            f.flush()  # Ensure immediate write to disk
            os.fsync(f.fileno())  # Force write to disk
    except Exception as e:
        logger.error(f"Failed to save judgment incrementally to {output_path}: {e}")
        raise


def append_summary_jsonl(record: Dict[str, Any], output_path: str):
    """
    Append a single summary record to JSONL file.
    
    Args:
        record: Summary record dictionary
        output_path: Path to save JSONL file
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_path), 'a', encoding='utf-8') as f:
            formatted_json = json.dumps(record, indent=2, ensure_ascii=False)
            f.write(formatted_json)
            f.write('\n\n')
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.error(f"Failed to append summary record to {output_path}: {e}")
        raise


def append_summary_csv(record: Dict[str, Any], output_path: str):
    """
    Append a single flattened summary record to CSV file.

    CSV columns:
    - data type (category): dataset category (one of six categories / dummy)
    - dataset: dataset name
    - bias type: injected bias type
    - bias injection mode: rewrite or word
    - data type (pairwise/scalar): evaluation data type
    - judge llm: judge model id
    - acc / acc_biased / cr / rr: core metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
    def _fmt_metric(val: Any):
        if val is None:
            return ""
        try:
            return f"{float(val) * 100:.2f}"
        except (TypeError, ValueError):
            return ""

    row = {
        "data type (category)": record.get("dataset_category", "uncategorized"),
        "dataset": record.get("dataset_name", ""),
        "bias type": record.get("bias_type", "none"),
        "bias injection mode": record.get("bias_injection_mode", ""),
        "data type (pairwise/scalar)": record.get("data_type", ""),
        "judge llm": record.get("judge_model_id", ""),
        "acc": _fmt_metric(metrics.get("accuracy_original")),
        "acc_biased": _fmt_metric(metrics.get("accuracy_biased")),
        "cr": _fmt_metric(metrics.get("cr")),
        "rr": _fmt_metric(metrics.get("rr")),
    }

    fieldnames = [
        "data type (category)",
        "dataset",
        "bias type",
        "bias injection mode",
        "data type (pairwise/scalar)",
        "judge llm",
        "acc",
        "acc_biased",
        "cr",
        "rr",
    ]

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def save_judgments_jsonl(judgments: List[Dict[str, Any]], output_path: str, append: bool = False):
    """
    Save judgments to JSONL file with formatted JSON (each key on a new line).
    
    Args:
        judgments: List of judgment dictionaries
        output_path: Path to save JSONL file
        append: If True, append to existing file; if False, overwrite
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if append else 'w'
    with open(str(output_path), mode, encoding='utf-8') as f:
        for judgment in judgments:
            # Format each JSON object with indentation (2 spaces)
            formatted_json = json.dumps(judgment, indent=2, ensure_ascii=False)
            f.write(formatted_json)
            f.write('\n')  # Add newline after each JSON object
    
    action = "appended" if append else "saved"
    logger.info(f"Judgments {action} to {output_path} ({len(judgments)} judgments)")


def print_metrics_summary(metrics: Dict[str, Any]):
    """
    Print metrics summary to console.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("Evaluation Metrics Summary")
    print("="*60)
    
    if "accuracy_original" in metrics:
        acc = metrics["accuracy_original"]
        print(f"\nAccuracy (Original R1 vs R2):")
        print(f"  Value: {acc.get('accuracy', 0.0):.4f}")
        print(f"  Proxy: {acc.get('proxy', False)}")
        print(f"  Has GT: {acc.get('has_gt', False)}")
        print(f"  Total: {acc.get('total', 0)}")
        if acc.get('valid_count') is not None:
            print(f"  Valid samples: {acc.get('valid_count', 0)}")
        if acc.get('invalid_count', 0) > 0:
            print(f"  ⚠️  Invalid samples excluded: {acc.get('invalid_count', 0)} (safety filter, quota errors, etc.)")
        if acc.get('metadata', {}).get('note'):
            print(f"  Note: {acc['metadata']['note']}")
    
    if "rr" in metrics:
        rr = metrics["rr"]
        print(f"\nRobustness Rate (RR):")
        print(f"  Value: {rr.get('rr', 0.0):.4f}")
        print(f"  GT Wins: {rr.get('gt_wins', 0)}/{rr.get('valid_count', rr.get('total', 0))}")
        print(f"  Has GT Labels: {rr.get('has_gt_labels', False)}")
        if rr.get('valid_count') is not None:
            print(f"  Valid samples: {rr.get('valid_count', 0)}")
        if rr.get('invalid_count', 0) > 0:
            print(f"  ⚠️  Invalid samples excluded: {rr.get('invalid_count', 0)} (safety filter, quota errors, etc.)")
        if rr.get('metadata', {}).get('note'):
            print(f"  Note: {rr['metadata']['note']}")
        if rr.get('metadata', {}).get('placeholder'):
            print(f"  ⚠️  Placeholder: {rr['metadata'].get('placeholder_note', '')}")

    if "accuracy_biased" in metrics:
        acc_biased = metrics["accuracy_biased"]
        print(f"\nAccuracy (Biased GT vs Biased Answer):")
        print(f"  Value: {acc_biased.get('accuracy_biased', 0.0):.4f}")
        print(f"  Has GT Labels: {acc_biased.get('has_gt_labels', False)}")
        print(f"  Total: {acc_biased.get('total', 0)}")
        if acc_biased.get('valid_count') is not None:
            print(f"  Valid samples: {acc_biased.get('valid_count', 0)}")
        if acc_biased.get('invalid_count', 0) > 0:
            print(f"  ⚠️  Invalid samples excluded: {acc_biased.get('invalid_count', 0)} (safety filter, quota errors, etc.)")
        if acc_biased.get('metadata', {}).get('note'):
            print(f"  Note: {acc_biased['metadata']['note']}")
    
    if "cr" in metrics:
        cr = metrics["cr"]
        print(f"\nConsistency Rate (CR):")
        print(f"  Value: {cr.get('cr', 0.0):.4f}")
        print(f"  Method: {cr.get('method', 'unknown')}")
        print(f"  Total: {cr.get('total', 0)}")
        if cr.get('valid_count') is not None:
            print(f"  Valid pairs: {cr.get('valid_count', 0)}")
        if cr.get('invalid_count', 0) > 0:
            print(f"  ⚠️  Invalid pairs excluded: {cr.get('invalid_count', 0)} (safety filter, quota errors, etc.)")
        if cr.get('metadata', {}).get('note'):
            print(f"  Note: {cr['metadata']['note']}")
        if cr.get('metadata', {}).get('todo'):
            print(f"  TODO: {cr['metadata']['todo']}")
    
    print("\n" + "="*60 + "\n")


