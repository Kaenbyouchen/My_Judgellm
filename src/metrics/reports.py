"""
Report generation for evaluation results.
"""
import json
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


def save_judgments_jsonl(judgments: List[Dict[str, Any]], output_path: str):
    """
    Save judgments to JSONL file.
    
    Args:
        judgments: List of judgment dictionaries
        output_path: Path to save JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import jsonlines
    with jsonlines.open(str(output_path), mode='w') as writer:
        for judgment in judgments:
            writer.write(judgment)
    
    logger.info(f"Judgments saved to {output_path} ({len(judgments)} judgments)")


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
        if acc.get('metadata', {}).get('note'):
            print(f"  Note: {acc['metadata']['note']}")
    
    if "rr" in metrics:
        rr = metrics["rr"]
        print(f"\nRobustness Rate (RR):")
        print(f"  Value: {rr.get('rr', 0.0):.4f}")
        print(f"  GT Wins: {rr.get('gt_wins', 0)}/{rr.get('total', 0)}")
        print(f"  Has GT Labels: {rr.get('has_gt_labels', False)}")
        if rr.get('metadata', {}).get('note'):
            print(f"  Note: {rr['metadata']['note']}")
        if rr.get('metadata', {}).get('placeholder'):
            print(f"  ⚠️  Placeholder: {rr['metadata'].get('placeholder_note', '')}")
    
    if "cr" in metrics:
        cr = metrics["cr"]
        print(f"\nConsistency Rate (CR):")
        print(f"  Value: {cr.get('cr', 0.0):.4f}")
        print(f"  Method: {cr.get('method', 'unknown')}")
        print(f"  Total: {cr.get('total', 0)}")
        if cr.get('metadata', {}).get('note'):
            print(f"  Note: {cr['metadata']['note']}")
        if cr.get('metadata', {}).get('todo'):
            print(f"  TODO: {cr['metadata']['todo']}")
    
    print("\n" + "="*60 + "\n")


