"""
Metrics for pairwise evaluation.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from ..dataset.schemas import PairwiseSample
from ..judge.base import JudgeResult


def compute_accuracy_original(
    samples: List[PairwiseSample],
    judgments: List[JudgeResult],
    use_proxy: bool = False
) -> Dict[str, Any]:
    """
    Compute accuracy on original R1 vs R2 comparison.
    
    Args:
        samples: Original samples
        judgments: Judge results for original R1 vs R2
        use_proxy: Whether to use proxy metric (when GT labels unavailable)
        
    Returns:
        Dictionary with accuracy metrics and metadata
    """
    if len(samples) != len(judgments):
        raise ValueError(f"Sample count mismatch: {len(samples)} samples vs {len(judgments)} judgments")
    
    # Exclude model-invalid judgments from denominator
    valid_pairs = [(s, j) for s, j in zip(samples, judgments) if j.is_valid]
    invalid_count = len(samples) - len(valid_pairs)
    
    if invalid_count > 0:
        logger.warning(
            f"Excluding {invalid_count} model-invalid judgments from accuracy denominator "
            "(e.g., safety filter blocks, quota errors, timeout)."
        )
    
    if not valid_pairs:
        return {
            "accuracy": 0.0,
            "proxy": use_proxy,
            "has_gt": False,
            "gt_count": 0,
            "total": len(samples),
            "valid_count": 0,
            "invalid_count": invalid_count,
            "answer_1_wins": 0,
            "answer_2_wins": 0,
            "ties": 0,
            "metadata": {
                "note": "No valid judgments available (all excluded due to model-side errors)",
                "invalid_reasons": "safety_filter, quota_error, timeout"
            }
        }
    
    valid_samples, valid_judgments = zip(*valid_pairs)
    valid_samples = list(valid_samples)
    valid_judgments = list(valid_judgments)
    
    # Check if we have GT labels
    has_gt = any(s.preferred is not None for s in valid_samples)
    gt_count = sum(1 for s in valid_samples if s.preferred is not None)
    
    if not has_gt or use_proxy:
        # Proxy: judge选择answer_1的比例
        answer_1_wins = sum(1 for j in valid_judgments if j.winner == "A")
        proxy_ratio = answer_1_wins / len(valid_judgments) if valid_judgments else 0.0
        
        return {
            "accuracy": proxy_ratio,
            "proxy": True,
            "has_gt": has_gt,
            "gt_count": gt_count,
            "total": len(samples),
            "valid_count": len(valid_judgments),
            "invalid_count": invalid_count,
            "answer_1_wins": answer_1_wins,
            "answer_2_wins": sum(1 for j in valid_judgments if j.winner == "B"),
            "ties": sum(1 for j in valid_judgments if j.winner == "tie"),
            "metadata": {
                "note": "Using proxy metric: proportion of judge selecting answer_1",
                "gt_available": has_gt,
                "invalid_excluded": True
            }
        }
    else:
        # Real accuracy: judge winner matches preferred
        correct = 0
        for sample, judgment in zip(valid_samples, valid_judgments):
            if sample.preferred == "1" and judgment.winner == "A":
                correct += 1
            elif sample.preferred == "2" and judgment.winner == "B":
                correct += 1
        
        accuracy = correct / len(valid_samples) if valid_samples else 0.0
        
        return {
            "accuracy": accuracy,
            "proxy": False,
            "has_gt": True,
            "gt_count": gt_count,
            "total": len(samples),
            "valid_count": len(valid_judgments),
            "invalid_count": invalid_count,
            "correct": correct,
            "answer_1_wins": sum(1 for j in valid_judgments if j.winner == "A"),
            "answer_2_wins": sum(1 for j in valid_judgments if j.winner == "B"),
            "ties": sum(1 for j in valid_judgments if j.winner == "tie"),
            "metadata": {
                "note": "Accuracy based on judge winner matching preferred label",
                "invalid_excluded": True
            }
        }


def compute_rr(
    samples: List[PairwiseSample],
    judgments: List[JudgeResult],
    biased_answers: List[str]
) -> Dict[str, Any]:
    """
    Compute Robustness Rate (RR): judge选择GT的比例（against biased answer）.
    
    Args:
        samples: Original samples
        judgments: Judge results for GT vs biased answer
        biased_answers: List of biased answers (for reference)
        
    Returns:
        Dictionary with RR metrics and metadata
    """
    if len(samples) != len(judgments):
        raise ValueError(f"Sample count mismatch: {len(samples)} samples vs {len(judgments)} judgments")
    
    # Exclude model-invalid judgments from denominator
    valid_pairs = [(s, j) for s, j in zip(samples, judgments) if j.is_valid]
    invalid_count = len(samples) - len(valid_pairs)
    
    if invalid_count > 0:
        logger.warning(
            f"Excluding {invalid_count} model-invalid judgments from RR denominator "
            "(e.g., safety filter blocks, quota errors, timeout)."
        )
    
    if not valid_pairs:
        return {
            "rr": 0.0,
            "gt_wins": 0,
            "total": len(samples),
            "valid_count": 0,
            "invalid_count": invalid_count,
            "has_gt_labels": False,
            "gt_wins_against_biased": 0,
            "biased_wins": 0,
            "ties": 0,
            "metadata": {
                "note": "No valid judgments available (all excluded due to model-side errors)",
                "invalid_reasons": "safety_filter, quota_error, timeout"
            }
        }
    
    valid_samples, valid_judgments = zip(*valid_pairs)
    valid_samples = list(valid_samples)
    valid_judgments = list(valid_judgments)
    
    # Determine GT for each sample
    gt_wins = 0
    total = len(samples)
    valid_total = len(valid_samples)
    has_gt_labels = any(s.preferred is not None for s in valid_samples)
    
    for sample, judgment in zip(valid_samples, valid_judgments):
        # Determine which answer is GT
        if sample.preferred == "1":
            gt_key = "A"  # answer_1 is GT, so it's answer A in judgment
        elif sample.preferred == "2":
            # answer_2 is GT, but we're comparing GT (answer_1) vs biased_answer_2
            # So GT should be answer A (which is answer_1)
            gt_key = "A"
        else:
            # Placeholder: assume answer_1 is GT
            gt_key = "A"
        
        if judgment.winner == gt_key:
            gt_wins += 1
    
    rr = gt_wins / valid_total if valid_total > 0 else 0.0
    
    return {
        "rr": rr,
        "gt_wins": gt_wins,
        "total": total,
        "valid_count": valid_total,
        "invalid_count": invalid_count,
        "has_gt_labels": has_gt_labels,
        "gt_wins_against_biased": gt_wins,
        "biased_wins": sum(1 for j in valid_judgments if j.winner == "B"),
        "ties": sum(1 for j in valid_judgments if j.winner == "tie"),
        "metadata": {
            "note": "RR = proportion of judge selecting GT answer against biased answer",
            "placeholder": not has_gt_labels,
            "placeholder_note": "Assuming answer_1 is GT when preferred label unavailable",
            "invalid_excluded": True
        }
    }


def compute_cr(
    judgments_round1: List[JudgeResult],
    judgments_round2: List[JudgeResult]
) -> Dict[str, Any]:
    """
    Compute Consistency Rate (CR): 一致性指标.
    
    真正的 CR 计算方式：
    - 对同一对答案评估两次
    - 如果两次结果相同，CR=1；否则 CR=0
    - 计算所有样本的平均值
    
    Args:
        judgments_round1: First round of judge results
        judgments_round2: Second round of judge results (same samples, same order)
        
    Returns:
        Dictionary with CR metrics and metadata
    """
    if len(judgments_round1) != len(judgments_round2):
        raise ValueError(
            f"Judgment count mismatch: round1 has {len(judgments_round1)} judgments, "
            f"round2 has {len(judgments_round2)} judgments"
        )
    
    total = len(judgments_round1)
    if total == 0:
        return {
            "cr": 0.0,
            "method": "consistency",
            "total": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "consistent_count": 0,
            "metadata": {"note": "No judgments available"}
        }
    
    # Exclude model-invalid judgment pairs from denominator
    valid_pairs = [(j1, j2) for j1, j2 in zip(judgments_round1, judgments_round2) if j1.is_valid and j2.is_valid]
    invalid_count = total - len(valid_pairs)
    
    if invalid_count > 0:
        logger.warning(
            f"Excluding {invalid_count} model-invalid judgment pairs from CR denominator "
            "(e.g., safety filter blocks, quota errors, timeout)."
        )
    
    if not valid_pairs:
        return {
            "cr": 0.0,
            "method": "consistency",
            "total": total,
            "valid_count": 0,
            "invalid_count": invalid_count,
            "consistent_count": 0,
            "metadata": {
                "note": "No valid judgment pairs available (all excluded due to model-side errors)",
                "invalid_reasons": "safety_filter, quota_error, timeout"
            }
        }
    
    valid_round1, valid_round2 = zip(*valid_pairs)
    valid_round1 = list(valid_round1)
    valid_round2 = list(valid_round2)
    
    # Compare winners from two rounds
    consistent_count = 0
    inconsistent_samples = []
    
    for idx, (j1, j2) in enumerate(zip(valid_round1, valid_round2)):
        # Compare winners: same winner means consistent
        if j1.winner == j2.winner:
            consistent_count += 1
        else:
            inconsistent_samples.append({
                "sample_index": idx,
                "round1_winner": j1.winner,
                "round2_winner": j2.winner
            })
    
    valid_total = len(valid_pairs)
    cr = consistent_count / valid_total if valid_total > 0 else 0.0
    
    return {
        "cr": cr,
        "method": "consistency",
        "total": total,
        "valid_count": valid_total,
        "invalid_count": invalid_count,
        "consistent_count": consistent_count,
        "inconsistent_count": valid_total - consistent_count,
        "inconsistent_samples": inconsistent_samples[:10],  # Limit to first 10 for brevity
        "metadata": {
            "note": "CR = proportion of samples with consistent judgment across two rounds",
            "definition": "For each sample, if round1 winner == round2 winner, CR=1; else CR=0. Final CR is the average.",
            "invalid_excluded": True
        }
    }

