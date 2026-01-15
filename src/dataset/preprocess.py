"""
Data preprocessing utilities.
"""
from typing import List
from .schemas import PairwiseSample


def validate_pairwise_samples(samples: List[PairwiseSample]) -> List[PairwiseSample]:
    """
    Validate and filter pairwise samples.
    
    Args:
        samples: List of samples to validate
        
    Returns:
        List of valid samples
    """
    valid_samples = []
    for sample in samples:
        # Basic validation
        if not sample.question or not sample.question.strip():
            continue
        if not sample.answer_1 or not sample.answer_1.strip():
            continue
        if not sample.answer_2 or not sample.answer_2.strip():
            continue
        
        valid_samples.append(sample)
    
    return valid_samples


def get_gt_statistics(samples: List[PairwiseSample]) -> dict:
    """
    Get statistics about ground truth labels.
    
    Args:
        samples: List of pairwise samples
        
    Returns:
        Dictionary with statistics
    """
    total = len(samples)
    with_gt = sum(1 for s in samples if s.preferred is not None)
    without_gt = total - with_gt
    
    preferred_1 = sum(1 for s in samples if s.preferred == "1")
    preferred_2 = sum(1 for s in samples if s.preferred == "2")
    
    return {
        "total": total,
        "with_gt": with_gt,
        "without_gt": without_gt,
        "preferred_1": preferred_1,
        "preferred_2": preferred_2,
        "gt_coverage": with_gt / total if total > 0 else 0.0
    }

