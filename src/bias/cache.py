"""
Bias injection cache management.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from ..dataset.schemas import PairwiseSample
from ..dataset.loaders import load_pairwise_jsonl


def get_cache_path(dataset_path: str, bias_type: str, model_name: str) -> Path:
    """
    Get the cache path for bias-injected dataset.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias (e.g., "jargon_overloading", "verbosity")
        model_name: Model name used for injection (e.g., "gpt4omini", "gemini")
        
    Returns:
        Path to cached bias-injected dataset
    """
    dataset_path_obj = Path(dataset_path)
    dataset_name = dataset_path_obj.parent.name
    
    # Construct cache path: data/<dataset_name>/bias_injection/<bias_type>/<bias_type>__<model_name>.jsonl
    cache_dir = dataset_path_obj.parent / "bias_injection" / bias_type
    cache_file = f"{bias_type}__{model_name}.jsonl"
    return cache_dir / cache_file


def get_meta_path(cache_path: Path) -> Path:
    """Get the meta.json path for a cached dataset."""
    return cache_path.with_suffix(".meta.json")


def check_cache_exists(dataset_path: str, bias_type: str, model_name: str) -> bool:
    """
    Check if cached bias-injected dataset exists.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        
    Returns:
        True if cache exists, False otherwise
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    meta_path = get_meta_path(cache_path)
    
    # Both files must exist
    return cache_path.exists() and meta_path.exists()


def load_cached_bias_dataset(
    dataset_path: str,
    bias_type: str,
    model_name: str
) -> Optional[List[PairwiseSample]]:
    """
    Load cached bias-injected dataset.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        
    Returns:
        List of PairwiseSample with bias injected, or None if cache doesn't exist
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    
    if not cache_path.exists():
        logger.debug(f"Cache not found: {cache_path}")
        return None
    
    try:
        logger.info(f"Loading cached bias-injected dataset from {cache_path}")
        samples = load_pairwise_jsonl(str(cache_path))
        logger.info(f"Loaded {len(samples)} samples from cache")
        return samples
    except Exception as e:
        logger.error(f"Error loading cached dataset: {e}")
        return None


def save_bias_dataset(
    samples: List[PairwiseSample],
    dataset_path: str,
    bias_type: str,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save bias-injected dataset to cache.
    
    Args:
        samples: List of PairwiseSample with bias injected
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        metadata: Additional metadata to save in meta.json
        
    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    meta_path = get_meta_path(cache_path)
    
    # Create directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSONL file
    import jsonlines
    with jsonlines.open(str(cache_path), mode='w') as writer:
        for sample in samples:
            writer.write(sample.to_dict())
    
    logger.info(f"Saved {len(samples)} bias-injected samples to {cache_path}")
    
    # Prepare metadata (no version field)
    meta_data = {
        "bias_type": bias_type,
        "model_name": model_name,
        "original_dataset": str(dataset_path),
        "num_samples": len(samples),
        "created_at": metadata.get("created_at") if metadata else None,
        "injector_config": metadata.get("injector_config") if metadata else None,
        "prompt_config": metadata.get("prompt_config") if metadata else None,
    }
    
    # Save meta.json
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {meta_path}")
    
    return cache_path


def apply_bias_to_samples(
    samples: List[PairwiseSample],
    bias_injector,
    inject_to: str = "non_gt"
) -> List[PairwiseSample]:
    """
    Apply bias injection to a list of samples.
    
    Args:
        samples: Original samples
        bias_injector: BiasInjector instance
        inject_to: Where to inject bias ("non_gt" or "gt")
        
    Returns:
        List of samples with bias injected
    """
    biased_samples = []
    
    for sample in samples:
        # Create a copy to avoid modifying original
        biased_sample = PairwiseSample.from_dict(sample.to_dict())
        
        # Get GT and non-GT answers
        gt_answer, gt_key = sample.get_gt_answer()
        non_gt_answer, non_gt_key = sample.get_non_gt_answer()
        
        # Inject bias based on inject_to setting
        if inject_to == "non_gt":
            # Inject bias into non-GT answer
            biased_answer = bias_injector.inject(non_gt_answer, question=sample.question)
            if gt_key == "answer_1":
                biased_sample.answer_2 = biased_answer
            else:
                biased_sample.answer_1 = biased_answer
        elif inject_to == "gt":
            # Inject bias into GT answer
            biased_answer = bias_injector.inject(gt_answer, question=sample.question)
            if gt_key == "answer_1":
                biased_sample.answer_1 = biased_answer
            else:
                biased_sample.answer_2 = biased_answer
        else:
            raise ValueError(f"Unknown inject_to value: {inject_to}")
        
        biased_samples.append(biased_sample)
    
    return biased_samples
