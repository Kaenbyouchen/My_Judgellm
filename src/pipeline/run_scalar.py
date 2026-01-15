"""
Pipeline for scalar evaluation (future use).
"""
from typing import List, Dict, Any
from loguru import logger

from ..dataset.schemas import ScalarSample
from ..dataset.loaders import load_scalar_jsonl


def run_scalar_evaluation(
    data_path: str,
    judge_type: str,
    judge_model_name: str,
    judge_config: Dict[str, Any],
    run_dir: str
) -> Dict[str, Any]:
    """
    Run scalar evaluation pipeline (placeholder for future implementation).
    
    Args:
        data_path: Path to scalar data file
        judge_type: Type of judge
        judge_model_name: Name of judge model
        judge_config: Judge model configuration
        run_dir: Run directory for all output files
        
    Returns:
        Dictionary with results
    """
    logger.info("Scalar evaluation pipeline (not yet implemented)")
    logger.warning("This is a placeholder. Scalar evaluation will be implemented in the future.")
    
    # Load data
    samples = load_scalar_jsonl(data_path)
    logger.info(f"Loaded {len(samples)} scalar samples")
    
    return {
        "samples": len(samples),
        "status": "not_implemented",
        "note": "Scalar evaluation is a placeholder for future implementation"
    }

