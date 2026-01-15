"""
Reproducibility utilities.
"""
import random
import numpy as np
import os
from typing import Optional
from loguru import logger


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Try to set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except (ImportError, AttributeError, TypeError, Exception):
        # Silently ignore if tensorflow is not available or has compatibility issues
        pass
    
    logger.info(f"Random seed set to {seed}")


def get_seed() -> Optional[int]:
    """
    Get current random seed from environment or return None.
    
    Returns:
        Seed value or None
    """
    seed_str = os.getenv("RANDOM_SEED")
    if seed_str:
        try:
            return int(seed_str)
        except ValueError:
            return None
    return None

