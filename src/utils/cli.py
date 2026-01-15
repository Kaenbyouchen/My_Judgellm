"""
CLI utilities.
"""
import argparse
from pathlib import Path


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="JudgeLLM Medical Bias Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path from config"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config"
    )
    
    return parser.parse_args()


