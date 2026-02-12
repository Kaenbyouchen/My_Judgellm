"""
Data loaders for medical bias evaluation dataset.
"""
import json
import jsonlines
from pathlib import Path
from typing import List, Iterator
from loguru import logger

from .schemas import PairwiseSample, ScalarSample


def load_pairwise_jsonl(file_path: str) -> List[PairwiseSample]:
    """
    Load pairwise samples from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of PairwiseSample objects
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    samples = []
    with jsonlines.open(str(file_path)) as reader:
        for idx, line in enumerate(reader):
            try:
                # Skip empty lines
                if not line or (isinstance(line, dict) and len(line) == 0):
                    continue
                
                # Normalize common field aliases (answer1/answer2/GT)
                if "answer_1" not in line and "answer1" in line:
                    line["answer_1"] = line.get("answer1")
                if "answer_2" not in line and "answer2" in line:
                    line["answer_2"] = line.get("answer2")
                if "preferred" not in line and "GT" in line:
                    gt = line.get("GT")
                    if gt == "answer1":
                        line["preferred"] = "1"
                    elif gt == "answer2":
                        line["preferred"] = "2"
                    elif gt == line.get("answer1"):
                        line["preferred"] = "1"
                    elif gt == line.get("answer2"):
                        line["preferred"] = "2"

                # Validate required fields
                required_fields = ["id", "question", "answer_1", "answer_2"]
                for field in required_fields:
                    if field not in line:
                        raise ValueError(f"Missing required field '{field}' in line {idx+1}")
                
                # Preserve additional fields (e.g., category, source, model names) in meta
                # so downstream evaluation can filter/group by dataset type.
                base_meta = line.get("meta") if isinstance(line.get("meta"), dict) else {}
                extra_meta = {
                    k: v
                    for k, v in line.items()
                    if k
                    not in {
                        "id",
                        "question",
                        "answer_1",
                        "answer_2",
                        "preferred",
                        "meta",
                        # alias/source fields already normalized above
                        "answer1",
                        "answer2",
                        "GT",
                    }
                }
                line["meta"] = {**base_meta, **extra_meta}

                sample = PairwiseSample.from_dict(line)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Error loading sample at line {idx+1}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} pairwise samples from {file_path}")
    return samples


def load_pairwise_jsonl_streaming(file_path: str) -> Iterator[PairwiseSample]:
    """
    Load pairwise samples from JSONL file as iterator (for large files).
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        PairwiseSample objects
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with jsonlines.open(str(file_path)) as reader:
        for idx, line in enumerate(reader):
            try:
                # Normalize common field aliases (answer1/answer2/GT)
                if "answer_1" not in line and "answer1" in line:
                    line["answer_1"] = line.get("answer1")
                if "answer_2" not in line and "answer2" in line:
                    line["answer_2"] = line.get("answer2")
                if "preferred" not in line and "GT" in line:
                    gt = line.get("GT")
                    if gt == "answer1":
                        line["preferred"] = "1"
                    elif gt == "answer2":
                        line["preferred"] = "2"
                    elif gt == line.get("answer1"):
                        line["preferred"] = "1"
                    elif gt == line.get("answer2"):
                        line["preferred"] = "2"

                required_fields = ["id", "question", "answer_1", "answer_2"]
                for field in required_fields:
                    if field not in line:
                        raise ValueError(f"Missing required field '{field}' in line {idx+1}")
                
                base_meta = line.get("meta") if isinstance(line.get("meta"), dict) else {}
                extra_meta = {
                    k: v
                    for k, v in line.items()
                    if k
                    not in {
                        "id",
                        "question",
                        "answer_1",
                        "answer_2",
                        "preferred",
                        "meta",
                        "answer1",
                        "answer2",
                        "GT",
                    }
                }
                line["meta"] = {**base_meta, **extra_meta}

                sample = PairwiseSample.from_dict(line)
                yield sample
            except Exception as e:
                logger.warning(f"Error loading sample at line {idx+1}: {e}")
                continue


def load_scalar_jsonl(file_path: str) -> List[ScalarSample]:
    """
    Load scalar samples from JSONL file (future use).
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of ScalarSample objects
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    samples = []
    with jsonlines.open(str(file_path)) as reader:
        for idx, line in enumerate(reader):
            try:
                required_fields = ["id", "question", "answer"]
                for field in required_fields:
                    if field not in line:
                        raise ValueError(f"Missing required field '{field}' in line {idx+1}")
                
                sample = ScalarSample.from_dict(line)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Error loading sample at line {idx+1}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} scalar samples from {file_path}")
    return samples

