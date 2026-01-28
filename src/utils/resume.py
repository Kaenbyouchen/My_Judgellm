"""
Resume functionality for checkpoint/resume evaluation.
"""
import json
from pathlib import Path
from typing import Set, Dict, Any, Optional, List
from loguru import logger


def load_completed_sample_ids(run_dir: Path, judgment_type: str = "original") -> Set[str]:
    """
    Load completed sample IDs from existing judgment files.
    
    Args:
        run_dir: Run directory containing output files
        judgment_type: "original" or "bias"
        
    Returns:
        Set of completed sample IDs
    """
    completed_ids = set()
    
    if judgment_type == "original":
        judgment_file = run_dir / "judge_raw_original.jsonl"
    elif judgment_type == "bias":
        judgment_file = run_dir / "judge_raw_bias.jsonl"
    else:
        return completed_ids
    
    if not judgment_file.exists():
        return completed_ids
    
    try:
        with open(judgment_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Improved parsing: handle both single-line and multi-line JSON
        import re
        
        # Strategy 1: Try to parse complete JSON objects (handles multi-line formatted JSON)
        # Match JSON objects from { to matching }
        json_objects = []
        current_obj = ""
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                escape_next = False
                current_obj += char
                continue
            
            if char == '\\':
                escape_next = True
                current_obj += char
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                current_obj += char
                continue
            
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        current_obj = ""
                    brace_count += 1
                    current_obj += char
                elif char == '}':
                    current_obj += char
                    brace_count -= 1
                    if brace_count == 0:
                        json_objects.append(current_obj.strip())
                        current_obj = ""
                else:
                    if brace_count > 0:
                        current_obj += char
            else:
                current_obj += char
        
        # Parse each JSON object
        for json_str in json_objects:
            if not json_str.strip():
                continue
            try:
                judgment = json.loads(json_str)
                sample_id = judgment.get("sample_id")
                if sample_id:
                    completed_ids.add(str(sample_id))
            except json.JSONDecodeError:
                # Fallback: try to extract sample_id with regex
                try:
                    match = re.search(r'"sample_id"\s*:\s*"([^"]+)"', json_str)
                    if match:
                        completed_ids.add(match.group(1))
                except Exception:
                    continue
        
        # Strategy 2: If no objects found, try line-by-line parsing (for single-line JSONL)
        if not completed_ids:
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    judgment = json.loads(line)
                    sample_id = judgment.get("sample_id")
                    if sample_id:
                        completed_ids.add(str(sample_id))
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        logger.warning(f"Error loading completed samples from {judgment_file}: {e}")
    
    return completed_ids


def check_resume_available(run_dir: Path) -> Dict[str, Any]:
    """
    Check if resume is available and return resume information.
    
    Args:
        run_dir: Run directory to check
        
    Returns:
        Dictionary with resume information:
        {
            "available": bool,
            "original_completed": int,
            "bias_completed": int,
            "original_file": str,
            "bias_file": str
        }
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        return {
            "available": False,
            "original_completed": 0,
            "bias_completed": 0,
            "original_file": None,
            "bias_file": None
        }
    
    original_file = run_dir / "judge_raw_original.jsonl"
    bias_file = run_dir / "judge_raw_bias.jsonl"
    
    original_completed = 0
    bias_completed = 0
    
    if original_file.exists():
        original_ids = load_completed_sample_ids(run_dir, "original")
        original_completed = len(original_ids)
    
    if bias_file.exists():
        bias_ids = load_completed_sample_ids(run_dir, "bias")
        bias_completed = len(bias_ids)
    
    available = original_completed > 0 or bias_completed > 0
    
    return {
        "available": available,
        "original_completed": original_completed,
        "bias_completed": bias_completed,
        "original_file": str(original_file) if original_file.exists() else None,
        "bias_file": str(bias_file) if bias_file.exists() else None
    }


def filter_completed_samples(
    samples: List[Any],
    completed_ids: Set[str],
    sample_id_key: str = "id"
) -> tuple[List[Any], List[Any]]:
    """
    Filter samples into completed and remaining.
    
    Args:
        samples: List of samples
        completed_ids: Set of completed sample IDs
        sample_id_key: Key to access sample ID (default: "id")
        
    Returns:
        Tuple of (remaining_samples, completed_samples)
    """
    remaining = []
    completed = []
    
    for sample in samples:
        sample_id = getattr(sample, sample_id_key, None) if hasattr(sample, 'id') else sample.get(sample_id_key) if isinstance(sample, dict) else None
        if sample_id and str(sample_id) in completed_ids:
            completed.append(sample)
        else:
            remaining.append(sample)
    
    return remaining, completed


def load_existing_judgments(run_dir: Path, judgment_type: str = "original") -> List[Dict[str, Any]]:
    """
    Load existing judgments from JSONL file.
    
    Args:
        run_dir: Run directory
        judgment_type: "original" or "bias"
        
    Returns:
        List of existing judgment dictionaries
    """
    if judgment_type == "original":
        judgment_file = run_dir / "judge_raw_original.jsonl"
    elif judgment_type == "bias":
        judgment_file = run_dir / "judge_raw_bias.jsonl"
    else:
        return []
    
    if not judgment_file.exists():
        return []
    
    judgments = []
    try:
        with open(judgment_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by newlines and parse each JSON object
            # Handle multi-line formatted JSON
            lines = content.split('\n')
            current_json = ""
            brace_count = 0
            
            for line in lines:
                if not line.strip() and brace_count == 0:
                    # Empty line between JSON objects, skip
                    continue
                    
                current_json += line + '\n'
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and current_json.strip():
                    try:
                        judgment = json.loads(current_json.strip())
                        judgments.append(judgment)
                        current_json = ""
                    except json.JSONDecodeError:
                        # Try to extract JSON from the text using regex
                        import re
                        json_match = re.search(r'\{.*\}', current_json, re.DOTALL)
                        if json_match:
                            try:
                                judgment = json.loads(json_match.group(0))
                                judgments.append(judgment)
                            except:
                                pass
                        current_json = ""
            
            # Handle any remaining JSON at the end
            if current_json.strip() and brace_count == 0:
                try:
                    judgment = json.loads(current_json.strip())
                    judgments.append(judgment)
                except:
                    pass
    except Exception as e:
        logger.warning(f"Error loading existing judgments from {judgment_file}: {e}")
    
    return judgments
