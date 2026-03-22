"""
Prompt config routing utilities.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

from .io import load_yaml


def _normalize_category(category: str) -> str:
    value = (category or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def get_prompt_filename_for_category(dataset_category: str) -> str:
    normalized = _normalize_category(dataset_category)
    mapping = {
        "one_turn_conversation": "one_turn_con_prompt.yaml",
        "multi_turn_conversation": "multi_turn_con_prompt.yaml",
        "semantic_q_a": "semantic_QA_prompt.yaml",
        "semantic_qa": "semantic_QA_prompt.yaml",
        "multi_choice_q_a": "MCQ_prompt.yaml",
        "multi_choice_qa": "MCQ_prompt.yaml",
        "open_q_a": "open_QA_prompt.yaml",
        "open_qa": "open_QA_prompt.yaml",
        "ehr": "EHR_prompt.yaml",
        "dummy": "MCQ_prompt.yaml",
    }
    return mapping.get(normalized, "MCQ_prompt.yaml")


def load_prompts_for_category(
    project_root: Path,
    dataset_category: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Load prompt config by dataset category.

    Returns:
        (prompt_config, source_description)
    """
    configs_dir = project_root / "configs"
    # Category prompt selected directly from dataset_category.
    target_name = get_prompt_filename_for_category(dataset_category)
    target_path = configs_dir / target_name

    if target_path.exists():
        return load_yaml(str(target_path)), str(target_path)

    # Fallback to MCQ prompt when category file is missing.
    fallback_path = configs_dir / "MCQ_prompt.yaml"
    if fallback_path.exists():
        return load_yaml(str(fallback_path)), str(fallback_path)

    return {}, "none"

