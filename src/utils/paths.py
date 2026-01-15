"""
Path utilities for dataset and run directory handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


def _sanitize(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.replace(" ", "_")


def _normalize_path(path_str: str, project_root: Optional[Path] = None) -> Path:
    path = Path(path_str)
    if project_root and not path.is_absolute():
        path = project_root / path
    try:
        return path.resolve()
    except Exception:
        return path


def get_dataset_name(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
    datasets_config: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> str:
    """
    Resolve dataset name from config or dataset registry, with a path-based fallback.
    """
    data_cfg = config.get("data", {}) if config else {}
    explicit_name = data_cfg.get("dataset_name")
    if explicit_name:
        return str(explicit_name)

    dataset_key = data_cfg.get("dataset")
    datasets_cfg = datasets_config or {}
    datasets = datasets_cfg.get("datasets", datasets_cfg) if isinstance(datasets_cfg, dict) else {}
    if dataset_key and isinstance(datasets, dict) and dataset_key in datasets:
        return str(dataset_key)

    if data_path and isinstance(datasets, dict) and datasets:
        data_path_norm = _normalize_path(data_path, project_root)
        for name, info in datasets.items():
            if not isinstance(info, dict):
                continue
            candidate_path = info.get("path")
            if not candidate_path:
                continue
            candidate_norm = _normalize_path(candidate_path, project_root)
            if candidate_norm == data_path_norm:
                return str(name)

    if data_path:
        inferred = Path(data_path).stem
        logger.warning(
            f"dataset_name not found in config; inferred from data_path: {inferred}"
        )
        return inferred

    logger.warning("dataset_name not found in config; falling back to 'unknown'")
    return "unknown"


def find_next_run_idx(base_output_dir: Path, dir_prefix: str) -> int:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_idx = 1
    while True:
        dir_name = f"{dir_prefix}_{run_idx:03d}"
        if not (base_output_dir / dir_name).exists():
            return run_idx
        run_idx += 1


def get_run_dir(
    config: Dict[str, Any],
    base_output_dir: Path,
    dataset_name: str,
    bias_name: str,
    judge_name: str,
    run_id: Optional[int] = None,
) -> Path:
    """
    Build the run directory as: {dataset_name}_{original_run_dirname}
    where original_run_dirname = {data_type}_{bias}_{judge}_{NNN}
    """
    data_type = (config.get("data", {}) or {}).get("type", "pairwise")

    dataset_name = _sanitize(dataset_name)
    data_type = _sanitize(data_type)
    bias_name = _sanitize(bias_name)
    judge_name = _sanitize(judge_name)

    original_prefix = f"{data_type}_{bias_name}_{judge_name}"
    dir_prefix = f"{dataset_name}_{original_prefix}"

    run_idx = run_id or find_next_run_idx(base_output_dir, dir_prefix)
    run_dir = base_output_dir / f"{dir_prefix}_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")
    return run_dir
