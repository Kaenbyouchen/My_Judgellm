#!/usr/bin/env python3
"""
Config consistency checker for JudgeLLM.

Checks:
- All model references in configs/*.yaml exist in configs/models.yaml
- Dataset name in configs/experiment.yaml exists in configs/datasets.yaml
- JUDGES array in scripts/slurm_batch_evaluate.sh uses valid model IDs
- models.yaml entries have model_name fields
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def collect_model_ids(models_cfg: Dict[str, Any]) -> Dict[str, str]:
    model_ids: Dict[str, str] = {}
    for provider, cfg in models_cfg.items():
        if not isinstance(cfg, dict):
            continue
        for k, v in cfg.items():
            if k == "defaults":
                continue
            if isinstance(v, dict):
                model_ids[k] = provider
    return model_ids


def check_model_refs_in_configs(model_ids: Dict[str, str]) -> List[Tuple[str, str, str]]:
    problems: List[Tuple[str, str, str]] = []
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = load_yaml(path)
        if not isinstance(cfg, dict):
            continue
        bias = cfg.get("bias", {}) if isinstance(cfg.get("bias", {}), dict) else {}
        judge = cfg.get("judge", {}) if isinstance(cfg.get("judge", {}), dict) else {}
        for label, value in [
            ("bias.model", bias.get("model")),
            ("bias.model_id", bias.get("model_id")),
            ("bias.model_name", bias.get("model_name")),
            ("judge.model", judge.get("model")),
            ("judge.model_id", judge.get("model_id")),
            ("judge.model_name", judge.get("model_name")),
        ]:
            if value and value not in model_ids:
                problems.append((str(path), label, value))
    return problems


def check_dataset_name() -> List[str]:
    problems: List[str] = []
    exp = load_yaml(CONFIG_DIR / "experiment.yaml")
    datasets = load_yaml(CONFIG_DIR / "datasets.yaml")
    datasets = datasets.get("datasets", datasets) if isinstance(datasets, dict) else {}
    dataset_name = exp.get("data", {}).get("dataset_name") if isinstance(exp.get("data", {}), dict) else None
    if dataset_name and dataset_name not in datasets:
        problems.append(f"dataset_name '{dataset_name}' not found in configs/datasets.yaml")
    return problems


def check_models_have_names(models_cfg: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for provider, cfg in models_cfg.items():
        if not isinstance(cfg, dict):
            continue
        for model_id, model_cfg in cfg.items():
            if model_id == "defaults":
                continue
            if not isinstance(model_cfg, dict):
                continue
            if "model_name" not in model_cfg:
                problems.append(f"models.yaml: provider '{provider}' model '{model_id}' missing 'model_name'")
    return problems


def check_slurm_judges(model_ids: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    slurm_path = ROOT / "scripts" / "slurm_batch_evaluate.sh"
    if not slurm_path.exists():
        return problems
    content = slurm_path.read_text()
    match = re.search(r"JUDGES=\\(([^)]*)\\)", content, re.MULTILINE)
    if not match:
        return problems
    judges_raw = match.group(1)
    judges = re.findall(r"\"([^\"]+)\"", judges_raw)
    for judge in judges:
        # judge can be provider:model_id or model_id
        if ":" in judge:
            _, model_id = judge.split(":", 1)
        else:
            model_id = judge
        if model_id not in model_ids:
            problems.append(f"slurm_batch_evaluate.sh: judge '{judge}' not found in models.yaml")
    return problems


def main() -> int:
    models_cfg = load_yaml(CONFIG_DIR / "models.yaml")
    model_ids = collect_model_ids(models_cfg)

    problems = []
    problems.extend(check_model_refs_in_configs(model_ids))
    problems.extend(check_dataset_name())
    problems.extend(check_models_have_names(models_cfg))
    problems.extend(check_slurm_judges(model_ids))

    print("Config Consistency Check")
    print("=" * 60)
    print(f"Model IDs loaded: {len(model_ids)}")
    print(f"Configs scanned: {len(list(CONFIG_DIR.glob('*.yaml')))}")
    print()

    if not problems:
        print("✅ No consistency issues found.")
        return 0

    print("❌ Issues found:")
    for p in problems:
        print(f" - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
