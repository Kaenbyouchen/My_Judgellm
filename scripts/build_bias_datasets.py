#!/usr/bin/env python
"""
Build bias-injected datasets for multiple biases and injection modes.

Usage:
  python scripts/build_bias_datasets.py \
    --config configs/experiment.yaml \
    --biases jargon_overloading clinical_formatting \
    --modes word rewrite
"""
import sys
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_yaml
from src.models.registry import ModelRegistry
from src.bias.injector import BiasInjector
from src.bias.cache import check_cache_exists, save_bias_dataset, apply_bias_to_samples
from src.dataset.loaders import load_pairwise_jsonl
from src.dataset.preprocess import validate_pairwise_samples


def resolve_data_path(
    config: Dict[str, Any],
    datasets_config: Dict[str, Any],
    project_root: Path,
    override_path: str | None = None,
) -> str:
    if override_path:
        path = override_path
    else:
        data_config = config.get("data", {})
        path = data_config.get("path")
        if not path:
            dataset_name = data_config.get("dataset_name")
            datasets = datasets_config.get("datasets", datasets_config) if datasets_config else {}
            if dataset_name and isinstance(datasets, dict) and dataset_name in datasets:
                dataset_info = datasets[dataset_name]
                if isinstance(dataset_info, dict):
                    path = dataset_info.get("path")
                elif isinstance(dataset_info, str):
                    path = dataset_info
    if not path:
        raise ValueError("Unable to resolve data path. Please set data.path or data.dataset_name in config.")
    if not Path(path).is_absolute():
        path = str(project_root / path)
    return path


def resolve_bias_injector(
    bias_config: Dict[str, Any],
    models_config: Dict[str, Any],
) -> tuple[str, str | None, Dict[str, Any]]:
    ModelRegistry.set_models_config(models_config)
    bias_allow_fallback_mock = bool(bias_config.get("allow_fallback_mock", False))

    injector_type = bias_config.get("injector_type")
    injector_model_id = None

    if not injector_type:
        if "model" in bias_config:
            injector_model_id = bias_config["model"]
            provider = ModelRegistry.infer_provider_from_model_id(injector_model_id)
            if provider is None:
                if bias_allow_fallback_mock:
                    injector_type = "mock"
                    bias_config["injector_type"] = "mock"
                else:
                    raise ValueError(
                        f"Could not infer provider for bias model '{injector_model_id}'. "
                        "Please add it to configs/models.yaml or specify bias.injector_type."
                    )
            else:
                injector_type = provider
                bias_config["injector_type"] = provider
        else:
            injector_type = "mock"
            bias_config["injector_type"] = "mock"

    injector_model_config: Dict[str, Any] = {}
    if injector_type != "mock":
        if injector_model_id is None:
            injector_model_id = (
                bias_config.get("model_id")
                or bias_config.get("model_name")
                or bias_config.get("model")
            )
        if not injector_model_id:
            raise ValueError("Bias injector model_id is required for non-mock injection.")
        injector_model_overrides = {
            k: v
            for k, v in bias_config.items()
            if k
            not in {
                "enabled",
                "type",
                "inject_to",
                "injection_mode",
                "injector_type",
                "model_id",
                "model_name",
                "model",
                "allow_fallback_mock",
            }
        }
        injector_model_config = {
            "model_id": injector_model_id,
            "allow_fallback_mock": bias_allow_fallback_mock,
        }
        injector_model_config.update(injector_model_overrides)

    return injector_type, injector_model_id, injector_model_config


def get_cache_model_name(injector_type: str, injector_model_config: Dict[str, Any]) -> str:
    if injector_type == "mock":
        return "mock"
    cache_model_name = (
        injector_model_config.get("model_id")
        or injector_model_config.get("model_name")
        or injector_type
    )
    return cache_model_name.lower().replace("-", "").replace("_", "")


def resolve_bias_list(bias_config: Dict[str, Any], override_biases: List[str] | None) -> List[str]:
    if override_biases:
        return override_biases
    bias_type = bias_config.get("type")
    if isinstance(bias_type, list):
        return [str(x) for x in bias_type]
    if isinstance(bias_type, str) and bias_type:
        return [bias_type]
    raise ValueError("No bias types provided. Use --biases or set bias.type in config.")


def main():
    parser = argparse.ArgumentParser(
        description="Build bias-injected datasets without running evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base experiment configuration file",
    )
    parser.add_argument(
        "--biases",
        nargs="+",
        default=None,
        help="Bias types to build. If omitted, use bias.type from config.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rewrite"],
        choices=["rewrite", "word"],
        help="Injection modes to build (rewrite, word).",
    )
    parser.add_argument(
        "--inject-to",
        type=str,
        default=None,
        choices=["gt", "non_gt"],
        help="Override inject_to (gt or non_gt).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache exists.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    bias_config = config.get("bias", {})

    # Load models.yaml
    models_yaml_path = project_root / "configs" / "models.yaml"
    models_config = load_yaml(str(models_yaml_path)) if models_yaml_path.exists() else {}

    # Load prompts.yaml
    prompts_yaml_path = project_root / "configs" / "prompts.yaml"
    prompts_config = load_yaml(str(prompts_yaml_path)) if prompts_yaml_path.exists() else {}

    # Load datasets.yaml
    datasets_yaml_path = project_root / "configs" / "datasets.yaml"
    datasets_config = load_yaml(str(datasets_yaml_path)) if datasets_yaml_path.exists() else {}

    data_path = resolve_data_path(config, datasets_config, project_root, args.data_path)
    logger.info(f"Using data path: {data_path}")

    bias_types = resolve_bias_list(bias_config, args.biases)
    injection_modes = args.modes
    inject_to = args.inject_to or bias_config.get("inject_to", "non_gt")

    injector_type, injector_model_id, injector_model_config = resolve_bias_injector(
        bias_config, models_config
    )
    cache_model_name = get_cache_model_name(injector_type, injector_model_config)

    # Load samples once
    samples = load_pairwise_jsonl(data_path)
    samples = validate_pairwise_samples(samples)
    if not samples:
        raise ValueError("No valid samples loaded.")

    for bias_type in bias_types:
        for mode in injection_modes:
            bias_variant_name = f"word_{bias_type}" if mode == "word" else bias_type
            logger.info("-" * 60)
            logger.info(f"Building bias dataset: {bias_variant_name} (mode={mode})")
            logger.info(f"Injector: {injector_type}, model: {injector_model_id or 'mock'}")

            if not args.force and check_cache_exists(data_path, bias_variant_name, cache_model_name):
                logger.info("Cache exists, skipping (use --force to rebuild).")
                continue

            injector_system_prompt = None
            injector_user_template = None
            if injector_type != "mock":
                bias_prompt_cfg = (prompts_config.get("bias_injection", {}) or {}).get(bias_type, {})
                if not bias_prompt_cfg:
                    raise ValueError(
                        f"Bias prompt not found for bias type '{bias_type}'. "
                        "Please add it under bias_injection in configs/prompts.yaml."
                    )
                injector_system_prompt = bias_prompt_cfg.get("system")
                injector_user_template = bias_prompt_cfg.get("user")

            bias_injector = BiasInjector(
                bias_type=bias_type,
                injector_type=injector_type,
                model_config=injector_model_config or {},
                system_prompt=injector_system_prompt,
                prompt_template=injector_user_template,
                injection_mode=mode,
            )

            biased_samples = apply_bias_to_samples(samples, bias_injector, inject_to=inject_to)

            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "injector_config": {
                    "injector_type": injector_type,
                    "model_config": injector_model_config,
                },
                "prompt_config": {
                    "system_prompt": injector_system_prompt,
                    "user_template": injector_user_template,
                },
                "injection_mode": mode,
            }
            save_bias_dataset(
                biased_samples,
                data_path,
                bias_variant_name,
                cache_model_name,
                metadata=metadata,
            )
            logger.info(f"Saved bias dataset: {bias_variant_name} (mode={mode})")

    logger.info("All bias datasets completed.")


if __name__ == "__main__":
    main()
