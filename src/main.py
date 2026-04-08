"""
Main entry point for the evaluation pipeline.
"""
import os
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# Add project root to path to enable absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io import load_yaml, save_yaml
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.utils.cli import parse_args
from src.utils.paths import get_dataset_name, get_run_dir
from src.utils.prompt_config import load_prompts_for_category
from src.pipeline.run_pairwise import run_pairwise_evaluation
from src.pipeline.run_scalar import run_scalar_evaluation
from src.models.registry import ModelRegistry
from src.metrics.reports import append_summary_jsonl, append_summary_csv


def get_bias_name(bias_config: dict) -> str:
    """
    Get bias name string from bias config.
    
    Args:
        bias_config: Bias configuration dictionary
        
    Returns:
        Bias name string (e.g., "jargon_overloading", "authority+jargon_overloading", or "none")
    """
    if not bias_config.get("enabled", False):
        return "none"
    
    bias_type = bias_config.get("type", "")
    # Support multiple biases if type is a list
    if isinstance(bias_type, list):
        base_name = "+".join(bias_type)
    elif isinstance(bias_type, str) and bias_type:
        base_name = bias_type
    else:
        return "none"

    injection_mode = bias_config.get("injection_mode", "rewrite")
    if injection_mode == "word":
        return f"word_{base_name}"
    return base_name


def get_judge_name(judge_type: str, judge_model_name: str, judge_model_config: dict) -> str:
    """
    Get judge name string for directory naming.
    
    Args:
        judge_type: Type of judge ("mock", "openai", "vllm", etc.)
        judge_model_name: Name of judge model
        judge_model_config: Judge model configuration
        
    Returns:
        Judge name string (e.g., "mock", "openai_gpt-4", "hf_llama-2-7b")
    """
    if judge_type == "mock":
        return "mock"
    elif judge_type in {"openai", "anthropic", "gemini"}:
        model = judge_model_config.get("model", judge_model_name)
        return f"{judge_type}_{model}"
    elif judge_type in {"hf", "vllm"}:
        model = judge_model_config.get("model_name", judge_model_name)
        # Extract short name from full path (e.g., "meta-llama/Llama-2-7b-chat-hf" -> "llama-2-7b")
        if "/" in model:
            model = model.split("/")[-1]
        return f"{judge_type}_{model}"
    else:
        # Fallback: use type and model name
        return f"{judge_type}_{judge_model_name}"


def main():
    """Main function."""
    args = parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "configs").exists():
        # Try to find project root by looking for configs directory
        current = Path.cwd()
        if (current / "configs").exists():
            project_root = current
        else:
            logger.error("Cannot find project root. Please run from project root directory.")
            sys.exit(1)
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(project_root / config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    
    # Load models.yaml separately (it's a separate config file)
    models_yaml_path = project_root / "configs" / "models.yaml"
    if models_yaml_path.exists():
        models_config = load_yaml(str(models_yaml_path))
    else:
        # Fallback: try to get from experiment.yaml if it has models section
        models_config = config.get("models", {})
        if not models_config:
            logger.warning(f"models.yaml not found at {models_yaml_path}, using empty config")
            models_config = {}

    # Make model pool available globally (so BiasInjector / ModelJudge can resolve provider+model_id)
    ModelRegistry.set_models_config(models_config)
    
    # Load datasets.yaml separately (optional)
    datasets_yaml_path = project_root / "configs" / "datasets.yaml"
    datasets_config = {}
    if datasets_yaml_path.exists():
        datasets_config = load_yaml(str(datasets_yaml_path))
    
    # Get experiment config
    exp_config = config.get("experiment", {})
    data_config = config.get("data", {})
    bias_config = config.get("bias", {})
    # Log raw bias config (before logging setup)
    logger.info(f"Bias config raw: {bias_config}")
    
    # Optional: allow fallback to mock for bias injection (default False when using external model)
    bias_allow_fallback_mock = bool(bias_config.get("allow_fallback_mock", False))

    # Determine bias injector type with proper priority:
    # 1. Explicit injector_type in config
    # 2. Infer from bias.model if present
    # 3. Default to "mock"
    bias_injector_type = bias_config.get("injector_type")
    injector_model_id = None
    bias_injection_mode = bias_config.get("injection_mode", "rewrite")
    
    if not bias_injector_type:
        # No explicit injector_type, check if bias.model is specified
        if "model" in bias_config:
            injector_model_id = bias_config["model"]
            # Infer provider from model_id using ModelRegistry
            injector_provider = ModelRegistry.infer_provider_from_model_id(injector_model_id)
            if injector_provider is None:
                if bias_allow_fallback_mock:
                    logger.warning(
                        f"Could not infer provider for bias injector model '{injector_model_id}'. "
                        "Falling back to 'mock' because bias.allow_fallback_mock=True"
                    )
                    bias_injector_type = "mock"
                    bias_config["injector_type"] = "mock"
                else:
                    raise ValueError(
                        f"Could not infer provider for bias injector model '{injector_model_id}'. "
                        "Please add it to configs/models.yaml or specify bias.injector_type explicitly."
                    )
            else:
                logger.info(f"Inferred bias injector provider '{injector_provider}' from model '{injector_model_id}'")
                bias_injector_type = injector_provider
                bias_config["injector_type"] = injector_provider
        else:
            # No model specified, use default mock
            bias_injector_type = "mock"
            bias_config["injector_type"] = "mock"
            logger.info("No bias.model specified, using default 'mock' injector")
    
    judge_config = config.get("judge", {})
    judge_allow_fallback_mock = bool(judge_config.get("allow_fallback_mock", False))
    eval_config = config.get("evaluation", {})
    
    # Resolve data path with priority: CLI arg > explicit path > dataset_name from datasets.yaml > default
    data_path = args.data_path
    dataset_name_from_config = data_config.get("dataset_name")
    datasets = datasets_config.get("datasets", datasets_config) if isinstance(datasets_config, dict) else {}
    dataset_info = datasets.get(dataset_name_from_config, {}) if isinstance(datasets, dict) and dataset_name_from_config else {}
    if not data_path:
        # Priority 1: Use explicit path if provided
        data_path = data_config.get("path")
        
        # Priority 2: If no explicit path, resolve from dataset_name via datasets.yaml
        if not data_path:
            if dataset_name_from_config and datasets_config:
                if isinstance(dataset_info, dict):
                    data_path = dataset_info.get("path")
                elif isinstance(dataset_info, str):
                    # Support simple format: dataset_name: "path/to/file.jsonl"
                    data_path = dataset_info
        
        # Priority 3: Fallback to default
        if not data_path:
            data_path = "data/dummy/pairwise.jsonl"
    
    if not os.path.isabs(data_path):
        data_path = str(project_root / data_path)

    # Dataset category is metadata only (does not affect evaluation logic).
    dataset_category = "uncategorized"
    if isinstance(dataset_info, dict):
        dataset_category = str(dataset_info.get("dataset_category", "uncategorized")).strip() or "uncategorized"

    # Load prompt config routed by dataset category.
    prompts_config, prompt_source = load_prompts_for_category(project_root, dataset_category)
    
    # Judge model selection
    # - New simplified format: judge.model (auto-infer provider from models.yaml)
    # - New format (recommended): judge.provider + judge.model_id (models.yaml is the model pool)
    # - Old format (compatible): judge.type + judge.model_name (treat model_name as model_id)
    
    # Support simplified format: just specify "model", auto-infer provider
    if "model" in judge_config and "provider" not in judge_config and "type" not in judge_config:
        judge_model_id = judge_config["model"]
        judge_provider = ModelRegistry.infer_provider_from_model_id(judge_model_id)
        if judge_provider is None:
            # Fallback: try common patterns
            if judge_model_id.startswith("gpt") or judge_model_id.startswith("gpt4") or judge_model_id.startswith("gpt5"):
                judge_provider = "openai"
            elif judge_model_id.startswith("gemini"):
                judge_provider = "gemini"
            elif judge_model_id.startswith("claude"):
                judge_provider = "anthropic"
            else:
                if judge_allow_fallback_mock:
                    logger.warning(
                        f"Could not infer provider for judge model '{judge_model_id}', "
                        "defaulting to 'mock' because allow_fallback_mock=True"
                    )
                    judge_provider = "mock"
                else:
                    raise ValueError(
                        f"Could not infer provider for judge model '{judge_model_id}'. "
                        "Please specify judge.provider explicitly or set judge.allow_fallback_mock: true."
                    )
        logger.info(f"Inferred judge provider '{judge_provider}' from model '{judge_model_id}'")
    else:
        # Use explicit provider/model_id format (backward compatible)
        judge_provider = judge_config.get("provider") or judge_config.get("type") or "mock"
        judge_model_id = judge_config.get("model_id") or judge_config.get("model_name") or judge_config.get("model") or "mock-judge-v1"

    # Per-run overrides (do NOT embed concrete model name strings here)
    judge_model_overrides = {
        k: v
        for k, v in judge_config.items()
        if k not in {"provider", "type", "model_id", "model_name"}
    }

    # Resolve once for logging + directory naming (actual model name may differ from model_id)
    resolved_judge_model_name, resolved_judge_model_config = ModelRegistry.resolve_model_config(
        provider=judge_provider,
        model_id=judge_model_id,
        overrides=judge_model_overrides,
    )

    judge_type = judge_provider
    judge_model_name = judge_model_id  # pass model_id; ModelRegistry will map it to actual model name

    # Build bias injector model config (so bias injection can actually use OpenAI/vLLM)
    injector_model_config: Dict[str, Any] = {}
    injector_system_prompt = None
    injector_user_template = None
    injector_resolved_model_name = None

    # Build judge prompt config (from routed prompt file)
    judge_system_prompt = None
    judge_user_template = None
    judge_prompt_cfg = (prompts_config.get("judge", {}) or {}).get("pairwise", {}) if prompts_config else {}
    if isinstance(judge_prompt_cfg, dict):
        judge_system_prompt = judge_prompt_cfg.get("system")
        judge_user_template = judge_prompt_cfg.get("user")
    
    # Build bias injector model config
    # injector_model_id was already determined above if bias.model was specified
    if bias_injector_type != "mock":
        # Get injector_model_id if not already determined
        if injector_model_id is None:
            injector_model_id = (
                bias_config.get("model_id") 
                or bias_config.get("model_name") 
                or bias_config.get("model") 
                or judge_model_id
            )
        
        injector_provider = bias_injector_type  # Already inferred above or explicitly set

        # Allow passing model overrides for the bias injector from experiment.yaml (e.g., api_key_env, temperature).
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
            # Keep model_id so BiasInjector can pass it into ModelRegistry.create_model
            "model_id": injector_model_id,
            "allow_fallback_mock": bias_allow_fallback_mock,
        }
        injector_model_config.update(injector_model_overrides)
        try:
            injector_resolved_model_name, _ = ModelRegistry.resolve_model_config(
                provider=injector_provider,
                model_id=injector_model_id,
                overrides={"allow_fallback_mock": bias_allow_fallback_mock},
            )
        except Exception:
            injector_resolved_model_name = None

        # Load prompt template for bias injection (required for AI-based injection)
        bias_type_name = bias_config.get("type", "jargon_overloading")
        bias_prompts_root = (prompts_config.get("bias_injection", {}) or {}) if prompts_config else {}
        bias_prompt_key = bias_type_name
        if bias_injection_mode == "word":
            for candidate in (f"{bias_type_name}_word", f"word_{bias_type_name}"):
                if candidate in bias_prompts_root:
                    bias_prompt_key = candidate
                    break
        else:
            for candidate in (f"{bias_type_name}_rewrite",):
                if candidate in bias_prompts_root:
                    bias_prompt_key = candidate
                    break
        bias_prompt_cfg = bias_prompts_root.get(bias_prompt_key, {})
        if not bias_prompt_cfg:
            raise ValueError(
                f"Bias prompt not found for bias type '{bias_type_name}' (mode='{bias_injection_mode}'). "
                "Please add it under bias_injection in the selected prompt config."
            )
        if isinstance(bias_prompt_cfg, dict):
            injector_system_prompt = bias_prompt_cfg.get("system")
            injector_user_template = bias_prompt_cfg.get("user")
        else:
            raise ValueError(
                f"Bias prompt for '{bias_type_name}' must be a mapping with 'system' and 'user' fields."
            )
    
    # Create run directory (before logging setup, so we can log to file)
    base_output_dir = Path(args.output_dir or config.get("experiment", {}).get("output_dir", "output"))
    if not base_output_dir.is_absolute():
        base_output_dir = project_root / base_output_dir
    
    dataset_name = get_dataset_name(
        config=config,
        data_path=data_path,
        datasets_config=datasets_config,
        project_root=project_root,
    )
    bias_name = get_bias_name(bias_config)
    judge_name = get_judge_name(judge_type, judge_model_name, resolved_judge_model_config)

    run_dir = get_run_dir(
        config=config,
        base_output_dir=base_output_dir,
        dataset_name=dataset_name,
        bias_name=bias_name,
        judge_name=judge_name,
    )
    
    # Setup logging (log file goes into run_dir)
    log_file = str(run_dir / "logs.txt")
    setup_logging(log_file=log_file, level="INFO")
    
    # Log configuration details (after logging setup)
    logger.info("="*60)
    logger.info("Judge Configuration Details")
    logger.info("="*60)
    logger.info(f"  Provider/Type: {judge_provider}")
    logger.info(f"  Model ID/Name: {judge_model_id}")
    logger.info(f"  Models Config Available: {bool(models_config)}")
    logger.info(f"  Provider in Models Config: {judge_provider in models_config}")
    if judge_provider in models_config:
        provider_cfg = models_config[judge_provider]
        if isinstance(provider_cfg, dict):
            logger.info(f"  Provider Config Keys: {list(provider_cfg.keys())}")
    logger.info(f"  Bias injector_type: {bias_injector_type} (allow_fallback_mock={bias_allow_fallback_mock})")
    if bias_injector_type != "mock":
        # Log bias injector model information
        bias_model_id = bias_config.get("model_id") or bias_config.get("model_name") or bias_config.get("model") or "N/A"
        logger.info(f"  Bias injector model_id: {bias_model_id}")
        if injector_resolved_model_name:
            logger.info(f"  Bias injector resolved model: {injector_resolved_model_name}")
    else:
        # Validation: warn if bias.model was specified but injector_type is still mock
        if "model" in bias_config:
            logger.warning(f"⚠️  WARNING: bias.model='{bias_config['model']}' was specified but injector_type is 'mock'. This may indicate a configuration issue.")
    if judge_system_prompt or judge_user_template:
        logger.info(f"  Judge prompt source: {prompt_source} (judge.pairwise)")
    logger.info(f"  API Key Env: {resolved_judge_model_config.get('api_key_env', 'N/A')}")
    
    # Check API key availability (without exposing the key)
    api_key_env_name = resolved_judge_model_config.get("api_key_env", "OPENAI_API_KEY")
    api_key_available = bool(os.getenv(api_key_env_name))
    logger.info(f"  API Key Available: {api_key_available}")
    logger.info(f"  Resolved Model Name: {resolved_judge_model_name}")
    logger.info(f"  Model Config Keys: {list(resolved_judge_model_config.keys())}")
    logger.info("="*60)
    
    logger.info("="*60)
    logger.info("JudgeLLM Medical Bias Evaluation")
    logger.info("="*60)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Dataset category: {dataset_category}")
    
    # Save resolved config to run_dir
    import copy
    config_resolved = copy.deepcopy(config)
    if "experiment" not in config_resolved:
        config_resolved["experiment"] = {}
    config_resolved["experiment"]["run_dir"] = str(run_dir)
    config_resolved["experiment"]["dataset_name"] = dataset_name
    config_resolved["experiment"]["dataset_category"] = dataset_category
    config_resolved["experiment"]["bias_name"] = bias_name
    config_resolved["experiment"]["judge_name"] = judge_name
    # Persist normalized bias injector type
    if "bias" not in config_resolved:
        config_resolved["bias"] = {}
    config_resolved["bias"]["injector_type"] = bias_injector_type
    config_resolved["bias"]["allow_fallback_mock"] = bias_allow_fallback_mock
    config_resolved["bias"]["injection_mode"] = bias_injection_mode
    # Persist resolved bias injector model_id (even if user didn't specify it, for reproducibility)
    if bias_injector_type != "mock":
        # Use the injector_model_id that was determined earlier, or fallback to config values
        resolved_bias_model_id = injector_model_id or bias_config.get("model_id") or bias_config.get("model_name") or bias_config.get("model") or judge_model_id
        config_resolved["bias"]["model_id"] = resolved_bias_model_id
    if injector_model_config:
        config_resolved["bias"]["injector_model_config_keys"] = list(injector_model_config.keys())
    save_yaml(config_resolved, str(run_dir / "config_resolved.yaml"))
    logger.info(f"Saved resolved config to {run_dir / 'config_resolved.yaml'}")
    
    # Set seed
    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    
    # Run pipeline based on data type
    data_type = data_config.get("type", "pairwise")
    
    if data_type == "pairwise":
        logger.info("Running pairwise evaluation pipeline")
        results = run_pairwise_evaluation(
            data_path=data_path,
            bias_type=bias_config.get("type", "jargon_overloading"),
            bias_variant_name=bias_name,
            bias_enabled=bias_config.get("enabled", True),
            injector_type=bias_config.get("injector_type", "mock"),
            injector_model_config=injector_model_config,
            injector_system_prompt=injector_system_prompt,
            injector_user_template=injector_user_template,
            injection_mode=bias_injection_mode,
            judge_system_prompt=judge_system_prompt,
            judge_user_template=judge_user_template,
            judge_type=judge_type,
            judge_model_name=judge_model_name,
            judge_config=judge_model_overrides,
            compute_original_acc=eval_config.get("compute_original_acc", True),
            compute_bias_metrics=eval_config.get("compute_bias_metrics", True),
            run_dir=str(run_dir),
            inject_to=bias_config.get("inject_to", "non_gt"),
            use_cache=bias_config.get("use_cache", True),  # Enable cache by default
            semantic_guard=bias_config.get("semantic_guard", {}),
            position_debias_pairwise=eval_config.get("position_debias_pairwise", True),
            request_batch_size=eval_config.get("request_batch_size", 1),
        )
    elif data_type == "scalar":
        logger.info("Running scalar evaluation pipeline (placeholder)")
        results = run_scalar_evaluation(
            data_path=data_path,
            judge_type=judge_type,
            judge_model_name=judge_model_name,
            judge_config=judge_model_overrides,
            run_dir=str(run_dir)
        )
    else:
        logger.error(f"Unknown data type: {data_type}")
        sys.exit(1)

    interrupted = bool(isinstance(results, dict) and results.get("interrupted", False))

    # Append summary records to outputs/results_summary.jsonl and outputs/results_summary.csv
    metrics = results.get("metrics", {}) if isinstance(results, dict) else {}
    summary_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "dataset_category": dataset_category,
        "data_path": data_path,
        "data_type": data_type,
        "bias_enabled": bias_config.get("enabled", True),
        "bias_type": bias_config.get("type", "none"),
        "bias_injection_mode": bias_injection_mode,
        "bias_inject_to": bias_config.get("inject_to", "non_gt"),
        "bias_injector_model_id": injector_model_id if bias_injector_type != "mock" else "mock",
        "judge_model_id": judge_model_id,
        "position_debias_pairwise": eval_config.get("position_debias_pairwise", True),
        "request_batch_size": eval_config.get("request_batch_size", 1),
        "interrupted": interrupted,
        "completed": not interrupted,
        "metrics": {
            "accuracy_original": metrics.get("accuracy_original", {}).get("accuracy"),
            "accuracy_biased": metrics.get("accuracy_biased", {}).get("accuracy_biased"),
            "rr": metrics.get("rr", {}).get("rr"),
            "cr": metrics.get("cr", {}).get("cr"),
        },
    }
    # Quality gate: detect runs where metrics are unreliable due to high error rate.
    # Check invalid_count from original and biased accuracy metrics.
    metrics_unreliable = False
    for metric_key in ("accuracy_original", "accuracy_biased"):
        m = metrics.get(metric_key, {})
        valid_count = m.get("valid_count", m.get("position_A_gt", {}).get("valid_count"))
        invalid_count = m.get("invalid_count", m.get("position_A_gt", {}).get("invalid_count"))
        if valid_count is not None and invalid_count is not None:
            total = valid_count + invalid_count
            if total > 0 and invalid_count / total > 0.5:
                metrics_unreliable = True
                logger.error(
                    f"Quality gate FAILED for {metric_key}: "
                    f"{invalid_count}/{total} judgments are invalid (>{50}%). "
                    "This run will NOT be written to results_summary."
                )

    if interrupted:
        logger.warning(
            "Run interrupted; skip appending results_summary.* to avoid treating partial run as completed."
        )
    elif metrics_unreliable:
        logger.error(
            "Run completed but metrics are unreliable (high error rate). "
            "Skipping results_summary.* to avoid polluting aggregate results. "
            "Check API quota / model availability and rerun."
        )
    else:
        summary_path = base_output_dir / "results_summary.jsonl"
        append_summary_jsonl(summary_record, str(summary_path))
        summary_csv_path = base_output_dir / "results_summary.csv"
        append_summary_csv(summary_record, str(summary_csv_path))

    # Append bias injection report into a global aggregated JSON file.
    if isinstance(results, dict) and results.get("bias_injection_report"):
        aggregated_path = base_output_dir / "bias_injection_report_all.json"
        aggregated_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir),
            "dataset_name": dataset_name,
            "dataset_category": dataset_category,
            "bias_type": bias_config.get("type", "none"),
            "bias_injection_mode": bias_injection_mode,
            "bias_inject_to": bias_config.get("inject_to", "non_gt"),
            "bias_injector_model_id": injector_model_id if bias_injector_type != "mock" else "mock",
            "judge_model_id": judge_model_id,
            "report_path": results.get("bias_injection_report_path"),
            "report": results.get("bias_injection_report"),
        }

        existing_entries = []
        if aggregated_path.exists():
            try:
                with aggregated_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_entries = loaded
                elif isinstance(loaded, dict):
                    existing_entries = [loaded]
            except Exception as e:
                logger.warning(f"Failed to read existing aggregated bias report, recreating file: {e}")

        existing_entries.append(aggregated_entry)
        with aggregated_path.open("w", encoding="utf-8") as f:
            json.dump(existing_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"Aggregated bias injection report updated: {aggregated_path}")
    
    logger.info("Evaluation completed successfully")
    if isinstance(results, dict) and results.get("interrupted", False):
        # When running under YAML batch runner, propagate interruption to stop the whole batch.
        if os.getenv("JUDGELLM_ABORT_BATCH_ON_INTERRUPT", "0") == "1":
            raise KeyboardInterrupt("Sub-run interrupted; aborting remaining batch runs.")
    return results


if __name__ == "__main__":
    main()

