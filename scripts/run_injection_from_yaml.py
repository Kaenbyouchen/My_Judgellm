#!/usr/bin/env python
"""
Run bias injection only (no judging/evaluation) from YAML config.

Usage:
  python scripts/run_injection_from_yaml.py --batch-config configs/batch_eval.yaml
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bias.cache import apply_bias_to_samples, check_cache_exists, get_cache_path, save_bias_dataset
from src.bias.injector import BiasInjector
from src.dataset.loaders import load_pairwise_jsonl
from src.dataset.preprocess import validate_pairwise_samples
from src.models.registry import ModelRegistry
from src.utils.io import load_yaml
from src.utils.prompt_config import load_prompts_for_category


def _setup_log_dir(log_dir_cfg: Any) -> Path:
    if isinstance(log_dir_cfg, str) and log_dir_cfg.strip():
        log_dir = Path(log_dir_cfg)
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / "outputs" / "injection_logs" / f"yaml_inject_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in {"rewrite", "word"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use rewrite/word.")
    return m


def _bias_base_name(key: str) -> str:
    if key.endswith("_rewrite"):
        return key[: -len("_rewrite")]
    if key.endswith("_word"):
        return key[: -len("_word")]
    if key.startswith("word_"):
        return key[len("word_") :]
    return key


def _available_bias_bases_from_prompts(prompts_cfg: Dict[str, Any]) -> Set[str]:
    bias_root = (prompts_cfg.get("bias_injection", {}) or {}) if isinstance(prompts_cfg, dict) else {}
    bases: Set[str] = set()
    for k, v in bias_root.items():
        if isinstance(v, dict):
            base = _bias_base_name(str(k))
            if base:
                bases.add(base)
    return bases


def _mode_supported(prompts_cfg: Dict[str, Any], bias_base: str, mode: str) -> bool:
    bias_root = (prompts_cfg.get("bias_injection", {}) or {}) if isinstance(prompts_cfg, dict) else {}
    if mode == "rewrite":
        return (f"{bias_base}_rewrite" in bias_root) or (bias_base in bias_root)
    return (f"{bias_base}_word" in bias_root) or (f"word_{bias_base}" in bias_root)


def _normalize_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        v = raw.strip()
        return [v] if v else []
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            v = str(x).strip()
            if v:
                out.append(v)
        return out
    return []


def _select_bias_prompt(prompts_cfg: Dict[str, Any], bias_type: str, mode: str) -> Tuple[str, Dict[str, Any]]:
    bias_root = (prompts_cfg.get("bias_injection", {}) or {}) if isinstance(prompts_cfg, dict) else {}
    if mode == "word":
        for key in (f"{bias_type}_word", f"word_{bias_type}"):
            if key in bias_root and isinstance(bias_root[key], dict):
                return key, bias_root[key]
    else:
        for key in (f"{bias_type}_rewrite", bias_type):
            if key in bias_root and isinstance(bias_root[key], dict):
                return key, bias_root[key]
    raise KeyError(f"Bias prompt not found: bias={bias_type}, mode={mode}")


def _load_base_experiment_bias_cfg(batch_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_exp_rel = str(batch_cfg.get("base_experiment_config", "configs/experiment.yaml")).strip()
    if not base_exp_rel:
        return {}
    base_exp_path = Path(base_exp_rel)
    if not base_exp_path.is_absolute():
        base_exp_path = PROJECT_ROOT / base_exp_path
    if not base_exp_path.exists():
        logger.warning(f"base_experiment_config not found: {base_exp_path}")
        return {}
    base_exp_yaml = load_yaml(str(base_exp_path))
    bias_cfg = base_exp_yaml.get("bias", {}) if isinstance(base_exp_yaml, dict) else {}
    return bias_cfg if isinstance(bias_cfg, dict) else {}


def _resolve_base_output_dir(batch_cfg: Dict[str, Any]) -> Path:
    base_exp_rel = str(batch_cfg.get("base_experiment_config", "configs/experiment.yaml")).strip()
    base_output_dir = PROJECT_ROOT / "outputs"
    if not base_exp_rel:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir

    base_exp_path = Path(base_exp_rel)
    if not base_exp_path.is_absolute():
        base_exp_path = PROJECT_ROOT / base_exp_path
    if base_exp_path.exists():
        base_exp_yaml = load_yaml(str(base_exp_path))
        exp_cfg = base_exp_yaml.get("experiment", {}) if isinstance(base_exp_yaml, dict) else {}
        out_cfg = exp_cfg.get("output_dir") if isinstance(exp_cfg, dict) else None
        if isinstance(out_cfg, str) and out_cfg.strip():
            out_path = Path(out_cfg.strip())
            base_output_dir = out_path if out_path.is_absolute() else (PROJECT_ROOT / out_path)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir


def _append_global_injection_report(
    base_output_dir: Path,
    entries: List[Dict[str, Any]],
) -> Path:
    aggregated_path = base_output_dir / "bias_injection_report_all.json"
    existing_entries: List[Dict[str, Any]] = []
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

    def _entry_key(x: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(x.get("dataset_name", "")),
            str(x.get("bias_type", "")),
            str(x.get("bias_injection_mode", "")),
            str(x.get("bias_injector_model_id", "")),
        )

    merged_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for x in existing_entries:
        if isinstance(x, dict):
            merged_map[_entry_key(x)] = x
    for x in entries:
        merged_map[_entry_key(x)] = x

    merged_entries = list(merged_map.values())
    with aggregated_path.open("w", encoding="utf-8") as f:
        json.dump(merged_entries, f, ensure_ascii=False, indent=2)
    return aggregated_path


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_existing_compact_entry(base_output_dir: Path, key_tuple: Tuple[str, str, str, str]) -> Dict[str, Any]:
    aggregated_path = base_output_dir / "bias_injection_report_all.json"
    if not aggregated_path.exists():
        return {}
    try:
        with aggregated_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        rows = loaded if isinstance(loaded, list) else ([loaded] if isinstance(loaded, dict) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_key = (
                str(row.get("dataset_name", "")),
                str(row.get("bias_type", "")),
                str(row.get("bias_injection_mode", "")),
                str(row.get("bias_injector_model_id", "")),
            )
            if row_key == key_tuple:
                return row
    except Exception:
        return {}
    return {}


def _build_compact_aggregated_entry(
    base_output_dir: Path,
    dataset_name: str,
    dataset_category: str,
    data_path: str,
    result: Dict[str, Any],
    summary_path: Path,
) -> Dict[str, Any]:
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    key_tuple = (
        str(dataset_name),
        str(result.get("bias_type", "")),
        str(result.get("mode", "")),
        str(result.get("injector_model_id", "")),
    )
    previous = _load_existing_compact_entry(base_output_dir, key_tuple)

    total_data_count = report.get("total_samples")
    if not isinstance(total_data_count, int):
        eff = report.get("injection_effective_count")
        unchg = report.get("injection_unchanged_count")
        if isinstance(eff, int) and isinstance(unchg, int):
            total_data_count = eff + unchg
        else:
            cache_path = Path(str(result.get("cache_path", "")).strip()) if result.get("cache_path") else None
            if cache_path and cache_path.exists():
                total_data_count = _count_jsonl_records(cache_path)
            elif isinstance(previous.get("total_data_count"), int):
                total_data_count = int(previous.get("total_data_count"))
            else:
                total_data_count = None

    guard_failed_count = report.get("guard_failed_count")
    semantic_pass_count = None
    if isinstance(total_data_count, int):
        if isinstance(guard_failed_count, int):
            semantic_pass_count = max(0, total_data_count - guard_failed_count)
        elif result.get("status") == "skipped_existing":
            if isinstance(previous.get("semantic_pass_count"), int):
                semantic_pass_count = int(previous.get("semantic_pass_count"))
            else:
                semantic_pass_count = None
        else:
            semantic_pass_count = total_data_count

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "dataset_category": dataset_category,
        "data_path": data_path,
        "bias_type": result.get("bias_type"),
        "bias_injection_mode": result.get("mode"),
        "bias_injector_model_id": result.get("injector_model_id"),
        "status": result.get("status"),
        "total_data_count": total_data_count,
        "semantic_pass_count": semantic_pass_count,
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run injection-only from YAML")
    parser.add_argument(
        "--batch-config",
        type=str,
        default="configs/batch_eval.yaml",
        help="Path to batch yaml config",
    )
    args = parser.parse_args()

    batch_cfg_path = Path(args.batch_config)
    if not batch_cfg_path.is_absolute():
        batch_cfg_path = PROJECT_ROOT / batch_cfg_path
    if not batch_cfg_path.exists():
        raise FileNotFoundError(f"Batch config not found: {batch_cfg_path}")

    batch_yaml = load_yaml(str(batch_cfg_path))
    batch_cfg = batch_yaml.get("batch", {})
    if not isinstance(batch_cfg, dict):
        raise ValueError("Invalid batch config: missing 'batch' mapping")

    # Logging level controls (clean terminal by default).
    # - console_log_level: default WARNING (show errors, hide INFO noise)
    # - file_log_level: default INFO (keep detailed run logs in file)
    console_log_level = str(batch_cfg.get("console_log_level", "WARNING")).strip().upper() or "WARNING"
    file_log_level = str(batch_cfg.get("file_log_level", "INFO")).strip().upper() or "INFO"

    # Load model pool for provider inference/resolution.
    models_yaml_path = PROJECT_ROOT / "configs" / "models.yaml"
    models_config = load_yaml(str(models_yaml_path)) if models_yaml_path.exists() else {}
    ModelRegistry.set_models_config(models_config)

    # Resolve dataset.
    dataset_name = str(batch_cfg.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError("batch.dataset_name is required")
    datasets_cfg = load_yaml(str(PROJECT_ROOT / "configs" / "datasets.yaml"))
    datasets_map = datasets_cfg.get("datasets", datasets_cfg) if isinstance(datasets_cfg, dict) else {}
    dataset_info = datasets_map.get(dataset_name) if isinstance(datasets_map, dict) else None
    if not isinstance(dataset_info, dict):
        raise ValueError(f"Unknown dataset_name '{dataset_name}' in configs/datasets.yaml")
    data_path = str(dataset_info.get("path", "")).strip()
    if not data_path:
        raise ValueError(f"Dataset '{dataset_name}' has no path")
    if not Path(data_path).is_absolute():
        data_path = str((PROJECT_ROOT / data_path).resolve())
    dataset_category = str(dataset_info.get("dataset_category", "uncategorized"))

    # Injection controls.
    injector_models = _normalize_str_list(batch_cfg.get("bias_injector_models"))
    if not injector_models:
        single_injector = str(batch_cfg.get("bias_injector_model", "")).strip()
        if single_injector:
            injector_models = [single_injector]
    if not injector_models:
        raise ValueError("batch.bias_injector_models (or bias_injector_model) is required for injection-only run")

    modes = [_normalize_mode(m) for m in (batch_cfg.get("modes", ["rewrite", "word"]) or ["rewrite", "word"])]
    use_all_biases = bool(batch_cfg.get("use_all_biases", True))
    bias_list_cfg = batch_cfg.get("eval_biases", None)
    if bias_list_cfg is None:
        bias_list_cfg = batch_cfg.get("bias_list", []) or []
    exclude_biases = {str(x).strip() for x in (batch_cfg.get("exclude_biases", []) or []) if str(x).strip()}
    inject_to = str(batch_cfg.get("inject_to", "non_gt")).strip().lower()
    dry_run = bool(batch_cfg.get("dry_run", False))
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))
    base_bias_cfg = _load_base_experiment_bias_cfg(batch_cfg)
    batch_guard = batch_cfg.get("semantic_guard", None)
    if isinstance(batch_guard, dict) and batch_guard:
        semantic_guard = batch_guard
        semantic_guard_source = "batch.semantic_guard"
    else:
        semantic_guard = base_bias_cfg.get("semantic_guard", {}) if isinstance(base_bias_cfg, dict) else {}
        semantic_guard_source = "base_experiment_config.bias.semantic_guard"

    prompts_cfg, prompt_source = load_prompts_for_category(PROJECT_ROOT, dataset_category)
    available_bases = _available_bias_bases_from_prompts(prompts_cfg)
    if use_all_biases:
        selected_biases = sorted([b for b in available_bases if b not in exclude_biases])
    else:
        selected_biases = [str(x).strip() for x in bias_list_cfg if str(x).strip()]
        missing = [b for b in selected_biases if b not in available_bases]
        if missing:
            raise ValueError(
                f"These biases are not found for category '{dataset_category}' in {prompt_source}: {missing}"
            )
        selected_biases = [b for b in selected_biases if b not in exclude_biases]
    if not selected_biases:
        raise ValueError("No biases selected to run.")

    combos: List[Tuple[str, str]] = []
    for b in selected_biases:
        for m in modes:
            if _mode_supported(prompts_cfg, b, m):
                combos.append((b, m))
            else:
                logger.warning(f"Skip unsupported combo: bias={b}, mode={m}")
    if not combos:
        raise ValueError("No runnable bias/mode combinations after prompt support filtering.")

    # Logging.
    log_dir = _setup_log_dir(batch_cfg.get("log_dir"))
    logger.remove()
    logger.add(sys.stderr, level=console_log_level)
    logger.add(log_dir / "yaml_injection.log", level=file_log_level)
    logger.info("=" * 72)
    logger.info("YAML injection-only start")
    logger.info(f"Batch config: {batch_cfg_path}")
    logger.info(f"Dataset: {dataset_name} ({dataset_category})")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Injector models: {injector_models}")
    logger.info(f"Prompt source: {prompt_source}")
    logger.info(f"Selected biases: {selected_biases}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Inject target: {inject_to}")
    logger.info(f"Semantic guard source: {semantic_guard_source}")
    logger.info(f"Semantic guard: {semantic_guard}")
    logger.info("=" * 72)

    # Load source samples once.
    samples = validate_pairwise_samples(load_pairwise_jsonl(data_path))
    logger.info(f"Loaded source samples: {len(samples)}")

    run_plan: List[Tuple[str, str, str, str]] = []
    for injector_model_id in injector_models:
        injector_provider = ModelRegistry.infer_provider_from_model_id(injector_model_id)
        if injector_provider is None:
            raise ValueError(
                f"Cannot infer provider for injector model '{injector_model_id}'. "
                "Please add it into configs/models.yaml."
            )
        for bias_type, mode in combos:
            run_plan.append((injector_model_id, injector_provider, bias_type, mode))

    results: List[Dict[str, Any]] = []
    total = len(run_plan)
    stop_early = False

    for idx, (injector_model_id, injector_provider, bias_type, mode) in enumerate(
        tqdm(run_plan, desc="Injection jobs", unit="job"),
        start=1,
    ):
        if stop_early:
            break

        try:
            record: Dict[str, Any] = {
                "index": idx,
                "total": total,
                "dataset": dataset_name,
                "injector_model_id": injector_model_id,
                "injector_provider": injector_provider,
                "bias_type": bias_type,
                "mode": mode,
                "inject_to": inject_to,
                "status": "pending",
                "error": None,
            }
            logger.info(f"[{idx}/{total}] inject model={injector_model_id} bias={bias_type} mode={mode}")

            prompt_key, prompt_cfg = _select_bias_prompt(prompts_cfg, bias_type, mode)
            system_prompt = prompt_cfg.get("system")
            user_template = prompt_cfg.get("user")

            # Keep model_id only; let BiasInjector -> ModelRegistry resolve to concrete model name once.
            # Passing resolved "model_name" here can trigger a second lookup using provider pool keys.
            merged_cfg = {
                "model_id": injector_model_id,
                "allow_fallback_mock": False,
            }

            bias_injector = BiasInjector(
                bias_type=bias_type,
                injector_type=injector_provider,
                model_config=merged_cfg,
                prompt_template=user_template,
                system_prompt=system_prompt,
                injection_mode=mode,
            )

            if dry_run:
                record["status"] = "dry_run"
            else:
                cache_model_name = injector_model_id.lower().replace("-", "").replace("_", "")
                bias_variant_name = f"word_{bias_type}" if mode == "word" else bias_type
                cache_path = get_cache_path(
                    dataset_path=data_path,
                    bias_type=bias_variant_name,
                    model_name=cache_model_name,
                )

                if check_cache_exists(
                    dataset_path=data_path,
                    bias_type=bias_variant_name,
                    model_name=cache_model_name,
                ):
                    record["status"] = "skipped_existing"
                    record["cache_path"] = str(cache_path)
                    record["report"] = {"skipped_reason": "cache_exists"}
                    results.append(record)
                    continue

                biased_samples, report = apply_bias_to_samples(
                    samples=samples,
                    bias_injector=bias_injector,
                    inject_to=inject_to,
                    semantic_guard=semantic_guard,
                )

                metadata = {
                    "created_at": datetime.datetime.now().isoformat(),
                    "injector_config": {
                        "injector_type": injector_provider,
                        "model_config": {"model_id": injector_model_id},
                    },
                    "prompt_config": {
                        "prompt_key": prompt_key,
                        "system_prompt": system_prompt,
                        "user_template": user_template,
                    },
                    "injection_mode": mode,
                    "semantic_guard": semantic_guard,
                }
                cache_path = save_bias_dataset(
                    samples=biased_samples,
                    dataset_path=data_path,
                    bias_type=bias_variant_name,
                    model_name=cache_model_name,
                    metadata=metadata,
                )
                record["status"] = "success"
                record["cache_path"] = str(cache_path)
                record["report"] = report
        except Exception as e:
            record["status"] = "failed"
            record["error"] = str(e)
            record["traceback"] = traceback.format_exc()
            logger.error(f"Injection failed: {e}")
            if not continue_on_error:
                stop_early = True
        results.append(record)

    summary = {
        "batch_config": str(batch_cfg_path),
        "dataset": dataset_name,
        "data_path": data_path,
        "dataset_category": dataset_category,
        "injector_models": injector_models,
        "combos": [{"bias_type": b, "mode": m} for b, m in combos],
        "total_planned": total,
        "executed": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "dry_run": sum(1 for r in results if r["status"] == "dry_run"),
        "skipped_existing": sum(1 for r in results if r["status"] == "skipped_existing"),
        "results": results,
    }
    summary_path = log_dir / "yaml_injection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Append per-job injection reports into a global aggregated JSON under outputs.
    # This keeps a single cross-run ledger (requested by user).
    base_output_dir = _resolve_base_output_dir(batch_cfg)
    aggregated_entries: List[Dict[str, Any]] = []
    for r in results:
        aggregated_entries.append(
            _build_compact_aggregated_entry(
                base_output_dir=base_output_dir,
                dataset_name=dataset_name,
                dataset_category=dataset_category,
                data_path=data_path,
                result=r,
                summary_path=summary_path,
            )
        )
    aggregated_path = _append_global_injection_report(base_output_dir, aggregated_entries)

    logger.info("=" * 72)
    logger.info(
        "Injection-only done. "
        f"success={summary['success']}, failed={summary['failed']}, "
        f"dry_run={summary['dry_run']}, skipped_existing={summary['skipped_existing']}"
    )
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Aggregated report: {aggregated_path}")
    logger.info("=" * 72)

    if summary["failed"] > 0 and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

