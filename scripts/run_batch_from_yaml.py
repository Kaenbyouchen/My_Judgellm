#!/usr/bin/env python
"""
Run batch evaluation from a single YAML config.

This runner supports:
- pick one dataset
- pick one or multiple judge models
- run all biases (or specified bias list) for that dataset category
- run chosen injection modes (rewrite/word)
- pick one or multiple bias injector models
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main as run_single_experiment
from src.utils.io import load_yaml
from src.utils.prompt_config import load_prompts_for_category

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Interrupt received. Current run will finish, then stop.")


def _setup_log_dir(log_dir_cfg: Any) -> Path:
    if isinstance(log_dir_cfg, str) and log_dir_cfg.strip():
        log_dir = Path(log_dir_cfg)
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / "outputs" / "batch_logs" / f"yaml_batch_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in {"rewrite", "word"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use rewrite/word.")
    return m


def _bias_base_name(key: str) -> str:
    """
    Convert prompt bias key to base bias name.
    Supports:
    - xxx_rewrite
    - xxx_word
    - word_xxx
    - xxx
    """
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
        if not isinstance(v, dict):
            continue
        base = _bias_base_name(str(k))
        if base:
            bases.add(base)
    return bases


def _mode_supported(prompts_cfg: Dict[str, Any], bias_base: str, mode: str) -> bool:
    bias_root = (prompts_cfg.get("bias_injection", {}) or {}) if isinstance(prompts_cfg, dict) else {}
    if mode == "rewrite":
        return (f"{bias_base}_rewrite" in bias_root) or (bias_base in bias_root)
    # word mode
    return (f"{bias_base}_word" in bias_root) or (f"word_{bias_base}" in bias_root)


def _build_run_config(
    base_cfg: Dict[str, Any],
    dataset_name: str,
    judge_model: str,
    injector_model: str,
    bias: str,
    mode: str,
    compute_original_acc: bool,
    semantic_guard_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)

    cfg.setdefault("data", {})
    cfg["data"]["dataset_name"] = dataset_name

    cfg.setdefault("judge", {})
    for k in ("provider", "type", "model_id", "model_name"):
        cfg["judge"].pop(k, None)
    cfg["judge"]["model"] = judge_model

    cfg.setdefault("bias", {})
    cfg["bias"]["enabled"] = True
    # Alias: "clinical_note" is treated as "clinical_formatting" internally.
    resolved_bias = "clinical_formatting" if bias == "clinical_note" else bias
    cfg["bias"]["type"] = resolved_bias
    cfg["bias"]["injection_mode"] = mode
    cfg["bias"]["model"] = injector_model
    for k in ("injector_type", "model_id", "model_name"):
        cfg["bias"].pop(k, None)

    # Avoid repeated original-judgment calls across many bias runs.
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["compute_original_acc"] = compute_original_acc
    cfg["evaluation"]["compute_bias_metrics"] = True

    # Pairwise position-bias correction: evaluate GT@A and GT@B, then average RR/CR/acc_biased.
    cfg["evaluation"]["position_debias_pairwise"] = True

    # Optional semantic guard override from dataset-bias plan.
    if isinstance(semantic_guard_override, dict) and semantic_guard_override:
        existing_guard = cfg["bias"].get("semantic_guard", {})
        if not isinstance(existing_guard, dict):
            existing_guard = {}
        merged_guard = deepcopy(existing_guard)
        # shallow merge is enough for current use; nested dicts merged one level deep
        for k, v in semantic_guard_override.items():
            if isinstance(v, dict) and isinstance(merged_guard.get(k), dict):
                merged_guard[k] = {**merged_guard[k], **v}
            else:
                merged_guard[k] = v
        cfg["bias"]["semantic_guard"] = merged_guard

    return cfg


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(obj, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


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


def _resolve_dataset_names(batch_cfg: Dict[str, Any], datasets_map: Dict[str, Any]) -> List[str]:
    dataset_names = _normalize_str_list(batch_cfg.get("dataset_names"))
    if not dataset_names:
        single_dataset = str(batch_cfg.get("dataset_name", "")).strip()
        if single_dataset:
            dataset_names = [single_dataset]
    if not dataset_names:
        raise ValueError("batch.dataset_names (or batch.dataset_name) is required")

    if len(dataset_names) == 1 and dataset_names[0].lower() == "all":
        all_names = sorted(list(datasets_map.keys())) if isinstance(datasets_map, dict) else []
        if not all_names:
            raise ValueError("No datasets found in configs/datasets.yaml")
        return all_names
    return dataset_names


def _load_dataset_bias_plan(plan_path: Path) -> Dict[str, Any]:
    if not plan_path.exists():
        raise FileNotFoundError(f"dataset_bias_plan not found: {plan_path}")
    plan = load_yaml(str(plan_path))
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid dataset_bias_plan format: {plan_path}")
    return plan


def _resolve_dataset_bias_modes_from_plan(
    dataset_name: str,
    plan_cfg: Dict[str, Any],
    fallback_modes: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    ds_map = plan_cfg.get("dataset_biases", {})
    if not isinstance(ds_map, dict):
        ds_map = {}
    global_cfg = plan_cfg.get("global", {})
    if not isinstance(global_cfg, dict):
        global_cfg = {}
    ds_entry = ds_map.get(dataset_name, {})
    if isinstance(ds_entry, list):
        ds_entry = {"biases": ds_entry}
    if not isinstance(ds_entry, dict):
        ds_entry = {}

    biases = _normalize_str_list(ds_entry.get("biases"))
    if not biases:
        biases = _normalize_str_list(global_cfg.get("default_biases"))
    if not biases:
        raise ValueError(
            f"No biases configured for dataset '{dataset_name}' in dataset_bias_plan and no global.default_biases."
        )

    modes = _normalize_str_list(ds_entry.get("modes"))
    if not modes:
        modes = _normalize_str_list(global_cfg.get("default_modes"))
    if not modes and fallback_modes:
        modes = list(fallback_modes)
    if not modes:
        modes = ["rewrite", "word"]
    modes = [_normalize_mode(m) for m in modes]

    guard_overrides = ds_entry.get("semantic_guard_overrides", {})
    if not isinstance(guard_overrides, dict):
        guard_overrides = {}
    return biases, modes, guard_overrides


def main():
    parser = argparse.ArgumentParser(description="Run batch eval from YAML")
    parser.add_argument(
        "--batch-config",
        type=str,
        default="configs/batch_eval.yaml",
        help="Path to batch yaml config",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    batch_cfg_path = Path(args.batch_config)
    if not batch_cfg_path.is_absolute():
        batch_cfg_path = PROJECT_ROOT / batch_cfg_path
    if not batch_cfg_path.exists():
        raise FileNotFoundError(f"Batch config not found: {batch_cfg_path}")

    batch_yaml = load_yaml(str(batch_cfg_path))
    batch_cfg = batch_yaml.get("batch", {})
    if not isinstance(batch_cfg, dict):
        raise ValueError("Invalid batch config: missing 'batch' mapping")

    base_config_path = Path(batch_cfg.get("base_experiment_config", "configs/experiment.yaml"))
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base experiment config not found: {base_config_path}")

    judge_models = _normalize_str_list(batch_cfg.get("judge_models"))
    # Backward compatibility: single judge_model
    if not judge_models:
        single_judge = str(batch_cfg.get("judge_model", "")).strip()
        if single_judge:
            judge_models = [single_judge]
    if not judge_models:
        raise ValueError("batch.judge_models (or batch.judge_model) is required")

    injector_models = _normalize_str_list(batch_cfg.get("bias_injector_models"))
    # Backward compatibility: single bias_injector_model
    if not injector_models:
        single_injector = str(batch_cfg.get("bias_injector_model", "")).strip()
        if single_injector:
            injector_models = [single_injector]
    # Default: follow each judge model
    if not injector_models:
        injector_models = ["same_as_judge"]
    use_all_biases = bool(batch_cfg.get("use_all_biases", True))
    # Preferred key: eval_biases; backward compatible key: bias_list
    bias_list_cfg = batch_cfg.get("eval_biases", None)
    if bias_list_cfg is None:
        bias_list_cfg = batch_cfg.get("bias_list", []) or []
    exclude_biases = {str(x).strip() for x in (batch_cfg.get("exclude_biases", []) or []) if str(x).strip()}
    modes = [_normalize_mode(m) for m in (batch_cfg.get("modes", ["rewrite", "word"]) or ["rewrite", "word"])]
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))
    dry_run = bool(batch_cfg.get("dry_run", False))
    reuse_original_per_judge = bool(batch_cfg.get("reuse_original_per_judge", True))

    base_cfg = load_yaml(str(base_config_path))
    datasets_cfg = load_yaml(str(PROJECT_ROOT / "configs" / "datasets.yaml"))
    datasets_map = datasets_cfg.get("datasets", datasets_cfg) if isinstance(datasets_cfg, dict) else {}
    if not isinstance(datasets_map, dict):
        raise ValueError("Invalid configs/datasets.yaml format.")
    dataset_names = _resolve_dataset_names(batch_cfg, datasets_map)
    dataset_bias_plan_path_cfg = str(batch_cfg.get("dataset_bias_plan", "")).strip()
    dataset_bias_plan_cfg: Dict[str, Any] = {}
    if dataset_bias_plan_path_cfg:
        dataset_bias_plan_path = Path(dataset_bias_plan_path_cfg)
        if not dataset_bias_plan_path.is_absolute():
            dataset_bias_plan_path = PROJECT_ROOT / dataset_bias_plan_path
        dataset_bias_plan_cfg = _load_dataset_bias_plan(dataset_bias_plan_path)

    dataset_run_items: List[Dict[str, Any]] = []
    for ds_name in dataset_names:
        dataset_info = datasets_map.get(ds_name)
        if not isinstance(dataset_info, dict):
            raise ValueError(f"Unknown dataset_name '{ds_name}' in configs/datasets.yaml")
        dataset_category = str(dataset_info.get("dataset_category", "uncategorized"))
        prompts_cfg, prompt_source = load_prompts_for_category(PROJECT_ROOT, dataset_category)
        available_bases = _available_bias_bases_from_prompts(prompts_cfg)

        if dataset_bias_plan_cfg:
            selected_biases, ds_modes, guard_overrides = _resolve_dataset_bias_modes_from_plan(
                ds_name,
                dataset_bias_plan_cfg,
                fallback_modes=modes,
            )
            selected_biases = [b for b in selected_biases if b not in exclude_biases]
        else:
            ds_modes = modes
            guard_overrides = {}
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
            logger.warning(f"No biases selected for dataset={ds_name}; skip.")
            continue

        combos: List[Tuple[str, str]] = []
        for b in selected_biases:
            # Alias support: clinical_note -> clinical_formatting prompt key lookup
            probe_bias = "clinical_formatting" if b == "clinical_note" else b
            for m in ds_modes:
                if _mode_supported(prompts_cfg, probe_bias, m):
                    combos.append((b, m))
                else:
                    logger.warning(f"Skip unsupported combo: dataset={ds_name}, bias={b}, mode={m}")
        if not combos:
            logger.warning(f"No runnable combos for dataset={ds_name}; skip.")
            continue

        dataset_run_items.append(
            {
                "dataset_name": ds_name,
                "dataset_category": dataset_category,
                "prompt_source": prompt_source,
                "selected_biases": selected_biases,
                "combos": combos,
                "guard_overrides": guard_overrides,
            }
        )

    if not dataset_run_items:
        raise ValueError("No runnable datasets/biases found from current batch config.")

    log_dir = _setup_log_dir(batch_cfg.get("log_dir"))
    run_cfg_dir = log_dir / "generated_configs"
    run_cfg_dir.mkdir(parents=True, exist_ok=True)

    logger.add(log_dir / "yaml_batch.log", level="INFO")
    logger.info("=" * 72)
    logger.info("YAML batch evaluation start")
    logger.info(f"Batch config: {batch_cfg_path}")
    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Datasets: {[x['dataset_name'] for x in dataset_run_items]}")
    logger.info(f"Judge models: {judge_models}")
    logger.info(f"Bias injector models: {injector_models}")
    for item in dataset_run_items:
        logger.info(
            f"Dataset={item['dataset_name']} category={item['dataset_category']} "
            f"biases={item['selected_biases']} modes={sorted(list(set([m for _, m in item['combos']])))}"
        )
    logger.info(f"Reuse original per judge: {reuse_original_per_judge}")
    total_combo_count = sum(
        len(judge_models) * len(injector_models) * len(item["combos"])
        for item in dataset_run_items
    )
    logger.info(f"Total planned runs: {total_combo_count}")
    logger.info(f"Log dir: {log_dir}")
    logger.info("=" * 72)

    results: List[Dict[str, Any]] = []
    run_index = 0
    stop_now = False
    original_done_for_judge_dataset: Set[Tuple[str, str]] = set()
    for judge_model in judge_models:
        for injector_spec in injector_models:
            injector_model = judge_model if injector_spec == "same_as_judge" else injector_spec
            for ds_item in dataset_run_items:
                ds_name = ds_item["dataset_name"]
                ds_category = ds_item["dataset_category"]
                guard_overrides_map = ds_item.get("guard_overrides", {})
                if not isinstance(guard_overrides_map, dict):
                    guard_overrides_map = {}
                for bias, mode in ds_item["combos"]:
                    if _shutdown_requested:
                        logger.warning("Stop requested; exiting loop.")
                        stop_now = True
                        break

                    run_index += 1
                    run_tag = f"{run_index:03d}_{ds_name}_{judge_model}_{injector_model}_{bias}_{mode}"
                    logger.info(f"[{run_index}/{total_combo_count}] {run_tag}")

                    result: Dict[str, Any] = {
                        "index": run_index,
                        "dataset": ds_name,
                        "dataset_category": ds_category,
                        "judge_model": judge_model,
                        "injector_model": injector_model,
                        "bias": bias,
                        "mode": mode,
                        "status": "pending",
                        "error": None,
                        "config_path": None,
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                    }

                    try:
                        if reuse_original_per_judge:
                            compute_original_acc = (judge_model, ds_name) not in original_done_for_judge_dataset
                        else:
                            compute_original_acc = True

                        semantic_guard_override = guard_overrides_map.get(f"{bias}:{mode}") or guard_overrides_map.get(bias)

                        # Built-in relax rule requested by user: clinical_formatting word mode.
                        if bias in {"clinical_note", "clinical_formatting"} and mode == "word":
                            if not isinstance(semantic_guard_override, dict):
                                semantic_guard_override = {}
                            pre_cfg = semantic_guard_override.get("precheck", {})
                            if not isinstance(pre_cfg, dict):
                                pre_cfg = {}
                            pre_cfg.setdefault("len_ratio_min", 0.30)
                            pre_cfg.setdefault("len_ratio_max", 3.50)
                            semantic_guard_override["precheck"] = pre_cfg

                        run_cfg = _build_run_config(
                            base_cfg,
                            ds_name,
                            judge_model,
                            injector_model,
                            bias,
                            mode,
                            compute_original_acc=compute_original_acc,
                            semantic_guard_override=semantic_guard_override,
                        )
                        run_cfg_path = run_cfg_dir / f"{run_tag}.yaml"
                        _write_yaml(run_cfg_path, run_cfg)
                        result["config_path"] = str(run_cfg_path)
                        result["compute_original_acc"] = compute_original_acc

                        if dry_run:
                            result["status"] = "dry_run"
                        else:
                            original_argv = sys.argv.copy()
                            prev_batch_flag = os.environ.get("JUDGELLM_ABORT_BATCH_ON_INTERRUPT")
                            try:
                                # In batch mode, force sub-run interrupt to bubble up immediately.
                                os.environ["JUDGELLM_ABORT_BATCH_ON_INTERRUPT"] = "1"
                                sys.argv = ["run_batch_from_yaml.py", "--config", str(run_cfg_path)]
                                run_output = run_single_experiment()
                            finally:
                                if prev_batch_flag is None:
                                    os.environ.pop("JUDGELLM_ABORT_BATCH_ON_INTERRUPT", None)
                                else:
                                    os.environ["JUDGELLM_ABORT_BATCH_ON_INTERRUPT"] = prev_batch_flag
                                sys.argv = original_argv
                            if isinstance(run_output, dict) and run_output.get("interrupted", False):
                                result["status"] = "interrupted"
                                logger.warning(f"Interrupted: {run_tag}. Stopping remaining batch runs.")
                                stop_now = True
                            elif _shutdown_requested:
                                result["status"] = "interrupted"
                                logger.warning(f"Batch interrupt flag detected: {run_tag}. Stopping remaining batch runs.")
                                stop_now = True
                            else:
                                result["status"] = "success"
                                if compute_original_acc:
                                    original_done_for_judge_dataset.add((judge_model, ds_name))
                    except KeyboardInterrupt:
                        result["status"] = "interrupted"
                        result["error"] = "KeyboardInterrupt"
                        logger.warning(f"KeyboardInterrupt: {run_tag}. Stopping remaining batch runs.")
                        stop_now = True
                    except Exception as e:
                        result["status"] = "failed"
                        result["error"] = str(e)
                        result["traceback"] = traceback.format_exc()
                        logger.error(f"Failed: {run_tag} | {e}")
                        if not continue_on_error:
                            results.append(result)
                            stop_now = True
                            break
                    finally:
                        # Sub-runs may overwrite signal handlers; restore batch-level handlers each iteration.
                        signal.signal(signal.SIGINT, _signal_handler)
                        signal.signal(signal.SIGTERM, _signal_handler)
                        result["end_time"] = datetime.now().isoformat()
                        results.append(result)
                    if stop_now:
                        break
                if stop_now:
                    break
            if stop_now:
                break
        if stop_now:
            break

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry = sum(1 for r in results if r["status"] == "dry_run")

    summary = {
        "batch_config": str(batch_cfg_path),
        "datasets": [x["dataset_name"] for x in dataset_run_items],
        "judge_models": judge_models,
        "injector_models": injector_models,
        "total_planned": total_combo_count,
        "executed": len(results),
        "success": success,
        "failed": failed,
        "interrupted": sum(1 for r in results if r["status"] == "interrupted"),
        "dry_run": dry,
        "results": results,
    }
    summary_path = log_dir / "yaml_batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 72)
    logger.info(f"Done. success={success}, failed={failed}, dry_run={dry}")
    logger.info(f"Summary: {summary_path}")
    logger.info(
        "Each run is executed by src.main, so outputs are stored under outputs/ and appended to results_summary.*"
    )
    logger.info("=" * 72)

    if failed > 0 and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

