#!/usr/bin/env python
"""
Run selected dataset/model/bias combinations in batch.

This script is designed for "pick one dataset, pick models, pick biases, run all combos".
Each run calls the existing main pipeline, so outputs are written to `outputs/` as usual.

Examples:
  python scripts/batch_selective_evaluate.py \
    --config configs/experiment.yaml \
    --dataset medical_eval_sphere \
    --models qwen3_4b_instruct llama31_8b_instruct \
    --biases plain_language authority \
    --modes rewrite word

  python scripts/batch_selective_evaluate.py \
    --config configs/experiment.yaml \
    --dataset open_patient \
    --models qwen3_4b_instruct \
    --biases none
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger

# Add project root to import path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main as run_single_experiment
from src.utils.io import load_yaml

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Interrupt received; current run will finish, then stop.")


def _setup_log_dir(user_log_dir: str | None) -> Path:
    if user_log_dir:
        log_dir = Path(user_log_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / "outputs" / "batch_logs" / f"selective_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _normalize_modes(modes: List[str]) -> List[str]:
    normalized = []
    for m in modes:
        mm = m.strip().lower()
        if mm not in {"rewrite", "word"}:
            raise ValueError(f"Unsupported mode '{m}'. Use 'rewrite' or 'word'.")
        normalized.append(mm)
    return normalized


def _build_config(
    base: Dict[str, Any],
    dataset_name: str,
    judge_model: str,
    bias_name: str,
    injection_mode: str,
    injector_model: str | None,
) -> Dict[str, Any]:
    cfg = deepcopy(base)

    cfg.setdefault("data", {})
    cfg["data"]["dataset_name"] = dataset_name

    cfg.setdefault("judge", {})
    # Use simplified format; provider auto-inferred from models.yaml.
    cfg["judge"].pop("provider", None)
    cfg["judge"].pop("model_id", None)
    cfg["judge"].pop("type", None)
    cfg["judge"].pop("model_name", None)
    cfg["judge"]["model"] = judge_model

    cfg.setdefault("bias", {})
    cfg["bias"]["injection_mode"] = injection_mode

    # Keep provider inferred from model pool by removing explicit injector_type.
    cfg["bias"].pop("injector_type", None)
    cfg["bias"].pop("model_id", None)
    cfg["bias"].pop("model_name", None)

    if bias_name == "none":
        cfg["bias"]["enabled"] = False
    else:
        cfg["bias"]["enabled"] = True
        cfg["bias"]["type"] = bias_name

    # If injector model not specified, reuse judge model.
    cfg["bias"]["model"] = injector_model if injector_model else judge_model
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Batch run selected dataset/model/bias combinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Base config path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name defined in configs/datasets.yaml")
    parser.add_argument("--models", nargs="+", required=True, help="Judge model IDs from configs/models.yaml")
    parser.add_argument(
        "--biases",
        nargs="+",
        required=True,
        help="Bias list. Use 'none' to run without bias injection.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rewrite", "word"],
        help="Injection modes to run (rewrite, word). Ignored when bias=none.",
    )
    parser.add_argument(
        "--injector-model",
        type=str,
        default=None,
        help="Optional fixed model ID for bias injection (default: same as judge model).",
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Batch log directory")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations only")
    parser.add_argument("--continue-on-error", action="store_true", default=True, help="Continue when one combo fails")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    base_config_path = Path(args.config)
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    base_config = load_yaml(str(base_config_path))
    log_dir = _setup_log_dir(args.log_dir)
    run_cfg_dir = log_dir / "generated_configs"
    run_cfg_dir.mkdir(parents=True, exist_ok=True)

    logger.add(log_dir / "batch_selective.log", level="INFO")
    modes = _normalize_modes(args.modes)

    combos: List[Dict[str, str]] = []
    for model in args.models:
        for bias in args.biases:
            if bias == "none":
                combos.append({"model": model, "bias": bias, "mode": "rewrite"})
            else:
                for mode in modes:
                    combos.append({"model": model, "bias": bias, "mode": mode})

    logger.info("=" * 72)
    logger.info("Selective batch evaluation start")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Biases: {args.biases}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Combinations: {len(combos)}")
    logger.info(f"Log dir: {log_dir}")
    logger.info("=" * 72)

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(combos, start=1):
        if _shutdown_requested:
            logger.warning("Shutdown requested; stopping batch loop.")
            break

        judge_model = c["model"]
        bias = c["bias"]
        mode = c["mode"]
        run_name = f"{idx:03d}_{args.dataset}_{judge_model}_{bias}_{mode}"
        logger.info(f"[{idx}/{len(combos)}] {run_name}")

        result = {
            "index": idx,
            "dataset": args.dataset,
            "judge_model": judge_model,
            "bias": bias,
            "mode": mode,
            "status": "pending",
            "error": None,
            "config_path": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }

        try:
            cfg = _build_config(
                base=base_config,
                dataset_name=args.dataset,
                judge_model=judge_model,
                bias_name=bias,
                injection_mode=mode,
                injector_model=args.injector_model,
            )
            cfg_path = run_cfg_dir / f"{run_name}.yaml"
            _write_yaml(cfg_path, cfg)
            result["config_path"] = str(cfg_path)

            if args.dry_run:
                result["status"] = "dry_run"
            else:
                original_argv = sys.argv.copy()
                try:
                    sys.argv = ["batch_selective_evaluate.py", "--config", str(cfg_path)]
                    run_single_experiment()
                finally:
                    sys.argv = original_argv
                result["status"] = "success"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            logger.error(f"Run failed: {run_name} | {e}")
            if not args.continue_on_error:
                results.append(result)
                break
        finally:
            result["end_time"] = datetime.now().isoformat()
            results.append(result)

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry = sum(1 for r in results if r["status"] == "dry_run")

    summary = {
        "dataset": args.dataset,
        "total_planned": len(combos),
        "executed": len(results),
        "success": success,
        "failed": failed,
        "dry_run": dry,
        "results": results,
    }
    summary_path = log_dir / "batch_selective_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 72)
    logger.info(f"Done. success={success}, failed={failed}, dry_run={dry}")
    logger.info(f"Summary: {summary_path}")
    logger.info(
        "Run outputs are in outputs/<dataset>_<data_type>_<bias>_<judge>/ and summary auto-appended to outputs/results_summary.*"
    )
    logger.info("=" * 72)

    if failed > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

