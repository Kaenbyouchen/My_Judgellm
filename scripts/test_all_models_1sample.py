#!/usr/bin/env python
"""
Quick smoke test: run 1 sample from each pairwise dataset with each available
model type (openai, anthropic, mock).  Verify results land in results_summary.csv.

Usage:
  python scripts/test_all_models_1sample.py
  python scripts/test_all_models_1sample.py --models openai        # test only openai
  python scripts/test_all_models_1sample.py --models openai,anthropic
  python scripts/test_all_models_1sample.py --datasets counselbench  # test only one dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main as run_single_experiment
from src.utils.io import load_yaml

# ── Test matrix ──────────────────────────────────────────────────────

# Map from test dataset name -> 1-sample data file
TEST_DATA = {
    "medical_eval_sphere": "data/test_1sample/medical_eval_sphere/medical_eval_sphere.jsonl",
    "mediq_askdocs":       "data/test_1sample/MediQ_AskDocs/mediq_askdocs_pairwise.jsonl",
    "counselbench":        "data/test_1sample/Counselbench/counselbench_pairwise.jsonl",
    "medval_bench":        "data/test_1sample/MedVAL_Bench/medval_bench_pairwise.jsonl",
    "medaesqa":            "data/test_1sample/MedAESQA/medaesqa_pairwise.jsonl",
}

# Dataset -> category (from datasets.yaml)
DATASET_CATEGORIES = {
    "medical_eval_sphere": "multi-choice qa",
    "mediq_askdocs":       "one turn conversation",
    "counselbench":        "one turn conversation",
    "medval_bench":        "multi-turn conversation",
    "medaesqa":            "semantic Q&A",
}

# Model configs to test: (label, judge_model, bias_model)
# - openai: uses gpt4omini (cheapest)
# - anthropic: uses claude35_sonnet
# - mock: always available
MODEL_CONFIGS = {
    "mock": {
        "judge_model": "mock-judge-v1",
        "judge_provider": "mock",
        "bias_model": None,  # bias disabled for mock test
        "bias_enabled": False,
    },
    "openai": {
        "judge_model": "gpt4omini",
        "judge_provider": None,  # auto-infer
        "bias_model": "gpt4omini",
        "bias_enabled": True,
    },
    "anthropic": {
        "judge_model": "claude3_haiku",
        "judge_provider": None,  # auto-infer
        "bias_model": "gpt4omini",  # use openai for bias injection (cheaper)
        "bias_enabled": True,
    },
}


def _check_api_key(model_type: str) -> bool:
    """Check if required API key is set."""
    if model_type == "mock":
        return True
    if model_type == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if model_type == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if model_type == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return False


def build_test_config(
    dataset_name: str,
    data_path: str,
    model_cfg: dict,
) -> dict:
    """Build a temporary experiment config for testing."""
    cfg = {
        "experiment": {
            "name": "smoke_test",
            "seed": 42,
            "output_dir": "outputs",
        },
        "data": {
            "type": "pairwise",
            "dataset_name": dataset_name,
            "path": data_path,
        },
        "bias": {
            "enabled": model_cfg["bias_enabled"],
            "type": "plain_language",
            "injection_mode": "word",
            "inject_to": "non_gt",
        },
        "judge": {
            "allow_fallback_mock": False,
        },
        "evaluation": {
            "compute_original_acc": True,
            "compute_bias_metrics": model_cfg["bias_enabled"],
            "request_batch_size": 1,
            "position_debias_pairwise": True,
        },
    }

    # Judge config
    if model_cfg["judge_provider"]:
        cfg["judge"]["provider"] = model_cfg["judge_provider"]
        cfg["judge"]["model_id"] = model_cfg["judge_model"]
    else:
        cfg["judge"]["model"] = model_cfg["judge_model"]

    # Bias config
    if model_cfg["bias_enabled"] and model_cfg["bias_model"]:
        cfg["bias"]["model"] = model_cfg["bias_model"]
        cfg["bias"]["semantic_guard"] = {"enabled": False}
    else:
        cfg["bias"]["enabled"] = False

    return cfg


def run_test(
    dataset_name: str,
    model_label: str,
    model_cfg: dict,
) -> dict:
    """Run a single test and return result dict."""
    data_path = TEST_DATA[dataset_name]
    result = {
        "dataset": dataset_name,
        "model": model_label,
        "status": "pending",
        "error": None,
        "duration_s": 0,
    }

    cfg = build_test_config(dataset_name, data_path, model_cfg)

    # Write temp config
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cfg_dir = PROJECT_ROOT / "outputs" / "test_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"test_{model_label}_{dataset_name}_{ts}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    original_argv = sys.argv.copy()
    t0 = time.time()
    try:
        sys.argv = ["test", "--config", str(cfg_path)]
        output = run_single_experiment()
        result["status"] = "success"
        if isinstance(output, dict):
            metrics = output.get("metrics", {})
            acc = metrics.get("accuracy_original", {}).get("accuracy")
            if acc is not None:
                result["acc"] = f"{acc:.4f}"
            acc_b = metrics.get("accuracy_biased", {}).get("accuracy_biased")
            if acc_b is not None:
                result["acc_biased"] = f"{acc_b:.4f}"
            rr = metrics.get("rr", {}).get("rr")
            if rr is not None:
                result["rr"] = f"{rr:.4f}"
            cr = metrics.get("cr", {}).get("cr")
            if cr is not None:
                result["cr"] = f"{cr:.4f}"
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        traceback.print_exc()
    finally:
        sys.argv = original_argv
        result["duration_s"] = round(time.time() - t0, 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test: 1 sample per dataset per model")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model types to test (default: all available)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to test (default: all)")
    args = parser.parse_args()

    # Determine which models to test
    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
    else:
        model_types = []
        for mt in ["mock", "openai", "anthropic"]:
            if _check_api_key(mt):
                model_types.append(mt)
            else:
                print(f"  [SKIP] {mt}: API key not set")

    # Determine which datasets to test
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    else:
        dataset_names = list(TEST_DATA.keys())

    total = len(model_types) * len(dataset_names)
    print(f"\n{'=' * 70}")
    print(f"  Smoke Test: {len(dataset_names)} datasets x {len(model_types)} model types = {total} runs")
    print(f"  Models: {model_types}")
    print(f"  Datasets: {dataset_names}")
    print(f"{'=' * 70}\n")

    results = []
    idx = 0
    for mt in model_types:
        if mt not in MODEL_CONFIGS:
            print(f"  [SKIP] Unknown model type: {mt}")
            continue
        model_cfg = MODEL_CONFIGS[mt]
        for ds in dataset_names:
            if ds not in TEST_DATA:
                print(f"  [SKIP] Unknown dataset: {ds}")
                continue
            idx += 1
            print(f"\n[{idx}/{total}] Testing {mt} + {ds} ...")
            r = run_test(ds, mt, model_cfg)
            results.append(r)
            status_icon = "OK" if r["status"] == "success" else "FAIL"
            print(f"  [{status_icon}] {mt} + {ds} ({r['duration_s']}s)")
            if r.get("acc"):
                print(f"       acc={r['acc']}", end="")
            if r.get("acc_biased"):
                print(f"  acc_biased={r['acc_biased']}", end="")
            if r.get("rr"):
                print(f"  rr={r['rr']}", end="")
            if r.get("cr"):
                print(f"  cr={r['cr']}", end="")
            print()
            if r["error"]:
                print(f"       Error: {r['error']}")

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {success} passed, {failed} failed, {len(results)} total")
    print(f"{'=' * 70}")

    if failed > 0:
        print("\n  FAILED tests:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"    - {r['model']} + {r['dataset']}: {r['error']}")
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        # Check CSV
        csv_path = PROJECT_ROOT / "outputs" / "results_summary.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"\n  results_summary.csv has {len(df)} rows")
        sys.exit(0)


if __name__ == "__main__":
    main()
