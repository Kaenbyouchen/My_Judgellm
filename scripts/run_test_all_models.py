#!/usr/bin/env python
"""
Quick-test batch: 5 pairwise datasets × 9 models, 10 samples each.

- API models (gpt52): call OpenAI API directly.
- vLLM serve models: auto start → evaluate → auto stop → next model.
- Results written to outputs/_test_batch_<ts>/ (main results_summary.csv untouched).

Usage:
    python scripts/run_test_all_models.py
    python scripts/run_test_all_models.py --num-samples 10 --seed 42
    python scripts/run_test_all_models.py --models gpt52 vllmserve_gemma3_4b
    python scripts/run_test_all_models.py --datasets counselbench medaesqa
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration tables
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS: Dict[str, Dict[str, str]] = {
    "counselbench": {
        "path": "data/Counselbench/counselbench_pairwise.jsonl",
        "category": "one turn conversation",
    },
    "medaesqa": {
        "path": "data/MedAESQA/medaesqa_pairwise.jsonl",
        "category": "semantic Q&A",
    },
    "medical_eval_sphere": {
        "path": "data/medical_eval_sphere/medical_eval_sphere.jsonl",
        "category": "multi-choice qa",
    },
    "mediq_askdocs": {
        "path": "data/MediQ_AskDocs/mediq_askdocs_pairwise.jsonl",
        "category": "one turn conversation",
    },
    "medval_bench": {
        "path": "data/MedVAL_Bench/medval_bench_pairwise.jsonl",
        "category": "multi-turn conversation",
    },
}

ALL_BIASES = [
    "jargon_overloading",
    "plain_language",
    "clinical_formatting",
    "empathy_tone",
    "authority_style",
    "fake_citation",
    "language_fluency",
]

MODELS: List[Dict[str, str]] = [
    {"id": "gpt52", "type": "api"},
    {"id": "vllmserve_medgemma_4b", "type": "vllm", "hf": "google/medgemma-4b-it"},
    {"id": "vllmserve_biomistral_7b", "type": "vllm", "hf": "BioMistral/BioMistral-7B"},
    {"id": "vllmserve_prometheus2_7b", "type": "vllm", "hf": "prometheus-eval/prometheus-7b-v2.0"},
    {"id": "vllmserve_gemma3_4b", "type": "vllm", "hf": "google/gemma-3-4b-it"},
    {"id": "vllmserve_qwen3_4b_instruct", "type": "vllm", "hf": "Qwen/Qwen3-4B-Instruct-2507"},
    {"id": "vllmserve_llama31_8b_instruct", "type": "vllm", "hf": "meta-llama/Llama-3.1-8B-Instruct"},
    {"id": "vllmserve_deepseek_r1_distill_qwen_7b", "type": "vllm", "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    {"id": "vllmserve_deepseek_r1_distill_llama_8b", "type": "vllm", "hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
]

BIAS_INJECTOR_MODEL = "gpt52"

# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file (supports both true JSONL and pretty-printed JSON)."""
    text = path.read_text(encoding="utf-8")
    records: List[Dict[str, Any]] = []
    # Try line-by-line first
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if records:
        return records
    # Fall back to brace-matching for pretty-printed JSON
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            records.append(obj)
            idx = next_idx
        except json.JSONDecodeError:
            break
    return records


def create_temp_data_files(
    num_samples: int,
    seed: int,
    tmp_dir: Path,
    dataset_filter: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Sample N records from each dataset and write to temp JSONL files."""
    rng = random.Random(seed)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Path] = {}
    for ds_name, ds_info in DATASETS.items():
        if dataset_filter and ds_name not in dataset_filter:
            continue
        src_path = PROJECT_ROOT / ds_info["path"]
        if not src_path.exists():
            print(f"  [WARN] Dataset file not found, skipping: {src_path}")
            continue
        records = _load_jsonl_records(src_path)
        if not records:
            print(f"  [WARN] No records in {src_path}, skipping")
            continue
        chosen = rng.sample(records, min(num_samples, len(records)))
        # Each dataset gets its own subdirectory to avoid bias injection
        # cache collisions (cache key is parent_dir + bias_type + model).
        ds_dir = tmp_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        out_path = ds_dir / f"test_{ds_name}_{num_samples}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in chosen:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result[ds_name] = out_path
        print(f"  {ds_name}: {len(chosen)} samples → {ds_name}/{out_path.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM serve management
# ═══════════════════════════════════════════════════════════════════════════════

def _kill_port_occupant(port: int) -> None:
    """Kill any process listening on the given port (never kills self)."""
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL, text=True
        )
        for pid_str in out.strip().split("\n"):
            pid_str = pid_str.strip()
            if pid_str.isdigit() and int(pid_str) != my_pid:
                os.kill(int(pid_str), signal.SIGKILL)
                print(f"  Killed existing process {pid_str} on port {port}")
        time.sleep(2)
    except (subprocess.CalledProcessError, OSError):
        pass


def start_vllm_server(
    hf_model: str,
    port: int,
    log_dir: Path,
    model_id: str,
    gpu_mem_util: float = 0.90,
    max_model_len: int = 2048,
) -> Optional[subprocess.Popen]:
    """Start `vllm serve` as a background subprocess."""
    _kill_port_occupant(port)

    log_file = log_dir / f"vllm_{model_id}.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--dtype", "auto",
        "--download-dir", str(PROJECT_ROOT / "src" / "models"),
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"  Starting vLLM serve: {hf_model}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    fh = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        start_new_session=True,
    )
    # Stash file handle for cleanup
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def wait_for_vllm_health(port: int, timeout: int = 600, interval: int = 5) -> bool:
    """Poll the vLLM health endpoint until ready or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"  vLLM server healthy after {attempt} checks ({time.time() + timeout - deadline:.0f}s)")
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        remaining = int(deadline - time.time())
        if attempt % 12 == 0:
            print(f"  Still waiting for vLLM... ({remaining}s remaining)")
        time.sleep(interval)
    return False


def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    """Terminate vLLM serve and all its worker child processes (frees GPU memory)."""
    if proc is None:
        return
    print("  Stopping vLLM server (process group)...")
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass

    # Step 1: SIGTERM the entire process group (graceful shutdown)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass

    # Step 2: Wait for main process to exit
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Step 3: Force-kill entire process group
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass

    fh = getattr(proc, "_log_fh", None)
    if fh:
        try:
            fh.close()
        except Exception:
            pass
    print("  vLLM server stopped (GPU memory released).")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation config & runner
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_experiment_config() -> Dict[str, Any]:
    """Load configs/experiment.yaml as the base config (same as batch runner)."""
    base_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_eval_config(
    base_cfg: Dict[str, Any],
    dataset_name: str,
    judge_model: str,
    bias_type: str,
    bias_injector_model: str = BIAS_INJECTOR_MODEL,
) -> Dict[str, Any]:
    """Override only the fields that differ from experiment.yaml.

    Judge prompts, semantic guard config, etc. are all inherited from
    experiment.yaml — exactly the same as the batch runner.
    """
    from copy import deepcopy
    cfg = deepcopy(base_cfg)

    cfg.setdefault("data", {})
    cfg["data"]["type"] = "pairwise"
    cfg["data"]["dataset_name"] = dataset_name

    cfg.setdefault("bias", {})
    cfg["bias"]["enabled"] = True
    cfg["bias"]["type"] = bias_type
    cfg["bias"]["injection_mode"] = "word"
    cfg["bias"]["inject_to"] = "non_gt"
    cfg["bias"]["model"] = bias_injector_model

    cfg.setdefault("judge", {})
    for k in ("provider", "type", "model_id", "model_name"):
        cfg["judge"].pop(k, None)
    cfg["judge"]["model"] = judge_model
    cfg["judge"]["allow_fallback_mock"] = False

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["compute_original_acc"] = True
    cfg["evaluation"]["compute_bias_metrics"] = True
    cfg["evaluation"]["position_debias_pairwise"] = True
    cfg["evaluation"]["request_batch_size"] = 8

    return cfg


def run_single_eval(
    config_dict: Dict[str, Any],
    data_path: str,
    output_dir: str,
    config_dir: Path,
    run_tag: str,
    python_exe: str,
    env_extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Write config to file, invoke src/main.py as subprocess, return parsed result."""
    cfg_path = config_dir / f"{run_tag}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    cmd = [
        python_exe, str(PROJECT_ROOT / "src" / "main.py"),
        "--config", str(cfg_path),
        "--data-path", data_path,
        "--output-dir", output_dir,
    ]

    env = {**os.environ}
    if env_extra:
        env.update(env_extra)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=600,
    )
    elapsed = time.time() - t0

    result: Dict[str, Any] = {
        "tag": run_tag,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "metrics": {},
    }

    # Parse metrics from stdout (look for "--- Pipeline done: ... ---")
    for line in proc.stdout.splitlines():
        if "Pipeline done:" in line:
            for part in line.split("|"):
                part = part.strip()
                for key in ("acc=", "acc_b=", "rr=", "cr="):
                    if part.startswith(key):
                        val_str = part[len(key):].rstrip("%").strip()
                        try:
                            result["metrics"][key.rstrip("=")] = float(val_str)
                        except ValueError:
                            pass

    if proc.returncode != 0:
        # Extract the actual exception line (last non-empty line of stderr)
        stderr_text = proc.stderr or ""
        stderr_lines = [l for l in stderr_text.strip().splitlines() if l.strip()]
        exception_line = stderr_lines[-1] if stderr_lines else "unknown error"
        result["error"] = exception_line[:300]
        stderr_log = config_dir / f"{run_tag}.stderr"
        stderr_log.write_text(stderr_text, encoding="utf-8")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Summary display
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(
    all_results: List[Dict[str, Any]],
    dataset_bias_map: Dict[str, str],
) -> None:
    """Print a formatted summary table of all test results."""
    print("\n" + "=" * 100)
    print("  TEST BATCH RESULTS SUMMARY")
    print("=" * 100)

    # Header: dataset → bias used
    print("\n  Dataset → Bias mapping:")
    for ds, bias in dataset_bias_map.items():
        print(f"    {ds}: {bias}")

    # Results table
    header = f"{'Model':<45} {'Dataset':<22} {'acc':>6} {'acc_b':>6} {'rr':>6} {'cr':>6} {'Time':>7} {'Status':>8}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        m = r.get("metrics", {})
        acc = f"{m['acc']:.1f}" if "acc" in m else "-"
        acc_b = f"{m['acc_b']:.1f}" if "acc_b" in m else "-"
        rr = f"{m['rr']:.1f}" if "rr" in m else "-"
        cr = f"{m['cr']:.1f}" if "cr" in m else "-"
        status = "OK" if r["returncode"] == 0 else "FAIL"
        elapsed = f"{r['elapsed_s']:.0f}s"

        tag_parts = r["tag"].split("__")
        model_name = tag_parts[0] if tag_parts else r["tag"]
        ds_name = tag_parts[1] if len(tag_parts) > 1 else "?"

        print(f"{model_name:<45} {ds_name:<22} {acc:>6} {acc_b:>6} {rr:>6} {cr:>6} {elapsed:>7} {status:>8}")

    print("-" * len(header))

    total = len(all_results)
    ok = sum(1 for r in all_results if r["returncode"] == 0)
    fail = total - ok
    print(f"\n  Total: {total}  |  OK: {ok}  |  Failed: {fail}")
    print("=" * 100 + "\n")

    # Also try to read the generated results_summary.csv
    if all_results:
        csv_candidates = set()
        for r in all_results:
            tag_parts = r["tag"].split("__")
            if len(tag_parts) >= 1:
                # output_dir is the same for all
                pass
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick-test: 5 datasets × 9 models, 10 samples each, auto vLLM serve management.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/run_test_all_models.py
          python scripts/run_test_all_models.py --num-samples 5 --seed 123
          python scripts/run_test_all_models.py --models gpt52 vllmserve_gemma3_4b
          python scripts/run_test_all_models.py --datasets counselbench medaesqa --models gpt52
        """),
    )
    p.add_argument("--num-samples", type=int, default=10, help="Samples per dataset (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling & bias selection")
    p.add_argument("--port", type=int, default=8000, help="Port for vLLM serve (default: 8000)")
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model IDs to test (default: all 9)",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Subset of dataset names to test (default: all 5 pairwise)",
    )
    p.add_argument(
        "--vllm-timeout", type=int, default=600,
        help="vLLM serve startup timeout in seconds (default: 600)",
    )
    p.add_argument(
        "--gpu-mem-util", type=float, default=0.90,
        help="GPU memory utilization for vLLM serve (default: 0.90)",
    )
    p.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Max model length for vLLM serve (default: 8192)",
    )
    p.add_argument(
        "--bias-injector", type=str, default=BIAS_INJECTOR_MODEL,
        help=f"Bias injector model (default: {BIAS_INJECTOR_MODEL})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = PROJECT_ROOT / "outputs" / f"_test_batch_{timestamp}"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = test_output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    log_dir = test_output_dir / "vllm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    print("=" * 72)
    print("  QUICK TEST BATCH — 5 Datasets × 9 Models")
    print("=" * 72)
    print(f"  Timestamp   : {timestamp}")
    print(f"  Output dir  : {test_output_dir}")
    print(f"  Num samples : {args.num_samples}")
    print(f"  Seed        : {args.seed}")
    print(f"  Python      : {python_exe}")
    print(f"  Port (vLLM) : {args.port}")
    print()

    # ── Load base experiment config (judge prompts, semantic guard, etc.) ──
    base_experiment_cfg = load_base_experiment_config()
    print(f"  Base config : configs/experiment.yaml")
    print()

    # ── Filter models ──
    models_to_run = MODELS
    if args.models:
        valid_ids = {m["id"] for m in MODELS}
        for mid in args.models:
            if mid not in valid_ids:
                print(f"  [WARN] Unknown model ID '{mid}', available: {sorted(valid_ids)}")
        models_to_run = [m for m in MODELS if m["id"] in args.models]
    if not models_to_run:
        print("  [ERROR] No models selected. Exiting.")
        sys.exit(1)

    dataset_filter = args.datasets
    if dataset_filter:
        valid_ds = set(DATASETS.keys())
        for ds in dataset_filter:
            if ds not in valid_ds:
                print(f"  [WARN] Unknown dataset '{ds}', available: {sorted(valid_ds)}")
        dataset_filter = [d for d in dataset_filter if d in valid_ds]

    # ── Step 1: Create temp data files ──
    print("─── Creating temp data files ───")
    tmp_data_dir = test_output_dir / "temp_data"
    temp_data_files = create_temp_data_files(args.num_samples, args.seed, tmp_data_dir, dataset_filter)
    if not temp_data_files:
        print("  [ERROR] No temp data files created. Exiting.")
        sys.exit(1)
    print()

    # ── Step 2: Random bias per dataset (fixed across models for fair comparison) ──
    rng = random.Random(args.seed)
    dataset_bias_map: Dict[str, str] = {}
    for ds_name in temp_data_files:
        dataset_bias_map[ds_name] = rng.choice(ALL_BIASES)

    print("─── Dataset → Bias assignment ───")
    for ds, bias in dataset_bias_map.items():
        print(f"  {ds}: {bias}")
    print()

    # ── Step 3: Run evaluations ──
    total_evals = len(models_to_run) * len(temp_data_files)
    all_results: List[Dict[str, Any]] = []
    eval_idx = 0
    batch_t0 = time.time()

    # Group: API models first, then vLLM models
    api_models = [m for m in models_to_run if m["type"] == "api"]
    vllm_models = [m for m in models_to_run if m["type"] == "vllm"]

    for model_group, group_label in [(api_models, "API"), (vllm_models, "vLLM")]:
        for model in model_group:
            model_id = model["id"]
            is_vllm = model["type"] == "vllm"
            vllm_proc: Optional[subprocess.Popen] = None

            print(f"\n{'═' * 72}")
            print(f"  MODEL: {model_id} ({group_label})")
            print(f"{'═' * 72}")

            # Start vLLM serve if needed
            if is_vllm:
                hf_name = model["hf"]
                vllm_proc = start_vllm_server(
                    hf_model=hf_name,
                    port=args.port,
                    log_dir=log_dir,
                    model_id=model_id,
                    gpu_mem_util=args.gpu_mem_util,
                    max_model_len=args.max_model_len,
                )
                if vllm_proc is None:
                    print(f"  [ERROR] Failed to start vLLM for {model_id}, skipping.")
                    for ds_name in temp_data_files:
                        eval_idx += 1
                        all_results.append({
                            "tag": f"{model_id}__{ds_name}",
                            "returncode": -1,
                            "elapsed_s": 0,
                            "metrics": {},
                            "error": "vLLM start failed",
                        })
                    continue

                print(f"  Waiting for vLLM health check (timeout={args.vllm_timeout}s)...")
                healthy = wait_for_vllm_health(args.port, timeout=args.vllm_timeout)
                if not healthy:
                    print(f"  [ERROR] vLLM failed to become healthy for {model_id}.")
                    stop_vllm_server(vllm_proc)
                    for ds_name in temp_data_files:
                        eval_idx += 1
                        all_results.append({
                            "tag": f"{model_id}__{ds_name}",
                            "returncode": -1,
                            "elapsed_s": 0,
                            "metrics": {},
                            "error": "vLLM health timeout",
                        })
                    continue
                print(f"  vLLM server ready for {model_id}!")

            # Run evaluations on all datasets
            env_extra: Dict[str, str] = {}
            if is_vllm:
                env_extra["VLLM_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"

            try:
                for ds_name, tmp_path in temp_data_files.items():
                    eval_idx += 1
                    bias_type = dataset_bias_map[ds_name]
                    run_tag = f"{model_id}__{ds_name}__{bias_type}"

                    progress = f"[{eval_idx}/{total_evals}]"
                    print(f"\n  {progress} {ds_name} | bias={bias_type}")

                    cfg = build_eval_config(
                        base_cfg=base_experiment_cfg,
                        dataset_name=ds_name,
                        judge_model=model_id,
                        bias_type=bias_type,
                        bias_injector_model=args.bias_injector,
                    )

                    try:
                        result = run_single_eval(
                            config_dict=cfg,
                            data_path=str(tmp_path),
                            output_dir=str(test_output_dir),
                            config_dir=config_dir,
                            run_tag=run_tag,
                            python_exe=python_exe,
                            env_extra=env_extra,
                        )
                    except subprocess.TimeoutExpired:
                        result = {
                            "tag": run_tag,
                            "returncode": -2,
                            "elapsed_s": 600,
                            "metrics": {},
                            "error": "eval timeout (600s)",
                        }
                    except Exception as e:
                        result = {
                            "tag": run_tag,
                            "returncode": -3,
                            "elapsed_s": 0,
                            "metrics": {},
                            "error": str(e),
                        }

                    all_results.append(result)
                    m = result.get("metrics", {})
                    status = "OK" if result["returncode"] == 0 else "FAIL"
                    metric_str = " | ".join(
                        f"{k}={v:.1f}%" for k, v in m.items()
                    ) if m else "no metrics"
                    print(
                        f"    → {status} ({result['elapsed_s']:.0f}s) {metric_str}"
                    )
                    if result["returncode"] != 0 and result.get("error"):
                        print(f"    Error: {result['error'][:200]}")

            finally:
                # Always stop vLLM after all datasets for this model.
                # Do NOT call _kill_port_occupant here — it uses lsof which
                # may return the main script's PID (lingering health-check
                # TCP connections in TIME_WAIT) and kill the script itself.
                # Port cleanup happens at the start of start_vllm_server().
                if is_vllm and vllm_proc is not None:
                    stop_vllm_server(vllm_proc)
                    time.sleep(3)

    batch_elapsed = time.time() - batch_t0
    elapsed_str = f"{batch_elapsed / 60:.1f} min" if batch_elapsed > 60 else f"{batch_elapsed:.0f}s"

    # ── Step 4: Summary ──
    print_summary_table(all_results, dataset_bias_map)
    print(f"  Total time: {elapsed_str}")
    print(f"  Results dir: {test_output_dir}")

    # Also try reading the generated results_summary.csv
    csv_path = test_output_dir / "results_summary.csv"
    if csv_path.exists():
        print(f"\n  Full results CSV: {csv_path}")

    # Save machine-readable summary
    summary_path = test_output_dir / "test_summary.json"
    summary = {
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "dataset_bias_map": dataset_bias_map,
        "models": [m["id"] for m in models_to_run],
        "total_evals": total_evals,
        "success": sum(1 for r in all_results if r["returncode"] == 0),
        "failed": sum(1 for r in all_results if r["returncode"] != 0),
        "total_elapsed_s": round(batch_elapsed, 1),
        "results": all_results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary JSON: {summary_path}\n")


if __name__ == "__main__":
    main()
