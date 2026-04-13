#!/home1/zrui6736/miniconda3/envs/Judgellm/bin/python
"""
Test all vLLM-serve models with 1 sample from each dataset.

For each model:
  1. Start vLLM serve (OpenAI-compatible API)
  2. Run 1-sample evaluation on all 5 datasets
  3. Stop vLLM serve
  4. Report pass/fail

Usage (run on a GPU node with compute capability >= 7.0):
  python scripts/test_vllm_serve_models.py
  python scripts/test_vllm_serve_models.py --models vllmserve_gemma3_4b
  python scripts/test_vllm_serve_models.py --port 8001 --gpu-ids 0
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main as run_single_experiment

# ── Models to test ──────────────────────────────────────────────────
# Each entry: (judge_id in models.yaml, HF model name for vllm serve)
VLLM_MODELS = [
    ("vllmserve_medgemma_4b",                  "google/medgemma-4b-it"),
    ("vllmserve_biomistral_7b",                "BioMistral/BioMistral-7B"),
    ("vllmserve_prometheus2_7b",               "prometheus-eval/prometheus-7b-v2.0"),
    ("vllmserve_gemma3_4b",                    "google/gemma-3-4b-it"),
    ("vllmserve_qwen3_4b_instruct",           "Qwen/Qwen3-4B-Instruct-2507"),
    ("vllmserve_llama31_8b_instruct",         "meta-llama/Llama-3.1-8B-Instruct"),
    ("vllmserve_deepseek_r1_distill_qwen_7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("vllmserve_deepseek_r1_distill_llama_8b","deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
]

# Also test gpt52 (API model, no vLLM needed)
API_MODELS = [
    ("gpt52", None),
]

# ── Datasets ────────────────────────────────────────────────────────
TEST_DATA = {
    "medical_eval_sphere": "data/test_1sample/medical_eval_sphere/medical_eval_sphere.jsonl",
    "mediq_askdocs":       "data/test_1sample/MediQ_AskDocs/mediq_askdocs_pairwise.jsonl",
    "counselbench":        "data/test_1sample/Counselbench/counselbench_pairwise.jsonl",
    "medval_bench":        "data/test_1sample/MedVAL_Bench/medval_bench_pairwise.jsonl",
    "medaesqa":            "data/test_1sample/MedAESQA/medaesqa_pairwise.jsonl",
}

# ── vLLM lifecycle ──────────────────────────────────────────────────
_vllm_proc = None
_vllm_log_fh = None
GPU_COOLDOWN = 15


def start_vllm(hf_model: str, port: int, gpu_ids: str) -> subprocess.Popen:
    global _vllm_proc, _vllm_log_fh
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model,
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",  # skip CUDA graph capture (faster startup on V100)
    ]
    log_dir = PROJECT_ROOT / "outputs" / "test_vllm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hf_model.replace("/", "_")
    log_path = log_dir / f"vllm_{safe_name}.log"
    log_fh = open(log_path, "w")
    _vllm_log_fh = log_fh
    print(f"    Starting vLLM serve: {hf_model}")
    print(f"    Log: {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    _vllm_proc = proc
    return proc


def wait_vllm_ready(port: int, timeout: int = 600, proc=None) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            print(f"    vLLM process exited with code {proc.returncode}")
            return False
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_vllm(proc) -> None:
    global _vllm_proc, _vllm_log_fh
    if proc is not None and proc.poll() is None:
        # Kill the entire process group (vLLM spawns child EngineCore processes)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            proc.wait(timeout=10)
    _vllm_proc = None
    if _vllm_log_fh is not None:
        _vllm_log_fh.close()
        _vllm_log_fh = None
    print(f"    Waiting {GPU_COOLDOWN}s for GPU memory release ...")
    time.sleep(GPU_COOLDOWN)


def _cleanup(signum, _frame):
    global _vllm_proc
    if _vllm_proc:
        print("\n  Interrupt — stopping vLLM ...")
        stop_vllm(_vllm_proc)
    sys.exit(1)


# ── Build experiment config ─────────────────────────────────────────

def build_config(dataset_name: str, data_path: str, judge_model_id: str) -> dict:
    """Build a minimal experiment config for 1-sample test."""
    return {
        "experiment": {
            "name": "vllm_smoke_test",
            "seed": 42,
            "output_dir": "outputs",
        },
        "data": {
            "type": "pairwise",
            "dataset_name": dataset_name,
            "path": data_path,
        },
        "bias": {
            "enabled": True,
            "type": "plain_language",
            "injection_mode": "word",
            "inject_to": "non_gt",
            "model": "gpt4omini",
            "semantic_guard": {"enabled": False},
        },
        "judge": {
            "model": judge_model_id,
            "allow_fallback_mock": False,
        },
        "evaluation": {
            "compute_original_acc": True,
            "compute_bias_metrics": True,
            "request_batch_size": 1,
            "position_debias_pairwise": True,
        },
    }


def run_one_test(dataset_name: str, judge_model_id: str) -> dict:
    """Run 1 sample and return result dict."""
    data_path = TEST_DATA[dataset_name]
    result = {
        "dataset": dataset_name,
        "judge": judge_model_id,
        "status": "pending",
        "error": None,
        "duration_s": 0,
    }

    cfg = build_config(dataset_name, data_path, judge_model_id)

    # Write temp config
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cfg_dir = PROJECT_ROOT / "outputs" / "test_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"vllm_test_{judge_model_id}_{dataset_name}_{ts}.yaml"
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
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        traceback.print_exc()
    finally:
        sys.argv = original_argv
        result["duration_s"] = round(time.time() - t0, 1)

    return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test vLLM serve models with 1 sample per dataset")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model IDs (e.g. vllmserve_gemma3_4b,gpt52)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all)")
    parser.add_argument("--vllm-timeout", type=int, default=600,
                        help="Seconds to wait for vLLM startup (default: 600)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API models (gpt52)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Filter models
    if args.models:
        requested = set(m.strip() for m in args.models.split(","))
        vllm_models = [(jid, hf) for jid, hf in VLLM_MODELS if jid in requested]
        api_models = [(jid, hf) for jid, hf in API_MODELS if jid in requested]
    else:
        vllm_models = list(VLLM_MODELS)
        api_models = [] if args.skip_api else list(API_MODELS)

    all_models = api_models + vllm_models

    # Filter datasets
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    else:
        dataset_names = list(TEST_DATA.keys())

    total_tests = len(all_models) * len(dataset_names)
    print(f"\n{'=' * 70}")
    print(f"  vLLM Serve Model Test")
    print(f"  {len(all_models)} models x {len(dataset_names)} datasets = {total_tests} tests")
    print(f"  Models: {[m[0] for m in all_models]}")
    print(f"  Datasets: {dataset_names}")
    print(f"{'=' * 70}\n")

    all_results = []
    global_idx = 0
    t_global = time.time()

    for model_idx, (judge_id, hf_model) in enumerate(all_models, 1):
        is_vllm = hf_model is not None
        model_start = time.time()

        print(f"\n{'─' * 70}")
        print(f"  [{model_idx}/{len(all_models)}] {judge_id}" +
              (f" ({hf_model})" if hf_model else " (API)"))
        print(f"{'─' * 70}")

        # Start vLLM if needed
        vllm_proc = None
        if is_vllm:
            vllm_proc = start_vllm(hf_model, args.port, args.gpu_ids)
            base_url = f"http://127.0.0.1:{args.port}/v1"
            os.environ["VLLM_BASE_URL"] = base_url
            print(f"    Waiting for vLLM ready (timeout={args.vllm_timeout}s) ...")
            if not wait_vllm_ready(args.port, args.vllm_timeout, proc=vllm_proc):
                print(f"    FAILED: vLLM did not start for {hf_model}")
                # Read last lines of log
                log_dir = PROJECT_ROOT / "outputs" / "test_vllm_logs"
                safe_name = hf_model.replace("/", "_")
                log_path = log_dir / f"vllm_{safe_name}.log"
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    print(f"    Last 10 log lines:")
                    for line in lines[-10:]:
                        print(f"      {line}")
                stop_vllm(vllm_proc)
                for ds in dataset_names:
                    global_idx += 1
                    all_results.append({
                        "dataset": ds, "judge": judge_id,
                        "status": "FAILED", "error": "vLLM failed to start",
                        "duration_s": 0,
                    })
                continue
            print(f"    vLLM ready at {base_url}")

        # Test each dataset
        model_results = []
        for ds in dataset_names:
            if ds not in TEST_DATA:
                print(f"    [SKIP] Unknown dataset: {ds}")
                continue
            global_idx += 1
            print(f"\n    [{global_idx}/{total_tests}] {judge_id} + {ds} ...", flush=True)
            r = run_one_test(ds, judge_id)
            model_results.append(r)
            all_results.append(r)

            icon = "OK" if r["status"] == "success" else "FAIL"
            print(f"    [{icon}] {ds} ({r['duration_s']}s)", end="")
            if r.get("acc"):
                print(f"  acc={r['acc']}", end="")
            print()
            if r["error"]:
                print(f"         Error: {r['error']}")

        # Stop vLLM
        if vllm_proc:
            print(f"\n    Stopping vLLM serve ...")
            stop_vllm(vllm_proc)

        model_elapsed = time.time() - model_start
        model_ok = sum(1 for r in model_results if r["status"] == "success")
        model_fail = sum(1 for r in model_results if r["status"] == "FAILED")
        print(f"    Model summary: {model_ok} passed, {model_fail} failed ({model_elapsed:.0f}s)")

    # ── Final summary ──────────────────────────────────────────────
    total_elapsed = time.time() - t_global
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "FAILED")

    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS: {success} passed, {failed} failed, {len(all_results)} total")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    # Per-model summary table
    print(f"\n  {'Model':<45} {'Pass':>5} {'Fail':>5}")
    print(f"  {'─' * 55}")
    for judge_id, _ in all_models:
        mr = [r for r in all_results if r["judge"] == judge_id]
        ok = sum(1 for r in mr if r["status"] == "success")
        fl = sum(1 for r in mr if r["status"] == "FAILED")
        status = "PASS" if fl == 0 else "FAIL"
        print(f"  {judge_id:<45} {ok:>5} {fl:>5}  {status}")

    if failed > 0:
        print(f"\n  Failed tests:")
        for r in all_results:
            if r["status"] == "FAILED":
                print(f"    - {r['judge']} + {r['dataset']}: {r['error']}")
        sys.exit(1)
    else:
        print(f"\n  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
