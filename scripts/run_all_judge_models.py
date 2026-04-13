#!/home1/zrui6736/miniconda3/envs/Judgellm/bin/python
"""
Run all judge models × all datasets × all biases.

Open-source models are served via vLLM (auto-started / stopped).
Already-completed evaluations (in outputs/results_summary.csv) are skipped.

Usage
-----
  # Full auto — start/stop vLLM for each open-source model:
  python scripts/run_all_judge_models.py --port 8000 --gpu-ids 0

  # Run one model only:
  python scripts/run_all_judge_models.py --model vllmserve_gemma3_4b

  # Manual vLLM — user manages the server; script only runs eval:
  python scripts/run_all_judge_models.py --no-auto-vllm --model vllmserve_gemma3_4b
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =====================================================================
# Evaluation matrix
# =====================================================================

DATASETS = [
    "medical_eval_sphere",
    "mediq_askdocs",
    "counselbench",
    "medval_bench",
    "medaesqa",
]

JUDGE_MODELS = [
    {"judge_id": "gpt52",                                  "type": "api"},
    {"judge_id": "vllmserve_medgemma_4b",                  "type": "vllm", "hf_model": "google/medgemma-4b-it"},
    {"judge_id": "vllmserve_biomistral_7b",                "type": "vllm", "hf_model": "BioMistral/BioMistral-7B"},
    {"judge_id": "vllmserve_prometheus2_7b",               "type": "vllm", "hf_model": "prometheus-eval/prometheus-7b-v2.0"},
    {"judge_id": "vllmserve_gemma3_4b",                    "type": "vllm", "hf_model": "google/gemma-3-4b-it"},
    {"judge_id": "vllmserve_qwen3_4b_instruct",           "type": "vllm", "hf_model": "Qwen/Qwen3-4B-Instruct-2507"},
    {"judge_id": "vllmserve_llama32_3b_instruct",         "type": "vllm", "hf_model": "meta-llama/Llama-3.2-3B-Instruct"},
    {"judge_id": "vllmserve_deepseek_r1_distill_qwen_7b", "type": "vllm", "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    {"judge_id": "vllmserve_deepseek_r1_distill_llama_8b","type": "vllm", "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
]

BIAS_INJECTOR = "gpt52"
MODE = "word"
INJECT_TO = "non_gt"

# =====================================================================
# CSV skip logic
# =====================================================================

def load_completed_from_csv(csv_path: Path) -> Set[Tuple[str, str, str]]:
    """Return {(dataset, bias_type, judge_id)} already in results_summary.csv."""
    done: Set[Tuple[str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row.get("dataset", "").strip()
            bias = row.get("bias type", "").strip()
            judge = row.get("judge llm", "").strip()
            if ds and bias and judge:
                done.add((ds, bias, judge))
    return done


# =====================================================================
# Bias plan
# =====================================================================

def load_bias_plan() -> Dict[str, List[str]]:
    """Return {dataset: [bias, ...]} from configs/dataset_bias_plan.yaml."""
    path = PROJECT_ROOT / "configs" / "dataset_bias_plan.yaml"
    with open(path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    result: Dict[str, List[str]] = {}
    for ds, info in (plan.get("dataset_biases") or {}).items():
        if isinstance(info, dict):
            result[ds] = info.get("biases", [])
        elif isinstance(info, list):
            result[ds] = info
    return result


def get_pending(
    judge_id: str,
    bias_plan: Dict[str, List[str]],
    done: Set[Tuple[str, str, str]],
) -> Dict[str, List[str]]:
    """Return {dataset: [pending_biases]} for *judge_id*."""
    pending: Dict[str, List[str]] = {}
    for ds in DATASETS:
        remaining = [b for b in bias_plan.get(ds, []) if (ds, b, judge_id) not in done]
        if remaining:
            pending[ds] = remaining
    return pending


# =====================================================================
# Temp config generation
# =====================================================================

def write_temp_configs(
    judge_id: str,
    pending: Dict[str, List[str]],
    run_dir: Path,
) -> Path:
    """Write a temp batch config + bias plan; return batch config path."""
    # ---- temp bias plan (only pending biases) ----
    plan_data = {"dataset_biases": {ds: {"biases": biases} for ds, biases in pending.items()}}
    plan_path = run_dir / "bias_plan.yaml"
    with open(plan_path, "w", encoding="utf-8") as f:
        yaml.dump(plan_data, f, allow_unicode=True, default_flow_style=False)

    # ---- temp batch config ----
    batch = {
        "batch": {
            "base_experiment_config": "configs/experiment.yaml",
            "dataset_names": list(pending.keys()),
            "judge_models": [judge_id],
            "bias_injector_models": [BIAS_INJECTOR],
            "dataset_bias_plan": str(plan_path),
            "modes": [MODE],
            "inject_to": INJECT_TO,
            "continue_on_error": True,
            "dry_run": False,
            "reuse_original_per_judge": True,
            "skip_existing_evals": False,
            "log_dir": str(run_dir),
        }
    }
    batch_path = run_dir / "batch_config.yaml"
    with open(batch_path, "w", encoding="utf-8") as f:
        yaml.dump(batch, f, allow_unicode=True, default_flow_style=False)

    return batch_path


# =====================================================================
# vLLM serve lifecycle
# =====================================================================

_vllm_proc: Optional[subprocess.Popen] = None
_vllm_log_fh = None                       # keep log file-handle for cleanup

GPU_COOLDOWN_SECONDS = 15                  # wait after killing vLLM so GPU memory is freed


def start_vllm(hf_model: str, port: int, gpu_ids: str,
               log_dir: Optional[Path] = None) -> subprocess.Popen:
    global _vllm_proc, _vllm_log_fh
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model,
        "--port", str(port),
        "--trust-remote-code",
    ]
    out_dir = log_dir or (PROJECT_ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each model gets its own log so previous logs aren't overwritten
    safe_name = hf_model.replace("/", "_")
    log_path = out_dir / f"vllm_serve_{safe_name}.log"
    log_fh = open(log_path, "w")
    _vllm_log_fh = log_fh
    print(f"  Starting vLLM serve: {hf_model}")
    print(f"  Log → {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    _vllm_proc = proc
    return proc


def vllm_is_alive(proc: Optional[subprocess.Popen]) -> bool:
    """Return True if the vLLM process is still running."""
    return proc is not None and proc.poll() is None


def wait_vllm_ready(port: int, timeout: int = 600,
                    proc: Optional[subprocess.Popen] = None) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        # If the vLLM process crashed, stop waiting immediately
        if proc is not None and proc.poll() is not None:
            print(f"  vLLM process exited with code {proc.returncode} during startup")
            return False
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_vllm(proc: Optional[subprocess.Popen]) -> None:
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
    # Let GPU memory fully release before starting next model
    print(f"  Waiting {GPU_COOLDOWN_SECONDS}s for GPU memory release …")
    time.sleep(GPU_COOLDOWN_SECONDS)


def _cleanup_handler(signum, _frame):
    global _vllm_proc
    if _vllm_proc:
        print("\nInterrupt received — stopping vLLM serve …")
        stop_vllm(_vllm_proc)
    sys.exit(1)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="vLLM serve port (default: 8000)")
    parser.add_argument("--gpu-ids", type=str, default="0", help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Only run this judge model")
    parser.add_argument("--no-auto-vllm", action="store_true",
                        help="Do NOT auto-start vLLM; user manages the server")
    parser.add_argument("--vllm-timeout", type=int, default=600,
                        help="Seconds to wait for vLLM readiness (default: 600)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)

    csv_path = PROJECT_ROOT / "outputs" / "results_summary.csv"
    bias_plan = load_bias_plan()

    # ---- select models ----
    models = JUDGE_MODELS
    if args.model:
        models = [m for m in models if m["judge_id"] == args.model]
        if not models:
            all_ids = [m["judge_id"] for m in JUDGE_MODELS]
            print(f"ERROR: model '{args.model}' not found. Available: {all_ids}")
            sys.exit(1)

    # ---- compute full pending matrix ----
    done = load_completed_from_csv(csv_path)
    total_all = sum(len(bias_plan.get(ds, [])) for ds in DATASETS) * len(models)
    total_done = 0
    for m in models:
        for ds in DATASETS:
            for b in bias_plan.get(ds, []):
                if (ds, b, m["judge_id"]) in done:
                    total_done += 1
    total_pending = total_all - total_done
    pct_done = total_done / total_all * 100 if total_all > 0 else 0
    bar_len = 40
    filled = int(bar_len * total_done / total_all) if total_all > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\n{'=' * 64}")
    print(f"  Evaluation matrix: {len(DATASETS)} datasets x {len(models)} judges")
    print(f"  [{bar}] {pct_done:.0f}%")
    print(f"  Completed: {total_done} / {total_all}  |  Pending: {total_pending}")
    print(f"{'=' * 64}\n")

    success_models = 0
    failed_models = 0
    skipped_models = 0
    all_start_time = time.time()

    for model_idx, model_info in enumerate(models, 1):
        judge_id = model_info["judge_id"]
        model_type = model_info["type"]
        hf_model = model_info.get("hf_model")

        # Re-read CSV every iteration — previous model run may have added rows
        done = load_completed_from_csv(csv_path)
        pending = get_pending(judge_id, bias_plan, done)
        n_pending = sum(len(v) for v in pending.values())

        if not pending:
            print(f"  [{model_idx}/{len(models)}] {judge_id}: All done -- skipping")
            skipped_models += 1
            continue

        elapsed_all = time.time() - all_start_time
        elapsed_str = f"{elapsed_all/60:.1f}m" if elapsed_all > 60 else f"{elapsed_all:.0f}s"
        print(f"\n{'=' * 64}")
        print(f"  [{model_idx}/{len(models)}] {judge_id}  |  {n_pending} pending evals  |  elapsed {elapsed_str}")
        for ds, biases in pending.items():
            print(f"    {ds}: {biases}")
        print(f"{'=' * 64}")

        # ---- prepare run dir (before vLLM so logs go here) ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "outputs" / "batch_logs" / f"auto_{judge_id}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---- vLLM lifecycle ----
        vllm_proc = None
        if model_type == "vllm" and not args.no_auto_vllm:
            vllm_proc = start_vllm(hf_model, args.port, args.gpu_ids, log_dir=run_dir)
            base_url = f"http://127.0.0.1:{args.port}/v1"
            os.environ["VLLM_BASE_URL"] = base_url
            print(f"  Waiting for vLLM to be ready (timeout={args.vllm_timeout}s) …")
            if not wait_vllm_ready(args.port, args.vllm_timeout, proc=vllm_proc):
                print(f"  ERROR: vLLM failed to start for {hf_model}")
                stop_vllm(vllm_proc)
                failed_models += 1
                continue
            print(f"  vLLM ready at {base_url}")
        elif model_type == "vllm" and args.no_auto_vllm:
            if not os.environ.get("VLLM_BASE_URL"):
                print(f"  WARNING: VLLM_BASE_URL not set — eval may fail")

        # ---- generate configs & run batch ----
        batch_path = write_temp_configs(judge_id, pending, run_dir)

        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_batch_from_yaml.py"),
                "--batch-config", str(batch_path),
            ]
            print(f"  Running batch …")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

            # Check if vLLM died mid-evaluation
            if vllm_proc and not vllm_is_alive(vllm_proc):
                print(f"  WARNING: vLLM process died during evaluation (exit code {vllm_proc.returncode})")

            if result.returncode == 0:
                print(f"  [{judge_id}] batch completed successfully")
                success_models += 1
            else:
                print(f"  [{judge_id}] batch exited with code {result.returncode}")
                failed_models += 1
        except Exception as e:
            print(f"  [{judge_id}] ERROR: {e}")
            failed_models += 1
        finally:
            if vllm_proc:
                print(f"  Stopping vLLM serve …")
                stop_vllm(vllm_proc)

    # ---- final summary ----
    done_final = load_completed_from_csv(csv_path)
    total_elapsed = time.time() - all_start_time
    elapsed_str = f"{total_elapsed/3600:.1f}h" if total_elapsed > 3600 else (f"{total_elapsed/60:.1f}m" if total_elapsed > 60 else f"{total_elapsed:.0f}s")
    pct_final = len(done_final) / total_all * 100 if total_all > 0 else 0
    bar_filled = int(40 * len(done_final) / total_all) if total_all > 0 else 0
    bar_final = "=" * bar_filled + "-" * (40 - bar_filled)
    print(f"\n{'=' * 64}")
    print(f"  FINISHED  |  Total time: {elapsed_str}")
    print(f"  [{bar_final}] {pct_final:.0f}%")
    print(f"  Models: success={success_models}  failed={failed_models}  skipped={skipped_models}")
    print(f"  CSV rows: {len(done_final)} / {total_all}")
    print(f"{'=' * 64}")

    if failed_models > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
