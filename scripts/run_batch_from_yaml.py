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
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import requests
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main as run_single_experiment
from src.utils.io import load_yaml
from src.utils.prompt_config import load_prompts_for_category

_shutdown_requested = False
_active_vllm_proc: Optional[subprocess.Popen] = None


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Interrupt received. Current run will finish, then stop.")


# ── vLLM serve auto-management ────────────────────────────────────────────────

def _is_vllm_serve_model(model_id: str) -> bool:
    return model_id.startswith("vllmserve_")


def _resolve_hf_model_name(model_id: str, models_cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve HF model name from models.yaml for a vllmserve_* model."""
    openai_section = models_cfg.get("openai", {})
    if isinstance(openai_section, dict):
        entry = openai_section.get(model_id, {})
        if isinstance(entry, dict) and "model_name" in entry:
            return entry["model_name"]
    return None


def _kill_port_occupant(port: int) -> None:
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL, text=True
        )
        for pid_str in out.strip().split("\n"):
            pid_str = pid_str.strip()
            if pid_str.isdigit() and int(pid_str) != my_pid:
                os.kill(int(pid_str), signal.SIGKILL)
                logger.info(f"Killed existing process {pid_str} on port {port}")
        time.sleep(2)
    except (subprocess.CalledProcessError, OSError):
        pass


def _start_vllm_server(
    hf_model: str,
    port: int,
    log_dir: Path,
    model_id: str,
    gpu_mem_util: float = 0.90,
    max_model_len: int = 8192,
) -> Optional[subprocess.Popen]:
    global _active_vllm_proc
    _kill_port_occupant(port)

    log_file = log_dir / f"vllm_{model_id}.log"
    # Note: --enforce-eager was previously set to avoid CUDA Graph issues on some
    # models, but it slows decode by 2-3x. All dense models in our list
    # (Llama / Qwen / Mistral / Gemma / DeepSeek-R1-Distill / BioMistral / Prometheus2-7B)
    # support CUDA Graph fine on modern vLLM, so we drop it for large speedups.
    # If a MoE / exotic model fails to start, re-add "--enforce-eager" here or
    # expose it as a per-model flag in models.yaml.
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--dtype", "auto",
        "--download-dir", str(PROJECT_ROOT / "src" / "models"),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"\n  [vLLM] Starting: {hf_model}", flush=True)
    print(f"  [vLLM] Command: {' '.join(cmd)}", flush=True)
    print(f"  [vLLM] Log: {log_file}", flush=True)
    logger.info(f"Starting vLLM serve for {model_id}: {hf_model}")

    fh = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        start_new_session=True,
    )
    proc._log_fh = fh  # type: ignore[attr-defined]
    _active_vllm_proc = proc
    return proc


def _wait_for_vllm_health(port: int, timeout: int = 600, interval: int = 5) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        if _shutdown_requested:
            return False
        attempt += 1
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                elapsed = timeout - int(deadline - time.time())
                print(f"  [vLLM] Healthy after {elapsed}s ({attempt} checks)", flush=True)
                logger.info(f"vLLM server healthy after {elapsed}s")
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        remaining = int(deadline - time.time())
        if attempt % 12 == 0:
            print(f"  [vLLM] Still waiting... ({remaining}s remaining)", flush=True)
        time.sleep(interval)
    return False


def _stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    global _active_vllm_proc
    if proc is None:
        return
    print("  [vLLM] Stopping server (process group)...", flush=True)
    logger.info("Stopping vLLM server")
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass

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

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
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
    _active_vllm_proc = None
    time.sleep(3)
    print("  [vLLM] Server stopped (GPU memory released).", flush=True)
    logger.info("vLLM server stopped")


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
    inject_to: Optional[str] = None,
    semantic_guard_override: Optional[Dict[str, Any]] = None,
    original_judgments_path: Optional[str] = None,
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
    if isinstance(inject_to, str) and inject_to.strip():
        target = inject_to.strip().lower()
        if target not in {"gt", "non_gt"}:
            raise ValueError(f"Unsupported inject_to '{inject_to}'. Use 'gt' or 'non_gt'.")
        cfg["bias"]["inject_to"] = target
    for k in ("injector_type", "model_id", "model_name"):
        cfg["bias"].pop(k, None)

    # Avoid repeated original-judgment calls across many bias runs.
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["compute_original_acc"] = compute_original_acc
    cfg["evaluation"]["compute_bias_metrics"] = True

    # Pairwise position-bias correction: evaluate GT@A and GT@B, then average RR/CR/acc_biased.
    cfg["evaluation"]["position_debias_pairwise"] = True

    # Pass cached original judgments path so RR can be computed as true
    # same-choice rate even when compute_original_acc is False.
    if original_judgments_path:
        cfg["evaluation"]["original_judgments_path"] = original_judgments_path

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


def _normalize_key_part(value: Any) -> str:
    return str(value).strip().lower()


def _build_eval_key(
    dataset_name: str,
    bias: str,
    mode: str,
    judge_model: str,
    injector_model: str,
    inject_to: str,
) -> Tuple[str, str, str, str, str, str]:
    return (
        _normalize_key_part(dataset_name),
        _normalize_key_part(bias),
        _normalize_key_part(mode),
        _normalize_key_part(judge_model),
        _normalize_key_part(injector_model),
        _normalize_key_part(inject_to),
    )


def _load_completed_eval_keys(summary_jsonl_path: Path) -> Set[Tuple[str, str, str, str, str, str]]:
    """
    Load successful eval keys from outputs/results_summary.jsonl.

    File is written as pretty-printed JSON objects separated by whitespace,
    so we parse by repeatedly raw-decoding JSON from text.
    """
    completed: Set[Tuple[str, str, str, str, str, str]] = set()
    if not summary_jsonl_path.exists():
        return completed
    try:
        raw_text = summary_jsonl_path.read_text(encoding="utf-8")
    except Exception:
        return completed

    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw_text)
    while idx < n:
        while idx < n and raw_text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, next_idx = decoder.raw_decode(raw_text, idx)
        except Exception:
            # Stop on parse error (likely trailing partial write).
            break
        idx = next_idx
        if not isinstance(obj, dict):
            continue

        # Explicit completion marker (newer records): required for safety.
        # This avoids counting interrupted/partial runs as completed.
        if obj.get("completed") is not True:
            continue

        metrics = obj.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        # Treat rows with valid core metrics as completed eval runs.
        if metrics.get("rr") is None or metrics.get("cr") is None or metrics.get("accuracy_biased") is None:
            continue

        key = _build_eval_key(
            dataset_name=obj.get("dataset_name", ""),
            bias=obj.get("bias_type", ""),
            mode=obj.get("bias_injection_mode", ""),
            judge_model=obj.get("judge_model_id", ""),
            injector_model=obj.get("bias_injector_model_id", ""),
            inject_to=obj.get("bias_inject_to", "non_gt"),
        )
        completed.add(key)
    return completed


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
    base_cfg = load_yaml(str(base_config_path))
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))
    dry_run = bool(batch_cfg.get("dry_run", False))
    reuse_original_per_judge = bool(batch_cfg.get("reuse_original_per_judge", True))

    # vLLM serve auto-management config
    vllm_auto_serve = bool(batch_cfg.get("vllm_auto_serve", True))
    vllm_port = int(batch_cfg.get("vllm_port", 8000))
    vllm_max_model_len = int(batch_cfg.get("vllm_max_model_len", 8192))
    vllm_gpu_mem_util = float(batch_cfg.get("vllm_gpu_memory_utilization", 0.90))
    vllm_startup_timeout = int(batch_cfg.get("vllm_startup_timeout", 600))
    models_cfg = load_yaml(str(PROJECT_ROOT / "configs" / "models.yaml"))
    skip_existing_evals = bool(batch_cfg.get("skip_existing_evals", False))
    inject_to_override = str(batch_cfg.get("inject_to", "")).strip().lower()
    if inject_to_override and inject_to_override not in {"gt", "non_gt"}:
        raise ValueError("batch.inject_to must be 'gt' or 'non_gt'")
    base_inject_to = str((base_cfg.get("bias", {}) or {}).get("inject_to", "non_gt")).strip().lower()
    if base_inject_to not in {"gt", "non_gt"}:
        base_inject_to = "non_gt"
    effective_inject_to_default = inject_to_override or base_inject_to

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
    completed_eval_keys = _load_completed_eval_keys(PROJECT_ROOT / "outputs" / "results_summary.jsonl") if skip_existing_evals else set()

    logger.add(log_dir / "yaml_batch.log", level="INFO")
    logger.info("=" * 72)
    logger.info("YAML batch evaluation start")
    logger.info(f"Batch config: {batch_cfg_path}")
    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Datasets: {[x['dataset_name'] for x in dataset_run_items]}")
    logger.info(f"Judge models: {judge_models}")
    logger.info(f"Bias injector models: {injector_models}")
    logger.info(f"Skip existing evals: {skip_existing_evals}")
    for item in dataset_run_items:
        logger.info(
            f"Dataset={item['dataset_name']} category={item['dataset_category']} "
            f"biases={item['selected_biases']} modes={sorted(list(set([m for _, m in item['combos']])))}"
        )
    logger.info(f"Reuse original per judge: {reuse_original_per_judge}")
    vllm_model_ids = [m for m in judge_models if _is_vllm_serve_model(m)]
    if vllm_model_ids and vllm_auto_serve:
        logger.info(
            f"vLLM auto-serve: port={vllm_port}, max_model_len={vllm_max_model_len}, "
            f"gpu_mem={vllm_gpu_mem_util}, timeout={vllm_startup_timeout}s, "
            f"models={vllm_model_ids}"
        )
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
    # Cache path to judge_raw_original.jsonl for each (judge, dataset) pair,
    # so subsequent bias runs can load it for true RR computation.
    original_judgments_cache: Dict[Tuple[str, str], str] = {}
    batch_start_time = time.time()
    success_count = 0
    failed_count = 0
    skipped_count = 0

    def _print_progress(idx: int, total: int, tag: str, status: str, elapsed_s: float = 0):
        """Print a visible progress line to terminal."""
        pct = idx / total * 100 if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * idx / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)
        batch_elapsed = time.time() - batch_start_time
        if idx > 0:
            eta_s = batch_elapsed / idx * (total - idx)
            eta_str = f"ETA {eta_s/60:.0f}m" if eta_s > 60 else f"ETA {eta_s:.0f}s"
        else:
            eta_str = "ETA --"
        elapsed_str = f"{batch_elapsed/60:.1f}m" if batch_elapsed > 60 else f"{batch_elapsed:.0f}s"
        print(
            f"\r[{bar}] {idx}/{total} ({pct:.0f}%) | "
            f"{status} | elapsed {elapsed_str} {eta_str}",
            flush=True,
        )

    print(f"\n{'=' * 72}")
    print(f"  BATCH PROGRESS: {total_combo_count} total runs planned")
    print(f"{'=' * 72}")

    _prev_vllm_base_url = os.environ.get("VLLM_BASE_URL")

    for judge_model in judge_models:
        # ── vLLM auto-serve: start server for vllmserve_* models ──────────
        vllm_proc: Optional[subprocess.Popen] = None
        if vllm_auto_serve and _is_vllm_serve_model(judge_model):
            hf_name = _resolve_hf_model_name(judge_model, models_cfg)
            if not hf_name:
                logger.error(f"Cannot resolve HF model name for '{judge_model}' in models.yaml. Skipping.")
                print(f"\n  [vLLM] ERROR: unknown model '{judge_model}', skipping.", flush=True)
                # count all combos for this model as failed
                for injector_spec in injector_models:
                    for ds_item in dataset_run_items:
                        for _ in ds_item["combos"]:
                            run_index += 1
                            failed_count += 1
                continue

            vllm_log_dir = log_dir / "vllm_logs"
            vllm_log_dir.mkdir(parents=True, exist_ok=True)
            vllm_proc = _start_vllm_server(
                hf_model=hf_name,
                port=vllm_port,
                log_dir=vllm_log_dir,
                model_id=judge_model,
                gpu_mem_util=vllm_gpu_mem_util,
                max_model_len=vllm_max_model_len,
            )
            if vllm_proc is None:
                logger.error(f"Failed to start vLLM for {judge_model}. Skipping.")
                for injector_spec in injector_models:
                    for ds_item in dataset_run_items:
                        for _ in ds_item["combos"]:
                            run_index += 1
                            failed_count += 1
                continue

            print(f"  [vLLM] Waiting for health check (timeout={vllm_startup_timeout}s)...", flush=True)
            healthy = _wait_for_vllm_health(vllm_port, timeout=vllm_startup_timeout)
            if not healthy:
                logger.error(f"vLLM health check timed out for {judge_model}. Skipping.")
                print(f"  [vLLM] Health check FAILED for {judge_model}.", flush=True)
                _stop_vllm_server(vllm_proc)
                vllm_proc = None
                for injector_spec in injector_models:
                    for ds_item in dataset_run_items:
                        for _ in ds_item["combos"]:
                            run_index += 1
                            failed_count += 1
                continue

            os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{vllm_port}/v1"
            logger.info(f"vLLM ready for {judge_model}, VLLM_BASE_URL={os.environ['VLLM_BASE_URL']}")

        try:
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
                        print(
                            f"\n>>> [{run_index}/{total_combo_count}] "
                            f"dataset={ds_name}  judge={judge_model}  bias={bias}({mode})",
                            flush=True,
                        )

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
                        run_t0 = time.time()

                        try:
                            eval_key = _build_eval_key(
                                dataset_name=ds_name,
                                bias=bias,
                                mode=mode,
                                judge_model=judge_model,
                                injector_model=injector_model,
                                inject_to=effective_inject_to_default,
                            )
                            if skip_existing_evals and eval_key in completed_eval_keys:
                                result["status"] = "skipped_existing_eval"
                                skipped_count += 1
                                print(f"    SKIP (already completed)", flush=True)
                                _print_progress(run_index, total_combo_count, run_tag, f"ok={success_count} fail={failed_count} skip={skipped_count}")
                                logger.info(
                                    f"Skip existing eval: dataset={ds_name}, bias={bias}, mode={mode}, "
                                    f"judge={judge_model}, injector={injector_model}, inject_to={effective_inject_to_default}"
                                )
                                continue

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

                            cached_orig_path = original_judgments_cache.get((judge_model, ds_name))
                            run_cfg = _build_run_config(
                                base_cfg,
                                ds_name,
                                judge_model,
                                injector_model,
                                bias,
                                mode,
                                compute_original_acc=compute_original_acc,
                                inject_to=inject_to_override or None,
                                semantic_guard_override=semantic_guard_override,
                                original_judgments_path=cached_orig_path if not compute_original_acc else None,
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
                                    success_count += 1
                                    if compute_original_acc:
                                        original_done_for_judge_dataset.add((judge_model, ds_name))
                                        if isinstance(run_output, dict):
                                            _run_dir = run_output.get("run_dir")
                                            if _run_dir:
                                                _orig_path = Path(_run_dir) / "judge_raw_original.jsonl"
                                                if _orig_path.exists():
                                                    original_judgments_cache[(judge_model, ds_name)] = str(_orig_path)
                                                    logger.info(
                                                        f"Cached original judgments path for "
                                                        f"({judge_model}, {ds_name}): {_orig_path}"
                                                    )
                                    if skip_existing_evals:
                                        completed_eval_keys.add(eval_key)
                        except KeyboardInterrupt:
                            result["status"] = "interrupted"
                            result["error"] = "KeyboardInterrupt"
                            logger.warning(f"KeyboardInterrupt: {run_tag}. Stopping remaining batch runs.")
                            stop_now = True
                        except Exception as e:
                            result["status"] = "failed"
                            result["error"] = str(e)
                            result["traceback"] = traceback.format_exc()
                            failed_count += 1
                            logger.error(f"Failed: {run_tag} | {e}")
                            if not continue_on_error:
                                results.append(result)
                                stop_now = True
                                break
                        finally:
                            signal.signal(signal.SIGINT, _signal_handler)
                            signal.signal(signal.SIGTERM, _signal_handler)
                            result["end_time"] = datetime.now().isoformat()
                            results.append(result)
                            run_elapsed = time.time() - run_t0
                            status_icon = {"success": "OK", "failed": "FAIL", "interrupted": "INT", "dry_run": "DRY", "skipped_existing_eval": "SKIP"}.get(result["status"], "?")
                            print(f"    {status_icon} ({run_elapsed:.1f}s)", flush=True)
                            _print_progress(run_index, total_combo_count, run_tag, f"ok={success_count} fail={failed_count} skip={skipped_count}")
                        if stop_now:
                            break
                    if stop_now:
                        break
                if stop_now:
                    break
        finally:
            if vllm_proc is not None:
                _stop_vllm_server(vllm_proc)
                vllm_proc = None
            if _prev_vllm_base_url is not None:
                os.environ["VLLM_BASE_URL"] = _prev_vllm_base_url
            else:
                os.environ.pop("VLLM_BASE_URL", None)
        if stop_now:
            break

    total_elapsed = time.time() - batch_start_time
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry = sum(1 for r in results if r["status"] == "dry_run")
    skipped = sum(1 for r in results if r["status"] == "skipped_existing_eval")

    # Print final summary to terminal
    elapsed_str = f"{total_elapsed/60:.1f}min" if total_elapsed > 60 else f"{total_elapsed:.0f}s"
    print(f"\n{'=' * 72}")
    print(f"  BATCH COMPLETE  |  Total time: {elapsed_str}")
    print(f"  Success: {success}  |  Failed: {failed}  |  Skipped: {skipped}  |  Dry: {dry}")
    print(f"{'=' * 72}\n")
    if failed > 0:
        print("  Failed runs:")
        for r in results:
            if r["status"] == "failed":
                print(f"    - [{r['index']}] {r['dataset']} / {r['bias']}({r['mode']}) / {r['judge_model']}")
                if r.get("error"):
                    print(f"      Error: {str(r['error'])[:120]}")
        print()

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
        "skipped_existing_eval": skipped,
        "results": results,
    }
    summary_path = log_dir / "yaml_batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 72)
    logger.info(f"Done. success={success}, failed={failed}, dry_run={dry}, skipped_existing_eval={skipped}")
    logger.info(f"Summary: {summary_path}")
    logger.info(
        "Each run is executed by src.main, so outputs are stored under outputs/ and appended to results_summary.*"
    )
    logger.info("=" * 72)

    if failed > 0 and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

