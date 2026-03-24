"""
Bias injection cache management.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from loguru import logger
from tqdm import tqdm

from ..dataset.schemas import PairwiseSample
from ..dataset.loaders import load_pairwise_jsonl
from ..models.registry import ModelRegistry


def _is_answer1_key(key: str) -> bool:
    """Normalize answer-key aliases to whether it points to answer_1."""
    return str(key).strip().lower() in {"1", "answer_1", "answer1"}


def _loose_precheck(
    original_text: str,
    candidate_text: str,
    injection_mode: str,
    pre_cfg: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Layer-1 loose semantic precheck.
    Only blocks obvious failures (empty/too short/template residue/extreme length drift).
    """
    mode = str(injection_mode or "rewrite").strip().lower()
    orig = original_text or ""
    cand = candidate_text or ""
    reasons: List[str] = []
    diag: Dict[str, Any] = {}

    if not cand.strip():
        reasons.append("empty_output")
    min_chars = int(pre_cfg.get("min_chars", 8))
    if len(cand.strip()) < min_chars:
        reasons.append("too_short")

    len_ratio = (len(cand) / max(1, len(orig))) if orig else 1.0
    diag["len_ratio"] = len_ratio
    len_min = float(pre_cfg.get("len_ratio_min", 0.35 if mode == "rewrite" else 0.45))
    len_max = float(pre_cfg.get("len_ratio_max", 2.60 if mode == "rewrite" else 1.80))
    if len_ratio < len_min or len_ratio > len_max:
        reasons.append("len_ratio_out_of_range")

    diag["reasons"] = reasons
    return len(reasons) == 0, diag


def _build_reviewer_model(guard_cfg: Dict[str, Any], bias_injector):
    reviewer_cfg = guard_cfg.get("reviewer", {}) if isinstance(guard_cfg, dict) else {}
    enabled = bool(reviewer_cfg.get("enabled", True))
    if not enabled:
        return None

    provider = str(reviewer_cfg.get("provider", "")).strip()
    model_id = (
        reviewer_cfg.get("model_id")
        or reviewer_cfg.get("model_name")
        or reviewer_cfg.get("model")
        or bias_injector.model_config.get("model_id")
        or bias_injector.model_config.get("model_name")
        or bias_injector.model_config.get("model")
    )
    if not provider and model_id:
        provider = ModelRegistry.infer_provider_from_model_id(str(model_id)) or ""
    if not provider:
        provider = str(getattr(bias_injector, "injector_type", "")).strip()
    if provider == "mock":
        return None
    if not model_id:
        return None

    overrides = reviewer_cfg.get("model_config", {}) if isinstance(reviewer_cfg.get("model_config"), dict) else {}
    try:
        return ModelRegistry.create_model(model_type=provider, model_name=str(model_id), config=overrides)
    except Exception as e:
        logger.warning(f"Semantic reviewer init failed: {e}. Fallback to precheck-only mode.")
        return None


def _review_semantic_equivalence(
    reviewer_model,
    question: str,
    original_text: str,
    candidate_text: str,
    reviewer_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, List[str]]:
    """
    Layer-2 strong-model review.
    Returns: (decision, confidence, reasons)
    """
    reviewer_cfg = reviewer_cfg or {}
    prompt_cfg = reviewer_cfg.get("prompt", {}) if isinstance(reviewer_cfg.get("prompt"), dict) else {}
    default_system_prompt = (
        "You are a strict semantic equivalence auditor for medical text rewriting. "
        "Only judge semantic meaning, not style. Output JSON only."
    )
    default_user_prompt = (
        "Task: Compare ORIGINAL and CANDIDATE.\n"
        "Decide if CANDIDATE preserves meaning without introducing new medical facts.\n\n"
        "Question:\n{question}\n\n"
        "ORIGINAL:\n{original_text}\n\n"
        "CANDIDATE:\n{candidate_text}\n\n"
        "Return JSON exactly with keys:\n"
        "{\"decision\":\"PASS|FAIL|UNCERTAIN\",\"confidence\":0.0-1.0,\"reasons\":[\"...\"]}\n"
    )
    system_prompt_template = str(prompt_cfg.get("system", default_system_prompt))
    user_prompt_template = str(prompt_cfg.get("user", default_user_prompt))
    try:
        system_prompt = system_prompt_template.format(
            question=question,
            original_text=original_text,
            candidate_text=candidate_text,
        )
    except Exception:
        system_prompt = system_prompt_template
    try:
        user_prompt = user_prompt_template.format(
            question=question,
            original_text=original_text,
            candidate_text=candidate_text,
        )
    except Exception:
        user_prompt = (
            f"{user_prompt_template}\n\nQuestion:\n{question}\n\nORIGINAL:\n{original_text}\n\n"
            f"CANDIDATE:\n{candidate_text}"
        )
    try:
        raw = reviewer_model.generate(user_prompt, system_prompt=system_prompt)
        text = (raw or "").strip()
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\\s*(.*?)\\s*```", text, re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        data = json.loads(text)
        decision = str(data.get("decision", "UNCERTAIN")).strip().upper()
        if decision not in {"PASS", "FAIL", "UNCERTAIN"}:
            decision = "UNCERTAIN"
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        reasons = [str(r) for r in reasons[:5]]
        return decision, confidence, reasons
    except Exception:
        return "UNCERTAIN", 0.5, ["review_parse_failed"]


def get_cache_path(dataset_path: str, bias_type: str, model_name: str) -> Path:
    """
    Get the cache path for bias-injected dataset.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias (e.g., "jargon_overloading", "verbosity")
        model_name: Model name used for injection (e.g., "gpt4omini", "gemini")
        
    Returns:
        Path to cached bias-injected dataset
    """
    dataset_path_obj = Path(dataset_path)
    dataset_name = dataset_path_obj.parent.name
    
    # Construct cache path: data/<dataset_name>/bias_injection/<bias_type>/<bias_type>__<model_name>.jsonl
    cache_dir = dataset_path_obj.parent / "bias_injection" / bias_type
    cache_file = f"{bias_type}__{model_name}.jsonl"
    return cache_dir / cache_file


def get_meta_path(cache_path: Path) -> Path:
    """Get the meta.json path for a cached dataset."""
    return cache_path.with_suffix(".meta.json")


def check_cache_exists(dataset_path: str, bias_type: str, model_name: str) -> bool:
    """
    Check if cached bias-injected dataset exists.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        
    Returns:
        True if cache exists, False otherwise
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    meta_path = get_meta_path(cache_path)
    
    # Both files must exist
    return cache_path.exists() and meta_path.exists()


def load_cached_bias_dataset(
    dataset_path: str,
    bias_type: str,
    model_name: str
) -> Optional[List[PairwiseSample]]:
    """
    Load cached bias-injected dataset.
    
    Args:
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        
    Returns:
        List of PairwiseSample with bias injected, or None if cache doesn't exist
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    
    if not cache_path.exists():
        logger.debug(f"Cache not found: {cache_path}")
        return None
    
    try:
        logger.info(f"Loading cached bias-injected dataset from {cache_path}")
        samples = load_pairwise_jsonl(str(cache_path))
        logger.info(f"Loaded {len(samples)} samples from cache")
        return samples
    except Exception as e:
        logger.error(f"Error loading cached dataset: {e}")
        return None


def save_bias_dataset(
    samples: List[PairwiseSample],
    dataset_path: str,
    bias_type: str,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save bias-injected dataset to cache.
    
    Args:
        samples: List of PairwiseSample with bias injected
        dataset_path: Original dataset path
        bias_type: Type of bias
        model_name: Model name used for injection
        metadata: Additional metadata to save in meta.json
        
    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(dataset_path, bias_type, model_name)
    meta_path = get_meta_path(cache_path)
    
    # Create directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSONL file
    import jsonlines
    with jsonlines.open(str(cache_path), mode='w') as writer:
        for sample in samples:
            writer.write(sample.to_dict())
    
    logger.info(f"Saved {len(samples)} bias-injected samples to {cache_path}")
    
    # Prepare metadata (no version field)
    meta_data = {
        "bias_type": bias_type,
        "model_name": model_name,
        "original_dataset": str(dataset_path),
        "num_samples": len(samples),
        "created_at": metadata.get("created_at") if metadata else None,
        "injector_config": metadata.get("injector_config") if metadata else None,
        "prompt_config": metadata.get("prompt_config") if metadata else None,
        "injection_mode": metadata.get("injection_mode") if metadata else None,
        "semantic_guard": metadata.get("semantic_guard") if metadata else None,
    }
    
    # Save meta.json
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {meta_path}")
    
    return cache_path


def apply_bias_to_samples(
    samples: List[PairwiseSample],
    bias_injector,
    inject_to: str = "non_gt",
    semantic_guard: Optional[Dict[str, Any]] = None,
) -> Tuple[List[PairwiseSample], Dict[str, Any]]:
    """
    Apply bias injection to a list of samples.
    
    Args:
        samples: Original samples
        bias_injector: BiasInjector instance
        inject_to: Where to inject bias ("non_gt" or "gt")
        
    Returns:
        (biased_samples, injection_report)
    """
    biased_samples = []
    guard_cfg = semantic_guard or {}
    guard_enabled = bool(guard_cfg.get("enabled", True))
    guard_max_attempts = max(1, int(guard_cfg.get("max_attempts", 2)))
    guard_on_fail = str(guard_cfg.get("on_fail", "use_original")).strip().lower()
    precheck_cfg = guard_cfg.get("precheck", {}) if isinstance(guard_cfg.get("precheck"), dict) else {}
    reviewer_cfg = guard_cfg.get("reviewer", {}) if isinstance(guard_cfg.get("reviewer"), dict) else {}
    fail_conf_threshold = float(reviewer_cfg.get("fail_confidence_threshold", 0.85))
    on_uncertain = str(reviewer_cfg.get("on_uncertain", "accept")).strip().lower()
    on_low_conf_fail = str(reviewer_cfg.get("on_low_conf_fail", "accept")).strip().lower()

    guard_reason_counter: Counter = Counter()
    guard_fail_count = 0
    guard_retry_count = 0
    reviewer_reject_count = 0
    precheck_reject_count = 0

    reviewer_model = _build_reviewer_model(guard_cfg, bias_injector) if guard_enabled else None
    sample_reports: List[Dict[str, Any]] = []

    def _inject_with_guard(target_text: str, question: str) -> Tuple[str, Dict[str, Any]]:
        nonlocal guard_retry_count, guard_fail_count, reviewer_reject_count, precheck_reject_count

        last_candidate = target_text
        detail: Dict[str, Any] = {
            "attempts_used": 0,
            "precheck_rejects": 0,
            "reviewer_rejects": 0,
            "reviewer_last_decision": None,
            "reviewer_last_confidence": None,
            "reviewer_last_reasons": [],
            "final_status": "accepted",
        }
        for attempt_idx in range(guard_max_attempts):
            detail["attempts_used"] = attempt_idx + 1
            if attempt_idx > 0:
                guard_retry_count += 1

            candidate = bias_injector.inject(target_text, question=question)
            last_candidate = candidate
            if not guard_enabled:
                detail["final_status"] = "accepted_no_guard"
                return candidate, detail

            # Layer-1: loose precheck
            pre_ok, pre_diag = _loose_precheck(
                original_text=target_text,
                candidate_text=candidate,
                injection_mode=getattr(bias_injector, "injection_mode", "rewrite"),
                pre_cfg=precheck_cfg,
            )
            if not pre_ok:
                precheck_reject_count += 1
                detail["precheck_rejects"] += 1
                for reason in pre_diag.get("reasons", []):
                    guard_reason_counter[f"precheck:{reason}"] += 1
                continue

            # Layer-2: strong model reviewer
            if reviewer_model is not None:
                decision, confidence, reasons = _review_semantic_equivalence(
                    reviewer_model=reviewer_model,
                    question=question,
                    original_text=target_text,
                    candidate_text=candidate,
                    reviewer_cfg=reviewer_cfg,
                )
                detail["reviewer_last_decision"] = decision
                detail["reviewer_last_confidence"] = confidence
                detail["reviewer_last_reasons"] = reasons
                if decision == "PASS":
                    detail["final_status"] = "accepted_reviewer_pass"
                    return candidate, detail
                if decision == "FAIL" and confidence >= fail_conf_threshold:
                    reviewer_reject_count += 1
                    detail["reviewer_rejects"] += 1
                    for reason in reasons or ["high_conf_fail"]:
                        guard_reason_counter[f"review:{reason}"] += 1
                    continue
                if decision == "UNCERTAIN" and on_uncertain == "retry":
                    for reason in reasons or ["uncertain_retry"]:
                        guard_reason_counter[f"review:{reason}"] += 1
                    continue
                if decision == "FAIL" and confidence < fail_conf_threshold and on_low_conf_fail == "retry":
                    for reason in reasons or ["low_conf_fail_retry"]:
                        guard_reason_counter[f"review:{reason}"] += 1
                    continue
                # UNCERTAIN or low-confidence FAIL: accept by default to avoid false rejects.
                detail["final_status"] = "accepted_reviewer_soft"
                return candidate, detail

            # Reviewer unavailable: precheck-only mode.
            detail["final_status"] = "accepted_precheck_only"
            return candidate, detail

        guard_fail_count += 1
        if guard_on_fail == "accept_last":
            detail["final_status"] = "failed_accept_last"
            return last_candidate, detail
        detail["final_status"] = "failed_use_original"
        return target_text, detail

    for sample in tqdm(samples, desc="Bias injection", total=len(samples)):
        # Create a copy to avoid modifying original
        biased_sample = PairwiseSample.from_dict(sample.to_dict())

        # Get GT and non-GT answers
        gt_answer, gt_key = sample.get_gt_answer()
        non_gt_answer, non_gt_key = sample.get_non_gt_answer()

        if inject_to == "non_gt":
            final_biased_answer, detail = _inject_with_guard(non_gt_answer, sample.question)
            if _is_answer1_key(non_gt_key):
                biased_sample.answer_1 = final_biased_answer
                target_key = "answer_1"
            else:
                biased_sample.answer_2 = final_biased_answer
                target_key = "answer_2"
        elif inject_to == "gt":
            final_biased_answer, detail = _inject_with_guard(gt_answer, sample.question)
            if _is_answer1_key(gt_key):
                biased_sample.answer_1 = final_biased_answer
                target_key = "answer_1"
            else:
                biased_sample.answer_2 = final_biased_answer
                target_key = "answer_2"
        else:
            raise ValueError(f"Unknown inject_to value: {inject_to}")

        original_target = non_gt_answer if inject_to == "non_gt" else gt_answer
        detail["sample_id"] = sample.id
        detail["inject_to"] = inject_to
        detail["target_key"] = target_key
        detail["changed"] = final_biased_answer != original_target
        detail["original_length"] = len(original_target or "")
        detail["final_length"] = len(final_biased_answer or "")
        sample_reports.append(detail)
        biased_samples.append(biased_sample)

    changed_count = sum(1 for r in sample_reports if r.get("changed"))
    failed_use_original = sum(1 for r in sample_reports if r.get("final_status") == "failed_use_original")
    failed_accept_last = sum(1 for r in sample_reports if r.get("final_status") == "failed_accept_last")
    report = {
        "total_samples": len(samples),
        "guard_enabled": guard_enabled,
        "max_attempts": guard_max_attempts,
        "on_fail": guard_on_fail,
        "injection_effective_count": changed_count,
        "injection_unchanged_count": len(samples) - changed_count,
        "guard_failed_count": guard_fail_count,
        "failed_use_original_count": failed_use_original,
        "failed_accept_last_count": failed_accept_last,
        "retries_total": guard_retry_count,
        "precheck_rejects_total": precheck_reject_count,
        "reviewer_rejects_total": reviewer_reject_count,
        "top_reasons": dict(guard_reason_counter),
        "samples": sample_reports,
    }

    if guard_enabled:
        logger.info(
            "Semantic guard summary: "
            f"failed={guard_fail_count}/{len(samples)}, retries={guard_retry_count}, "
            f"precheck_rejects={precheck_reject_count}, reviewer_rejects={reviewer_reject_count}, "
            f"top_reasons={dict(guard_reason_counter)}"
        )
    return biased_samples, report
