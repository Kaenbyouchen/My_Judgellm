"""
Pipeline for pairwise evaluation.
"""
import datetime
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from loguru import logger

from ..dataset.schemas import PairwiseSample
from ..dataset.loaders import load_pairwise_jsonl
from ..dataset.preprocess import validate_pairwise_samples, get_gt_statistics
from ..bias.injector import BiasInjector
from ..bias.cache import check_cache_exists, load_cached_bias_dataset, save_bias_dataset, apply_bias_to_samples
from ..judge.judge_runner import create_judge
from ..judge.base import JudgeResult
from ..metrics.pairwise_metrics import compute_accuracy_original, compute_accuracy_biased, compute_rr, compute_cr
from ..metrics.reports import save_metrics_json, save_metrics_csv, save_judgments_jsonl, save_judgment_jsonl_single, print_metrics_summary
from ..utils.resume import (
    check_resume_available,
    load_completed_sample_ids,
    filter_completed_samples,
    load_existing_judgments
)

# Global flag for graceful shutdown
_shutdown_requested = False


def _is_answer1_key(key: str) -> bool:
    """Normalize answer-key aliases to whether it points to answer_1."""
    return str(key).strip().lower() in {"1", "answer_1", "answer1"}


def _answer_by_key(sample: PairwiseSample, key: str) -> str:
    """Read answer text by normalized key alias."""
    return sample.answer_1 if _is_answer1_key(key) else sample.answer_2


def _iter_chunks(items, size: int):
    """Yield fixed-size chunks from a sequence-like object."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("\n" + "="*60)
    logger.warning("⚠️  Interrupt signal received (Ctrl+C or SIGTERM)")
    logger.warning("⚠️  Current progress will be saved. You can resume later by running the same command.")
    logger.warning("="*60)
    # Don't exit immediately - let the current iteration finish and save


def run_pairwise_evaluation(
    data_path: str,
    bias_type: str,
    bias_variant_name: str,
    bias_enabled: bool,
    injector_type: str,
    judge_type: str,
    judge_model_name: str,
    judge_config: Dict[str, Any],
    compute_original_acc: bool,
    compute_bias_metrics: bool,
    run_dir: str,
    injector_model_config: Optional[Dict[str, Any]] = None,
    injector_system_prompt: Optional[str] = None,
    injector_user_template: Optional[str] = None,
    injection_mode: str = "rewrite",
    judge_system_prompt: Optional[str] = None,
    judge_user_template: Optional[str] = None,
    inject_to: str = "non_gt",
    use_cache: bool = True,
    semantic_guard: Optional[Dict[str, Any]] = None,
    position_debias_pairwise: bool = True,
    request_batch_size: int = 1,
    original_judgments_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run pairwise evaluation pipeline.
    
    Args:
        data_path: Path to pairwise data file
        bias_type: Type of bias to inject
        bias_variant_name: Cache name for bias variant (e.g., "jargon_overloading" or "word_jargon_overloading")
        bias_enabled: Whether to enable bias injection
        injector_type: Type of bias injector ("mock", "openai", "vllm")
        injector_model_config: Model config for bias injector (if using AI-based injection)
        injector_system_prompt: Optional system prompt for bias injection
        injector_user_template: Optional user prompt template for bias injection
        injection_mode: "rewrite" for full rewrite, "word" for word/phrase replacement
        judge_system_prompt: Optional system prompt for judge pairwise prompt
        judge_user_template: Optional user prompt template for judge pairwise prompt
        judge_type: Type of judge ("mock", "openai", "vllm")
        judge_model_name: Name of judge model
        judge_config: Judge model configuration
        compute_original_acc: Whether to compute accuracy on original R1 vs R2
        compute_bias_metrics: Whether to compute RR/CR on R1 vs biased_R2
        run_dir: Run directory for all output files
        semantic_guard: Optional semantic guard config for bias injection
        position_debias_pairwise: If true, evaluate biased set twice with GT@A and GT@B and average metrics
        
    Returns:
        Dictionary with all metrics and results
    """
    logger.info("Starting pairwise evaluation pipeline")
    inject_to = str(inject_to).strip().lower()
    if inject_to not in {"non_gt", "gt"}:
        raise ValueError(f"Invalid inject_to='{inject_to}'. Expected one of: non_gt, gt")
    request_batch_size = max(1, int(request_batch_size or 1))
    logger.info(
        "Pairwise protocol: "
        f"inject_to={inject_to}, "
        f"position_debias_pairwise={position_debias_pairwise}, "
        f"compute_original_acc={compute_original_acc}, "
        f"compute_bias_metrics={compute_bias_metrics}, "
        f"request_batch_size={request_batch_size}"
    )
    
    # Setup signal handlers for graceful shutdown
    global _shutdown_requested
    _shutdown_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Check for resume
    run_dir_path = Path(run_dir)
    resume_info = check_resume_available(run_dir_path)
    
    if resume_info["available"]:
        logger.info("=" * 60)
        logger.info("🔄 Resume mode detected!")
        logger.info(f"  Original judgments completed: {resume_info['original_completed']}")
        logger.info(f"  Bias judgments completed: {resume_info['bias_completed']}")
        logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    samples = load_pairwise_jsonl(data_path)
    samples = validate_pairwise_samples(samples)
    
    if not samples:
        raise ValueError("No valid samples loaded")
    
    # Print statistics
    stats = get_gt_statistics(samples)
    logger.info(f"Loaded {stats['total']} samples")
    logger.info(f"GT coverage: {stats['gt_coverage']:.2%} ({stats['with_gt']} with GT, {stats['without_gt']} without)")
    
    if stats['without_gt'] > 0:
        logger.warning("Some samples lack GT labels. Using placeholder assumption: answer_1 is GT")
    
    # Initialize bias injector
    bias_injector = None
    if bias_enabled:
        logger.info(f"Initializing bias injector: type={bias_type}, injector={injector_type}")
        bias_injector = BiasInjector(
            bias_type=bias_type,
            injector_type=injector_type,
            model_config=injector_model_config or {},
            system_prompt=injector_system_prompt,
            prompt_template=injector_user_template,
            injection_mode=injection_mode,
        )
    
    # Initialize judge
    logger.info(f"Initializing judge: type={judge_type}, model={judge_model_name}")
    judge = create_judge(
        judge_type=judge_type,
        model_name=judge_model_name,
        config=judge_config,
        system_prompt=judge_system_prompt,
        user_template=judge_user_template,
    )
    
    # Run evaluation
    results = {
        "original_judgments": [],
        "bias_judgments": [],
        "biased_answers": [],
        "samples": []
    }
    
    # Load existing judgments if resuming
    existing_original_judgments = []
    existing_bias_judgments = []
    completed_original_ids = set()
    completed_bias_ids = set()
    
    if resume_info["available"]:
        if resume_info["original_file"]:
            existing_original_judgments = load_existing_judgments(run_dir_path, "original")
            logger.info(f"Loaded {len(existing_original_judgments)} existing original judgments")
            completed_original_ids = load_completed_sample_ids(run_dir_path, "original")
        if resume_info["bias_file"]:
            existing_bias_judgments = load_existing_judgments(run_dir_path, "bias")
            logger.info(f"Loaded {len(existing_bias_judgments)} existing bias judgments")
            completed_bias_ids = load_completed_sample_ids(run_dir_path, "bias")
    
    # Step 1: Evaluate original R1 vs R2 (for Acc)
    if compute_original_acc:
        logger.info("Evaluating original R1 vs R2...")
        
        # Prepare output file path for incremental saving
        original_judgments_path = run_dir_path / "judge_raw_original.jsonl"
        
        # Filter completed samples
        remaining_samples_original, completed_samples_original = filter_completed_samples(samples, completed_original_ids)
        
        if completed_original_ids:
            logger.info(f"Resuming: {len(completed_samples_original)} samples already completed, {len(remaining_samples_original)} remaining")
            # Add existing judgments to results
            results["original_judgments"].extend(existing_original_judgments)
        
        original_judgments = []
        # Convert existing judgments to JudgeResult objects for metrics calculation
        for existing_judgment in existing_original_judgments:
            original_judgments.append(JudgeResult.from_dict({
                "winner": existing_judgment.get("winner", "tie"),
                "score_A": existing_judgment.get("score_A"),
                "score_B": existing_judgment.get("score_B"),
                "explanation": existing_judgment.get("explanation"),
                "raw": existing_judgment.get("raw", {})
            }))
        
        pbar = tqdm(total=len(remaining_samples_original), desc="Original judgment")
        try:
            for sample_batch in _iter_chunks(remaining_samples_original, request_batch_size):
                # Check for shutdown request
                if _shutdown_requested:
                    logger.warning("Shutdown requested. Saving progress and exiting gracefully...")
                    break

                if request_batch_size == 1:
                    batch_judgments = [
                        judge.judge_pairwise(
                            question=sample_batch[0].question,
                            answer_A=sample_batch[0].answer_1,
                            answer_B=sample_batch[0].answer_2,
                        )
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=request_batch_size) as executor:
                        futures = [
                            executor.submit(
                                judge.judge_pairwise,
                                question=s.question,
                                answer_A=s.answer_1,
                                answer_B=s.answer_2,
                            )
                            for s in sample_batch
                        ]
                        batch_judgments = [f.result() for f in futures]

                for sample, judgment in zip(sample_batch, batch_judgments):
                    original_judgments.append(judgment)

                    # Save judgment record
                    judgment_record = {
                        "sample_id": sample.id,
                        "type": "original",
                        "question": sample.question,
                        "answer_A": sample.answer_1,
                        "answer_B": sample.answer_2,
                        "preferred": sample.preferred,
                        **judgment.to_dict(),
                    }
                    results["original_judgments"].append(judgment_record)

                    # Immediately save to file (incremental save to prevent data loss)
                    try:
                        save_judgment_jsonl_single(judgment_record, str(original_judgments_path))
                    except Exception as e:
                        logger.error(f"Failed to save judgment for sample {sample.id}: {e}")
                        raise

                pbar.update(len(sample_batch))

                # Check again after save
                if _shutdown_requested:
                    logger.warning("Shutdown requested after saving. Exiting gracefully...")
                    break
        finally:
            pbar.close()
        
        results["original_judgments_list"] = original_judgments

    # Load cached original judgments from a previous run (for RR computation
    # when compute_original_acc=False but we still need same-choice comparison).
    if not compute_original_acc and original_judgments_path:
        _ext_path = Path(original_judgments_path)
        if _ext_path.exists():
            logger.info(
                f"Loading cached original judgments from {original_judgments_path} "
                "for true RR computation"
            )
            _ext_records = load_existing_judgments(_ext_path.parent, "original")
            if _ext_records:
                _ext_objs = [
                    JudgeResult.from_dict({
                        "winner": r.get("winner", "tie"),
                        "score_A": r.get("score_A"),
                        "score_B": r.get("score_B"),
                        "explanation": r.get("explanation"),
                        "raw": r.get("raw", {}),
                        "is_valid": r.get("is_valid", True),
                    })
                    for r in _ext_records
                ]
                results["original_judgments_list"] = _ext_objs
                logger.info(
                    f"Loaded {len(_ext_objs)} cached original judgments "
                    f"(samples expected: {len(samples)})"
                )
            else:
                logger.warning(
                    f"Cached original judgments file exists but contains no records: "
                    f"{original_judgments_path}"
                )
        else:
            logger.warning(
                f"Cached original judgments path does not exist: {original_judgments_path}"
            )

    # Step 2: Inject bias and evaluate GT vs biased answer (for RR/CR)
    if compute_bias_metrics and bias_enabled:
        logger.info("Injecting bias and evaluating GT vs biased answer...")
        
        # Determine model name for cache
        if injector_type == "mock":
            cache_model_name = "mock"
        else:
            # Extract model name from injector_model_config
            cache_model_name = (
                injector_model_config.get("model_id") 
                or injector_model_config.get("model_name")
                or injector_type
            )
            # Normalize model name (remove special characters, use lowercase)
            cache_model_name = cache_model_name.lower().replace("-", "").replace("_", "")
        
        # Check cache
        cached_samples = None
        if use_cache:
            logger.info(
                f"Checking cache for bias-injected dataset: bias_type={bias_variant_name}, model={cache_model_name}"
            )
            if check_cache_exists(data_path, bias_variant_name, cache_model_name):
                logger.info("Cache found! Loading cached bias-injected dataset...")
                cached_samples = load_cached_bias_dataset(data_path, bias_variant_name, cache_model_name)
                if cached_samples:
                    logger.info(f"Successfully loaded {len(cached_samples)} samples from cache")
                else:
                    logger.warning("Cache file exists but failed to load, will regenerate")
            else:
                logger.info("Cache not found, will generate and save bias-injected dataset")
        
        # If cache not available, inject bias
        if cached_samples is None:
            logger.info("Injecting bias into samples...")
            biased_samples, guard_report = apply_bias_to_samples(
                samples,
                bias_injector,
                inject_to=inject_to,
                semantic_guard=semantic_guard,
            )
            
            # Save to cache
            if use_cache:
                metadata = {
                    "created_at": datetime.datetime.now().isoformat(),
                    "injector_config": {
                        "injector_type": injector_type,
                        "model_config": injector_model_config,
                    },
                    "prompt_config": {
                        "system_prompt": injector_system_prompt,
                        "user_template": injector_user_template,
                    },
                    "injection_mode": injection_mode,
                    "semantic_guard": semantic_guard or {},
                }
                save_bias_dataset(
                    biased_samples,
                    data_path,
                    bias_variant_name,
                    cache_model_name,
                    metadata=metadata
                )
                logger.info("Bias-injected dataset saved to cache")
        else:
            biased_samples = cached_samples
            guard_report = {
                "total_samples": len(cached_samples),
                "from_cache": True,
                "note": "Loaded from cache; per-sample injection checks are unavailable in this run."
            }

        # Save detailed bias injection report to run output directory
        report_payload = {
            "dataset_path": data_path,
            "bias_type": bias_type,
            "bias_variant_name": bias_variant_name,
            "injection_mode": injection_mode,
            "inject_to": inject_to,
            "injector_type": injector_type,
            "injector_model_config": injector_model_config or {},
            "semantic_guard_config": semantic_guard or {},
            "report": guard_report,
        }
        report_path = run_dir_path / "bias_injection_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Bias injection report saved to {report_path}")
        results["bias_injection_report"] = report_payload
        results["bias_injection_report_path"] = str(report_path)
        
        if len(biased_samples) != len(samples):
            raise RuntimeError(
                f"Biased sample count mismatch: original={len(samples)} vs biased={len(biased_samples)}"
            )

        # Position-debias mode evaluates both orders:
        # - GT@A:  answer_A=GT,    answer_B=biased(non-GT)
        # - GT@B:  answer_A=biased, answer_B=GT
        if completed_bias_ids and position_debias_pairwise:
            logger.warning(
                "Resume data for bias judgments is ignored in position_debias_pairwise mode. "
                "Bias judgments will be recomputed for all samples to keep metric definitions consistent."
            )

        # Prepare output file path for incremental saving
        bias_judgments_path = run_dir_path / "judge_raw_bias.jsonl"

        bias_judgments_posA_round1 = []
        bias_judgments_posA_round2 = []
        bias_judgments_posB_round1 = []
        bias_judgments_posB_round2 = []
        biased_answers = []
        target_changed_count = 0
        target_unchanged_count = 0

        pbar_bias = tqdm(total=len(samples), desc="Judgment (GT@A + GT@B)")
        try:
            paired_samples = list(zip(samples, biased_samples))
            for sample_batch in _iter_chunks(paired_samples, request_batch_size):
                if _shutdown_requested:
                    logger.warning("Shutdown requested. Saving progress and exiting gracefully...")
                    break

                prepared = []
                for original_sample, biased_sample in sample_batch:
                    # Get GT and biased answers
                    gt_answer, gt_key = original_sample.get_gt_answer()
                    non_gt_answer, non_gt_key = original_sample.get_non_gt_answer()
                    if original_sample.id != biased_sample.id:
                        raise RuntimeError(
                            f"Sample alignment mismatch between original and biased datasets: "
                            f"original_id={original_sample.id}, biased_id={biased_sample.id}"
                        )

                    # Get biased answer from biased sample according to inject_to target.
                    # - inject_to=non_gt: compare original GT vs biased non-GT
                    # - inject_to=gt:     compare original GT vs biased GT
                    if inject_to == "gt":
                        target_key = gt_key
                        other_key = non_gt_key
                        original_target = gt_answer
                        original_other = non_gt_answer
                    else:
                        target_key = non_gt_key
                        other_key = gt_key
                        original_target = non_gt_answer
                        original_other = gt_answer

                    biased_answer = _answer_by_key(biased_sample, target_key)
                    other_side_after = _answer_by_key(biased_sample, other_key)
                    if other_side_after != original_other:
                        raise RuntimeError(
                            "Bias injection target-side violation detected: "
                            f"inject_to={inject_to}, sample_id={original_sample.id}. "
                            "Non-target answer was modified unexpectedly."
                        )
                    if biased_answer == original_target:
                        target_unchanged_count += 1
                    else:
                        target_changed_count += 1

                    biased_answers.append(biased_answer)
                    prepared.append(
                        {
                            "original_sample": original_sample,
                            "gt_answer": gt_answer,
                            "non_gt_answer": non_gt_answer,
                            "biased_answer": biased_answer,
                            "target_key": target_key,
                        }
                    )

                def _judge_one(ctx: Dict[str, Any]):
                    original_sample = ctx["original_sample"]
                    gt_answer = ctx["gt_answer"]
                    biased_answer = ctx["biased_answer"]
                    # GT@A
                    judgment_posA_round1 = judge.judge_pairwise(
                        question=original_sample.question,
                        answer_A=gt_answer,
                        answer_B=biased_answer,
                    )
                    judgment_posA_round2 = judge.judge_pairwise(
                        question=original_sample.question,
                        answer_A=gt_answer,
                        answer_B=biased_answer,
                    )
                    # GT@B
                    judgment_posB_round1 = judge.judge_pairwise(
                        question=original_sample.question,
                        answer_A=biased_answer,
                        answer_B=gt_answer,
                    )
                    judgment_posB_round2 = judge.judge_pairwise(
                        question=original_sample.question,
                        answer_A=biased_answer,
                        answer_B=gt_answer,
                    )
                    return (
                        judgment_posA_round1,
                        judgment_posA_round2,
                        judgment_posB_round1,
                        judgment_posB_round2,
                    )

                if request_batch_size == 1:
                    judged_batch = [_judge_one(prepared[0])]
                else:
                    with ThreadPoolExecutor(max_workers=request_batch_size) as executor:
                        futures = [executor.submit(_judge_one, ctx) for ctx in prepared]
                        judged_batch = [f.result() for f in futures]

                for ctx, judgments in zip(prepared, judged_batch):
                    original_sample = ctx["original_sample"]
                    gt_answer = ctx["gt_answer"]
                    non_gt_answer = ctx["non_gt_answer"]
                    biased_answer = ctx["biased_answer"]
                    target_key = ctx["target_key"]
                    (
                        judgment_posA_round1,
                        judgment_posA_round2,
                        judgment_posB_round1,
                        judgment_posB_round2,
                    ) = judgments

                    bias_judgments_posA_round1.append(judgment_posA_round1)
                    bias_judgments_posA_round2.append(judgment_posA_round2)
                    bias_judgments_posB_round1.append(judgment_posB_round1)
                    bias_judgments_posB_round2.append(judgment_posB_round2)

                    judgment_record = {
                        "sample_id": original_sample.id,
                        "type": "bias",
                        "question": original_sample.question,
                        "gt_answer": gt_answer,
                        "biased_answer": biased_answer,
                        "original_non_gt": non_gt_answer,
                        "inject_to": inject_to,
                        "target_key": "answer_1" if _is_answer1_key(target_key) else "answer_2",
                        "bias_type": bias_type,
                        "preferred": original_sample.preferred,
                        "posA_round1_winner": judgment_posA_round1.winner,
                        "posA_round2_winner": judgment_posA_round2.winner,
                        "posB_round1_winner": judgment_posB_round1.winner,
                        "posB_round2_winner": judgment_posB_round2.winner,
                        "consistent_posA": judgment_posA_round1.winner == judgment_posA_round2.winner,
                        "consistent_posB": judgment_posB_round1.winner == judgment_posB_round2.winner,
                        "from_cache": cached_samples is not None,
                        **judgment_posA_round1.to_dict(),
                    }
                    results["bias_judgments"].append(judgment_record)

                    # Immediately save to file (incremental save to prevent data loss)
                    try:
                        save_judgment_jsonl_single(judgment_record, str(bias_judgments_path))
                    except Exception as e:
                        logger.error(f"Failed to save bias judgment for sample {original_sample.id}: {e}")
                        raise

                pbar_bias.update(len(sample_batch))

                # Check again after save
                if _shutdown_requested:
                    logger.warning("Shutdown requested after saving. Exiting gracefully...")
                    break
        finally:
            pbar_bias.close()
        
        # Keep backward-compatible fields while exposing position-debias outputs.
        results["bias_judgments_list"] = bias_judgments_posA_round1
        results["bias_judgments_round1"] = bias_judgments_posA_round1
        results["bias_judgments_round2"] = bias_judgments_posA_round2
        results["bias_judgments_posA_round1"] = bias_judgments_posA_round1
        results["bias_judgments_posA_round2"] = bias_judgments_posA_round2
        results["bias_judgments_posB_round1"] = bias_judgments_posB_round1
        results["bias_judgments_posB_round2"] = bias_judgments_posB_round2
        results["biased_answers"] = biased_answers
        results["injection_target_audit"] = {
            "inject_to": inject_to,
            "target_changed_count": target_changed_count,
            "target_unchanged_count": target_unchanged_count,
            "total_samples": len(samples),
            "request_batch_size": request_batch_size,
        }
        logger.info(
            "Injection target audit: "
            f"inject_to={inject_to}, changed={target_changed_count}, "
            f"unchanged={target_unchanged_count}, total={len(samples)}"
        )
    
    # Compute metrics
    metrics = {}

    # Keep run non-blocking: if local interruption causes count mismatch, patch with
    # explicit local-invalid placeholders instead of raising and aborting.
    if compute_original_acc and results.get("original_judgments_list") is not None:
        expected = len(samples)
        actual = len(results["original_judgments_list"])
        if actual < expected:
            missing = expected - actual
            logger.warning(
                f"Original judgments incomplete due to local interruption: expected {expected}, got {actual}. "
                f"Padding {missing} local-invalid placeholders."
            )
            results["original_judgments_list"].extend(
                [
                    JudgeResult(
                        winner="tie",
                        score_A=0.0,
                        score_B=0.0,
                        explanation="Local pipeline interruption: judgment missing.",
                        raw={"error": "local_missing_judgment"},
                        is_valid=False,
                    )
                    for _ in range(missing)
                ]
            )
        elif actual > expected:
            logger.warning(
                f"Original judgments exceed sample count: expected {expected}, got {actual}. Truncating extras."
            )
            results["original_judgments_list"] = results["original_judgments_list"][:expected]

    if compute_bias_metrics and results.get("bias_judgments_list") is not None:
        expected = len(samples)
        actual_bias = len(results["bias_judgments_list"])
        if actual_bias < expected:
            missing = expected - actual_bias
            logger.warning(
                f"Bias round1 judgments incomplete due to local interruption: expected {expected}, got {actual_bias}. "
                f"Padding {missing} local-invalid placeholders."
            )
            results["bias_judgments_list"].extend(
                [
                    JudgeResult(
                        winner="tie",
                        score_A=0.0,
                        score_B=0.0,
                        explanation="Local pipeline interruption: judgment missing.",
                        raw={"error": "local_missing_judgment"},
                        is_valid=False,
                    )
                    for _ in range(missing)
                ]
            )
        elif actual_bias > expected:
            logger.warning(
                f"Bias round1 judgments exceed sample count: expected {expected}, got {actual_bias}. Truncating extras."
            )
            results["bias_judgments_list"] = results["bias_judgments_list"][:expected]

        round2 = results.get("bias_judgments_round2", [])
        actual_round2 = len(round2)
        if actual_round2 < expected:
            missing = expected - actual_round2
            logger.warning(
                f"Bias round2 judgments incomplete due to local interruption: expected {expected}, got {actual_round2}. "
                f"Padding {missing} local-invalid placeholders."
            )
            round2.extend(
                [
                    JudgeResult(
                        winner="tie",
                        score_A=0.0,
                        score_B=0.0,
                        explanation="Local pipeline interruption: round2 judgment missing.",
                        raw={"error": "local_missing_judgment_round2"},
                        is_valid=False,
                    )
                    for _ in range(missing)
                ]
            )
            results["bias_judgments_round2"] = round2
        elif actual_round2 > expected:
            logger.warning(
                f"Bias round2 judgments exceed sample count: expected {expected}, got {actual_round2}. Truncating extras."
            )
            results["bias_judgments_round2"] = round2[:expected]

        answers = results.get("biased_answers", [])
        if len(answers) < expected:
            answers.extend([""] * (expected - len(answers)))
            results["biased_answers"] = answers
        elif len(answers) > expected:
            results["biased_answers"] = answers[:expected]
    
    if compute_original_acc and results.get("original_judgments_list"):
        logger.info("Computing accuracy on original R1 vs R2...")
        use_proxy = stats['without_gt'] > 0
        acc_metrics = compute_accuracy_original(
            samples=samples,
            judgments=results["original_judgments_list"],
            use_proxy=use_proxy
        )
        metrics["accuracy_original"] = acc_metrics
    
    if compute_bias_metrics and results.get("bias_judgments_posA_round1"):
        logger.info("Computing position-debiased biased accuracy, RR and CR...")

        def _target_win_stats(judgments: List[JudgeResult], target_winner: str) -> Dict[str, Any]:
            total = len(judgments)
            valid = [j for j in judgments if j.is_valid]
            invalid_count = total - len(valid)
            wins = sum(1 for j in valid if j.winner == target_winner)
            ties = sum(1 for j in valid if j.winner == "tie")
            rate = wins / len(valid) if valid else 0.0
            return {
                "rate": rate,
                "wins": wins,
                "ties": ties,
                "total": total,
                "valid_count": len(valid),
                "invalid_count": invalid_count,
            }

        posA_r1 = results.get("bias_judgments_posA_round1", [])
        posA_r2 = results.get("bias_judgments_posA_round2", [])
        posB_r1 = results.get("bias_judgments_posB_round1", [])
        posB_r2 = results.get("bias_judgments_posB_round2", [])

        # In posA runs GT is answer_A; in posB runs GT is answer_B.
        accA = _target_win_stats(posA_r1, "A")
        accB = _target_win_stats(posB_r1, "B")
        acc_avg = (accA["rate"] + accB["rate"]) / 2.0
        metrics["accuracy_biased"] = {
            "accuracy_biased": acc_avg,
            "position_A_gt": accA,
            "position_B_gt": accB,
            "metadata": {
                "note": "Average biased accuracy across GT@A and GT@B evaluations",
                "position_debias_pairwise": position_debias_pairwise,
            },
        }

        # ── Robustness Rate (RR) ──
        # RR = proportion of samples where the judge makes the SAME choice
        # before and after bias injection (regardless of correctness).
        # "Same choice" is normalized to answer identity (GT vs non-GT),
        # not positional label (A vs B), because biased evaluation swaps positions.
        orig_judgments = results.get("original_judgments_list", [])
        if orig_judgments and len(orig_judgments) == len(samples):
            # Normalize original choice to "gt" / "non_gt" / "tie"
            orig_choices = []  # per-sample: "gt", "non_gt", "tie", or None (invalid)
            for s, j in zip(samples, orig_judgments):
                if not j.is_valid:
                    orig_choices.append(None)
                elif j.winner == "tie":
                    orig_choices.append("tie")
                elif (s.preferred == "1" and j.winner == "A") or \
                     (s.preferred == "2" and j.winner == "B"):
                    orig_choices.append("gt")
                else:
                    orig_choices.append("non_gt")

            def _rr_same_choice(biased_judgments: List[JudgeResult],
                                gt_winner: str) -> Dict[str, Any]:
                """Compute RR as same-choice rate.

                gt_winner: the positional label that corresponds to GT in this
                biased run ("A" for GT@A, "B" for GT@B).
                """
                same = 0
                total_valid = 0
                invalid_count = 0
                for oc, bj in zip(orig_choices, biased_judgments):
                    if oc is None or not bj.is_valid:
                        invalid_count += 1
                        continue
                    # Normalize biased choice to "gt" / "non_gt" / "tie"
                    if bj.winner == "tie":
                        biased_choice = "tie"
                    elif bj.winner == gt_winner:
                        biased_choice = "gt"
                    else:
                        biased_choice = "non_gt"
                    total_valid += 1
                    if oc == biased_choice:
                        same += 1
                rate = same / total_valid if total_valid > 0 else 0.0
                return {
                    "rate": rate,
                    "same": same,
                    "total_valid": total_valid,
                    "invalid_count": invalid_count,
                }

            rrA = _rr_same_choice(posA_r1, "A")
            rrB = _rr_same_choice(posB_r1, "B")
            rr_avg = (rrA["rate"] + rrB["rate"]) / 2.0
            metrics["rr"] = {
                "rr": rr_avg,
                "rr_gt_at_A": rrA["rate"],
                "rr_gt_at_B": rrB["rate"],
                "position_A_gt": rrA,
                "position_B_gt": rrB,
                "metadata": {
                    "note": (
                        "RR = proportion of samples where judge makes the same choice "
                        "(GT/non-GT/tie) before and after bias injection, "
                        "averaged across GT@A and GT@B to reduce position bias"
                    ),
                    "position_debias_pairwise": position_debias_pairwise,
                },
            }
        else:
            _orig_len = len(orig_judgments) if orig_judgments else 0
            logger.warning(
                f"Cannot compute true RR: original judgments "
                f"{'unavailable' if not orig_judgments else f'count mismatch ({_orig_len} vs {len(samples)})'}"
                f". Falling back to acc_biased as RR proxy. "
                f"To fix: ensure original_judgments_path is set in batch config, "
                f"or re-run with compute_original_acc=true."
            )
            rrA = accA
            rrB = accB
            rr_avg = (rrA["rate"] + rrB["rate"]) / 2.0
            metrics["rr"] = {
                "rr": rr_avg,
                "rr_gt_at_A": rrA["rate"],
                "rr_gt_at_B": rrB["rate"],
                "position_A_gt": rrA,
                "position_B_gt": rrB,
                "metadata": {
                    "note": "Fallback RR (= acc_biased); original judgments unavailable",
                    "position_debias_pairwise": position_debias_pairwise,
                },
            }

        if posA_r1 and posA_r2 and posB_r1 and posB_r2:
            crA = compute_cr(judgments_round1=posA_r1, judgments_round2=posA_r2)
            crB = compute_cr(judgments_round1=posB_r1, judgments_round2=posB_r2)
            cr_avg = (crA.get("cr", 0.0) + crB.get("cr", 0.0)) / 2.0
            metrics["cr"] = {
                "cr": cr_avg,
                "cr_gt_at_A": crA.get("cr", 0.0),
                "cr_gt_at_B": crB.get("cr", 0.0),
                "position_A_gt": crA,
                "position_B_gt": crB,
                "metadata": {
                    "note": "CR averaged across GT@A and GT@B evaluations",
                    "position_debias_pairwise": position_debias_pairwise,
                },
            }
        else:
            logger.warning("CR cannot be computed: missing position-debias rounds")
            metrics["cr"] = {
                "cr": 0.0,
                "method": "consistency",
                "total": 0,
                "metadata": {"note": "CR not computed: missing position-debias rounds"}
            }
    
    results["metrics"] = metrics
    if "injection_target_audit" in results:
        metrics.setdefault("metadata", {})
        metrics["metadata"]["injection_target_audit"] = results["injection_target_audit"]
    
    # Save results to run_dir
    run_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Note: Judgments are already saved incrementally during the loop
    # Here we just log the final status
    if results["original_judgments"]:
        judgments_path = run_dir_path / "judge_raw_original.jsonl"
        if judgments_path.exists():
            logger.info(f"Original judgments saved incrementally to {judgments_path} ({len(results['original_judgments'])} total)")
        else:
            # Fallback: save all if file doesn't exist (shouldn't happen with incremental save)
            save_judgments_jsonl(results["original_judgments"], str(judgments_path), append=False)
            logger.warning(f"Original judgments file not found, saved all at once to {judgments_path}")
    
    if results["bias_judgments"]:
        judgments_path = run_dir_path / "judge_raw_bias.jsonl"
        if judgments_path.exists():
            logger.info(f"Bias judgments saved incrementally to {judgments_path} ({len(results['bias_judgments'])} total)")
        else:
            # Fallback: save all if file doesn't exist (shouldn't happen with incremental save)
            save_judgments_jsonl(results["bias_judgments"], str(judgments_path), append=False)
            logger.warning(f"Bias judgments file not found, saved all at once to {judgments_path}")
    
    # Save all judgments in one file for convenience
    if results["original_judgments"] or results["bias_judgments"]:
        all_judgments = []
        if results["original_judgments"]:
            all_judgments.extend(results["original_judgments"])
        if results["bias_judgments"]:
            all_judgments.extend(results["bias_judgments"])
        if all_judgments:
            judgments_path = run_dir_path / "judge_raw.jsonl"
            save_judgments_jsonl(all_judgments, str(judgments_path))
            logger.info(f"All judgments saved to {judgments_path}")
    
    # Save metrics
    metrics_path = run_dir_path / "metrics.json"
    save_metrics_json(metrics, str(metrics_path))
    logger.info(f"Metrics saved to {metrics_path}")
    
    csv_path = run_dir_path / "results.csv"
    save_metrics_csv(metrics, str(csv_path))
    logger.info(f"Metrics summary saved to {csv_path}")
    
    # Print summary
    print_metrics_summary(metrics)
    results["interrupted"] = _shutdown_requested
    
    if _shutdown_requested:
        logger.warning("="*60)
        logger.warning("⚠️  Evaluation interrupted by user")
        logger.warning(f"⚠️  Progress saved to: {run_dir_path}")
        logger.warning("⚠️  You can resume by running the same command again")
        logger.warning("="*60)
        # Don't raise exception, just return partial results
    else:
        logger.info("Pipeline completed successfully")
    
    return results

