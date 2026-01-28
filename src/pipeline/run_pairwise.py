"""
Pipeline for pairwise evaluation.
"""
import datetime
import signal
import sys
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
from ..metrics.pairwise_metrics import compute_accuracy_original, compute_rr, compute_cr
from ..metrics.reports import save_metrics_json, save_metrics_csv, save_judgments_jsonl, save_judgment_jsonl_single, print_metrics_summary
from ..utils.resume import (
    check_resume_available,
    load_completed_sample_ids,
    filter_completed_samples,
    load_existing_judgments
)

# Global flag for graceful shutdown
_shutdown_requested = False


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
    judge_system_prompt: Optional[str] = None,
    judge_user_template: Optional[str] = None,
    inject_to: str = "non_gt",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run pairwise evaluation pipeline.
    
    Args:
        data_path: Path to pairwise data file
        bias_type: Type of bias to inject
        bias_enabled: Whether to enable bias injection
        injector_type: Type of bias injector ("mock", "openai", "hf")
        injector_model_config: Model config for bias injector (if using AI-based injection)
        injector_system_prompt: Optional system prompt for bias injection
        injector_user_template: Optional user prompt template for bias injection
        judge_system_prompt: Optional system prompt for judge pairwise prompt
        judge_user_template: Optional user prompt template for judge pairwise prompt
        judge_type: Type of judge ("mock", "openai", "hf")
        judge_model_name: Name of judge model
        judge_config: Judge model configuration
        compute_original_acc: Whether to compute accuracy on original R1 vs R2
        compute_bias_metrics: Whether to compute RR/CR on R1 vs biased_R2
        run_dir: Run directory for all output files
        
    Returns:
        Dictionary with all metrics and results
    """
    logger.info("Starting pairwise evaluation pipeline")
    
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
        
        for sample in tqdm(remaining_samples_original, desc="Original judgment"):
            # Check for shutdown request
            if _shutdown_requested:
                logger.warning("Shutdown requested. Saving progress and exiting gracefully...")
                break
            
            judgment = judge.judge_pairwise(
                question=sample.question,
                answer_A=sample.answer_1,
                answer_B=sample.answer_2
            )
            original_judgments.append(judgment)
            
            # Save judgment record
            judgment_record = {
                "sample_id": sample.id,
                "type": "original",
                "question": sample.question,
                "answer_A": sample.answer_1,
                "answer_B": sample.answer_2,
                "preferred": sample.preferred,
                **judgment.to_dict()
            }
            results["original_judgments"].append(judgment_record)
            
            # Immediately save to file (incremental save to prevent data loss)
            try:
                save_judgment_jsonl_single(judgment_record, str(original_judgments_path))
            except Exception as e:
                logger.error(f"Failed to save judgment for sample {sample.id}: {e}")
                raise
            
            # Check again after save
            if _shutdown_requested:
                logger.warning("Shutdown requested after saving. Exiting gracefully...")
                break
        
        results["original_judgments_list"] = original_judgments
    
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
            logger.info(f"Checking cache for bias-injected dataset: bias_type={bias_type}, model={cache_model_name}")
            if check_cache_exists(data_path, bias_type, cache_model_name):
                logger.info("Cache found! Loading cached bias-injected dataset...")
                cached_samples = load_cached_bias_dataset(data_path, bias_type, cache_model_name)
                if cached_samples:
                    logger.info(f"Successfully loaded {len(cached_samples)} samples from cache")
                else:
                    logger.warning("Cache file exists but failed to load, will regenerate")
            else:
                logger.info("Cache not found, will generate and save bias-injected dataset")
        
        # If cache not available, inject bias
        if cached_samples is None:
            logger.info("Injecting bias into samples...")
            biased_samples = apply_bias_to_samples(samples, bias_injector, inject_to=inject_to)
            
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
                }
                save_bias_dataset(
                    biased_samples,
                    data_path,
                    bias_type,
                    cache_model_name,
                    metadata=metadata
                )
                logger.info("Bias-injected dataset saved to cache")
        else:
            biased_samples = cached_samples
        
        # Filter completed samples for bias evaluation
        # Create mapping of sample_id to biased_sample for remaining samples
        remaining_samples_bias = []
        remaining_biased_samples = []
        completed_bias_samples = []
        completed_bias_judgments_round1 = []
        completed_bias_judgments_round2 = []
        completed_biased_answers = []
        
        for original_sample, biased_sample in zip(samples, biased_samples):
            if str(original_sample.id) in completed_bias_ids:
                completed_bias_samples.append(original_sample)
                # Find existing judgment for this sample
                existing_judgment = next(
                    (j for j in existing_bias_judgments if str(j.get("sample_id")) == str(original_sample.id)),
                    None
                )
                if existing_judgment:
                    # Reconstruct JudgeResult from existing judgment
                    completed_bias_judgments_round1.append(JudgeResult.from_dict({
                        "winner": existing_judgment.get("round1_winner") or existing_judgment.get("winner", "tie"),
                        "score_A": existing_judgment.get("score_A"),
                        "score_B": existing_judgment.get("score_B"),
                        "explanation": existing_judgment.get("explanation"),
                        "raw": existing_judgment.get("raw", {})
                    }))
                    completed_bias_judgments_round2.append(JudgeResult.from_dict({
                        "winner": existing_judgment.get("round2_winner") or existing_judgment.get("winner", "tie"),
                        "score_A": existing_judgment.get("score_A"),
                        "score_B": existing_judgment.get("score_B"),
                        "explanation": existing_judgment.get("explanation"),
                        "raw": existing_judgment.get("raw", {})
                    }))
                    completed_biased_answers.append(existing_judgment.get("answer_B") or existing_judgment.get("original_non_gt", ""))
            else:
                remaining_samples_bias.append(original_sample)
                remaining_biased_samples.append(biased_sample)
        
        if completed_bias_ids:
            logger.info(f"Resuming bias evaluation: {len(completed_bias_samples)} samples already completed, {len(remaining_samples_bias)} remaining")
            # Add existing bias judgments to results
            results["bias_judgments"].extend(existing_bias_judgments)
        
        # Prepare output file path for incremental saving
        bias_judgments_path = run_dir_path / "judge_raw_bias.jsonl"
        
        # Now evaluate using biased samples (only remaining ones)
        bias_judgments_round1 = completed_bias_judgments_round1.copy()
        bias_judgments_round2 = completed_bias_judgments_round2.copy()
        biased_answers = completed_biased_answers.copy()
        
        for original_sample, biased_sample in tqdm(zip(remaining_samples_bias, remaining_biased_samples), desc="Judgment (2 rounds)", total=len(remaining_samples_bias)):
            # Check for shutdown request
            if _shutdown_requested:
                logger.warning("Shutdown requested. Saving progress and exiting gracefully...")
                break
            
            # Get GT and biased answers
            gt_answer, gt_key = original_sample.get_gt_answer()
            non_gt_answer, non_gt_key = original_sample.get_non_gt_answer()
            
            # Get biased answer from biased sample
            if gt_key == "answer_1":
                biased_answer = biased_sample.answer_2
            else:
                biased_answer = biased_sample.answer_1
            
            biased_answers.append(biased_answer)
            
            # Judge: GT vs biased answer (Round 1)
            judgment_round1 = judge.judge_pairwise(
                question=original_sample.question,
                answer_A=gt_answer,  # GT answer
                answer_B=biased_answer  # Biased answer
            )
            bias_judgments_round1.append(judgment_round1)
            
            # Judge: GT vs biased answer (Round 2) - for CR consistency check
            judgment_round2 = judge.judge_pairwise(
                question=original_sample.question,
                answer_A=gt_answer,  # GT answer
                answer_B=biased_answer  # Biased answer
            )
            bias_judgments_round2.append(judgment_round2)
            
            # Save judgment record (use round1 for RR calculation)
            judgment_record = {
                "sample_id": original_sample.id,
                "type": "bias",
                "question": original_sample.question,
                "answer_A": gt_answer,
                "answer_B": biased_answer,
                "original_non_gt": non_gt_answer,
                "bias_type": bias_type,
                "preferred": original_sample.preferred,
                "round1_winner": judgment_round1.winner,
                "round2_winner": judgment_round2.winner,
                "consistent": judgment_round1.winner == judgment_round2.winner,
                "from_cache": cached_samples is not None,
                **judgment_round1.to_dict()
            }
            results["bias_judgments"].append(judgment_record)
            
            # Immediately save to file (incremental save to prevent data loss)
            try:
                save_judgment_jsonl_single(judgment_record, str(bias_judgments_path))
            except Exception as e:
                logger.error(f"Failed to save bias judgment for sample {original_sample.id}: {e}")
                raise
            
            # Check again after save
            if _shutdown_requested:
                logger.warning("Shutdown requested after saving. Exiting gracefully...")
                break
        
        # Use round1 for RR calculation
        results["bias_judgments_list"] = bias_judgments_round1
        results["bias_judgments_round1"] = bias_judgments_round1
        results["bias_judgments_round2"] = bias_judgments_round2
        results["biased_answers"] = biased_answers
    
    # Compute metrics
    metrics = {}
    
    if compute_original_acc and results.get("original_judgments_list"):
        logger.info("Computing accuracy on original R1 vs R2...")
        use_proxy = stats['without_gt'] > 0
        acc_metrics = compute_accuracy_original(
            samples=samples,
            judgments=results["original_judgments_list"],
            use_proxy=use_proxy
        )
        metrics["accuracy_original"] = acc_metrics
    
    if compute_bias_metrics and results.get("bias_judgments_list"):
        logger.info("Computing RR and CR...")
        rr_metrics = compute_rr(
            samples=samples,
            judgments=results["bias_judgments_list"],
            biased_answers=results["biased_answers"]
        )
        metrics["rr"] = rr_metrics
        
        # CR: consistency across two rounds of evaluation
        if results.get("bias_judgments_round1") and results.get("bias_judgments_round2"):
            cr_metrics = compute_cr(
                judgments_round1=results["bias_judgments_round1"],
                judgments_round2=results["bias_judgments_round2"]
            )
            metrics["cr"] = cr_metrics
        else:
            logger.warning("CR cannot be computed: missing round2 judgments")
            metrics["cr"] = {
                "cr": 0.0,
                "method": "consistency",
                "total": 0,
                "metadata": {"note": "CR not computed: missing round2 judgments"}
            }
    
    results["metrics"] = metrics
    
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

