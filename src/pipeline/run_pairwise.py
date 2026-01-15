"""
Pipeline for pairwise evaluation.
"""
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from loguru import logger

from ..dataset.schemas import PairwiseSample
from ..dataset.loaders import load_pairwise_jsonl
from ..dataset.preprocess import validate_pairwise_samples, get_gt_statistics
from ..bias.injector import BiasInjector
from ..judge.judge_runner import create_judge
from ..judge.base import JudgeResult
from ..metrics.pairwise_metrics import compute_accuracy_original, compute_rr, compute_cr
from ..metrics.reports import save_metrics_json, save_metrics_csv, save_judgments_jsonl, print_metrics_summary


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
    
    # Step 1: Evaluate original R1 vs R2 (for Acc)
    if compute_original_acc:
        logger.info("Evaluating original R1 vs R2...")
        original_judgments = []
        
        for sample in tqdm(samples, desc="Original judgment"):
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
        
        results["original_judgments_list"] = original_judgments
    
    # Step 2: Inject bias and evaluate GT vs biased answer (for RR/CR)
    if compute_bias_metrics and bias_enabled:
        logger.info("Injecting bias and evaluating GT vs biased answer...")
        bias_judgments_round1 = []
        bias_judgments_round2 = []
        biased_answers = []
        
        for sample in tqdm(samples, desc="Bias injection + judgment (2 rounds)"):
            # Get GT and non-GT answers
            gt_answer, gt_key = sample.get_gt_answer()
            non_gt_answer, non_gt_key = sample.get_non_gt_answer()
            
            # Inject bias into non-GT answer
            biased_answer = bias_injector.inject(non_gt_answer, question=sample.question)
            biased_answers.append(biased_answer)
            
            # Judge: GT vs biased answer (Round 1)
            judgment_round1 = judge.judge_pairwise(
                question=sample.question,
                answer_A=gt_answer,  # GT answer
                answer_B=biased_answer  # Biased answer
            )
            bias_judgments_round1.append(judgment_round1)
            
            # Judge: GT vs biased answer (Round 2) - for CR consistency check
            judgment_round2 = judge.judge_pairwise(
                question=sample.question,
                answer_A=gt_answer,  # GT answer
                answer_B=biased_answer  # Biased answer
            )
            bias_judgments_round2.append(judgment_round2)
            
            # Save judgment record (use round1 for RR calculation)
            judgment_record = {
                "sample_id": sample.id,
                "type": "bias",
                "question": sample.question,
                "answer_A": gt_answer,
                "answer_B": biased_answer,
                "original_non_gt": non_gt_answer,
                "bias_type": bias_type,
                "preferred": sample.preferred,
                "round1_winner": judgment_round1.winner,
                "round2_winner": judgment_round2.winner,
                "consistent": judgment_round1.winner == judgment_round2.winner,
                **judgment_round1.to_dict()
            }
            results["bias_judgments"].append(judgment_record)
        
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
    from pathlib import Path
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save judgments
    if results["original_judgments"]:
        judgments_path = run_dir_path / "judge_raw_original.jsonl"
        save_judgments_jsonl(results["original_judgments"], str(judgments_path))
        logger.info(f"Original judgments saved to {judgments_path}")
    
    if results["bias_judgments"]:
        judgments_path = run_dir_path / "judge_raw_bias.jsonl"
        save_judgments_jsonl(results["bias_judgments"], str(judgments_path))
        logger.info(f"Bias judgments saved to {judgments_path}")
    
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
    
    logger.info("Pipeline completed successfully")
    
    return results

