#!/usr/bin/env python
"""
Batch evaluation script for running multiple judge models sequentially.
Designed for USC CARC (Slurm) environment.

Usage:
    python scripts/batch_evaluate.py --config configs/experiment.yaml --judges gpt4omini gpt52 gemini3_pro
"""
import sys
import argparse
import signal
import yaml
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger

# Global flag for graceful shutdown
_batch_shutdown_requested = False


def _batch_signal_handler(signum, frame):
    """Handle interrupt signals gracefully in batch mode."""
    global _batch_shutdown_requested
    _batch_shutdown_requested = True
    logger.warning("\n" + "="*60)
    logger.warning("⚠️  Batch evaluation interrupted (Ctrl+C or SIGTERM)")
    logger.warning("⚠️  Current judge evaluation will finish, then batch will stop")
    logger.warning("⚠️  Completed evaluations are saved and can be resumed")
    logger.warning("="*60)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import main as run_single_experiment
from src.utils.io import load_yaml


def setup_logging(log_dir: Path, judge_name: str):
    """Setup logging for a specific judge run."""
    log_file = log_dir / f"batch_{judge_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days"
    )
    return log_file


def run_judge_experiment(
    base_config_path: str,
    judge_provider: str,
    judge_model_id: str,
    log_dir: Path,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run a single judge experiment.
    
    Args:
        base_config_path: Path to base experiment config
        judge_provider: Judge provider (e.g., "openai", "gemini")
        judge_model_id: Judge model ID (e.g., "gpt4omini", "gemini3_pro")
        log_dir: Directory for logs
        dry_run: If True, only print what would be run
        
    Returns:
        Dict with status and results
    """
    judge_name = f"{judge_provider}_{judge_model_id}"
    log_file = setup_logging(log_dir, judge_name)
    
    result = {
        "judge_provider": judge_provider,
        "judge_model_id": judge_model_id,
        "judge_name": judge_name,
        "status": "pending",
        "log_file": str(log_file),
        "error": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    }
    
    try:
        logger.info(f"=" * 80)
        logger.info(f"Starting evaluation with judge: {judge_provider}/{judge_model_id}")
        logger.info(f"=" * 80)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would run: judge.provider={judge_provider}, judge.model_id={judge_model_id}")
            result["status"] = "dry_run"
            return result
        
        # Load base config
        base_config = load_yaml(base_config_path)
        
        # Modify judge configuration
        # Use new simplified format: just specify "model" (provider will be auto-inferred)
        if "judge" not in base_config:
            base_config["judge"] = {}
        # Remove old format keys if present
        base_config["judge"].pop("provider", None)
        base_config["judge"].pop("model_id", None)
        base_config["judge"].pop("type", None)
        base_config["judge"].pop("model_name", None)
        # Set new simplified format
        base_config["judge"]["model"] = judge_model_id
        
        # Create temporary config file for this run
        temp_config_path = log_dir / f"config_{judge_name}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Using temporary config: {temp_config_path}")
        
        # Modify sys.argv to pass config to main()
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["batch_evaluate.py", "--config", str(temp_config_path)]
            run_single_experiment()
            result["status"] = "success"
            logger.info(f"✓ Successfully completed evaluation with {judge_name}")
        finally:
            sys.argv = original_argv
        
        result["end_time"] = datetime.now().isoformat()
        
    except KeyboardInterrupt:
        logger.warning(f"Interrupted by user for {judge_name}")
        result["status"] = "interrupted"
        result["error"] = "KeyboardInterrupt"
        result["end_time"] = datetime.now().isoformat()
        raise  # Re-raise to allow outer handler
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"✗ Failed to run evaluation with {judge_name}")
        logger.error(f"Error: {error_msg}")
        logger.error(f"Traceback:\n{error_traceback}")
        
        result["status"] = "failed"
        result["error"] = error_msg
        result["traceback"] = error_traceback
        result["end_time"] = datetime.now().isoformat()
        
        # Save error details to separate file
        error_log_file = log_dir / f"error_{judge_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Judge: {judge_provider}/{judge_model_id}\n")
            f.write(f"Time: {result['start_time']}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"\nTraceback:\n{error_traceback}\n")
        result["error_log_file"] = str(error_log_file)
    
    return result


def parse_judge_spec(judge_spec: str) -> tuple[str, str]:
    """
    Parse judge specification.
    
    Formats:
    - "gpt4omini" -> ("openai", "gpt4omini")
    - "openai:gpt4omini" -> ("openai", "gpt4omini")
    - "gemini:gemini3_pro" -> ("gemini", "gemini3_pro")
    - "qwen3_4b_instruct" -> ("vllm", "qwen3_4b_instruct") (inferred from models.yaml)
    
    Args:
        judge_spec: Judge specification string
        
    Returns:
        (provider, model_id) tuple
    """
    if ":" in judge_spec:
        provider, model_id = judge_spec.split(":", 1)
        return provider.strip(), model_id.strip()
    else:
        # Try to infer provider from model_id using ModelRegistry
        model_id = judge_spec.strip()
        
        # First, try to use ModelRegistry to infer from models.yaml
        try:
            from src.models.registry import ModelRegistry
            # Load models.yaml if not already loaded
            from src.utils.io import load_yaml
            project_root = Path(__file__).parent.parent
            models_yaml_path = project_root / "configs" / "models.yaml"
            if models_yaml_path.exists():
                models_config = load_yaml(str(models_yaml_path))
                ModelRegistry.set_models_config(models_config)
            
            provider = ModelRegistry.infer_provider_from_model_id(model_id)
            if provider:
                logger.info(f"Inferred provider '{provider}' for model '{model_id}' from models.yaml")
                return provider, model_id
        except Exception as e:
            logger.debug(f"Could not use ModelRegistry to infer provider: {e}")
        
        # Fallback: Common patterns
        if model_id.startswith("gpt") or model_id.startswith("gpt4") or model_id.startswith("gpt5"):
            return "openai", model_id
        elif model_id.startswith("gemini"):
            return "gemini", model_id
        elif model_id.startswith("claude"):
            return "anthropic", model_id
        else:
            # For unknown models, default to vLLM (most unlisted models are open-source)
            logger.warning(f"Could not infer provider for {model_id}, defaulting to 'vllm'")
            return "vllm", model_id


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for multiple judge models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with specific judges
  python scripts/batch_evaluate.py --config configs/experiment.yaml --judges gpt4omini gpt52 gemini3_pro
  
  # Run with explicit provider:model format
  python scripts/batch_evaluate.py --config configs/experiment.yaml --judges openai:gpt4omini gemini:gemini3_pro
  
  # Dry run (show what would be run)
  python scripts/batch_evaluate.py --config configs/experiment.yaml --judges gpt4omini --dry-run
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base experiment configuration file"
    )
    
    parser.add_argument(
        "--judges",
        nargs="+",
        required=True,
        help="List of judge models to run. Format: 'model_id' or 'provider:model_id'"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs (default: outputs/batch_logs/<timestamp>)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue to next judge if one fails (default: True)"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    global _batch_shutdown_requested
    _batch_shutdown_requested = False
    signal.signal(signal.SIGINT, _batch_signal_handler)
    signal.signal(signal.SIGTERM, _batch_signal_handler)
    
    # Setup log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = project_root / "outputs" / "batch_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup main batch log
    batch_log_file = log_dir / "batch_summary.log"
    logger.add(
        batch_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("Starting batch evaluation")
    logger.info("=" * 80)
    logger.info(f"Base config: {args.config}")
    logger.info(f"Judges: {args.judges}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Continue on error: {args.continue_on_error}")
    
    # Parse judge specifications
    judge_configs = []
    for judge_spec in args.judges:
        provider, model_id = parse_judge_spec(judge_spec)
        judge_configs.append((provider, model_id))
        logger.info(f"  - {provider}:{model_id}")
    
    # Run experiments
    results = []
    for i, (provider, model_id) in enumerate(judge_configs, 1):
        # Check for shutdown request
        if _batch_shutdown_requested:
            logger.warning("Batch evaluation interrupted. Stopping after current judge completes.")
            break
        
        logger.info("")
        logger.info(f"[{i}/{len(judge_configs)}] Running judge: {provider}/{model_id}")
        
        try:
            result = run_judge_experiment(
                base_config_path=args.config,
                judge_provider=provider,
                judge_model_id=model_id,
                log_dir=log_dir,
                dry_run=args.dry_run
            )
            results.append(result)
            
            if result["status"] == "failed" and not args.continue_on_error:
                logger.error(f"Stopping batch evaluation due to failure (--continue-on-error is False)")
                break
            
            # Check again after judge completes
            if _batch_shutdown_requested:
                logger.warning("Batch evaluation interrupted. Stopping after current judge.")
                break
                
        except KeyboardInterrupt:
            logger.warning("Batch evaluation interrupted by user")
            _batch_shutdown_requested = True
            break
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Batch evaluation summary")
    logger.info("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    total_count = len(results)
    
    logger.info(f"Total: {total_count}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("")
    
    # Print detailed results
    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"{status_icon} {result['judge_name']}: {result['status']}")
        if result["status"] == "failed":
            logger.info(f"    Error: {result.get('error', 'Unknown error')}")
            logger.info(f"    Error log: {result.get('error_log_file', 'N/A')}")
        logger.info(f"    Log: {result['log_file']}")
    
    # Save summary JSON
    import json
    summary_file = log_dir / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total": total_count,
            "success": success_count,
            "failed": failed_count,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"All logs saved to: {log_dir}")
    
    # Exit with error code if any failed
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
