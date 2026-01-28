#!/bin/bash
#SBATCH --job-name=judgellm_batch
#SBATCH --output=outputs/slurm_logs/judgellm_batch_%j.out
#SBATCH --error=outputs/slurm_logs/judgellm_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu  # Adjust partition name as needed (e.g., gpu, cpu, shared)
#SBATCH --gres=gpu:1  # Remove this line if not using GPU

# USC CARC Slurm batch evaluation script
# This script runs multiple judge models sequentially on a given dataset + bias configuration

# ============================================================================
# CONFIGURATION - Modify these variables for your experiment
# ============================================================================

# Base experiment configuration file
CONFIG_FILE="configs/experiment.yaml"

# List of judge models to evaluate (space-separated)
# Format: "model_id" or "provider:model_id"
# Examples:
#   - "gpt4omini" (will infer provider as openai)
#   - "openai:gpt4omini" (explicit provider)
#   - "gemini:gemini3_pro" (explicit provider)
JUDGES=("gpt4omini" "gpt52" "gemini3_pro")

# Optional: Custom log directory (default: outputs/batch_logs/<timestamp>)
# LOG_DIR="outputs/batch_logs/custom_run"

# ============================================================================
# SCRIPT SETUP
# ============================================================================

set -e  # Exit on error (but continue-on-error is handled by Python script)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Create log directories
mkdir -p outputs/slurm_logs
mkdir -p outputs/batch_logs

# Activate virtual environment if it exists
if [ -d "Judgellm" ]; then
    echo "Activating virtual environment..."
    source Judgellm/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# ============================================================================
# RUN BATCH EVALUATION
# ============================================================================

echo "============================================================================"
echo "Batch Evaluation Job Started"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Judges: ${JUDGES[*]}"
echo "Time: $(date)"
echo "============================================================================"

# Build command
CMD="python scripts/batch_evaluate.py"
CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --judges ${JUDGES[@]}"
CMD="$CMD --continue-on-error"

# Add custom log directory if specified
if [ -n "$LOG_DIR" ]; then
    CMD="$CMD --log-dir $LOG_DIR"
fi

# Run the batch evaluation
echo "Running: $CMD"
echo ""

$CMD

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Batch Evaluation Job Completed"
echo "============================================================================"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "============================================================================"

exit $EXIT_CODE
