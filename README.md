# JudgeLLM — Medical Bias Evaluation

A lightweight framework for evaluating LLM-as-a-Judge robustness in medical QA by injecting controlled bias styles into answers.

## Overview

This project tests whether judge models can still identify the best answer when one candidate is modified with specific bias styles (e.g., excessive medical jargon, authority references). It measures:

- **Accuracy**: Can the judge pick the correct answer in original comparisons?
- **Robustness Rate (RR)**: Does the judge still prefer the ground-truth answer when bias is injected?
- **Consistency Rate (CR)**: Are judgments consistent across repeated evaluations?

## Features

- ✅ **Multiple bias types**: `jargon_overloading`, `authority`, `complexity`
- ✅ **Multiple judge backends**: Mock (offline), OpenAI, HuggingFace, Gemini, Anthropic
- ✅ **Latest models supported**: Gemini 3 Pro/Flash, Claude Opus 4.5, Claude Sonnet 4.5
- ✅ **Configuration-driven**: All settings in YAML files
- ✅ **Pairwise evaluation**: Compare two answers side-by-side
- ✅ **Comprehensive metrics**: Accuracy, RR, CR with detailed outputs

## Installation

### Prerequisites

- Python 3.9+
- pip

### Create Virtual Environment

```bash
python3 -m venv Judgellm
source Judgellm/bin/activate  # On Windows: Judgellm\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Set Environment Variables (Optional)

If using API-based models:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 2. Run with Mock Mode (No API Key Required)

```bash
python scripts/run_experiment.py --config configs/test_experiment.yaml
```

This uses mock judge and bias injector, perfect for testing the framework.

### 3. Run with Real Models

Edit `configs/experiment.yaml` to use real models:

```yaml
judge:
  provider: "openai"  # or "gemini", "anthropic"
  model_id: "gpt52"  # or "gemini3_pro", "claude_opus_45", etc.

bias:
  injector_type: "openai"  # or "mock", "gemini", "anthropic"
  model_id: "gpt4omini"  # or "gemini3_flash", etc.
```

Then run:

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

## Configuration

The main configuration file is `configs/experiment.yaml`:

```yaml
# Data
data:
  type: "pairwise"
  path: "data/dummy/pairwise.jsonl"
  dataset_name: "dummy"

# Bias Injection
bias:
  enabled: true
  type: "jargon_overloading"  # jargon_overloading, authority, complexity
  injector_type: "mock"  # mock, openai, hf, gemini

# Judge Model
judge:
  provider: "mock"  # mock, openai, hf, gemini, anthropic
  model_id: "mock-judge-v1"

# Evaluation
evaluation:
  compute_original_acc: true
  compute_bias_metrics: true
```

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `bias.type` | Bias style to inject | `jargon_overloading`, `authority`, `complexity` |
| `bias.injector_type` | How to inject bias | `mock` (offline), `openai`, `hf`, `gemini` |
| `judge.provider` | Judge backend | `mock`, `openai`, `hf`, `gemini`, `anthropic` |
| `judge.model_id` | Specific model to use | See `configs/models.yaml` |

## Supported Models

The framework supports multiple model providers. Available models are configured in `configs/models.yaml`:

### OpenAI
- `gpt4omini` → `gpt-4o-mini`
- `gpt41` → `gpt-4.1`
- `gpt52` → `gpt-5.2`

### Anthropic (Claude)
- `claude35_sonnet` → `claude-3-5-sonnet-20240620`
- `claude_opus_45` → `claude-opus-4.5` ⭐ **Latest**
- `claude_sonnet_45` → `claude-sonnet-4.5` ⭐ **Latest**

### Gemini
- `gemini15_pro` → `gemini-1.5-pro`
- `med_gemini` → `med-gemini`
- `gemini3_pro` → `gemini-3-pro` ⭐ **Latest**
- `gemini3_flash` → `gemini-3-flash` ⭐ **Latest**

### HuggingFace
- `llama2_7b_chat` → `meta-llama/Llama-2-7b-chat-hf`

### Mock (Offline)
- `mock-judge-v1` → For testing without API keys

To use a model, set `judge.model_id` or `bias.model_id` in `configs/experiment.yaml` to the model ID listed above.

## Project Structure

```
Ruishan_Judgellm_Ours/
├── configs/              # Configuration files
│   ├── experiment.yaml   # Main experiment config
│   ├── models.yaml       # Model registry
│   └── prompts.yaml      # Prompt templates
├── data/                 # Datasets
│   └── dummy/            # Test data
├── outputs/              # Experiment results
├── scripts/              # Entry point scripts
│   └── run_experiment.py # Main script
└── src/                  # Source code
    ├── bias/             # Bias injection
    ├── judge/            # Judge models
    ├── metrics/          # Evaluation metrics
    └── pipeline/         # Evaluation pipelines
```

## Expected Output

After running, you'll find results in `outputs/<run_name>/`:

- **`results.csv`** - Summary metrics (Accuracy, RR, CR)
- **`metrics.json`** - Detailed metrics with metadata
- **`judge_raw_original.jsonl`** - Original comparisons
- **`judge_raw_bias.jsonl`** - Bias-injected comparisons
- **`logs.txt`** - Full execution logs
- **`config_resolved.yaml`** - Final resolved configuration

Example output:

```
Accuracy (Original): 1.0000
Robustness Rate (RR): 1.0000
Consistency Rate (CR): 1.0000
```

## Extending the Framework

### Adding a New Bias Type

1. Create a new class in `src/bias/builtin_biases.py`:

```python
class MyBias(BaseBias):
    def apply(self, text: str, context: Optional[Dict] = None) -> str:
        # Your bias injection logic
        return modified_text
```

2. Register it in `BUILTIN_BIASES`:

```python
BUILTIN_BIASES = {
    "my_bias": MyBias,
    # ... existing biases
}
```

### Adding a New Judge Backend

1. Create a client in `src/models/` (e.g., `my_client.py`)
2. Register it in `src/models/registry.py`
3. Add model configs to `configs/models.yaml`

## FAQ

### Q: How do I use my own dataset?

A: Create a JSONL file with format:
```json
{"id": "sample_1", "question": "...", "answer_1": "...", "answer_2": "...", "preferred": "1"}
```

Then set `data.path` in `configs/experiment.yaml`.

### Q: API key not working?

A: Make sure environment variables are set:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

If not set, add to `~/.zshrc` or `~/.bashrc`:
```bash
export OPENAI_API_KEY="your-key"
```

### Q: Can I run without API keys?

A: Yes! Use `mock` mode for both judge and bias injector. See `configs/test_experiment.yaml`.

### Q: How to change the judge model?

A: Edit `judge.model_id` in `configs/experiment.yaml`. Available models are listed in `configs/models.yaml`.

### Q: What if I get import errors?

A: Make sure virtual environment is activated and dependencies are installed:
```bash
source Judgellm/bin/activate
pip install -r requirements.txt
```

## License

[Add your license here]

## Citation

[Add citation if applicable]
