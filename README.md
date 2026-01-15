# JudgeLLM Medical Bias Evaluation (Medical QA)

This repository provides a **configuration-driven evaluation framework** for studying **LLM-as-a-Judge bias in medical question answering (QA)**. The core idea is to **inject a controllable “bias style”** into one answer and then measure whether a judge model still prefers the ground-truth (preferred) answer.

The documentation below is **strictly based on the current code and configs** in this repo. If a feature is not wired into the code (even if a config file exists), it is explicitly marked as such.

## 1. Project Overview

### What problem does this project address?

In medical QA, answers can be persuasive for the wrong reasons (e.g., overly technical jargon, authoritative tone). When using **LLM-as-a-Judge** to compare answers, the judge itself can be **biased toward certain surface styles**, which may reduce reliability in safety-critical domains.

This project evaluates:

- **How robust a judge model is** when one candidate answer is modified to include a specific bias style.
- How often the judge’s preference remains consistent across repeated evaluations.

### What does “bias” mean here?

In this repository, **bias refers to controlled transformations applied to an answer’s wording/style**, not demographic or societal bias. The built-in bias types currently implemented are:

- `jargon_overloading`: adds excessive medical jargon (rule-based or LLM-based rewriting)
- `authority`: adds authority references (rule-based)
- `complexity`: makes text unnecessarily complex (rule-based)

### What is the role of JudgeLLM in this evaluation?

The “JudgeLLM” is the **judge model** that performs **pairwise comparison** between two answers and outputs a winner:

- `A`, `B`, or `tie`

The judge can be:

- `mock`: heuristic judge (no external dependencies)
- `openai`: OpenAI chat model
- `hf`: local HuggingFace model (optional dependencies required)

## 2. Evaluation Pipeline (Step-by-Step)

The main pipeline is implemented in `src/main.py` and `src/pipeline/run_pairwise.py`.

1. **Load configs**
   - Load experiment config from `--config` (default `configs/experiment.yaml`).
   - Load model pool from `configs/models.yaml` (provider → model_id → config).
   - Load prompt templates from `configs/prompts.yaml` (currently used for **bias injection only**).
   - Load optional dataset registry from `configs/datasets.yaml`.

2. **Create a run directory**
   - Output directory is `experiment.output_dir` (or CLI `--output-dir` override).
   - A run directory is created with an auto-increment index:
     - `outputs/<dataset>_<data_type>_<bias>_<judge>_<NNN>/`

3. **Set random seed**
   - Controlled by `experiment.seed` or CLI `--seed`.

4. **Load input data (pairwise JSONL)**
   - `data.path` points to a JSONL file with fields: `id`, `question`, `answer_1`, `answer_2`, optional `preferred`.
   - If `preferred` is missing (`null`), the code uses a placeholder assumption:
     - `answer_1` is treated as GT, `answer_2` as non-GT.

5. **Initialize bias injector**
   - If `bias.enabled=true`, the pipeline injects bias into the **non-GT answer** (this is the only behavior implemented; `bias.inject_to` is not used to branch logic).
   - Bias injection backends:
     - `bias.injector_type=mock`: rule-based injection (`src/bias/builtin_biases.py`)
     - `bias.injector_type=openai|hf`: LLM-based rewriting (`src/bias/injector.py`), using `configs/prompts.yaml` if available for the given bias type, otherwise a generic prompt.

6. **Initialize judge**
   - Create a judge instance with `judge.provider` and `judge.model_id`.
   - For pairwise judging, the system/user prompt can be configured via `configs/prompts.yaml` under `judge.pairwise` (and falls back to the defaults in `src/judge/prompts.py` if not provided).

7. **Run judgments**
   - **Original comparison** (for Accuracy):
     - Judge compares `answer_1` vs `answer_2` once per sample.
   - **Bias comparison** (for RR/CR):
     - Inject bias into the non-GT answer → `biased_answer`.
     - Judge compares `GT` vs `biased_answer` **twice** (Round 1 and Round 2) to measure consistency.

8. **Compute metrics**
   - `Accuracy (original)`: whether the judge winner matches `preferred` (or a proxy if GT unavailable).
   - `RR (Robustness Rate)`: proportion of samples where the judge selects the GT answer against the biased answer.
   - `CR (Consistency Rate)`: proportion of samples where Round 1 winner == Round 2 winner.

9. **Save outputs**
   - Judgments and metrics are saved into the run directory (see Outputs section).

## 3. How to Run an Evaluation

### Entry point (main script)

The evaluation entry point is:

- `scripts/run_experiment.py` → calls `src.main.main()`

### Install dependencies

```bash
pip install -r requirements.txt
```

Notes:

- OpenAI / HuggingFace backends are listed as optional dependencies, but they are in `requirements.txt` by default in this repo.

### Required environment variables (only if using OpenAI)

- `OPENAI_API_KEY`

### Run command

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

### What should you edit for each experiment?

Typically you only need to edit:

- `configs/experiment.yaml`: selects dataset, judge, bias, and metrics toggles

You may also edit:

- `configs/models.yaml`: registers available models (model pool). For switching models **between runs**, you should only change `experiment.yaml` (model_id), not `models.yaml`.
- `configs/prompts.yaml`: bias injection prompt templates (only used when `bias.injector_type != mock`).

## 4. Key Configuration Parameters (Table)

The parameters below are the ones most researchers/engineers will modify in practice.

| Parameter | File | Example | What it does | When/why to change |
|---|---|---:|---|---|
| `--config` | `src/utils/cli.py` | `configs/experiment.yaml` | Select experiment config file | Run different experiments without editing the default file |
| `--output-dir` | `src/utils/cli.py` | `outputs` | Override `experiment.output_dir` | Separate outputs by machine/user/job |
| `--data-path` | `src/utils/cli.py` | `data/dummy/pairwise.jsonl` | Override `data.path` | Quick dataset switching for debugging |
| `--seed` | `src/utils/cli.py` | `42` | Override `experiment.seed` | Reproduce or test sensitivity to randomness |
| `experiment.seed` | `configs/experiment.yaml` | `42` | Random seed used by pipeline | Reproducibility |
| `experiment.output_dir` | `configs/experiment.yaml` | `outputs` | Base directory for run folders | Organize experiment outputs |
| `data.type` | `configs/experiment.yaml` | `pairwise` | Select pipeline type (`pairwise` implemented; `scalar` is placeholder) | Use `pairwise` for current evaluation |
| `data.path` | `configs/experiment.yaml` | `data/dummy/pairwise.jsonl` | Input JSONL path | Evaluate on your dataset |
| `data.dataset_name` | `configs/experiment.yaml` | `medical_eval_sphere` | Dataset name used for run dir prefix | Keep output dirs readable and stable |
| `configs/datasets.yaml` | `configs/datasets.yaml` | see below | Dataset registry (name → path) | Centralize dataset name/path mapping |
| `bias.enabled` | `configs/experiment.yaml` | `true` | Enable bias injection + RR/CR evaluation | Turn off to compute only original accuracy |
| `bias.type` | `configs/experiment.yaml` | `jargon_overloading` | Bias type used for injection | Study different bias styles (`jargon_overloading`, `authority`, `complexity`) |
| `bias.injector_type` | `configs/experiment.yaml` | `mock` / `openai` / `hf` | Bias injection backend | Use `mock` for offline; use LLM for more realistic rewriting |
| `bias.model_id` | `configs/experiment.yaml` | `gpt4omini` | **Model used for bias injection** (when injector_type is not `mock`) | Fix bias injector while changing judge, or vice versa |
| `judge.provider` | `configs/experiment.yaml` | `openai` / `hf` / `mock` | Judge backend provider | Compare different judges |
| `judge.model_id` | `configs/experiment.yaml` | `gpt52` | Which judge model to use (from model pool) | Main knob for judge model comparisons |
| `judge.allow_fallback_mock` | `configs/experiment.yaml` | `false` | If true, falls back to mock when judge model unavailable | Useful for CI or offline debugging |
| `evaluation.compute_original_acc` | `configs/experiment.yaml` | `true` | Compute accuracy on original `answer_1` vs `answer_2` | Disable to save cost/time if not needed |
| `evaluation.compute_bias_metrics` | `configs/experiment.yaml` | `true` | Compute RR/CR on `GT` vs `biased_answer` | Disable to run only original comparisons |
| `openai.<model_id>.model_name` | `configs/models.yaml` | `gpt-4o-mini` | Actual OpenAI model name | Register new models without changing code |
| `openai.defaults.api_key_env` | `configs/models.yaml` | `OPENAI_API_KEY` | Which env var holds the API key | Multiple keys / environments |
| `openai.defaults.max_tokens` | `configs/models.yaml` | `1000` | Token limit (mapped to `max_completion_tokens` for some models) | Control cost/length |
| `hf.<model_id>.model_name` | `configs/models.yaml` | `meta-llama/Llama-2-7b-chat-hf` | HuggingFace model id/path | Switch local models |
| `hf.defaults.device` | `configs/models.yaml` | `cuda` | Device for HF backend | CPU/GPU selection |
| `bias_injection.<type>.system/user` | `configs/prompts.yaml` | `{question}`, `{answer}` | Prompt templates for LLM-based bias rewriting | Customize injection behavior |
| `judge.pairwise.system/user` | `configs/prompts.yaml` | `{question}`, `{answer_a}`, `{answer_b}` | Prompt templates for pairwise judging | Tune judging criteria, output style, and robustness |

## 5. Outputs

After a successful **pairwise** evaluation run, a new run folder is created under:

- `experiment.output_dir/` (default: `outputs/`)

Folder naming pattern:

- `outputs/<dataset>_<data_type>_<bias>_<judge>_<NNN>/`
  - Example: `outputs/medical_eval_sphere_pairwise_jargon_overloading_openai_gpt-5.2_002/`
  - If `data.dataset_name` is missing, it will be inferred from `data.path` (file stem).

Files written to the run folder (pairwise pipeline):

- **`logs.txt`**: full run logs (console + debug logs)
- **`config_resolved.yaml`**: the resolved experiment config saved at runtime (includes derived fields such as run_dir; also persists resolved bias injector model_id)
- **`judge_raw_original.jsonl`**: per-sample judgments for original `answer_1` vs `answer_2`
- **`judge_raw_bias.jsonl`**: per-sample judgments for `GT` vs `biased_answer` (includes round1/round2 winners and a `consistent` flag)
- **`judge_raw.jsonl`**: concatenation of original + bias judgments
- **`metrics.json`**: full metrics object (Accuracy/RR/CR with metadata)
- **`results.csv`**: a flattened metrics summary (one row per metric)

Recommended usage:

- **Post-hoc analysis / visualization**: `judge_raw_*.jsonl`, `metrics.json`, `results.csv`
- **Paper/report writing**: `results.csv` (summary table) + selected examples from `judge_raw_bias.jsonl`
- **Debugging / audit trail**: `logs.txt`, `config_resolved.yaml`

## 6. Demo: Minimal Working Example

This demo evaluates the **robustness of a judge** when the non-GT answer is injected with **medical jargon overloading**.

### Command

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

### Minimal config snippet (pairwise + bias + judge)

```yaml
data:
  type: "pairwise"
  path: "data/dummy/pairwise.jsonl"
  dataset_name: "dummy_pairwise"

bias:
  enabled: true
  type: "jargon_overloading"
  injector_type: "openai"
  model_id: "gpt4omini"   # fixed bias injector model

judge:
  provider: "openai"
  model_id: "gpt52"       # judge model to evaluate

evaluation:
  compute_original_acc: true
  compute_bias_metrics: true

### Optional dataset registry

You can define dataset name ↔ path in `configs/datasets.yaml`:

```yaml
datasets:
  medical_eval_sphere:
    path: "data/medical_eval_sphere/medical_eval_sphere.jsonl"
  dummy_pairwise:
    path: "data/dummy/pairwise.jsonl"
```
```

### What you should see after running

In a newly created run directory under `outputs/`, you should see (at least):

- `logs.txt`
- `config_resolved.yaml`
- `judge_raw_original.jsonl`
- `judge_raw_bias.jsonl`
- `judge_raw.jsonl`
- `metrics.json`
- `results.csv`

