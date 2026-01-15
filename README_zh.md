# JudgeLLM 医学问答偏置评测框架（Medical Bias）

本仓库提供一个**配置驱动**的评测框架，用于研究 **LLM-as-a-Judge 在医学问答（Medical QA）场景下的偏置问题**。核心思路是：对某个候选答案进行可控的“偏置风格”注入，然后评估 **Judge 模型**在面对带偏置答案时是否仍能稳定选择更优/更接近 GT（preferred）的答案。

本文档**严格基于仓库当前代码与配置文件**进行总结：若某个配置存在但未在代码路径中使用，会明确标注为“当前未接入/未使用”，而不是猜测其行为。

## 1. 项目概述（Project Overview）

### 本项目解决什么问题？

在医学问答中，答案可能因为**表面风格**而显得更“可信”（例如术语堆砌、权威背书、复杂句式），即使其医学内容并不更好。当我们使用 **LLM-as-a-Judge** 做答案比较时，Judge 本身可能对这些风格产生偏好，从而导致评测结果不可靠。

本项目评测的重点是：

- **Judge 模型在带偏置答案干扰下的鲁棒性**（是否仍选择 GT/更优答案）。
- Judge 在重复评估同一对答案时的**一致性**。

### 这里所说的 bias 指什么？

本仓库中的 bias 指的是对答案进行**可控的文本风格/措辞变换**，并非人口统计学或社会公平意义上的偏置。当前内置的 bias 类型包括：

- `jargon_overloading`：医学术语过载（可规则注入，也可用 LLM 重写）
- `authority`：权威背书/指南引用（规则注入）
- `complexity`：不必要的复杂化表达（规则注入）

### JudgeLLM 在整体评测流程中扮演什么角色？

JudgeLLM 指 **Judge 模型**本身，它对两段答案做 **pairwise 比较**并输出赢家：

- `A`、`B` 或 `tie`

Judge 可选后端：

- `mock`：启发式规则 Judge（无需外部依赖）
- `openai`：OpenAI Chat 模型
- `hf`：本地 HuggingFace 模型（需安装可选依赖）

## 2. 整体评测流程（Evaluation Pipeline）

主流程位于 `src/main.py` 与 `src/pipeline/run_pairwise.py`。

1. **加载配置**
   - 从 `--config` 加载实验配置（默认 `configs/experiment.yaml`）。
   - 从 `configs/models.yaml` 加载模型池（provider → model_id → config）。
   - 从 `configs/prompts.yaml` 加载 prompt 模板（当前仅用于 **bias 注入**）。
   - 可选：从 `configs/datasets.yaml` 加载数据集注册表。

2. **创建本次 run 目录**
   - 输出根目录为 `experiment.output_dir`（或 CLI `--output-dir` 覆盖）。
   - 自动创建带递增编号的 run 目录：
     - `outputs/<dataset>_<data_type>_<bias>_<judge>_<NNN>/`

3. **设置随机种子**
   - 来自 `experiment.seed`（或 CLI `--seed` 覆盖）。

4. **加载输入数据（pairwise JSONL）**
   - `data.path` 指向 JSONL 文件，每行包含：`id`、`question`、`answer_1`、`answer_2`，可选 `preferred`。
   - 若 `preferred` 缺失（`null`），代码使用占位假设：
     - `answer_1` 视为 GT，`answer_2` 视为非 GT。

5. **初始化 bias 注入器**
   - 当 `bias.enabled=true` 时，流程会将 bias 注入到**非 GT 答案**（当前唯一实现；`bias.inject_to` 参数不会影响逻辑分支）。
   - 注入后端：
     - `bias.injector_type=mock`：规则注入（`src/bias/builtin_biases.py`）
     - `bias.injector_type=openai|hf`：LLM 重写注入（`src/bias/injector.py`），若 `configs/prompts.yaml` 为该 bias 类型提供模板则使用模板，否则使用通用 prompt。

6. **初始化 Judge**
   - 通过 `judge.provider` + `judge.model_id` 创建 Judge。
   - pairwise judge 的 system/user prompt 可通过 `configs/prompts.yaml` 的 `judge.pairwise` 配置（若未提供则回退到 `src/judge/prompts.py` 中的默认值）。

7. **进行判断（judgments）**
   - **原始比较（用于 Accuracy）**：
     - 对每个样本判断一次 `answer_1` vs `answer_2`。
   - **偏置比较（用于 RR/CR）**：
     - 对非 GT 答案注入偏置得到 `biased_answer`。
     - 对 `GT` vs `biased_answer` 判断**两次**（Round 1 + Round 2）用于一致性计算。

8. **汇总与统计指标**
   - `Accuracy (original)`：原始 `answer_1` vs `answer_2` 的赢家是否与 `preferred` 一致（无 GT 时使用 proxy）。
   - `RR (Robustness Rate)`：面对 `GT` vs `biased_answer`，Judge 选择 GT 的比例。
   - `CR (Consistency Rate)`：Round 1 winner 与 Round 2 winner 相同的比例。

9. **保存输出**
   - judgments 与 metrics 统一写入 run 目录（见 Outputs 章节）。

## 3. 如何运行一次 Evaluation（How to Run）

### 主入口脚本

评测入口为：

- `scripts/run_experiment.py` → 调用 `src.main.main()`

### 安装依赖

```bash
pip install -r requirements.txt
```

说明：

- OpenAI / HuggingFace 后端属于可选能力，但在本仓库的 `requirements.txt` 中默认列出。

### 环境变量（使用 OpenAI 时需要）

- `OPENAI_API_KEY`

### 运行命令

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

### 每次跑实验前应该改哪些配置？

通常只需要修改：

- `configs/experiment.yaml`：选择数据集、bias、judge、是否计算各项指标等

必要时还会修改：

- `configs/models.yaml`：登记可用模型（模型池）。**切换模型进行对比实验时，推荐只改 `experiment.yaml` 的 model_id**，而不是改 `models.yaml`。
- `configs/prompts.yaml`：仅在 `bias.injector_type != mock` 时用于 bias 注入的 prompt 模板。
- `configs/datasets.yaml`：可选的数据集注册表（name → path）。

## 4. 可修改参数说明（重点）

下表列出评测中最常用、最重要、需要关注的参数。

| 参数名 | 所在文件 | 示例取值 | 作用说明 | 何时/为何修改 |
|---|---|---:|---|---|
| `--config` | `src/utils/cli.py` | `configs/experiment.yaml` | 指定实验配置文件路径 | 同一套代码跑多组实验配置 |
| `--output-dir` | `src/utils/cli.py` | `outputs` | 覆盖 `experiment.output_dir` | 按机器/用户/任务分离输出 |
| `--data-path` | `src/utils/cli.py` | `data/dummy/pairwise.jsonl` | 覆盖 `data.path` | 临时切换数据集做调试 |
| `--seed` | `src/utils/cli.py` | `42` | 覆盖 `experiment.seed` | 控制/复现实验随机性 |
| `experiment.seed` | `configs/experiment.yaml` | `42` | 评测全流程使用的随机种子 | 可复现性控制 |
| `experiment.output_dir` | `configs/experiment.yaml` | `outputs` | 输出根目录 | 统一管理实验输出 |
| `data.type` | `configs/experiment.yaml` | `pairwise` | 评测模式（当前实现 `pairwise`；`scalar` 为占位） | 当前应使用 `pairwise` |
| `data.path` | `configs/experiment.yaml` | `data/dummy/pairwise.jsonl` | 输入 JSONL 路径 | 替换为你的数据集 |
| `data.dataset_name` | `configs/experiment.yaml` | `medical_eval_sphere` | 运行目录前缀中的数据集名 | 输出目录可读、稳定 |
| `configs/datasets.yaml` | `configs/datasets.yaml` | 见下方 | 数据集注册表（name → path） | 集中管理数据集名称与路径 |
| `bias.enabled` | `configs/experiment.yaml` | `true` | 是否启用偏置注入与 RR/CR 评测 | 只算原始 Accuracy 时可关掉 |
| `bias.type` | `configs/experiment.yaml` | `jargon_overloading` | 要注入的偏置类型 | 对比不同偏置风格（`jargon_overloading`/`authority`/`complexity`） |
| `bias.injector_type` | `configs/experiment.yaml` | `mock/openai/hf` | 偏置注入后端 | 离线用 `mock`；更真实重写用 LLM |
| `bias.model_id` | `configs/experiment.yaml` | `gpt4omini` | **用于 bias 注入的模型**（当 injector_type 非 mock） | 固定注入模型，单独对比不同 judge |
| `judge.provider` | `configs/experiment.yaml` | `openai/hf/mock` | Judge 后端类型 | 对比不同 Judge 体系 |
| `judge.model_id` | `configs/experiment.yaml` | `gpt52` | Judge 模型选择（来自模型池） | 主要用于对比不同 Judge 模型 |
| `judge.allow_fallback_mock` | `configs/experiment.yaml` | `false` | Judge 不可用时是否回退到 mock | CI/离线调试时可开启 |
| `evaluation.compute_original_acc` | `configs/experiment.yaml` | `true` | 是否计算原始 `answer_1` vs `answer_2` 的 Accuracy | 节省成本/时间时可关闭 |
| `evaluation.compute_bias_metrics` | `configs/experiment.yaml` | `true` | 是否计算 RR/CR（GT vs biased） | 仅关注原始比较时可关闭 |
| `openai.<model_id>.model_name` | `configs/models.yaml` | `gpt-4o-mini` | OpenAI 真实模型名 | 在不改代码的情况下注册新模型 |
| `openai.defaults.api_key_env` | `configs/models.yaml` | `OPENAI_API_KEY` | API key 的环境变量名 | 多 key/多环境切换 |
| `openai.defaults.max_tokens` | `configs/models.yaml` | `1000` | token 上限（对部分模型内部映射为 `max_completion_tokens`） | 控制生成长度/成本 |
| `hf.<model_id>.model_name` | `configs/models.yaml` | `meta-llama/Llama-2-7b-chat-hf` | HuggingFace 模型名/路径 | 切换本地模型 |
| `hf.defaults.device` | `configs/models.yaml` | `cuda` | HF 设备选择 | CPU/GPU 切换 |
| `bias_injection.<type>.system/user` | `configs/prompts.yaml` | `{question}` `{answer}` | LLM bias 注入 prompt 模板 | 控制注入方式与风格 |
| `judge.pairwise.system/user` | `configs/prompts.yaml` | `{question}` `{answer_a}` `{answer_b}` | pairwise judge prompt 模板 | 调整 judge 的评判标准与输出风格 |

## 5. 输出结果说明（Outputs）

完成一次 **pairwise** 评测后，会在以下目录下创建一个新的 run 文件夹：

- `experiment.output_dir/`（默认 `outputs/`）

目录命名格式：

- `outputs/<dataset>_<data_type>_<bias>_<judge>_<NNN>/`
  - 示例：`outputs/medical_eval_sphere_pairwise_jargon_overloading_openai_gpt-5.2_002/`
  - 若 `data.dataset_name` 缺失，将从 `data.path` 的文件名推断。

pairwise 流程会输出以下文件：

- **`logs.txt`**：本次运行日志（包含 debug 信息）
- **`config_resolved.yaml`**：运行时保存的解析后配置（包含 run_dir、并记录实际使用的 bias 注入 model_id）
- **`judge_raw_original.jsonl`**：原始 `answer_1` vs `answer_2` 的逐样本判决
- **`judge_raw_bias.jsonl`**：`GT` vs `biased_answer` 的逐样本判决（包含 round1/round2 winner 与 `consistent` 标记）
- **`judge_raw.jsonl`**：原始判决 + 偏置判决的合并文件
- **`metrics.json`**：完整指标（Accuracy/RR/CR 及其 metadata）
- **`results.csv`**：指标摘要（每个指标一行，便于表格化分析）

推荐用途：

- **后续分析/可视化**：`judge_raw_*.jsonl`、`metrics.json`、`results.csv`
- **论文/报告撰写**：`results.csv`（汇总表）+ 从 `judge_raw_bias.jsonl` 挑选典型案例
- **调试/审计**：`logs.txt`、`config_resolved.yaml`

## 6. Demo 示例（非常重要）

该 demo 评测内容：当“非 GT 答案”被注入 **医学术语过载**后，Judge 是否仍能选择 GT（并统计 RR/CR）。

### 运行命令

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

### 最小配置片段（pairwise + bias + judge）

```yaml
data:
  type: "pairwise"
  path: "data/dummy/pairwise.jsonl"
  dataset_name: "dummy_pairwise"

bias:
  enabled: true
  type: "jargon_overloading"
  injector_type: "openai"
  model_id: "gpt4omini"   # 固定注入模型

judge:
  provider: "openai"
  model_id: "gpt52"       # 被评测的 judge 模型

evaluation:
  compute_original_acc: true
  compute_bias_metrics: true

### 可选：数据集注册表

你可以在 `configs/datasets.yaml` 中定义数据集名称与路径：

```yaml
datasets:
  medical_eval_sphere:
    path: "data/medical_eval_sphere/medical_eval_sphere.jsonl"
  dummy_pairwise:
    path: "data/dummy/pairwise.jsonl"
```
```

### 运行后你应该看到哪些输出文件

在 `outputs/` 下新生成的 run 目录中，至少会包含：

- `logs.txt`
- `config_resolved.yaml`
- `judge_raw_original.jsonl`
- `judge_raw_bias.jsonl`
- `judge_raw.jsonl`
- `metrics.json`
- `results.csv`


