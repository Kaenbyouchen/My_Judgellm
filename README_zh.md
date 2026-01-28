# JudgeLLM — 医疗问答偏置评测框架

一个轻量级框架，用于评估 LLM-as-a-Judge 在医疗问答场景下的鲁棒性，通过向答案注入可控的偏置风格进行测试。

## 项目概述

本项目测试当候选答案被注入特定偏置风格（如过度医学术语、权威引用）时，判断模型是否仍能识别出最佳答案。评测指标包括：

- **准确率 (Accuracy)**：判断模型在原始对比中能否选择正确答案？
- **鲁棒率 (RR)**：当注入偏置后，判断模型是否仍偏好真实答案？
- **一致率 (CR)**：重复评估时判断结果是否一致？

## 核心功能

- ✅ **多种偏置类型**：`jargon_overloading`（术语过载）、`authority`（权威背书）、`complexity`（复杂化）
- ✅ **多种判断后端**：Mock（离线）、OpenAI、HuggingFace、Gemini、Anthropic
- ✅ **支持最新模型**：Gemini 3 Pro/Flash、Claude Opus 4.5、Claude Sonnet 4.5
- ✅ **配置驱动**：所有设置通过 YAML 文件管理
- ✅ **成对评估**：并排比较两个答案
- ✅ **完整指标**：准确率、RR、CR 及详细输出

## 安装步骤

### 环境要求

- Python 3.9+
- pip

### 创建虚拟环境

```bash
python3 -m venv Judgellm
source Judgellm/bin/activate  # Windows: Judgellm\Scripts\activate
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 设置环境变量（可选）

如果使用 API 模型：

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 2. 使用 Mock 模式运行（无需 API Key）

```bash
python scripts/run_experiment.py --config configs/test_experiment.yaml
```

这将使用 mock 判断器和偏置注入器，适合测试框架功能。

### 3. 使用真实模型运行

编辑 `configs/experiment.yaml` 使用真实模型：

```yaml
judge:
  provider: "openai"  # 或 "gemini", "anthropic"
  model_id: "gpt52"  # 或 "gemini3_pro", "claude_opus_45" 等

bias:
  injector_type: "openai"  # 或 "mock", "gemini", "anthropic"
  model_id: "gpt4omini"  # 或 "gemini3_flash" 等
```

然后运行：

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

## 配置说明

主配置文件是 `configs/experiment.yaml`：

```yaml
# 数据配置
data:
  type: "pairwise"
  path: "data/dummy/pairwise.jsonl"
  dataset_name: "dummy"

# 偏置注入
bias:
  enabled: true
  type: "jargon_overloading"  # jargon_overloading, authority, complexity
  injector_type: "mock"  # mock, openai, hf, gemini

# 判断模型
judge:
  provider: "mock"  # mock, openai, hf, gemini, anthropic
  model_id: "mock-judge-v1"

# 评估配置
evaluation:
  compute_original_acc: true
  compute_bias_metrics: true
```

### 关键参数

| 参数 | 说明 | 选项 |
|------|------|------|
| `bias.type` | 要注入的偏置类型 | `jargon_overloading`, `authority`, `complexity` |
| `bias.injector_type` | 偏置注入方式 | `mock`（离线）, `openai`, `hf`, `gemini` |
| `judge.provider` | 判断后端 | `mock`, `openai`, `hf`, `gemini`, `anthropic` |
| `judge.model_id` | 使用的具体模型 | 见 `configs/models.yaml` |

## 支持的模型

框架支持多个模型提供商。可用模型在 `configs/models.yaml` 中配置：

### OpenAI
- `gpt4omini` → `gpt-4o-mini`
- `gpt41` → `gpt-4.1`
- `gpt52` → `gpt-5.2`
- `gpt5mini` → `gpt-5-mini`

### Anthropic (Claude)
- `claude35_sonnet` → `claude-3-5-sonnet-20240620`
- `claude_opus_45` → `claude-opus-4.5` ⭐ **最新**
- `claude_sonnet_45` → `claude-sonnet-4.5` ⭐ **最新**

### Gemini
- `gemini3_pro` → `gemini-3-pro` ⭐ **最新**
- `gemini3_flash` → `gemini-3-flash` ⭐ **最新**

### HuggingFace

#### General SOTA (通用 SOTA 模型)
- `qwen3_next_80b_a3b_instruct` → `Qwen/Qwen3-Next-80B-A3B-Instruct`

#### Medical SOTA (医疗领域 SOTA 模型)
- `medgemma_4b` → `google/medgemma-4b-it`
- `medgemma_27b` → `google/medgemma-27b-it`
- `biomistral_7b` → `BioMistral/BioMistral-7B`
- `medical_qwen3_14b_1218` → `zjydiary/Medical-Qwen3-14B-1218` (ModelScope)

#### Judge Expert (判断专家模型)
- `m_prometheus_3b` → `Unbabel/M-Prometheus-3B`
- `m_prometheus_7b` → `Unbabel/M-Prometheus-7B`
- `m_prometheus_14b` → `Unbabel/M-Prometheus-14B`

#### Small Models (小型模型)
- `gemma3_4b` → `google/gemma-3-4b-it`
- `gemma3_12b` → `google/gemma-3-12b-it`
- `gemma3_27b` → `google/gemma-3-27b-it`
- `qwen3_4b_instruct` → `Qwen/Qwen3-4B-Instruct-2507`
- `qwen3_14b_instruct` → `OpenPipe/Qwen3-14B-Instruct`
- `llama32_3b_instruct` → `meta-llama/Llama-3.2-3B-Instruct`
- `llama31_8b_instruct` → `meta-llama/Llama-3.1-8B-Instruct`
- `llama4_scout_instruct` → `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `llama33_70b_instruct` → `meta-llama/Llama-3.3-70B-Instruct`
- `deepseek_r1_distill_llama_8b` → `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek_r1_distill_qwen_7b` → `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek_r1_distill_qwen_14b` → `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

### Mock（离线）
- `mock-judge-v1` → 无需 API key 的测试模型

要使用某个模型，在 `configs/experiment.yaml` 中设置 `judge.model_id` 或 `bias.model_id` 为上面列出的模型 ID。

## 项目结构

```
Ruishan_Judgellm_Ours/
├── configs/              # 配置文件
│   ├── experiment.yaml   # 主实验配置
│   ├── models.yaml       # 模型注册表
│   └── prompts.yaml      # 提示词模板
├── data/                 # 数据集
│   └── dummy/            # 测试数据
├── outputs/              # 实验结果
├── scripts/              # 入口脚本
│   └── run_experiment.py # 主脚本
└── src/                  # 源代码
    ├── bias/             # 偏置注入
    ├── judge/            # 判断模型
    ├── metrics/          # 评估指标
    └── pipeline/         # 评估流程
```

## 预期输出

运行后，结果保存在 `outputs/<运行名称>/`：

- **`results.csv`** - 指标摘要（准确率、RR、CR）
- **`metrics.json`** - 详细指标及元数据
- **`judge_raw_original.jsonl`** - 原始对比结果
- **`judge_raw_bias.jsonl`** - 偏置注入后的对比结果
- **`logs.txt`** - 完整执行日志
- **`config_resolved.yaml`** - 最终解析的配置

示例输出：

```
Accuracy (Original): 1.0000
Robustness Rate (RR): 1.0000
Consistency Rate (CR): 1.0000
```

## 扩展框架

### 添加新的偏置类型

1. 在 `src/bias/builtin_biases.py` 中创建新类：

```python
class MyBias(BaseBias):
    def apply(self, text: str, context: Optional[Dict] = None) -> str:
        # 你的偏置注入逻辑
        return modified_text
```

2. 在 `BUILTIN_BIASES` 中注册：

```python
BUILTIN_BIASES = {
    "my_bias": MyBias,
    # ... 现有偏置类型
}
```

### 添加新的判断后端

1. 在 `src/models/` 中创建客户端（如 `my_client.py`）
2. 在 `src/models/registry.py` 中注册
3. 在 `configs/models.yaml` 中添加模型配置

## 常见问题

### Q: 如何使用自己的数据集？

A: 创建 JSONL 文件，格式如下：
```json
{"id": "sample_1", "question": "...", "answer_1": "...", "answer_2": "...", "preferred": "1"}
```

然后在 `configs/experiment.yaml` 中设置 `data.path`。

### Q: API Key 不工作？

A: 确保环境变量已设置：
```bash
echo $OPENAI_API_KEY  # 应该显示你的 key
```

如果未设置，添加到 `~/.zshrc` 或 `~/.bashrc`：
```bash
export OPENAI_API_KEY="your-key"
```

### Q: 可以不使用 API Key 运行吗？

A: 可以！对判断器和偏置注入器都使用 `mock` 模式。参考 `configs/test_experiment.yaml`。

### Q: 如何更换判断模型？

A: 编辑 `configs/experiment.yaml` 中的 `judge.model_id`。可用模型列表见 `configs/models.yaml`。

### Q: 遇到导入错误怎么办？

A: 确保虚拟环境已激活且依赖已安装：
```bash
source Judgellm/bin/activate
pip install -r requirements.txt
```

## 许可证

[添加你的许可证]

## 引用

[如适用，添加引用信息]
