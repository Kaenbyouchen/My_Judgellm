# 项目依赖检查报告

## ✅ 必需依赖（已在 requirements.txt）

以下依赖是项目运行所必需的，已在 `requirements.txt` 中列出：

| 包名 | 用途 | 代码位置 |
|------|------|---------|
| `pyyaml>=6.0` | 读取/写入 YAML 配置文件 | `src/utils/io.py` |
| `jsonlines>=3.1.0` | 读取/写入 JSONL 数据文件 | `src/dataset/loaders.py`, `src/metrics/reports.py` |
| `numpy>=1.24.0,<2.0.0` | 数值计算、随机种子设置 | `src/utils/reproducibility.py` |
| `pandas>=2.0.0` | 数据处理、CSV 导出 | `src/metrics/reports.py` |
| `tqdm>=4.65.0` | 进度条显示 | `src/pipeline/run_pairwise.py` |
| `loguru>=0.7.0` | 日志记录 | 所有模块 |
| `matplotlib>=3.7.0` | 数据可视化（预留） | 未直接使用，但可能在分析脚本中使用 |
| `seaborn>=0.12.0` | 数据可视化（预留） | 未直接使用，但可能在分析脚本中使用 |

## ⚠️ 可选依赖（未在 requirements.txt）

以下依赖是可选的，只有在使用特定功能时才需要：

### 1. OpenAI API（用于调用 GPT-4 等模型）

```bash
pip install openai
```

**用途**：
- 作为 JudgeLLM（`judge.type: "openai"`）
- 作为 Bias 注入的 GenAI（`bias.injector_type: "openai"`）

**代码位置**：`src/models/openai_client.py`

**说明**：如果未安装，代码会自动回退到 mock 模式，不会报错。

### 2. HuggingFace Transformers（用于本地开源模型）

```bash
pip install transformers torch
```

**用途**：
- 作为 JudgeLLM（`judge.type: "hf"`）
- 作为 Bias 注入的 GenAI（`bias.injector_type: "hf"`）

**代码位置**：`src/models/hf_client.py`

**说明**：
- 如果未安装，代码会自动回退到 mock 模式
- 如果使用 GPU，需要安装支持 CUDA 的 PyTorch 版本

### 3. vLLM（用于高效推理，当前未实现）

```bash
pip install vllm
```

**状态**：配置文件中已有占位，但代码中尚未实现。

## 📦 标准库（无需安装）

以下模块是 Python 标准库，无需额外安装：

- `json` - JSON 处理
- `os`, `sys` - 系统操作
- `pathlib` - 路径处理
- `typing` - 类型注解
- `abc` - 抽象基类
- `dataclasses` - 数据类
- `argparse` - 命令行参数解析
- `random` - 随机数生成
- `re` - 正则表达式

## 🔍 依赖检查命令

### 检查已安装的包

```bash
# 检查必需依赖
pip list | grep -E "pyyaml|jsonlines|numpy|pandas|tqdm|loguru|matplotlib|seaborn"

# 检查可选依赖
pip list | grep -E "openai|transformers|torch"
```

### 验证安装

```bash
python -c "import yaml, jsonlines, numpy, pandas, tqdm, loguru; print('✓ 所有必需依赖已安装')"
```

### 安装所有必需依赖

```bash
pip install -r requirements.txt
```

### 安装可选依赖（如果需要）

```bash
# 安装 OpenAI（用于 GPT-4 等）
pip install openai

# 安装 HuggingFace（用于本地模型）
pip install transformers torch

# 如果需要 GPU 支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 建议

### 当前状态

✅ **所有必需依赖已在 requirements.txt 中**

项目可以在 mock 模式下运行，无需安装可选依赖。

### 如果需要使用真实模型

1. **使用 OpenAI API**：
   ```bash
   pip install openai
   export OPENAI_API_KEY="your-api-key"
   ```

2. **使用 HuggingFace 模型**：
   ```bash
   pip install transformers torch
   ```

### 依赖版本说明

- `numpy<2.0.0`：限制 NumPy 版本以避免与其他包的兼容性问题
- 其他包使用 `>=` 指定最低版本，允许使用更新版本

## 🚨 已知问题

1. **NumPy 版本冲突**：
   - 如果安装了 NumPy 2.0+，可能导致 pandas/scipy 不兼容
   - 解决方案：`pip install "numpy>=1.24.0,<2.0.0"`

2. **可选依赖警告**：
   - 如果未安装 `openai` 或 `transformers`，运行时会显示警告
   - 这是正常的，代码会自动回退到 mock 模式


