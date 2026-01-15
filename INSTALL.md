# 安装指南

## 快速安装（推荐）

一键安装所有依赖（包括可选依赖）：

```bash
pip install -r requirements.txt
```

## 分步安装

### 1. 仅安装必需依赖（Mock 模式）

如果只需要运行 mock 模式，可以只安装必需依赖：

```bash
pip install pyyaml jsonlines "numpy>=1.24.0,<2.0.0" pandas tqdm loguru matplotlib seaborn
```

### 2. 安装可选依赖（真实模型）

如果需要使用真实模型，额外安装：

```bash
# OpenAI API
pip install openai

# HuggingFace 模型
pip install transformers torch
```

## 验证安装

运行以下命令验证所有依赖是否安装成功：

```bash
python -c "
import yaml, jsonlines, numpy, pandas, tqdm, loguru
print('✓ 必需依赖已安装')

try:
    import openai
    print('✓ OpenAI 已安装')
except ImportError:
    print('⚠ OpenAI 未安装（可选）')

try:
    import transformers, torch
    print('✓ HuggingFace 已安装')
except ImportError:
    print('⚠ HuggingFace 未安装（可选）')
"
```

## 常见问题

### 1. NumPy 版本冲突

如果遇到 NumPy 版本问题：

```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### 2. PyTorch GPU 支持

如果需要 GPU 支持，安装 CUDA 版本的 PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 依赖冲突

如果遇到依赖冲突，建议使用虚拟环境：

```bash
# 使用 conda
conda create -n judge_llm python=3.10 -y
conda activate judge_llm
pip install -r requirements.txt

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 依赖说明

- **必需依赖**：项目运行所必需，未安装会导致运行失败
- **可选依赖**：用于真实模型调用，未安装时会自动使用 mock 模式

详细信息请查看 `DEPENDENCIES.md`。


