# 运行指南

## 方法一：从项目根目录运行（推荐）

### 1. 进入项目目录

```bash
cd judge_llm_medical_bias
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行实验

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

## 方法二：从当前目录运行

如果你在 `Judge_LLM_Cursor` 目录下，可以直接运行：

```bash
# 安装依赖
pip install -r judge_llm_medical_bias/requirements.txt

# 运行实验
python judge_llm_medical_bias/scripts/run_experiment.py --config judge_llm_medical_bias/configs/experiment.yaml
```

## 方法三：使用绝对路径

```bash
python "D:\USC Research Ruishan Liu\LLM\Judge_LLM_Cursor\judge_llm_medical_bias\scripts\run_experiment.py" --config "D:\USC Research Ruishan Liu\LLM\Judge_LLM_Cursor\judge_llm_medical_bias\configs\experiment.yaml"
```

## 预期输出

运行成功后会看到：

1. **控制台输出**：
   - 加载数据的日志
   - 评估进度条
   - 指标摘要表格

2. **生成的文件**（在 `outputs/` 目录）：
   - `outputs/judgments/original_<timestamp>.jsonl` - 原始判断结果
   - `outputs/judgments/bias_<timestamp>.jsonl` - 偏见判断结果
   - `outputs/metrics/metrics_<timestamp>.json` - 完整指标
   - `outputs/metrics/summary_<timestamp>.csv` - 指标摘要
   - `outputs/runs/<timestamp>/run.log` - 详细日志

## 常见问题

### 问题1：ModuleNotFoundError

如果遇到 `ModuleNotFoundError: No module named 'xxx'`，请安装依赖：

```bash
pip install -r requirements.txt
```

### 问题2：找不到配置文件

确保在项目根目录运行，或使用完整路径。

### 问题3：路径包含空格

如果路径包含空格（如 "USC Research"），使用引号包裹路径。

## 验证安装

运行以下命令验证项目结构：

```bash
python -c "import sys; sys.path.insert(0, 'judge_llm_medical_bias'); from src.dataset.loaders import load_pairwise_jsonl; print('✓ 导入成功')"
```


