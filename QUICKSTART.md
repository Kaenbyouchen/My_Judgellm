# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行 Dummy Demo

在项目根目录下运行：

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

或者使用完整路径：

```bash
python "D:\USC Research Ruishan Liu\LLM\Judge_LLM_Cursor\judge_llm_medical_bias\scripts\run_experiment.py" --config "D:\USC Research Ruishan Liu\LLM\Judge_LLM_Cursor\judge_llm_medical_bias\configs\experiment.yaml"
```

## 3. 查看结果

运行成功后，结果会保存在 `outputs/` 目录：

- `outputs/judgments/` - 判断结果（JSONL格式）
- `outputs/metrics/` - 指标结果（JSON和CSV格式）
- `outputs/runs/<timestamp>/run.log` - 运行日志

## 4. 预期输出

运行成功后会看到类似以下输出：

```
============================================================
Evaluation Metrics Summary
============================================================

Accuracy (Original R1 vs R2):
  Value: 1.0000
  Proxy: False
  Has GT: True
  Total: 8

Robustness Rate (RR):
  Value: 1.0000
  GT Wins: 8/8
  Has GT Labels: True

Consistency Rate (CR):
  Value: 0.xxxx
  Method: score_difference
  Total: 8
```

## 注意事项

- 首次运行需要安装依赖（`pip install -r requirements.txt`）
- Mock 模式无需 GPU 或 API key
- 所有路径都是相对于项目根目录的


