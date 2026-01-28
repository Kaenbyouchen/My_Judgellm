# 断点续训功能说明

## 功能概述

断点续训功能允许你在程序意外中断后，重新运行相同的命令继续从未完成的样本开始评估，而无需从头开始。

## 工作原理

1. **自动检测**：程序会自动检测输出目录中是否已有部分结果
2. **跳过已完成**：自动识别已完成的样本 ID，跳过这些样本
3. **继续运行**：从未完成的样本继续评估
4. **追加保存**：新结果会追加到现有文件，不会覆盖已有结果

## 使用方法

### 基本使用

**无需任何额外操作！** 只需重新运行相同的命令：

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

程序会自动：
- 检测到输出目录中已有部分结果
- 显示恢复信息（已完成多少样本，剩余多少样本）
- 继续从未完成的样本开始评估

### 示例输出

当检测到可恢复的运行时，你会看到：

```
============================================================
🔄 Resume mode detected!
  Original judgments completed: 250
  Bias judgments completed: 0
============================================================
Loaded 400 samples
Resuming: 250 samples already completed, 150 remaining
```

### 输出文件

程序会在以下位置保存结果：

- `outputs/<run_name>/judge_raw_original.jsonl` - 原始评估结果
- `outputs/<run_name>/judge_raw_bias.jsonl` - 偏差评估结果
- `outputs/<run_name>/judge_raw.jsonl` - 合并所有结果
- `outputs/<run_name>/metrics.json` - 评估指标
- `outputs/<run_name>/results.csv` - 指标摘要

## 注意事项

1. **配置一致性**：恢复运行时，请确保使用**相同的配置**（数据集、模型、bias 类型等），否则可能导致结果不一致。

2. **输出目录**：程序通过输出目录来识别可恢复的运行。确保：
   - 使用相同的 `output_dir` 配置
   - 或者使用相同的 `run_name`（如果配置了）

3. **部分完成**：
   - 如果只完成了 Step 1（原始评估），程序会继续完成 Step 1，然后执行 Step 2
   - 如果只完成了 Step 2 的部分样本，程序会继续完成 Step 2

4. **指标计算**：最终指标会在所有样本完成后重新计算，确保准确性。

## 故障排除

### 如果程序没有检测到可恢复的运行

1. 检查输出目录是否存在：`outputs/<run_name>/`
2. 检查是否有 `judge_raw_original.jsonl` 或 `judge_raw_bias.jsonl` 文件
3. 确保文件格式正确（有效的 JSONL 格式）

### 如果恢复后结果不正确

1. 检查配置是否与原始运行一致
2. 检查数据集是否相同
3. 如果问题持续，可以删除输出目录重新开始

## 技术细节

- 程序通过读取 JSONL 文件中的 `sample_id` 字段来识别已完成的样本
- 支持多行格式的 JSON（pretty-printed JSONL）
- 新结果以追加模式写入，不会覆盖已有结果
- 最终指标基于所有样本（包括已完成的和新完成的）重新计算
