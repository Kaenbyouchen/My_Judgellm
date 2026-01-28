# 安全过滤器触发问题修复

## 问题描述

当 Gemini API 的安全过滤器触发时（`finish_reason: 2`），程序会返回 `[Response blocked by safety filter]`，然后 `judge_runner.py` 会返回 `tie` 结果。这些 `tie` 结果会被计入最终指标，**污染评估结果**。

## 解决方案

### 1. 添加无效标记

在 `JudgeResult` 类中添加了 `is_valid` 字段：
- `is_valid=True`：正常的判断结果
- `is_valid=False`：无效的判断结果（安全过滤器阻止、配额错误、超时等）

### 2. 标记无效结果

当检测到以下错误时，自动标记为无效：
- **安全过滤器阻止**：`[Response blocked by safety filter]`
- **配额错误**：`[Response blocked - API quota exceeded...]`
- **超时错误**：`[Response blocked - timeout...]`

### 3. 排除无效样本

所有指标计算函数（`compute_accuracy_original`, `compute_rr`, `compute_cr`）现在会：
- 自动过滤掉 `is_valid=False` 的判断
- 只基于有效判断计算指标
- 在报告中显示被排除的无效样本数量

## 修改的文件

1. **`src/judge/base.py`**
   - 添加 `is_valid` 字段到 `JudgeResult`
   - 更新 `to_dict()` 和 `from_dict()` 方法

2. **`src/judge/judge_runner.py`**
   - 安全过滤器/配额/超时错误时返回 `is_valid=False` 的结果

3. **`src/metrics/pairwise_metrics.py`**
   - `compute_accuracy_original()`：排除无效判断
   - `compute_rr()`：排除无效判断
   - `compute_cr()`：排除无效判断对

4. **`src/metrics/reports.py`**
   - 在指标报告中显示无效样本数量

## 使用效果

### 之前（污染评估）
```
Accuracy: 0.7500
Total: 400 samples
Ties: 50  ← 包含被安全过滤器阻止的样本
```

### 现在（排除无效样本）
```
Accuracy: 0.8000
Total: 400 samples
Valid samples: 350
⚠️  Invalid samples excluded: 50 (safety filter, quota errors, etc.)
```

## 指标计算逻辑

- **Accuracy (Acc)**：只基于有效判断计算
- **Robustness Rate (RR)**：只基于有效判断计算
- **Consistency Rate (CR)**：只基于两轮都有效的判断对计算

## 注意事项

1. **无效样本仍会保存**：无效的判断结果仍会保存到 JSONL 文件中，但标记了 `"is_valid": false`
2. **指标更准确**：最终指标只基于有效的判断，不会被错误响应污染
3. **可见性**：报告中会明确显示有多少样本被排除，以及排除的原因

## 示例输出

```
============================================================
Evaluation Metrics Summary
============================================================

Accuracy (Original R1 vs R2):
  Value: 0.7850
  Proxy: False
  Has GT: True
  Total: 400
  Valid samples: 350
  ⚠️  Invalid samples excluded: 50 (safety filter, quota errors, etc.)
  Note: Accuracy based on judge winner matching preferred label

Robustness Rate (RR):
  Value: 0.7200
  GT Wins: 252/350
  Has GT Labels: True
  Valid samples: 350
  ⚠️  Invalid samples excluded: 50 (safety filter, quota errors, etc.)
  Note: RR = proportion of judge selecting GT answer against biased answer
```

现在，即使 Gemini API 的安全过滤器频繁触发，评估结果也不会被污染！
