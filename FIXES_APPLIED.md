# 修复应用总结

## 修复内容

### 1. ✅ 批量脚本 Provider 推断修复

**问题**：
- 批量脚本 `batch_evaluate.py` 中的 `parse_judge_spec()` 使用简单字符串匹配推断 provider
- HF 模型（如 `qwen3_4b_instruct`, `gemma3_4b` 等）会被错误推断为 `openai`

**修复**：
- 修改 `parse_judge_spec()` 使用 `ModelRegistry.infer_provider_from_model_id()` 从 `models.yaml` 推断
- 添加 fallback 逻辑：如果无法推断，对于未知模型默认使用 `hf`（因为大部分未匹配的模型是 HF 模型）

**验证**：
```bash
✓ qwen3_4b_instruct -> ('hf', 'qwen3_4b_instruct')
✓ gemma3_4b -> ('hf', 'gemma3_4b')
✓ gpt4omini -> ('openai', 'gpt4omini')
✓ gemini3_pro -> ('gemini', 'gemini3_pro')
```

### 2. ✅ 断点续训 JSON 解析改进

**问题**：
- `load_completed_sample_ids()` 对多行格式 JSON 的解析不够健壮
- 可能无法正确解析格式化的 JSONL 文件（每个 key 一行）

**修复**：
- 重写 JSON 解析逻辑，支持：
  1. **多行格式 JSON**：正确处理嵌套的 JSON 对象（考虑字符串中的括号）
  2. **单行格式 JSONL**：作为 fallback，如果多行解析失败
- 使用状态机正确跟踪字符串、转义字符和括号匹配

**验证**：
```bash
✓ 可以正确解析多行格式的 JSONL 文件
✓ 可以正确提取 sample_id
```

### 3. ✅ 信号处理防止意外中断

**问题**：
- 程序被 Ctrl+C 或 SIGTERM 中断时，可能丢失正在处理的样本进度

**修复**：
- 在 `run_pairwise.py` 中添加信号处理器：
  - 捕获 `SIGINT` (Ctrl+C) 和 `SIGTERM`
  - 设置全局标志 `_shutdown_requested`
  - 在循环中检查标志，允许当前样本完成并保存后再退出
- 在 `batch_evaluate.py` 中添加批量模式的信号处理：
  - 允许当前 judge 完成后再停止
  - 已完成的 judge 结果会被保存

**行为**：
- 按 Ctrl+C 时：
  1. 显示警告信息
  2. 当前样本完成判断并保存
  3. 优雅退出，提示可以恢复
- 不会丢失已保存的进度

### 4. ✅ 增量保存功能确认

**功能**：
- 每个判断完成后立即保存到文件
- 使用 `os.fsync()` 强制写入磁盘
- 添加错误处理和日志

**位置**：
- `src/metrics/reports.py`: `save_judgment_jsonl_single()`
- `src/pipeline/run_pairwise.py`: 在 Step 1 和 Step 2 循环中调用

## 使用说明

### 单个模型运行

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

**中断处理**：
- 按 Ctrl+C 时，当前样本会完成并保存
- 重新运行相同命令会自动恢复

### 批量运行

```bash
python scripts/batch_evaluate.py --config configs/experiment.yaml --judges gpt4omini qwen3_4b_instruct gemini3_pro
```

**中断处理**：
- 按 Ctrl+C 时，当前 judge 会完成并保存
- 已完成的 judge 结果会被保存
- 可以继续运行剩余的 judge

### 断点续训

**自动检测**：
- 程序启动时自动检测输出目录中是否已有部分结果
- 显示恢复信息（已完成多少样本，剩余多少样本）
- 自动跳过已完成的样本

**示例输出**：
```
============================================================
🔄 Resume mode detected!
  Original judgments completed: 250
  Bias judgments completed: 0
============================================================
Loaded 400 samples
Resuming: 250 samples already completed, 150 remaining
```

## 修复的文件

1. `scripts/batch_evaluate.py`
   - 修复 provider 推断逻辑
   - 添加信号处理

2. `src/utils/resume.py`
   - 改进 JSON 解析逻辑

3. `src/pipeline/run_pairwise.py`
   - 添加信号处理
   - 在循环中检查中断标志

4. `src/metrics/reports.py`
   - 已包含增量保存功能（之前已修复）

## 测试验证

✅ Provider 推断测试通过
✅ JSON 解析测试通过
✅ 信号处理导入成功
✅ 无语法错误

## 注意事项

1. **模型配置**：确保 `configs/models.yaml` 中包含所有要使用的模型
2. **API 密钥**：确保环境变量中设置了相应的 API 密钥
3. **中断恢复**：中断后重新运行相同命令即可自动恢复
4. **批量运行**：批量脚本现在可以正确处理所有类型的模型（OpenAI, Gemini, Anthropic, HF）
