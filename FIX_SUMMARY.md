# 修复总结：Mock 回退问题修复

## 问题定位

### 1. Mock 回退的触发点

找到以下回退点：

1. **`ModelRegistry.create_model()`** (src/models/registry.py:41-43)
   - 当 `model.is_available()` 返回 False 时，会回退到 MockModel
   - 当创建模型时发生异常，也会回退到 MockModel

2. **`ModelJudge.__init__()`** (src/judge/judge_runner.py:97-99)
   - 当模型不可用时，设置 `self.model = None`
   - 在 `judge_pairwise()` 中，如果 `self.model is None`，会使用 MockJudge

3. **`OpenAIModel.is_available()`** (src/models/openai_client.py:62-68)
   - 如果 `openai` 包未导入，返回 False
   - 如果 API key 不存在，返回 False

## 修复内容

### 2. 配置解析日志

在 `main.py` 中添加了详细的配置解析日志：
- Provider/Type
- Model ID/Name
- API Key 环境变量名
- API Key 是否可用（True/False，不暴露 key）
- Model Config 的键列表

在 `ModelRegistry.create_model()` 中添加了日志：
- 记录创建模型的参数（model_type, model_name, config keys）

在 `OpenAIModel.__init__()` 中添加了日志：
- 记录 API key 是否找到
- 记录 openai 包是否导入成功

### 3. 配置读取逻辑修复

**支持两种配置格式：**

1. **新格式（Model Pool）**：
   ```yaml
   # experiment.yaml
   judge:
     provider: "openai"
     model_id: "gpt52"
   
   # models.yaml
   openai:
     defaults:
       api_key_env: "OPENAI_API_KEY"
     gpt52:
       model: "gpt-4o-mini"
   ```

2. **旧格式（Flat Structure，向后兼容）**：
   ```yaml
   # experiment.yaml
   judge:
     type: "openai"
     model_name: "gpt-4o-mini"
   
   # models.yaml
   openai:
     api_key_env: "OPENAI_API_KEY"
     model: "gpt-4o-mini"
   ```

**代码逻辑：**
- 优先读取 `provider`/`model_id`，如果没有则读取 `type`/`model_name`（向后兼容）
- 支持 model pool 结构：`models[provider][model_id]`
- 支持 flat 结构：`models[provider]`（直接包含配置）
- 对于 flat 结构，如果 config 中没有 `model` 字段，使用 experiment.yaml 中的 `model_id`

### 4. API Key 读取和模型可用性检查

**修复点：**
- `OpenAIModel.__init__()` 中正确读取 `api_key_env` 配置
- 添加详细的日志，显示 API key 是否找到
- `is_available()` 方法添加调试日志

### 5. Fallback 机制改进

**之前：** Silent fallback（静默回退到 mock）

**现在：** 显式错误，除非配置 `allow_fallback_mock: true`

**修改位置：**
1. `ModelRegistry.create_model()`: 如果模型不可用，默认抛出 `RuntimeError`，除非 `allow_fallback_mock: true`
2. `ModelJudge.__init__()`: 如果模型初始化失败，默认抛出 `RuntimeError`，除非 `allow_fallback_mock: true`
3. `ModelJudge.judge_pairwise()`: 运行时错误仍然允许 fallback（这是合理的，因为运行时错误不应该中断整个流程）

**配置示例：**
```yaml
judge:
  provider: "openai"
  model_id: "gpt52"
  allow_fallback_mock: false  # 默认 false，设置为 true 允许 fallback
```

## 修改的文件

1. **src/main.py**
   - 修复配置读取逻辑，支持 provider/model_id 格式
   - 添加详细的配置解析日志
   - 调整日志输出顺序（先创建 run_dir，再设置日志）

2. **src/models/registry.py**
   - 添加详细的模型创建日志
   - 修改 fallback 逻辑：默认抛出错误，除非 `allow_fallback_mock: true`

3. **src/models/openai_client.py**
   - 添加 API key 状态日志
   - 添加 openai 包导入状态日志
   - 改进 `is_available()` 的调试日志

4. **src/judge/judge_runner.py**
   - 修改 `ModelJudge.__init__()`: 默认抛出错误，除非 `allow_fallback_mock: true`
   - 添加详细的初始化日志
   - 改进错误处理逻辑

## 验证方法

运行程序后，检查日志中应该出现：

1. **配置解析日志**：
   ```
   ============================================================
   Judge Configuration Details
   ============================================================
     Provider/Type: openai
     Model ID/Name: gpt-4o-mini
     API Key Env: OPENAI_API_KEY
     API Key Available: True
     Model Config Keys: ['api_key_env', 'model', 'temperature', 'max_tokens']
   ============================================================
   ```

2. **模型创建日志**：
   ```
   ModelRegistry.create_model called:
     model_type: openai
     model_name: gpt-4o-mini
     config keys: ['api_key_env', 'model', 'temperature', 'max_tokens']
   OpenAIModel: API key found from environment variable 'OPENAI_API_KEY'
   OpenAIModel: OpenAI package imported successfully, API key set
   Created openai model instance: gpt-4o-mini
   Model openai:gpt-4o-mini is available and ready to use
   ModelJudge: Successfully initialized openai model 'gpt-4o-mini'
   ```

3. **如果 API key 缺失**：
   ```
   OpenAIModel: API key NOT found in environment variable 'OPENAI_API_KEY'
   Model openai:gpt-4o-mini is not available. Check API key, package installation, and configuration.
   RuntimeError: Model openai:gpt-4o-mini is not available. Set 'allow_fallback_mock: true' in config to allow fallback to mock.
   ```

4. **如果 openai 包未安装**：
   ```
   OpenAIModel: openai package not installed. Install with: pip install openai
   Model openai:gpt-4o-mini is not available. Check API key, package installation, and configuration.
   RuntimeError: Model openai:gpt-4o-mini is not available. Set 'allow_fallback_mock: true' in config to allow fallback to mock.
   ```

## 关键代码片段

### 配置读取（main.py）
```python
# 支持新格式（provider/model_id）和旧格式（type/model_name）
judge_provider = judge_config.get("provider") or judge_config.get("type", "mock")
judge_model_id = judge_config.get("model_id") or judge_config.get("model_name", "mock-judge-v1")

# 支持 model pool 和 flat structure
if judge_model_id in provider_config:
    # Model pool structure
    judge_model_config = provider_config[judge_model_id].copy()
else:
    # Flat structure
    judge_model_config = provider_config.copy()
```

### Fallback 逻辑（registry.py）
```python
if not model.is_available():
    allow_fallback = config.get("allow_fallback_mock", False)
    if allow_fallback:
        logger.warning("Falling back to mock")
        return MockModel(...)
    else:
        raise RuntimeError("Model not available. Set 'allow_fallback_mock: true' to allow fallback.")
```

## 使用说明

1. **设置 API key**：
   ```powershell
   $env:OPENAI_API_KEY = "your-api-key"
   ```

2. **配置 experiment.yaml**：
   ```yaml
   judge:
     provider: "openai"
     model_id: "gpt-4o-mini"  # 或使用 model pool 中的 model_id
     allow_fallback_mock: false  # 设置为 true 允许 fallback
   ```

3. **运行程序**：
   ```bash
   python scripts/run_experiment.py --config configs/experiment.yaml
   ```

4. **检查日志**：查看 `outputs/<run_dir>/logs.txt` 确认模型是否正确初始化。


