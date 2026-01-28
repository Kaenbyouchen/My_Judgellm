# 批量评测脚本使用指南

本目录包含用于在 USC CARC (Slurm) 上批量运行多个 judge 模型的脚本。

## 文件说明

- `batch_evaluate.py`: Python 批量评测脚本，支持按顺序运行多个 judge 模型
- `slurm_batch_evaluate.sh`: Slurm 批处理脚本，用于在 CARC 上提交作业

## 快速开始

### 1. 本地测试（不使用 Slurm）

```bash
# 运行多个 judge 模型
python scripts/batch_evaluate.py \
    --config configs/experiment.yaml \
    --judges gpt5mini gemini25_flash

# 使用显式 provider:model 格式
python scripts/batch_evaluate.py \
    --config configs/experiment.yaml \
    --judges openai:gpt4omini gemini:gemini3_pro anthropic:claude_opus_45

#  dry run（查看将要运行的内容）
python scripts/batch_evaluate.py \
    --config configs/experiment.yaml \
    --judges gpt4omini --dry-run
```

### 2. 在 USC CARC (Slurm) 上运行

#### 步骤 1: 修改配置

编辑 `scripts/slurm_batch_evaluate.sh`，修改以下变量：

```bash
# 基础配置文件
CONFIG_FILE="configs/experiment.yaml"

# 要运行的 judge 模型列表
JUDGES=("gpt4omini" "gpt52" "gemini3_pro")

# 可选：自定义日志目录
# LOG_DIR="outputs/batch_logs/custom_run"
```

#### 步骤 2: 调整 Slurm 资源（如需要）

根据你的需求修改 `#SBATCH` 指令：

```bash
#SBATCH --time=24:00:00      # 作业时间限制
#SBATCH --mem=16G            # 内存需求
#SBATCH --cpus-per-task=4    # CPU 核心数
#SBATCH --partition=gpu      # 分区名称（根据 CARC 调整）
#SBATCH --gres=gpu:1         # GPU 需求（如不需要可删除此行）
```

#### 步骤 3: 提交作业

```bash
# 提交到 Slurm
sbatch scripts/slurm_batch_evaluate.sh

# 查看作业状态
squeue -u $USER

# 查看作业输出
tail -f outputs/slurm_logs/judgellm_batch_<JOB_ID>.out
```

## 功能特性

### 1. 错误处理

- ✅ 任意一个 judge 运行失败不会中断整体流程
- ✅ 每个失败的运行都会保存详细的错误日志
- ✅ 继续运行下一个 judge 模型

### 2. 日志管理

每个 judge 运行都会生成：
- **运行日志**: `batch_<judge_name>_<timestamp>.log`
- **错误日志**（如果失败）: `error_<judge_name>_<timestamp>.log`
- **临时配置文件**: `config_<judge_name>.yaml`
- **批量总结**: `batch_summary.json` 和 `batch_summary.log`

### 3. 输出结构

```
outputs/
├── batch_logs/
│   └── <timestamp>/
│       ├── batch_summary.json          # JSON 格式的总结
│       ├── batch_summary.log           # 文本格式的总结
│       ├── batch_openai_gpt4omini_*.log
│       ├── batch_gemini_gemini3_pro_*.log
│       ├── error_gemini_gemini3_pro_*.log  # 如果失败
│       └── config_openai_gpt4omini.yaml    # 临时配置文件
└── slurm_logs/
    ├── judgellm_batch_<JOB_ID>.out
    └── judgellm_batch_<JOB_ID>.err
```

## Judge 模型格式

支持两种格式：

1. **简化格式**（自动推断 provider）:
   ```bash
   --judges gpt4omini gpt52 gemini3_pro
   ```
   - `gpt*` → 推断为 `openai`
   - `gemini*` → 推断为 `gemini`
   - `claude*` → 推断为 `anthropic`

2. **显式格式**（明确指定 provider）:
   ```bash
   --judges openai:gpt4omini gemini:gemini3_pro anthropic:claude_opus_45
   ```

## 命令行参数

### `batch_evaluate.py`

```
--config CONFIG         基础实验配置文件路径（必需）
--judges JUDGES [JUDGES ...]  要运行的 judge 模型列表（必需）
--log-dir LOG_DIR       日志目录（可选，默认：outputs/batch_logs/<timestamp>）
--dry-run               仅显示将要运行的内容，不实际运行
--continue-on-error     遇到错误时继续运行下一个（默认：True）
```

## 示例场景

### 场景 1: 测试多个 OpenAI 模型

```bash
python scripts/batch_evaluate.py \
    --config configs/experiment.yaml \
    --judges gpt4omini gpt52 gpt5mini
```

### 场景 2: 跨提供商对比

```bash
python scripts/batch_evaluate.py \
    --config configs/experiment.yaml \
    --judges openai:gpt4omini gemini:gemini3_pro anthropic:claude_opus_45
```

### 场景 3: 在 Slurm 上运行长时间作业

1. 编辑 `slurm_batch_evaluate.sh`，设置：
   ```bash
   #SBATCH --time=48:00:00  # 48 小时
   JUDGES=("gpt4omini" "gpt52" "gemini3_pro" "gemini3_flash" "claude_opus_45")
   ```

2. 提交作业：
   ```bash
   sbatch scripts/slurm_batch_evaluate.sh
   ```

3. 监控进度：
   ```bash
   # 查看作业状态
   squeue -u $USER
   
   # 实时查看输出
   tail -f outputs/slurm_logs/judgellm_batch_<JOB_ID>.out
   
   # 查看批量总结
   cat outputs/batch_logs/<timestamp>/batch_summary.json
   ```

## 故障排查

### 问题 1: 某个 judge 运行失败

检查对应的错误日志：
```bash
cat outputs/batch_logs/<timestamp>/error_<judge_name>_*.log
```

### 问题 2: Slurm 作业被取消

检查 Slurm 日志：
```bash
cat outputs/slurm_logs/judgellm_batch_<JOB_ID>.err
```

### 问题 3: 配置问题

检查临时配置文件：
```bash
cat outputs/batch_logs/<timestamp>/config_<judge_name>.yaml
```

## 注意事项

1. **API 密钥**: 确保在运行环境中设置了必要的 API 密钥：
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `ANTHROPIC_API_KEY`

2. **资源限制**: 根据数据集大小和模型数量调整 Slurm 资源分配

3. **缓存机制**: 如果启用了 bias 注入缓存，第一次运行会较慢，后续运行会更快

4. **并发控制**: 当前脚本是顺序运行的，如需并行运行多个 judge，需要修改脚本或使用多个 Slurm 作业
