# 示例：Retool lite

[English](./README.md)

这是 [verl's retool](https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py) 的最小复现版本，展示了如何在 slime 中使用工具调用功能。

## 环境设置

使用 `zhuzilin/slime:latest` 镜像并初始化环境：

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
```

## 模型设置

### 选项 1：Qwen3-4B（推荐）

下载并设置 Qwen3-4B：

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# 训练数据（DAPO 数据集）
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# 评估数据
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

### 选项 2：Qwen2.5-3B

下载并设置 Qwen2.5-3B：

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

## 数据准备

### 使用 DAPO 数据集（推荐）

DAPO 数据集已经准备好并包含工具调用示例。您可以直接使用：

```bash
# 数据集已下载到 /root/dapo-math-17k/dapo-math-17k.jsonl
# 无需额外准备
```

### 生成自定义数据

或者，您可以使用提供的脚本自动生成训练数据：

```bash
bash examples/retool/generate_data.sh
```

或者手动准备工具调用训练数据，格式如下：

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "编写一个函数来计算 5 的阶乘。"
    }
  ],
  "ground_truth": "5 的阶乘是 120"
}
```

## 本地测试

在运行完整训练之前，您可以本地测试工具调用功能：

### 快速测试

运行工具的快速测试：

```bash
cd examples/retool
python3 quick_test.py
```

这将测试：
- 安全代码的 Python 工具执行
- 数学表达式的计算器工具
- 从模拟响应中提取工具调用
- 危险代码的安全检查

### 完整测试

运行综合本地测试器：

```bash
cd examples/retool
python3 test_retool_local.py
```

这包括：
- 单轮对话测试
- 多轮对话测试
- 工具安全性测试
- 模拟 LLM 响应生成

### 测试脚本

您也可以使用提供的脚本：

```bash
bash examples/retool/run_test.sh
```

## 运行脚本

### 对于使用 DAPO 数据集的 Qwen3-4B：

```bash
cd slime/
bash examples/retool/run_qwen3_4B.sh
```

### 对于使用生成数据的 Qwen2.5-3B：

```bash
cd slime/
bash examples/retool/run_qwen2.5_3B.sh
```

## 代码结构

要在 slime 中实现工具调用，您只需要实现自定义数据生成函数和任务的奖励模型。这些对应于启动脚本中的以下 2 个配置项：

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)
```

这些是 `generate_with_retool.py` 中的 `generate` 和 `reward_func` 函数。

## 工具调用格式

模型被训练为使用以下格式进行工具调用：

1. **工具调用**：`<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>`
2. **工具结果**：`<tool_results>...</tool_results>`
3. **最终答案**：`<answer>...</answer>`

### 可用工具

该示例包含两个工具：

1. **Python 解释器**：在安全沙盒中执行 Python 代码
   ```json
   {"name": "python", "arguments": {"code": "print('Hello, World!')"}}
   ```

2. **计算器**：执行数学计算
   ```json
   {"name": "calculator", "arguments": {"expression": "15 * 23 + 7"}}
   ```

## Python 代码解释器

Python 解释器工具提供了用于执行 Python 代码的安全沙盒环境：

### 安全特性

- **代码安全检查**：验证代码是否包含危险模式
- **模块限制**：仅允许安全模块（math、random、datetime 等）
- **超时保护**：防止无限循环（默认：10 秒）
- **隔离执行**：在临时目录中运行
- **输出捕获**：安全地捕获 stdout 和 stderr

### 允许的模块

沙盒允许以下 Python 模块：
- `math`、`random`、`datetime`、`collections`、`itertools`
- `functools`、`operator`、`statistics`、`decimal`、`fractions`

### 受限操作

以下操作因安全原因被阻止：
- 文件系统访问（`os`、`sys`、`subprocess` 等）
- 动态代码执行（`eval`、`exec`、`__import__`）
- 反射和内省（`getattr`、`setattr`、`globals` 等）
- 输入/输出操作（`input`、`open`、`file`）

## 计算器工具

计算器工具提供安全的数学表达式求值：

### 特性

- **安全求值**：仅允许数学运算
- **字符验证**：限制输入为数字、运算符和括号
- **错误处理**：为无效表达式提供清晰的错误消息

### 支持的操作

- 基本算术：`+`、`-`、`*`、`/`
- 分组括号：`(`、`)`
- 小数和整数

## 自定义

您可以通过以下方式自定义工具注册表：

1. 在 `ToolRegistry._register_default_tools()` 方法中添加新工具
2. 在相应的 `_execute_*` 方法中实现工具执行逻辑
3. 修改奖励函数以更好地适应您的特定任务
4. 在 `RETOOL_CONFIGS` 中调整沙盒设置

## 示例对话流程

```
用户：计算 15 * 23 + 7，然后编写一个函数来找到结果的阶乘。

助手：我来帮您解决这个问题。首先，让我计算 15 * 23 + 7。

<tool_call>{"name": "calculator", "arguments": {"expression": "15 * 23 + 7"}}</tool_call>

<tool_results>
Tool 'calculator' result: Result: 352
</tool_results>

现在我将编写一个 Python 函数来计算 352 的阶乘。

<tool_call>{"name": "python", "arguments": {"code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(352))"}}</tool_call>

<tool_results>
Tool 'python' result: Output:
[结果将在这里显示]
</tool_results>

<answer>15 * 23 + 7 的结果是 352。我还提供了一个 Python 函数来计算它的阶乘。</answer>
```

## 模型兼容性

- **Qwen3-4B**：✅ 完全支持工具调用功能
- **Qwen2.5-3B**：✅ 支持工具调用功能
- **其他 Qwen 模型**：应该可以通过适当的模型配置工作 