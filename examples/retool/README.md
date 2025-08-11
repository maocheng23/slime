# Example: Retool lite

[中文版](./README_zh.md)

This is a minimal reproduction of [verl's retool](https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py) and an example of using tool calling in slime.

## Environment Setup

Use the `zhuzilin/slime:latest` image and initialize the environment:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
```

## Model Setup

### Option 1: Qwen3-4B (Recommended)

Download and setup Qwen3-4B:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# train data (DAPO dataset)
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
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

### Option 2: Qwen2.5-3B

Download and setup Qwen2.5-3B:

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

## Data Preparation

### Using DAPO Dataset (Recommended)

The DAPO dataset is already prepared and contains tool calling examples. You can use it directly:

```bash
# The dataset is already downloaded to /root/dapo-math-17k/dapo-math-17k.jsonl
# No additional preparation needed
```

### Generating Custom Data

Alternatively, you can generate training data automatically using the provided script:

```bash
bash examples/retool/generate_data.sh
```

Or manually prepare your tool calling training data in the following format:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Write a function to calculate the factorial of 5."
    }
  ],
  "ground_truth": "The factorial of 5 is 120"
}
```

## Local Testing

Before running the full training, you can test the tool calling functionality locally:

### Quick Test

Run a quick test of the tools:

```bash
cd examples/retool
python3 quick_test.py
```

This will test:
- Python tool execution with safe code
- Calculator tool with mathematical expressions
- Tool call extraction from mock responses
- Security checks for dangerous code

### Full Test

Run the comprehensive local tester:

```bash
cd examples/retool
python3 test_retool_local.py
```

This includes:
- Single-turn conversation testing
- Multi-turn conversation testing
- Tool safety testing
- Mock LLM response generation

### Test Script

You can also use the provided script:

```bash
bash examples/retool/run_test.sh
```

## Running the Script

### For Qwen3-4B with DAPO dataset:

```bash
cd slime/
bash examples/retool/run_qwen3_4B.sh
```

### For Qwen2.5-3B with generated data:

```bash
cd slime/
bash examples/retool/run_qwen2.5_3B.sh
```

## Code Structure

To implement tool calling in slime, you only need to implement a custom data generation function and a reward model for the task. These correspond to the following 2 configuration items in the startup script:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)
```

These are the `generate` and `reward_func` functions in `generate_with_retool.py`.

## Tool Calling Format

The model is trained to use the following format for tool calls:

1. **Tool Call**: `<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>`
2. **Tool Results**: `<tool_results>...</tool_results>`
3. **Final Answer**: `<answer>...</answer>`

### Available Tools

The example includes two tools:

1. **Python Interpreter**: Execute Python code in a safe sandbox
   ```json
   {"name": "python", "arguments": {"code": "print('Hello, World!')"}}
   ```

2. **Calculator**: Perform mathematical calculations
   ```json
   {"name": "calculator", "arguments": {"expression": "15 * 23 + 7"}}
   ```

## Python Code Interpreter

The Python interpreter tool provides a secure sandbox environment for executing Python code:

### Security Features

- **Code Safety Check**: Validates code against dangerous patterns
- **Module Restrictions**: Only allows safe modules (math, random, datetime, etc.)
- **Timeout Protection**: Prevents infinite loops (default: 10 seconds)
- **Isolated Execution**: Runs in temporary directories
- **Output Capture**: Captures stdout and stderr safely

### Allowed Modules

The sandbox allows the following Python modules:
- `math`, `random`, `datetime`, `collections`, `itertools`
- `functools`, `operator`, `statistics`, `decimal`, `fractions`

### Restricted Operations

The following operations are blocked for security:
- File system access (`os`, `sys`, `subprocess`, etc.)
- Dynamic code execution (`eval`, `exec`, `__import__`)
- Reflection and introspection (`getattr`, `setattr`, `globals`, etc.)
- Input/output operations (`input`, `open`, `file`)

## Calculator Tool

The calculator tool provides safe mathematical expression evaluation:

### Features

- **Safe Evaluation**: Only allows mathematical operations
- **Character Validation**: Restricts input to numbers, operators, and parentheses
- **Error Handling**: Provides clear error messages for invalid expressions

### Supported Operations

- Basic arithmetic: `+`, `-`, `*`, `/`
- Parentheses for grouping: `(`, `)`
- Decimal numbers and integers

## Customization

You can customize the tool registry by:

1. Adding new tools in the `ToolRegistry._register_default_tools()` method
2. Implementing tool execution logic in the corresponding `_execute_*` methods
3. Modifying the reward function to better suit your specific task
4. Adjusting sandbox settings in `RETOOL_CONFIGS`

## Example Conversation Flow

```
User: Calculate 15 * 23 + 7 and then write a function to find the factorial of the result.

Assistant: I'll help you with that. First, let me calculate 15 * 23 + 7.

<tool_call>{"name": "calculator", "arguments": {"expression": "15 * 23 + 7"}}</tool_call>

<tool_results>
Tool 'calculator' result: Result: 352
</tool_results>

Now I'll write a Python function to calculate the factorial of 352.

<tool_call>{"name": "python", "arguments": {"code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(352))"}}</tool_call>

<tool_results>
Tool 'python' result: Output:
[Result would be displayed here]
</tool_results>

<answer>The result of 15 * 23 + 7 is 352. I've also provided a Python function to calculate its factorial.</answer>
```

## Model Compatibility

- **Qwen3-4B**: ✅ Fully supported with tool calling capabilities
- **Qwen2.5-3B**: ✅ Supported with tool calling capabilities
- **Other Qwen models**: Should work with appropriate model configuration 