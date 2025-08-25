# Retool with Jinja2 Template Support

这个示例展示了如何在SLIME的retool功能中集成Jinja2模板，为对话添加system prompt和工具调用支持。

## 功能特性

1. **Jinja2模板支持**: 使用Jinja2模板引擎格式化对话
2. **System Prompt**: 可以为对话添加系统提示
3. **工具调用**: 支持在对话中注册和使用工具
4. **多轮对话**: 支持多轮对话的格式化

## 安装依赖

```bash
pip install jinja2>=3.0.0
```

或者使用提供的requirements文件：

```bash
pip install -r requirements.txt
```

## 主要修改

### 1. 添加Jinja2依赖

在 `generate_with_retool.py` 中添加了Jinja2导入：

```python
try:
    from jinja2 import Template
except ImportError:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2")
```

### 2. 定义模板

添加了 `TOOL_TEMPLATE` 常量，支持工具调用和系统提示：

```python
TOOL_TEMPLATE = """{%- if tools %}
<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{%- endif %}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|im_start|>system
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""
```

### 3. 格式化函数

添加了 `format_conversation_with_tools` 函数：

```python
def format_conversation_with_tools(prompt: str, tools: List[Dict[str, Any]] = None, system_prompt: str = None) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)
    
    # Prepare messages
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Render template
    formatted_text = template.render(
        messages=messages,
        tools=tools or []
    )
    
    return formatted_text
```

### 4. 修改generate函数

在 `generate` 函数中集成了Jinja2模板：

```python
for turn in range(RETOOL_CONFIGS["max_turns"]):
    # Build current input using Jinja2 template
    if turn == 0:
        # First turn: format with tools and system prompt
        tool_specs = tool_registry.get_tool_specs()
        system_prompt = "You are a helpful assistant that can use Python tools to solve mathematical problems. When you need to perform calculations, use the Python tool to execute code and get results."
        current_input = format_conversation_with_tools(
            prompt=prompt,
            tools=tool_specs,
            system_prompt=system_prompt
        )
    else:
        # Subsequent turns: append to existing conversation
        current_input = prompt + response
```

## 使用方法

### 1. 基本使用

```python
from generate_with_retool import format_conversation_with_tools, tool_registry

# 获取工具列表
tools = tool_registry.get_tool_specs()

# 格式化对话
formatted = format_conversation_with_tools(
    prompt="Solve this math problem: What is 2 + 2?",
    tools=tools,
    system_prompt="You are a helpful assistant that can use Python tools to solve mathematical problems."
)
```

### 2. 自定义系统提示

```python
system_prompt = """You are a helpful assistant that can use Python tools to solve mathematical problems. 
When you need to perform calculations, use the Python tool to execute code and get results. 
Always show your work step by step."""
```

### 3. 多轮对话

```python
from jinja2 import Template

MULTI_TURN_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|im_start|>system
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

template = Template(MULTI_TURN_TEMPLATE)
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "Let me calculate that for you."},
    {"role": "user", "content": "Please show your work."}
]

formatted = template.render(messages=conversation)
```

## 测试

运行测试脚本：

```bash
# 测试Jinja2模板功能
python simple_jinja2_test.py

# 测试完整的retool功能
python test_retool_local.py
```

## 输出格式

使用Jinja2模板后，对话将被格式化为以下格式：

```
<|im_start|>system
You are a helpful assistant that can use Python tools to solve mathematical problems.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "python", "description": "Execute Python code in a safe sandbox environment", "parameters": {...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>system
You are a helpful assistant that can use Python tools to solve mathematical problems.<|im_end|>
<|im_start|>user
Solve this math problem: What is 2 + 2?<|im_end|>
<|im_start|>assistant
```

## 优势

1. **标准化格式**: 使用Jinja2模板确保对话格式的一致性
2. **灵活性**: 可以轻松自定义系统提示和工具描述
3. **可扩展性**: 可以轻松添加新的工具和对话格式
4. **可读性**: 模板化的代码更容易理解和维护

## 注意事项

1. 确保安装了Jinja2依赖
2. 模板中的特殊字符需要正确转义
3. 工具定义需要符合JSON Schema格式
4. 系统提示应该清晰明确，指导模型如何使用工具 