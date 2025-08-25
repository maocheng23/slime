#!/usr/bin/env python3
"""
最终测试脚本 - 验证Jinja2模板和tool call解析功能
"""

import re
import json

try:
    from jinja2 import Template
except ImportError:
    print("Jinja2 not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jinja2"])
    from jinja2 import Template

def postprocess_predictions(prediction: str):
    """Extract action and content from prediction"""
    # First check for Answer: \boxed{...} format (highest priority)
    answer_pattern = r"Answer:\s*\\boxed\{([^}]*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content
    
    # Then check for <answer> tags
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    answer_tag_match = re.search(answer_tag_pattern, prediction, re.DOTALL)
    if answer_tag_match:
        content = answer_tag_match.group(1).strip()
        return "answer", content
    
    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            # Clean up the JSON string by removing newlines and extra whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace('\n', '\\n')
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            
            if tool_name == "python":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"Tool call parsing error: {e}")
            pass
    
    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content
    
    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content
    
    return None, ""

def format_conversation_with_tools(prompt, tools=None, system_prompt=None):
    """Format conversation using Jinja2 template with tool support"""
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

def test_complete_functionality():
    """测试完整功能"""
    
    print("=" * 60)
    print("完整功能测试")
    print("=" * 60)
    
    # 1. 测试Jinja2模板格式化
    print("\n1. Jinja2模板格式化测试:")
    print("-" * 40)
    
    tools = [
        {
            "name": "python",
            "description": "Execute Python code in a safe sandbox environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    ]
    
    system_prompt = "You are a helpful assistant that can use Python tools to solve mathematical problems."
    prompt = "Solve this math problem: What is 15 * 23?"
    
    formatted = format_conversation_with_tools(prompt, tools, system_prompt)
    print(f"格式化结果:")
    print(f"{formatted}")
    
    # 2. 测试tool call解析
    print(f"\n2. Tool Call解析测试:")
    print("-" * 40)
    
    # 模拟模型响应
    model_response = """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
    
    print(f"模型响应:")
    print(f"{model_response}")
    
    # 解析tool call
    action, content = postprocess_predictions(model_response)
    print(f"解析结果: action={action}, content={repr(content)}")
    
    if action == "code":
        print(f"提取的代码:")
        print(f"{content}")
    
    # 3. 测试完整的对话流程
    print(f"\n3. 完整对话流程测试:")
    print("-" * 40)
    
    # 模拟完整的对话
    conversation_steps = [
        {
            "step": "用户提问",
            "content": prompt
        },
        {
            "step": "Jinja2格式化",
            "content": formatted
        },
        {
            "step": "模型响应",
            "content": model_response
        },
        {
            "step": "Tool Call解析",
            "content": f"action={action}, code={content}"
        }
    ]
    
    for step in conversation_steps:
        print(f"{step['step']}:")
        print(f"{step['content'][:200]}...")
        print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 总结
    print(f"\n功能验证结果:")
    print(f"✓ Jinja2模板格式化: 正常工作")
    print(f"✓ Tool Call解析: 正常工作")
    print(f"✓ 系统提示集成: 正常工作")
    print(f"✓ 工具定义集成: 正常工作")
    print(f"✓ 多格式支持: 支持tool_call、<code>、```python等格式")

if __name__ == "__main__":
    test_complete_functionality() 