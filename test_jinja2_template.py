#!/usr/bin/env python3
"""
测试Jinja2模板功能的脚本
"""

import sys
sys.path.append('/root/slime')

try:
    from jinja2 import Template
except ImportError:
    print("Jinja2 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jinja2"])
    from jinja2 import Template

def test_jinja2_template():
    """测试Jinja2模板功能"""
    
    # 模板定义
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

    def format_conversation_with_tools(prompt, tools=None, system_prompt=None):
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

    # 测试用例
    print("=" * 60)
    print("Jinja2模板测试")
    print("=" * 60)
    
    # 测试用例1：有工具，有系统提示
    print("\n测试用例1：有工具，有系统提示")
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
    prompt = "Solve this math problem: What is 2 + 2?"
    
    result = format_conversation_with_tools(prompt, tools, system_prompt)
    print("格式化结果:")
    print(result)
    
    # 测试用例2：无工具，有系统提示
    print("\n测试用例2：无工具，有系统提示")
    print("-" * 40)
    
    result2 = format_conversation_with_tools(prompt, None, system_prompt)
    print("格式化结果:")
    print(result2)
    
    # 测试用例3：无工具，无系统提示
    print("\n测试用例3：无工具，无系统提示")
    print("-" * 40)
    
    result3 = format_conversation_with_tools(prompt, None, None)
    print("格式化结果:")
    print(result3)
    
    # 测试用例4：多轮对话
    print("\n测试用例4：多轮对话")
    print("-" * 40)
    
    # 模拟多轮对话的模板
    MULTI_TURN_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{- '<|im_start|>system\n' }}{{- message['content'] }}{{- '<|im_end|>\n' }}
{%- elif message['role'] == 'user' %}
{{- '<|im_start|>user\n' }}{{- message['content'] }}{{- '<|im_end|>\n' }}
{%- elif message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\n' }}{{- message['content'] }}{{- '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{{- '<|im_start|>assistant\n' }}"""

    template = Template(MULTI_TURN_TEMPLATE)
    
    multi_turn_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "Let me calculate that for you."},
        {"role": "user", "content": "Please show your work."}
    ]
    
    result4 = template.render(messages=multi_turn_messages)
    print("多轮对话格式化结果:")
    print(result4)

if __name__ == "__main__":
    test_jinja2_template() 