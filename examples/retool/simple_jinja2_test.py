#!/usr/bin/env python3
"""
简化的Jinja2模板测试
"""

try:
    from jinja2 import Template
except ImportError:
    print("Jinja2 not installed. Installing...")
    import subprocess
    import sys
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

    print("=" * 60)
    print("Jinja2模板测试")
    print("=" * 60)
    
    # 测试工具定义
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
    
    # 测试用例
    test_cases = [
        {
            "name": "数学计算问题",
            "prompt": "Solve this math problem: What is 15 * 23?",
            "system_prompt": "You are a helpful assistant that can use Python tools to solve mathematical problems."
        },
        {
            "name": "复杂计算问题", 
            "prompt": "Calculate the area of a circle with radius 5 units.",
            "system_prompt": "You are a helpful assistant that can use Python tools to solve mathematical problems. Use Python code to perform calculations and show your work step by step."
        },
        {
            "name": "无系统提示的问题",
            "prompt": "What is the square root of 144?",
            "system_prompt": None
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"   Prompt: {case['prompt']}")
        print(f"   System Prompt: {case['system_prompt']}")
        
        # 格式化对话
        formatted = format_conversation_with_tools(
            prompt=case['prompt'],
            tools=tools,
            system_prompt=case['system_prompt']
        )
        
        print(f"   格式化结果:")
        print(f"   {formatted}")
        print(f"   " + "-" * 50)
    
    # 多轮对话测试
    print(f"\n多轮对话测试:")
    
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
        {
            "role": "system",
            "content": "You are a helpful assistant that can use Python tools to solve mathematical problems."
        },
        {
            "role": "user", 
            "content": "What is 2 + 3?"
        },
        {
            "role": "assistant",
            "content": "Let me calculate that for you using Python."
        },
        {
            "role": "user",
            "content": "Please show your work."
        }
    ]
    
    print("   对话历史:")
    for msg in conversation:
        print(f"     {msg['role']}: {msg['content']}")
    
    formatted_conversation = template.render(messages=conversation)
    print(f"   格式化结果:")
    print(f"   {formatted_conversation}")

if __name__ == "__main__":
    test_jinja2_template() 