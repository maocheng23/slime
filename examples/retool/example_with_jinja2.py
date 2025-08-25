#!/usr/bin/env python3
"""
使用Jinja2模板的Retool示例
"""

import sys
import asyncio
sys.path.append('/root/slime')

# 直接导入函数，避免模块路径问题
sys.path.append('examples/retool')
from generate_with_retool import (
    format_conversation_with_tools,
    tool_registry,
    execute_predictions
)

async def test_jinja2_retool():
    """测试使用Jinja2模板的retool功能"""
    
    print("=" * 60)
    print("Jinja2 Retool示例")
    print("=" * 60)
    
    # 1. 查看注册的工具
    print("\n1. 注册的工具:")
    tools = tool_registry.get_tool_specs()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # 2. 测试不同的prompt格式
    test_cases = [
        {
            "name": "数学计算问题",
            "prompt": "Solve this math problem: What is 15 * 23?",
            "system_prompt": "You are a helpful assistant that can use Python tools to solve mathematical problems. When you need to perform calculations, use the Python tool to execute code and get results."
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
        print(f"   {formatted[:200]}...")
    
    # 3. 测试工具调用执行
    print(f"\n3. 工具调用测试:")
    
    # 模拟一个包含工具调用的响应
    test_response = """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
    
    print(f"   测试响应: {test_response}")
    
    # 执行工具调用
    next_obs, done = await execute_predictions(test_response)
    print(f"   执行结果: {next_obs}")
    print(f"   是否完成: {done}")
    
    # 4. 完整的对话流程示例
    print(f"\n4. 完整对话流程示例:")
    
    # 模拟多轮对话
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
    
    # 使用模板格式化多轮对话
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
    formatted_conversation = template.render(messages=conversation)
    
    print(f"   格式化结果:")
    print(f"   {formatted_conversation}")

if __name__ == "__main__":
    asyncio.run(test_jinja2_retool()) 