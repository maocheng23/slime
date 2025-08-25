#!/usr/bin/env python3
"""
测试tool call解析功能
"""

import sys
import asyncio
sys.path.append('/root/slime')

# 直接导入函数，避免模块路径问题
sys.path.append('examples/retool')
from generate_with_retool import (
    postprocess_predictions,
    postprocess_responses,
    execute_predictions,
    tool_registry
)

async def test_tool_call_parsing():
    """测试tool call解析功能"""
    
    print("=" * 60)
    print("Tool Call解析测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            "name": "新的tool_call格式",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
        },
        {
            "name": "旧的<code>格式",
            "prediction": """Let me calculate this step by step.

<code>
result = 15 * 23
print(result)
</code>"""
        },
        {
            "name": "旧的```python格式",
            "prediction": """Let me calculate this step by step.

```python
result = 15 * 23
print(result)
```"""
        },
        {
            "name": "答案格式 - Answer: \\boxed{}",
            "prediction": """Let me calculate this step by step.

Answer: \\boxed{345}"""
        },
        {
            "name": "答案格式 - <answer>标签",
            "prediction": """Let me calculate this step by step.

<answer>345</answer>"""
        },
        {
            "name": "无效的tool_call格式",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"invalid": "format"}
</tool_call>"""
        },
        {
            "name": "混合格式",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>

The result is 345.

Answer: \\boxed{345}"""
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"   预测内容: {case['prediction'][:100]}...")
        
        # 测试postprocess_predictions
        action, content = postprocess_predictions(case['prediction'])
        print(f"   解析结果: action={action}, content={repr(content)}")
        
        # 测试postprocess_responses
        processed = postprocess_responses(case['prediction'])
        print(f"   处理后: {processed[:100]}...")
        
        # 测试execute_predictions（如果是代码）
        if action == "code":
            print(f"   执行代码...")
            try:
                next_obs, done = await execute_predictions(case['prediction'])
                print(f"   执行结果: done={done}")
                print(f"   输出: {next_obs[:100]}...")
            except Exception as e:
                print(f"   执行错误: {e}")
        
        print(f"   " + "-" * 50)
    
    # 测试工具注册
    print(f"\n工具注册测试:")
    tools = tool_registry.get_tool_specs()
    print(f"   注册的工具数量: {len(tools)}")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    # 测试工具执行
    print(f"\n工具执行测试:")
    try:
        result = await tool_registry.execute_tool("python", {"code": "print('Hello, World!')"})
        print(f"   执行结果: {result}")
    except Exception as e:
        print(f"   执行错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_call_parsing()) 