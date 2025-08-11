#!/usr/bin/env python3
"""
快速测试脚本 - 测试 Python 工具调用功能
"""

import asyncio
from generate_with_retool import ToolRegistry, extract_tool_calls, execute_tool_calls


async def test_python_tool():
    """测试 Python 工具"""
    print("测试 Python 工具...")
    
    tool_registry = ToolRegistry()
    
    # 测试安全的 Python 代码
    safe_codes = [
        "print('Hello, World!')",
        "print(2 + 3 * 4)",
        "import math\nprint(math.pi)",
        "numbers = [1, 2, 3, 4, 5]\nprint(sum(numbers))",
        ("def factorial(n):\n    if n <= 1:\n        return 1\n    "
         "return n * factorial(n-1)\nprint(factorial(5))")
    ]
    
    for i, code in enumerate(safe_codes, 1):
        print(f"\n测试 {i}: {code}")
        result = await tool_registry.execute_tool("python", {"code": code})
        print(f"结果: {result}")
    
    # 测试危险代码
    print("\n测试危险代码...")
    dangerous_codes = [
        "import os",
        "eval('print(1)')",
        "open('/etc/passwd')"
    ]
    
    for code in dangerous_codes:
        print(f"\n危险代码: {code}")
        result = await tool_registry.execute_tool("python", {"code": code})
        print(f"结果: {result}")


async def test_calculator_tool():
    """测试计算器工具"""
    print("\n\n测试计算器工具...")
    
    tool_registry = ToolRegistry()
    
    # 测试正常表达式
    expressions = [
        "2 + 3",
        "10 * 5",
        "100 / 4",
        "(2 + 3) * 4",
        "15 * 23 + 7"
    ]
    
    for expr in expressions:
        print(f"\n表达式: {expr}")
        result = await tool_registry.execute_tool("calculator", {"expression": expr})
        print(f"结果: {result}")
    
    # 测试危险表达式
    print("\n测试危险表达式...")
    dangerous_exprs = [
        "os.system('rm -rf /')",
        "eval('print(1)')",
        "import os"
    ]
    
    for expr in dangerous_exprs:
        print(f"\n危险表达式: {expr}")
        result = await tool_registry.execute_tool("calculator", {"expression": expr})
        print(f"结果: {result}")


async def test_tool_extraction():
    """测试工具调用提取"""
    print("\n\n测试工具调用提取...")
    
    # 模拟 LLM 响应
    mock_responses = [
        """Let me calculate that for you.
<tool_call>{"name": "calculator", "arguments": {"expression": "15 * 23 + 7"}}</tool_call>
<tool_results>
Tool 'calculator' result: Result: 352
</tool_results>
<answer>The result is 352.</answer>""",
        
        """I'll write a Python function.
<tool_call>{"name": "python", "arguments": {"code": "print('Hello')"}}</tool_call>
<tool_results>
Tool 'python' result: Output:
Hello
</tool_results>
<answer>Function executed successfully.</answer>"""
    ]
    
    for i, response in enumerate(mock_responses, 1):
        print(f"\n响应 {i}:")
        print(response)
        
        tool_calls = extract_tool_calls(response)
        print(f"提取的工具调用: {tool_calls}")
        
        if tool_calls:
            results = await execute_tool_calls(tool_calls)
            print(f"执行结果: {results}")


async def main():
    """主函数"""
    print("开始快速测试...")
    
    await test_python_tool()
    await test_calculator_tool()
    await test_tool_extraction()
    
    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(main()) 