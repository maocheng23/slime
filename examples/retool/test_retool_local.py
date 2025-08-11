#!/usr/bin/env python3
"""
本地测试器，用于测试 retool 的 Python 工具调用功能
模拟 LLM 输出而不实际运行 LLM
"""

import asyncio
import re
from typing import Dict, List, Any
from dataclasses import dataclass

# 导入我们的工具调用实现
from generate_with_retool import (
    ToolRegistry, 
    extract_tool_calls, 
    extract_final_answer, 
    execute_tool_calls,
    postprocess_response
)


@dataclass
class MockSample:
    """模拟 Sample 类"""
    prompt: str
    response: str = ""
    tokens: List[int] = None
    response_length: int = 0
    loss_masks: List[int] = None
    status: str = "PENDING"


class MockLLM:
    """模拟 LLM 输出"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
    
    def add_conversation(self, user_input: str, assistant_response: str):
        """添加对话历史"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
    
    def generate_response(self, prompt: str, max_turns: int = 3) -> str:
        """生成模拟的 LLM 响应"""
        # 基于 prompt 内容生成相应的工具调用
        if "calculate" in prompt.lower() or "math" in prompt.lower():
            return self._generate_calculator_response(prompt)
        elif ("python" in prompt.lower() or "function" in prompt.lower() or 
              "code" in prompt.lower()):
            return self._generate_python_response(prompt)
        elif "factorial" in prompt.lower():
            return self._generate_factorial_response(prompt)
        elif "fibonacci" in prompt.lower():
            return self._generate_fibonacci_response(prompt)
        else:
            return self._generate_general_response(prompt)
    
    def _generate_calculator_response(self, prompt: str) -> str:
        """生成计算器工具调用响应"""
        # 提取数字和运算符
        numbers = re.findall(r'\d+', prompt)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            if "*" in prompt:
                expression = f"{a} * {b}"
                result = a * b
            elif "+" in prompt:
                expression = f"{a} + {b}"
                result = a + b
            elif "-" in prompt:
                expression = f"{a} - {b}"
                result = a - b
            elif "/" in prompt:
                expression = f"{a} / {b}"
                result = a / b
            else:
                expression = f"{a} + {b}"
                result = a + b
            
            return f"""Let me calculate that for you.

<tool_call>{{"name": "calculator", "arguments": {{"expression": "{expression}"}}}}</tool_call>

<tool_results>
Tool 'calculator' result: Result: {result}
</tool_results>

<answer>The result of {expression} is {result}.</answer>"""
        else:
            return """I need more specific numbers to calculate. Please provide the numbers you want me to work with."""
    
    def _generate_python_response(self, prompt: str) -> str:
        """生成 Python 工具调用响应"""
        if "factorial" in prompt.lower():
            return self._generate_factorial_response(prompt)
        elif "fibonacci" in prompt.lower():
            return self._generate_fibonacci_response(prompt)
        else:
            return """I can help you write Python code. What specific function or calculation would you like me to implement?"""
    
    def _generate_factorial_response(self, prompt: str) -> str:
        """生成阶乘计算响应"""
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            n = int(numbers[0])
            code = f"""def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial({n}))"""
            
            return f"""I'll write a Python function to calculate the factorial of {n}.

<tool_call>{{"name": "python", "arguments": {{"code": "{code}"}}}}</tool_call>

<tool_results>
Tool 'python' result: Output:
{self._calculate_factorial(n)}
</tool_results>

<answer>The factorial of {n} is {self._calculate_factorial(n)}. I used a recursive function that multiplies n by the factorial of n-1 until it reaches the base case of 1.</answer>"""
        else:
            return "Please provide a number to calculate its factorial."
    
    def _generate_fibonacci_response(self, prompt: str) -> str:
        """生成斐波那契数列响应"""
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            n = int(numbers[0])
            code = f"""def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

print(fibonacci({n}))"""
            
            fib_sequence = self._calculate_fibonacci(n)
            return f"""I'll create a Python function to generate the first {n} Fibonacci numbers.

<tool_call>{{"name": "python", "arguments": {{"code": "{code}"}}}}</tool_call>

<tool_results>
Tool 'python' result: Output:
{fib_sequence}
</tool_results>

<answer>The first {n} Fibonacci numbers are: {fib_sequence}</answer>"""
        else:
            return "Please provide a number to generate Fibonacci sequence."
    
    def _generate_general_response(self, prompt: str) -> str:
        """生成通用响应"""
        return """I can help you with mathematical calculations and Python programming. What would you like me to do?"""
    
    def _calculate_factorial(self, n: int) -> int:
        """计算阶乘"""
        if n <= 1:
            return 1
        return n * self._calculate_factorial(n - 1)
    
    def _calculate_fibonacci(self, n: int) -> List[int]:
        """计算斐波那契数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        else:
            fib = [0, 1]
            for i in range(2, n):
                fib.append(fib[i-1] + fib[i-2])
            return fib


class RetoolLocalTester:
    """Retool 本地测试器"""
    
    def __init__(self):
        self.mock_llm = MockLLM()
        self.tool_registry = ToolRegistry()
    
    async def test_single_turn(self, user_input: str) -> Dict[str, Any]:
        """测试单轮对话"""
        print(f"\n{'='*60}")
        print(f"用户输入: {user_input}")
        print(f"{'='*60}")
        
        # 模拟 LLM 生成响应
        llm_response = self.mock_llm.generate_response(user_input)
        print(f"LLM 响应:\n{llm_response}")
        
        # 后处理响应
        processed_response = postprocess_response(llm_response)
        print(f"\n后处理后的响应:\n{processed_response}")
        
        # 提取工具调用
        tool_calls = extract_tool_calls(processed_response)
        print(f"\n提取的工具调用: {tool_calls}")
        
        # 执行工具调用
        tool_results = None
        if tool_calls:
            tool_results = await execute_tool_calls(tool_calls)
            print(f"\n工具执行结果:\n{tool_results}")
        
        # 提取最终答案
        final_answer = extract_final_answer(processed_response)
        print(f"\n最终答案: {final_answer}")
        
        return {
            "user_input": user_input,
            "llm_response": llm_response,
            "processed_response": processed_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_answer": final_answer
        }
    
    async def test_multi_turn(self, conversation: List[str]) -> List[Dict[str, Any]]:
        """测试多轮对话"""
        results = []
        full_context = ""
        
        for i, user_input in enumerate(conversation):
            print(f"\n{'#'*80}")
            print(f"第 {i+1} 轮对话")
            print(f"{'#'*80}")
            
            # 构建完整上下文
            current_input = full_context + user_input if full_context else user_input
            
            # 测试当前轮次
            result = await self.test_single_turn(current_input)
            results.append(result)
            
            # 更新上下文
            full_context = current_input + "\n" + result["llm_response"]
        
        return results
    
    async def test_tool_safety(self):
        """测试工具安全性"""
        print(f"\n{'='*60}")
        print("测试工具安全性")
        print(f"{'='*60}")
        
        dangerous_codes = [
            "import os\nos.system('rm -rf /')",
            "eval('print(1)')",
            "exec('print(1)')",
            "__import__('os')",
            "open('/etc/passwd', 'r')",
            "input('Enter password:')"
        ]
        
        for code in dangerous_codes:
            print(f"\n测试危险代码: {code}")
            result = await self.tool_registry.execute_tool("python", {"code": code})
            print(f"结果: {result}")
    
    async def test_calculator_safety(self):
        """测试计算器安全性"""
        print(f"\n{'='*60}")
        print("测试计算器安全性")
        print(f"{'='*60}")
        
        dangerous_expressions = [
            "os.system('rm -rf /')",
            "eval('print(1)')",
            "import os",
            "open('/etc/passwd')"
        ]
        
        for expr in dangerous_expressions:
            print(f"\n测试危险表达式: {expr}")
            result = await self.tool_registry.execute_tool("calculator", {"expression": expr})
            print(f"结果: {result}")


async def main():
    """主函数"""
    tester = RetoolLocalTester()
    
    # 测试用例
    test_cases = [
        "Calculate 15 * 23 + 7",
        "Write a function to calculate the factorial of 5",
        "Generate the first 10 Fibonacci numbers",
        "What is 100 / 4?",
        "Create a Python function to find the sum of numbers from 1 to 10"
    ]
    
    # 单轮测试
    print("开始单轮测试...")
    for test_case in test_cases:
        await tester.test_single_turn(test_case)
    
    # 多轮测试
    print("\n\n开始多轮测试...")
    multi_turn_conversation = [
        "Calculate 10 + 5",
        "Now write a function to calculate the factorial of that result",
        "What is the result of the factorial?"
    ]
    await tester.test_multi_turn(multi_turn_conversation)
    
    # 安全性测试
    print("\n\n开始安全性测试...")
    await tester.test_tool_safety()
    await tester.test_calculator_safety()


if __name__ == "__main__":
    asyncio.run(main()) 