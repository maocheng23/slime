#!/usr/bin/env python3
"""
调试奖励计算过程的脚本
"""

import sys
import os
sys.path.append('/root/slime')

from examples.retool.generate_with_retool import reward_func
from slime.utils.types import Sample
import asyncio

async def debug_reward():
    """调试奖励计算"""
    
    # 测试用例1：正确答案
    sample1 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="Let me calculate this step by step.\n\n```python\nresult = 2 + 2\nprint(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}",
        label="4"
    )
    sample1.tool_call_count = 1
    
    # 测试用例2：错误答案
    sample2 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="Let me calculate this step by step.\n\n```python\nresult = 2 + 3\nprint(result)\n```\n\n<interpreter>\n5\n</interpreter>\n\nAnswer: \\boxed{5}",
        label="4"
    )
    sample2.tool_call_count = 1
    
    # 测试用例3：没有工具调用
    sample3 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="The answer is 4.\n\nAnswer: \\boxed{4}",
        label="4"
    )
    sample3.tool_call_count = 0
    
    # 测试用例4：多个工具调用
    sample4 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="Let me calculate this step by step.\n\n```python\nresult = 2 + 2\nprint(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\n```python\nverify = result * 1\nprint(verify)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}",
        label="4"
    )
    sample4.tool_call_count = 2
    
    class MockArgs:
        pass
    
    args = MockArgs()
    
    test_cases = [
        ("正确答案，1次工具调用", sample1),
        ("错误答案，1次工具调用", sample2),
        ("正确答案，无工具调用", sample3),
        ("正确答案，2次工具调用", sample4),
    ]
    
    print("=" * 60)
    print("奖励计算调试")
    print("=" * 60)
    
    for name, sample in test_cases:
        print(f"\n测试用例: {name}")
        print(f"Prompt: {sample.prompt}")
        print(f"Response: {sample.response}")
        print(f"Label: {sample.label}")
        print(f"Tool call count: {sample.tool_call_count}")
        
        try:
            result = await reward_func(args, sample)
            print(f"Reward result: {result}")
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Accuracy: {result.get('acc', 'N/A')}")
            print(f"Prediction: {result.get('pred', 'N/A')}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(debug_reward()) 