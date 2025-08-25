#!/usr/bin/env python3
"""
调试奖励计算过程的脚本
"""

import sys
import asyncio

sys.path.append('/root/slime')

from examples.retool.generate_with_retool import reward_func
from slime.utils.types import Sample


async def debug_reward():
    """调试奖励计算"""
    
    # 测试用例1：正确答案，1次工具调用
    sample1 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response=("Let me calculate this step by step.\n\n```python\nresult = 2 + 2\n"
                 "print(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}"),
        label="4"
    )
    sample1.tool_call_count = 1
    
    # 测试用例2：错误答案，1次工具调用
    sample2 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="Let me calculate this step by step.\n\n```python\nresult = 2 + 3\nprint(result)\n```\n\n<interpreter>\n5\n</interpreter>\n\nAnswer: \\boxed{5}",
        label="4"
    )
    sample2.tool_call_count = 1
    
    # 测试用例3：正确答案，无工具调用
    sample3 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="The answer is 4.\n\nAnswer: \\boxed{4}",
        label="4"
    )
    sample3.tool_call_count = 0
    
    # 测试用例4：正确答案，3次工具调用
    sample4 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="Let me calculate this step by step.\n\n```python\nresult = 2 + 2\nprint(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\n```python\nverify = result * 1\nprint(verify)\n```\n\n<interpreter>\n4\n</interpreter>\n\n```python\nfinal_check = verify + 0\nprint(final_check)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}",
        label="4"
    )
    sample4.tool_call_count = 3
    
    # 测试用例5：错误答案，无工具调用
    sample5 = Sample(
        prompt="Solve this math problem: What is 2 + 2?",
        response="The answer is 5.\n\nAnswer: \\boxed{5}",
        label="4"
    )
    sample5.tool_call_count = 0
    
    class MockArgs:
        pass
    
    args = MockArgs()
    
    test_cases = [
        ("正确答案，1次工具调用", sample1),
        ("错误答案，1次工具调用", sample2),
        ("正确答案，无工具调用", sample3),
        ("正确答案，3次工具调用", sample4),
        ("错误答案，无工具调用", sample5),
    ]
    
    print("=" * 60)
    print("奖励计算调试")
    print("=" * 60)
    
    for case_name, sample in test_cases:
        print(f"\n{case_name}:")
        print(f"  工具调用次数: {sample.tool_call_count}")
        print(f"  完整响应: {sample.response[:100]}...")
        
        try:
            reward_result = await reward_func(args, sample)
            print(f"  奖励结果: {reward_result}")
            
            if isinstance(reward_result, dict):
                print(f"    分数: {reward_result.get('score', 'N/A')}")
                print(f"    准确率: {reward_result.get('acc', 'N/A')}")
                print(f"    预测: {reward_result.get('pred', 'N/A')}")
            else:
                print(f"    总分数: {reward_result}")
                
        except Exception as e:
            print(f"    错误: {e}")
    
    print("\n" + "=" * 60)
    print("奖励归一化模拟")
    print("=" * 60)
    
    # 模拟reward归一化过程
    rewards = [1.2, 1.2, -1.05, -1.05, 1.2, -1.05, 1.2, -1.05]  # 模拟8个样本的奖励
    n_samples_per_prompt = 4
    
    print(f"原始奖励: {rewards}")
    print(f"每组样本数: {n_samples_per_prompt}")
    
    import torch
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    rewards_reshaped = rewards_tensor.reshape(-1, n_samples_per_prompt)
    
    print(f"重塑后形状: {rewards_reshaped.shape}")
    print(f"重塑后数据:\n{rewards_reshaped}")
    
    # 计算组内均值
    mean = rewards_reshaped.mean(dim=-1, keepdim=True)
    print(f"组内均值:\n{mean}")
    
    # 归一化
    normalized_rewards = rewards_reshaped - mean
    print(f"归一化后:\n{normalized_rewards}")
    
    # 标准差归一化
    std = normalized_rewards.std(dim=-1, keepdim=True)
    print(f"组内标准差:\n{std}")
    
    final_rewards = normalized_rewards / (std + 1e-6)
    print(f"最终归一化:\n{final_rewards}")
    
    print(f"最终奖励列表: {final_rewards.flatten().tolist()}")

if __name__ == "__main__":
    asyncio.run(debug_reward()) 