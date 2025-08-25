#!/usr/bin/env python3
"""
简化的奖励计算测试脚本
"""

def test_math_dapo_compute_score():
    """测试math_dapo_compute_score函数的基本逻辑"""
    
    # 模拟math_dapo_compute_score函数
    def mock_math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True):
        # 简单的答案提取逻辑
        if "\\boxed{4}" in solution_str and ground_truth == "4":
            return {"score": 1.0, "acc": True, "pred": "4"}
        elif "\\boxed{5}" in solution_str and ground_truth == "4":
            return {"score": -1.0, "acc": False, "pred": "5"}
        else:
            return {"score": -1.0, "acc": False, "pred": ""}
    
    def test_reward_func(prompt, response, label, tool_call_count):
        """测试reward函数逻辑"""
        solution_str = prompt + response
        result = mock_math_dapo_compute_score(solution_str, label, strict_box_verify=True)
        
        base_score = result["score"]
        
        # 改进的工具调用奖励逻辑
        if base_score > 0:  # 正确答案
            if 1 <= tool_call_count <= 3:
                tool_bonus = 0.2
            elif tool_call_count > 3:
                tool_bonus = 0.1
            else:
                tool_bonus = 0.0
            result["score"] = base_score + tool_bonus
        else:  # 错误答案
            if tool_call_count == 0:
                result["score"] = base_score - 0.1
            else:
                result["score"] = base_score + 0.05
        
        return result
    
    # 测试用例
    test_cases = [
        {
            "name": "正确答案，1次工具调用",
            "prompt": "Solve this math problem: What is 2 + 2?",
            "response": "Let me calculate this step by step.\n\n```python\nresult = 2 + 2\nprint(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}",
            "label": "4",
            "tool_call_count": 1
        },
        {
            "name": "错误答案，1次工具调用",
            "prompt": "Solve this math problem: What is 2 + 2?",
            "response": "Let me calculate this step by step.\n\n```python\nresult = 2 + 3\nprint(result)\n```\n\n<interpreter>\n5\n</interpreter>\n\nAnswer: \\boxed{5}",
            "label": "4",
            "tool_call_count": 1
        },
        {
            "name": "正确答案，无工具调用",
            "prompt": "Solve this math problem: What is 2 + 2?",
            "response": "The answer is 4.\n\nAnswer: \\boxed{4}",
            "label": "4",
            "tool_call_count": 0
        },
        {
            "name": "正确答案，3次工具调用",
            "prompt": "Solve this math problem: What is 2 + 2?",
            "response": "Let me calculate this step by step.\n\n```python\nresult = 2 + 2\nprint(result)\n```\n\n<interpreter>\n4\n</interpreter>\n\n```python\nverify = result * 1\nprint(verify)\n```\n\n<interpreter>\n4\n</interpreter>\n\n```python\nfinal_check = verify + 0\nprint(final_check)\n```\n\n<interpreter>\n4\n</interpreter>\n\nAnswer: \\boxed{4}",
            "label": "4",
            "tool_call_count": 3
        },
        {
            "name": "错误答案，无工具调用",
            "prompt": "Solve this math problem: What is 2 + 2?",
            "response": "The answer is 5.\n\nAnswer: \\boxed{5}",
            "label": "4",
            "tool_call_count": 0
        }
    ]
    
    print("=" * 60)
    print("奖励计算测试")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  工具调用次数: {case['tool_call_count']}")
        
        result = test_reward_func(
            case['prompt'], 
            case['response'], 
            case['label'], 
            case['tool_call_count']
        )
        
        print(f"  奖励结果: {result}")
        print(f"    分数: {result['score']}")
        print(f"    准确率: {result['acc']}")
        print(f"    预测: {result['pred']}")
    
    print("\n" + "=" * 60)
    print("奖励归一化模拟")
    print("=" * 60)
    
    # 模拟reward归一化过程
    rewards = [1.2, 1.2, -0.95, -0.95, 1.2, -0.95, 1.2, -0.95]  # 模拟8个样本的奖励
    n_samples_per_prompt = 4
    
    print(f"原始奖励: {rewards}")
    print(f"每组样本数: {n_samples_per_prompt}")
    
    # 重塑为组
    num_groups = len(rewards) // n_samples_per_prompt
    rewards_reshaped = []
    for i in range(num_groups):
        group = rewards[i * n_samples_per_prompt:(i + 1) * n_samples_per_prompt]
        rewards_reshaped.append(group)
    
    print(f"重塑后数据:")
    for i, group in enumerate(rewards_reshaped):
        print(f"  组 {i}: {group}")
    
    # 计算组内均值
    means = []
    for group in rewards_reshaped:
        mean = sum(group) / len(group)
        means.append(mean)
    
    print(f"组内均值: {means}")
    
    # 归一化
    normalized_rewards = []
    for i, group in enumerate(rewards_reshaped):
        mean = means[i]
        normalized_group = [r - mean for r in group]
        normalized_rewards.extend(normalized_group)
    
    print(f"归一化后: {normalized_rewards}")
    
    # 标准差归一化
    stds = []
    for group in rewards_reshaped:
        mean = sum(group) / len(group)
        variance = sum((r - mean) ** 2 for r in group) / len(group)
        std = variance ** 0.5
        stds.append(std)
    
    print(f"组内标准差: {stds}")
    
    # 最终归一化
    final_rewards = []
    for i, group in enumerate(rewards_reshaped):
        mean = means[i]
        std = stds[i]
        for r in group:
            normalized = (r - mean) / (std + 1e-6)
            final_rewards.append(normalized)
    
    print(f"最终归一化: {final_rewards}")

if __name__ == "__main__":
    test_math_dapo_compute_score() 