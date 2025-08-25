#!/usr/bin/env python3
"""
简化的tool call解析测试
"""

import re
import json

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
            tool_call_data = json.loads(tool_call_match.group(1))
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            
            if tool_name == "python":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
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

def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r'<tool_call>\s*\{.*?\}\s*</tool_call>'
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]
    
    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"
    
    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r'```python\s*.*?```'
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]
    
    # Handle <answer> tags
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    
    # Handle Answer: \boxed{...} format
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...}
        answer_pattern = r'Answer:\s*\\boxed\{[^}]*\}'
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]
    
    return resp

def test_tool_call_parsing():
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
        
        print(f"   " + "-" * 50)
    
    # 测试JSON解析
    print(f"\nJSON解析测试:")
    test_json = '{"name": "python", "arguments": {"code": "print(123)"}}'
    try:
        data = json.loads(test_json)
        print(f"   解析成功: {data}")
        print(f"   tool_name: {data.get('name')}")
        print(f"   arguments: {data.get('arguments')}")
    except json.JSONDecodeError as e:
        print(f"   解析失败: {e}")

if __name__ == "__main__":
    test_tool_call_parsing() 