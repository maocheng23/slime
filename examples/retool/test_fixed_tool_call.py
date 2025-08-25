#!/usr/bin/env python3
"""
测试修复后的tool call解析
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
            # Clean up the JSON string by removing newlines and extra whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = re.sub(r'\n(?=[^"]*"[^"]*$)', '\\n', json_str)
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            
            if tool_name == "python":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"Tool call parsing error: {e}")
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

def test_fixed_tool_call():
    """测试修复后的tool call解析"""
    
    print("=" * 60)
    print("修复后的Tool Call解析测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            "name": "新的tool_call格式（多行）",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
        },
        {
            "name": "新的tool_call格式（单行）",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "print('Hello World')"}}
</tool_call>"""
        },
        {
            "name": "复杂的tool_call格式",
            "prediction": """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "import math\nresult = math.sqrt(144)\nprint(result)"}}
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
            "name": "答案格式",
            "prediction": """Let me calculate this step by step.

Answer: \\boxed{345}"""
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"   预测内容: {case['prediction'][:100]}...")
        
        # 测试postprocess_predictions
        action, content = postprocess_predictions(case['prediction'])
        print(f"   解析结果: action={action}, content={repr(content)}")
        
        if action == "code":
            print(f"   代码内容: {content}")
        
        print(f"   " + "-" * 50)
    
    # 测试JSON清理功能
    print(f"\nJSON清理测试:")
    test_json = '{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}'
    print(f"   原始JSON: {test_json}")
    
    # 应用清理逻辑
    cleaned_json = re.sub(r'\n(?=[^"]*"[^"]*$)', '\\n', test_json)
    print(f"   清理后JSON: {cleaned_json}")
    
    try:
        data = json.loads(cleaned_json)
        print(f"   解析成功: {data}")
        print(f"   代码: {data['arguments']['code']}")
    except json.JSONDecodeError as e:
        print(f"   解析失败: {e}")

if __name__ == "__main__":
    test_fixed_tool_call() 