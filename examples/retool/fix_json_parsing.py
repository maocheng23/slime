#!/usr/bin/env python3
"""
修复JSON解析问题
"""

import re
import json

def fix_json_string(json_str):
    """修复JSON字符串中的换行符问题"""
    # 方法1: 直接替换换行符
    fixed = json_str.replace('\n', '\\n')
    return fixed

def fix_json_string_v2(json_str):
    """修复JSON字符串中的换行符问题 - 方法2"""
    # 方法2: 使用正则表达式替换字符串值中的换行符
    def replace_newlines(match):
        return match.group(0).replace('\n', '\\n')
    
    # 匹配JSON字符串值并替换其中的换行符
    pattern = r'"([^"]*)"'
    fixed = re.sub(pattern, replace_newlines, json_str)
    return fixed

def test_json_fixing():
    """测试JSON修复功能"""
    
    print("=" * 60)
    print("JSON修复测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        '{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}',
        '{"name": "python", "arguments": {"code": "import math\nresult = math.sqrt(144)\nprint(result)"}}',
        '{"name": "python", "arguments": {"code": "print(\'Hello World\')"}}'
    ]
    
    for i, test_json in enumerate(test_cases, 1):
        print(f"\n{i}. 测试用例:")
        print(f"   原始JSON: {test_json}")
        
        # 方法1
        fixed1 = fix_json_string(test_json)
        print(f"   方法1修复: {fixed1}")
        try:
            data1 = json.loads(fixed1)
            print(f"   方法1解析成功: {data1}")
        except json.JSONDecodeError as e:
            print(f"   方法1解析失败: {e}")
        
        # 方法2
        fixed2 = fix_json_string_v2(test_json)
        print(f"   方法2修复: {fixed2}")
        try:
            data2 = json.loads(fixed2)
            print(f"   方法2解析成功: {data2}")
        except json.JSONDecodeError as e:
            print(f"   方法2解析失败: {e}")
        
        print(f"   " + "-" * 50)

def test_tool_call_parsing():
    """测试tool call解析"""
    
    print("=" * 60)
    print("Tool Call解析测试")
    print("=" * 60)
    
    test_prediction = """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
    
    print(f"测试内容:")
    print(f"{test_prediction}")
    print(f"=" * 60)
    
    # 提取tool_call内容
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    match = re.search(tool_call_pattern, test_prediction, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        print(f"提取的JSON: {json_str}")
        
        # 修复JSON
        fixed_json = fix_json_string(json_str)
        print(f"修复后JSON: {fixed_json}")
        
        try:
            data = json.loads(fixed_json)
            print(f"解析成功: {data}")
            print(f"tool_name: {data.get('name')}")
            print(f"arguments: {data.get('arguments')}")
            print(f"code: {data.get('arguments', {}).get('code')}")
        except json.JSONDecodeError as e:
            print(f"解析失败: {e}")

if __name__ == "__main__":
    test_json_fixing()
    test_tool_call_parsing() 