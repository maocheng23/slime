#!/usr/bin/env python3
"""
调试tool_call解析问题
"""

import re
import json

def debug_tool_call_parsing():
    """调试tool_call解析"""
    
    print("=" * 60)
    print("Tool Call解析调试")
    print("=" * 60)
    
    # 测试用例
    test_prediction = """Let me calculate this step by step.

<tool_call>
{"name": "python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}
</tool_call>"""
    
    print(f"测试内容:")
    print(f"{test_prediction}")
    print(f"=" * 60)
    
    # 测试正则表达式
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    print(f"正则表达式: {tool_call_pattern}")
    
    match = re.search(tool_call_pattern, test_prediction, re.DOTALL)
    print(f"匹配结果: {match}")
    
    if match:
        print(f"匹配组: {match.groups()}")
        json_str = match.group(1)
        print(f"JSON字符串: {json_str}")
        
        try:
            tool_call_data = json.loads(json_str)
            print(f"解析成功: {tool_call_data}")
            print(f"tool_name: {tool_call_data.get('name')}")
            print(f"arguments: {tool_call_data.get('arguments')}")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
    else:
        print("没有匹配到tool_call")
    
    # 测试不同的正则表达式
    print(f"\n" + "=" * 60)
    print("测试不同的正则表达式")
    print("=" * 60)
    
    patterns = [
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        r"<tool_call>(.*?)</tool_call>",
        r"<tool_call>\s*(\{.*\})\s*</tool_call>",
        r"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. 模式: {pattern}")
        match = re.search(pattern, test_prediction, re.DOTALL)
        if match:
            print(f"   匹配成功: {match.group(1)}")
        else:
            print(f"   匹配失败")
    
    # 测试JSON解析
    print(f"\n" + "=" * 60)
    print("JSON解析测试")
    print("=" * 60)
    
    json_str = '{"name": "python", "arguments": {"code": "result = 15 * 23\\nprint(result)"}}'
    print(f"JSON字符串: {json_str}")
    
    try:
        data = json.loads(json_str)
        print(f"解析成功: {data}")
    except json.JSONDecodeError as e:
        print(f"解析失败: {e}")

if __name__ == "__main__":
    debug_tool_call_parsing() 