import re

def extract_code_blocks(text: str):
    """Extract code blocks and interpreter output from text"""
    code_blocks = []
    
    # Match code block pattern: <code>\n```python\ncode\n```\n</code>
    code_pattern = r'<code>\s*```python\s*(.*?)\s*```\s*</code>'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    
    # Also match ```python code blocks without <code> tags
    python_code_pattern = r'```python\s*(.*?)\s*```'
    python_code_matches = re.findall(python_code_pattern, text, re.DOTALL)
    
    # Combine both types of matches
    all_code_matches = code_matches + python_code_matches
    
    # Match interpreter output pattern: <interpreter>output</interpreter>
    interpreter_pattern = r'<interpreter>(.*?)</interpreter>'
    interpreter_matches = re.findall(interpreter_pattern, text, re.DOTALL)
    
    # Pair code blocks with interpreter outputs
    for i, code in enumerate(all_code_matches):
        code_block = {
            "code": code.strip(),
            "output": (interpreter_matches[i] if i < len(interpreter_matches) 
                      else "")
        }
        code_blocks.append(code_block)
    
    return code_blocks

# Test the function
test_text = """Finally, compute the distance squared between \\(F\\) and \\(G\\):
```python
f = -11/2
g = 1/2
FG_squared = (g - f)**2 + (12 - 0)**2
print(FG_squared)
```
Answer: \\boxed{169}<|im_end|>"""

print("Testing code extraction:")
code_blocks = extract_code_blocks(test_text)
print(f"Found {len(code_blocks)} code blocks")
for i, block in enumerate(code_blocks):
    print(f"\nCode Block {i+1}:")
    print(f"Code: {repr(block['code'])}")
    print(f"Output: {repr(block['output'])}") 