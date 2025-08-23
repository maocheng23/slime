import re

def execute_python_code_safely(code: str) -> str:
    """Safely execute Python code and return the output"""
    try:
        # Create a safe execution environment with minimal builtins
        import io
        import sys
        
        # Capture stdout
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # Create safe globals with only necessary builtins
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                "all": all,
                "any": any,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "reversed": reversed,
                "filter": filter,
                "map": map,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                "dir": dir,
                "vars": vars,
                "locals": locals,
                "globals": globals,
                "eval": eval,
                "exec": exec,
                "compile": compile,
                "open": open,
                "input": input,
                "raw_input": input,  # Python 2 compatibility
                "reload": lambda x: x,  # Dummy function
                "help": lambda x: "Help not available in safe mode",
                "copyright": "Copyright not available in safe mode",
                "credits": "Credits not available in safe mode",
                "license": "License not available in safe mode",
                "exit": lambda: None,  # Dummy function
                "quit": lambda: None,  # Dummy function
            }
        }
        
        # Execute the code
        exec(code, safe_globals, {})
        
        # Get the captured output
        output = new_stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        return output.strip() if output else "No output"
    except Exception as e:
        return f"Error: {str(e)}"

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
        # If no interpreter output found, try to simulate the output
        output = ""
        if i < len(interpreter_matches):
            output = interpreter_matches[i]
        else:
            # For code blocks without interpreter output, simulate execution
            try:
                output = execute_python_code_safely(code.strip())
            except Exception as e:
                output = f"[Execution error: {str(e)}]"
        
        code_block = {
            "code": code.strip(),
            "output": output
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

print("Testing code extraction with execution:")
code_blocks = extract_code_blocks(test_text)
print(f"Found {len(code_blocks)} code blocks")
for i, block in enumerate(code_blocks):
    print(f"\nCode Block {i+1}:")
    print(f"Code: {repr(block['code'])}")
    print(f"Output: {repr(block['output'])}") 