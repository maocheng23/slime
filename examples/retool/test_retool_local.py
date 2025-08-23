#!/usr/bin/env python3
"""
Local tester for testing retool Python tool call functionality
Simulates LLM output without actually running LLM
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import our tool call implementation
from generate_with_retool import (
    ToolRegistry, 
    postprocess_responses,
    postprocess_predictions
)


@dataclass
class MockSample:
    """Mock Sample class"""
    prompt: str
    response: str = ""
    tokens: List[int] = None
    response_length: int = 0
    loss_masks: List[int] = None
    status: str = "PENDING"


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
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


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract final answer in \boxed{} format from text"""
    # Match \boxed{answer} pattern
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


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





class RetoolLocalTester:
    """Retool local tester"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
    

    
    async def test_new_format(self):
        """Test new format response"""
        print(f"\n{'='*60}")
        print("Testing New Format Response")
        print(f"{'='*60}")
        
        # Test the format you provided
        test_response = """Okay, let's tackle this geometry problem step by step. First, let me understand what's given and what's being asked.

We have a rectangle \\(ABCD\\) with \\(AB = 5\\) and \\(AD = 12\\). So, \\(AB\\) is the length, and \\(AD\\) is the width. Let me visualize this rectangle. Let's consider \\(A\\) at the origin \\((0, 0)\\), \\(B\\) at \\((5, 0)\\), \\(D\\) at \\((0, 12)\\), and \\(C\\) at \\((5, 12)\\). That's the standard coordinate system setup.

Now, there's a point \\(E\\) outside the rectangle such that the perpendicular distances from \\(E\\) to the lines \\(AB\\) and \\(AD\\) are 12 and 1, respectively. Also, triangle \\(ABE\\) is acute. We need to find \\(FG^2\\), where \\(F\\) and \\(G\\) are the intersections of a line passing through \\(E\\) that splits the rectangle into two equal areas. The line intersects \\(\\overline{AB}\\) at \\(F\\) and \\(\\overline{CD}\\) at \\(G\\).

First, let's figure out the coordinates of point \\(E\\). The distances from \\(E\\) to lines \\(AB\\) and \\(AD\\) are 12 and 1. The line \\(AB\\) is the x-axis, so the distance from \\(E\\) to \\(AB\\) is the vertical distance, which is 12. Similarly, the line \\(AD\\) is the y-axis, so the distance from \\(E\\) to \\(AD\\) is the horizontal distance, which is 1. However, since \\(E\\) is outside the rectangle, we need to determine the correct quadrant.

Given that triangle \\(ABE\\) is acute, let's analyze the possible positions of \\(E\\). If \\(E\\) is at \\((1, 12)\\), the triangle would have sides \\(AB = 5\\), \\(BE = \\sqrt{(5-1)^2 + (0-12)^2} = \\sqrt{16 + 144} = \\sqrt{160}\\), and \\(AE = \\sqrt{1^2 + 12^2} = \\sqrt{145}\\). Checking angles using the Law of Cosines, angle at \\(B\\) would be obtuse, violating the acute condition. Similarly, other quadrants for \\(E\\) either place it inside the rectangle or result in an obtuse triangle. After verifying all possibilities, \\(E\\) must be at \\((-1, 1)\\) to satisfy the acute triangle condition.

Next, the line passing through \\(E\\) must split the rectangle into two equal areas. The area of the rectangle is \\(5 \\times 12 = 60\\), so each part must have area 30. The line intersects \\(\\overline{AB}\\) at \\(F(f, 0)\\) and \\(\\overline{CD}\\) at \\(G(f + 6, 12)\\). The equation of the line is determined by passing through \\(E(-1, 1)\\) and has a slope calculated to ensure the area condition. Solving the system of equations for the line and the rectangle's boundaries, we find \\(f = -\\frac{11}{2}\\), leading to \\(F\\left(-\\frac{11}{2}, 0\\right)\\) and \\(G\\left(\\frac{1}{2}, 12\\right)\\).

Finally, compute the distance squared between \\(F\\) and \\(G\\):
```python
f = -11/2
g = 1/2
FG_squared = (g - f)**2 + (12 - 0)**2
print(FG_squared)
```
Answer: \\boxed{169}<|im_end|>
"""
        
        print("Original Response:")
        print(test_response)
        
        # Test postprocess_predictions
        print(f"\n{'='*60}")
        print("Testing postprocess_predictions")
        print(f"{'='*60}")
        
        action, content = postprocess_predictions(test_response)
        print(f"Action: {action}")
        print(f"Content: {repr(content)}")
        
        # Test postprocess_responses
        print(f"\n{'='*60}")
        print("Testing postprocess_responses")
        print(f"{'='*60}")
        
        processed_response = postprocess_responses(test_response)
        print(f"Processed Response: {repr(processed_response)}")
        
        # Extract code blocks
        print(f"\n{'='*60}")
        print("Testing Code Block Extraction")
        print(f"{'='*60}")
        
        code_blocks = extract_code_blocks(test_response)
        print(f"Number of Extracted Code Blocks: {len(code_blocks)}")
        for i, block in enumerate(code_blocks):
            print(f"\nCode Block {i+1}:")
            print(f"Code:\n{block['code']}")
            print(f"Output:\n{block['output']}")
        
        # Extract final answer
        print(f"\n{'='*60}")
        print("Testing Answer Extraction")
        print(f"{'='*60}")
        
        final_answer = extract_boxed_answer(test_response)
        print(f"Final Answer: {final_answer}")
        
        # Simulate tool call count
        print(f"\n{'='*60}")
        print("Testing Tool Call Analysis")
        print(f"{'='*60}")
        
        tool_call_count = len(code_blocks)
        print(f"Tool Call Count: {tool_call_count}")
        
        # Simulate reward calculation
        print(f"\n{'='*60}")
        print("Testing Reward Calculation")
        print(f"{'='*60}")
        
        # Create a mock sample for reward calculation
        from slime.utils.types import Sample
        
        mock_sample = Sample(
            prompt="Solve this geometry problem: Find FG² where F and G are...",
            response=test_response,
            label="169"  # Ground truth
        )
        mock_sample.tool_call_count = tool_call_count
        
        # Import and test reward function
        try:
            from generate_with_retool import reward_func
            
            # Create mock args
            class MockArgs:
                pass
            args = MockArgs()
            
            # Calculate reward
            reward_result = await reward_func(args, mock_sample)
            print(f"Reward Result: {reward_result}")
            
            if isinstance(reward_result, dict):
                print(f"Score: {reward_result.get('score', 'N/A')}")
                print(f"Accuracy: {reward_result.get('acc', 'N/A')}")
                print(f"Prediction: {reward_result.get('pred', 'N/A')}")
            else:
                print(f"Total Score: {reward_result}")
                
        except Exception as e:
            print(f"Error calculating reward: {e}")
            reward_result = None
        
        return {
            "action": action,
            "content": content,
            "processed_response": processed_response,
            "code_blocks": code_blocks,
            "final_answer": final_answer,
            "tool_call_count": tool_call_count,
            "reward_result": reward_result
        }


async def main():
    """Main function"""
    tester = RetoolLocalTester()
    
    # Test new format (geometry problem example)
    print("Starting Geometry Problem Test...")
    await tester.test_new_format()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 