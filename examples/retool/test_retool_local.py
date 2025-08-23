#!/usr/bin/env python3
"""
Local tester for testing retool Python tool call functionality
Simulates LLM output without actually running LLM
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
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


class MockLLM:
    """Mock LLM output"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
    
    def add_conversation(self, user_input: str, assistant_response: str):
        """Add conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
    
    def generate_response(self, prompt: str, max_turns: int = 3) -> str:
        """Generate mock LLM response"""
        # Generate corresponding tool calls based on prompt content
        if "calculate" in prompt.lower() or "math" in prompt.lower():
            return self._generate_calculator_response(prompt)
        elif ("python" in prompt.lower() or "function" in prompt.lower() or 
              "code" in prompt.lower()):
            return self._generate_python_response(prompt)
        elif "factorial" in prompt.lower():
            return self._generate_factorial_response(prompt)
        elif "fibonacci" in prompt.lower():
            return self._generate_fibonacci_response(prompt)
        elif "students" in prompt.lower() and "class" in prompt.lower():
            return self._generate_students_problem_response(prompt)
        else:
            return self._generate_general_response(prompt)
    
    def _generate_calculator_response(self, prompt: str) -> str:
        """Generate calculator tool call response"""
        # Extract numbers and operators
        numbers = re.findall(r'\d+', prompt)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            if "*" in prompt:
                expression = f"{a} * {b}"
                result = a * b
            elif "+" in prompt:
                expression = f"{a} + {b}"
                result = a + b
            elif "-" in prompt:
                expression = f"{a} - {b}"
                result = a - b
            elif "/" in prompt:
                expression = f"{a} / {b}"
                result = a / b
            else:
                expression = f"{a} + {b}"
                result = a + b
            
            # Execute the code to get actual output
            code = f"print({expression})"
            output = execute_python_code_safely(code)
            
            return f"""Let me calculate that for you.

<code>
```python
{code}
```
</code>

<interpreter>
{output}
</interpreter>

<answer>
\\boxed{{{result}}}
</answer>"""
        else:
            return """I need more specific numbers to calculate. Please provide the numbers you want me to work with."""
    
    def _generate_python_response(self, prompt: str) -> str:
        """Generate Python tool call response"""
        if "factorial" in prompt.lower():
            return self._generate_factorial_response(prompt)
        elif "fibonacci" in prompt.lower():
            return self._generate_fibonacci_response(prompt)
        else:
            return """I can help you write Python code. What specific function or calculation would you like me to implement?"""
    
    def _generate_factorial_response(self, prompt: str) -> str:
        """Generate factorial calculation response"""
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            n = int(numbers[0])
            code = f"""def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial({n}))"""
            
            # Execute the code to get actual output
            output = execute_python_code_safely(code)
            result = self._calculate_factorial(n)
            
            return f"""I'll write a Python function to calculate the factorial of {n}.

<code>
```python
{code}
```
</code>

<interpreter>
{output}
</interpreter>

<answer>
\\boxed{{{result}}}
</answer>"""
        else:
            return "Please provide a number to calculate its factorial."
    
    def _generate_fibonacci_response(self, prompt: str) -> str:
        """Generate Fibonacci sequence response"""
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            n = int(numbers[0])
            code = f"""def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

print(fibonacci({n}))"""
            
            # Execute the code to get actual output
            output = execute_python_code_safely(code)
            fib_sequence = self._calculate_fibonacci(n)
            
            return f"""I'll create a Python function to generate the first {n} Fibonacci numbers.

<code>
```python
{code}
```
</code>

<interpreter>
{output}
</interpreter>

<answer>
\\boxed{{{fib_sequence}}}
</answer>"""
        else:
            return "Please provide a number to generate Fibonacci sequence."
    
    def _generate_students_problem_response(self, prompt: str) -> str:
        """Generate students problem response (set theory problem)"""
        # This is a complex set theory problem requiring multi-step calculation
        code1 = """M = 100
S = 94
E = 57
M_S = 73
M_E = 24
S_E = 27
M_S_E = 22

total_in_any = M + S + E - M_S - M_E - S_E + M_S_E
students_neither = 152 - total_in_any
print(students_neither)"""

        code2 = """only_M_S = 73 - 22
only_M_E = 24 - 22
only_S_E = 27 - 22

only_M = 100 - only_M_S - only_M_E - 22
only_S = 94 - only_M_S - only_S_E - 22
only_E = 57 - only_M_E - only_S_E - 22

total_verify = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22
print(total_verify)"""

        # Execute the code to get actual outputs
        output1 = execute_python_code_safely(code1)
        output2 = execute_python_code_safely(code2)

        return (
            "Okay, so I need to find out how many students at Dala High School "
            "are not taking any of the three classes: Math, Science, or English. "
            "The total number of students is 152. Let me see... I remember this "
            "is a problem about sets and maybe using the principle of "
            "inclusion-exclusion. Let me recall how that works.\n\n"
            "First, the inclusion-exclusion principle for three sets says that "
            "the total number of students taking at least one of the classes is "
            "equal to the sum of the numbers in each individual class, minus "
            "the sum of the numbers in each pair of classes, plus the number in "
            "all three classes. Then, subtract that total from the overall "
            "number of students to get those not taking any of the three.\n\n"
            "Let me write that down step by step. Let M be the set of students "
            "taking Math, S for Science, and E for English. Then:\n\n"
            "Total in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - "
            "|M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|\n\n"
            "Given the numbers:\n\n"
            "|M| = 100\n\n"
            "|S| = 94\n\n"
            "|E| = 57\n\n"
            "|M ∩ S| = 73\n\n"
            "|M ∩ E| = 24\n\n"
            "|S ∩ E| = 27\n\n"
            "|M ∩ S ∩ E| = 22\n\n"
            "So plugging these into the formula:\n\n"
            "Total = 100 + 94 + 57 - 73 - 24 - 27 + 22\n\n"
            "Let me compute that step by step using code to ensure accuracy.\n\n"
            "<code>\n"
            "```python\n\n"
            "M = 100\n"
            "S = 94\n"
            "E = 57\n"
            "M_S = 73\n"
            "M_E = 24\n"
            "S_E = 27\n"
            "M_S_E = 22\n\n"
            "total_in_any = M + S + E - M_S - M_E - S_E + M_S_E\n"
            "students_neither = 152 - total_in_any\n"
            "print(students_neither)\n"
            "```\n"
            "</code>\n\n"
            "<interpreter>\n"
            f"{output1}\n"
            "</interpreter>\n\n"
            "The code returns 3, indicating that 3 students take none of the "
            "classes. To verify, I'll break down each category to ensure the "
            "total matches:\n\n"
            "- Only Math and Science: 73 - 22 = 51\n"
            "- Only Math and English: 24 - 22 = 2\n"
            "- Only Science and English: 27 - 22 = 5\n"
            "- Only Math: 100 - 51 - 2 - 22 = 25\n"
            "- Only Science: 94 - 51 - 5 - 22 = 16\n"
            "- Only English: 57 - 2 - 5 - 22 = 28\n\n"
            "Summing all categories:\n\n"
            "<code>\n"
            "```python\n\n"
            "only_M_S = 73 - 22\n"
            "only_M_E = 24 - 22\n"
            "only_S_E = 27 - 22\n\n"
            "only_M = 100 - only_M_S - only_M_E - 22\n"
            "only_S = 94 - only_M_S - only_S_E - 22\n"
            "only_E = 57 - only_M_E - only_S_E - 22\n\n"
            "total_verify = only_M + only_S + only_E + only_M_S + "
            "only_M_E + only_S_E + 22\n"
            "print(total_verify)\n"
            "```\n"
            "</code>\n\n"
            "<interpreter>\n"
            f"{output2}\n"
            "</interpreter>\n\n"
            "The verification total is 149, so students not taking any classes "
            "are 152 - 149 = 3, confirming the initial result.\n\n"
            "<answer>\n"
            "\\boxed{3}\n"
            "</answer>"
        )
    
    def _generate_general_response(self, prompt: str) -> str:
        """Generate general response"""
        return """I can help you with mathematical calculations and Python programming. What would you like me to do?"""
    
    def _calculate_factorial(self, n: int) -> int:
        """Calculate factorial"""
        if n <= 1:
            return 1
        return n * self._calculate_factorial(n - 1)
    
    def _calculate_fibonacci(self, n: int) -> List[int]:
        """Calculate Fibonacci sequence"""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        else:
            fib = [0, 1]
            for i in range(2, n):
                fib.append(fib[i-1] + fib[i-2])
            return fib


class RetoolLocalTester:
    """Retool local tester"""
    
    def __init__(self):
        self.mock_llm = MockLLM()
        self.tool_registry = ToolRegistry()
    
    async def test_single_turn(self, user_input: str) -> Dict[str, Any]:
        """Test single turn conversation"""
        print(f"\n{'='*60}")
        print(f"User Input: {user_input}")
        print(f"{'='*60}")
        
        # Simulate LLM response generation
        llm_response = self.mock_llm.generate_response(user_input)
        print(f"LLM Response:\n{llm_response}")
        
        # Test postprocess_predictions
        print(f"\n{'='*60}")
        print("Testing postprocess_predictions")
        print(f"{'='*60}")
        
        action, content = postprocess_predictions(llm_response)
        print(f"Action: {action}")
        print(f"Content: {repr(content)}")
        
        # Post-process response
        processed_response = postprocess_responses(llm_response)
        print(f"\nPost-processed Response:\n{processed_response}")
        
        # Extract code blocks
        code_blocks = extract_code_blocks(processed_response)
        print(f"\nExtracted Code Blocks: {code_blocks}")
        
        # Extract final answer
        final_answer = extract_boxed_answer(processed_response)
        print(f"\nFinal Answer: {final_answer}")
        
        # Calculate tool call count
        tool_call_count = len(code_blocks)
        print(f"\nTool Call Count: {tool_call_count}")
        
        # Execute code blocks (simulation)
        code_results = []
        for code_block in code_blocks:
            code = code_block["code"]
            expected_output = code_block["output"]
            actual_output = execute_python_code_safely(code)
            print(f"\nExecuting Code:\n{code}")
            print(f"Expected Output: {expected_output}")
            print(f"Actual Output: {actual_output}")
            print(f"Output Match: {expected_output.strip() == actual_output.strip()}")
            code_results.append({
                "code": code,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "match": expected_output.strip() == actual_output.strip()
            })
        
        # Simulate reward calculation
        print(f"\n{'='*60}")
        print("Testing Reward Calculation")
        print(f"{'='*60}")
        
        try:
            from slime.utils.types import Sample
            from generate_with_retool import reward_func
            
            # Create mock sample
            mock_sample = Sample(
                prompt=user_input,
                response=llm_response,
                label=final_answer if final_answer else "unknown"
            )
            mock_sample.tool_call_count = tool_call_count
            
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
            "user_input": user_input,
            "llm_response": llm_response,
            "action": action,
            "content": content,
            "processed_response": processed_response,
            "code_blocks": code_blocks,
            "code_results": code_results,
            "final_answer": final_answer,
            "tool_call_count": tool_call_count,
            "reward_result": reward_result
        }
    
    async def test_multi_turn(self, conversation: List[str]) -> List[Dict[str, Any]]:
        """Test multi-turn conversation"""
        results = []
        full_context = ""
        
        for i, user_input in enumerate(conversation):
            print(f"\n{'#'*80}")
            print(f"Turn {i+1}")
            print(f"{'#'*80}")
            
            # Build complete context
            current_input = (full_context + user_input if full_context 
                           else user_input)
            
            # Test current turn
            result = await self.test_single_turn(current_input)
            results.append(result)
            
            # Update context
            full_context = current_input + "\n" + result["llm_response"]
        
        return results
    
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
            import asyncio
            
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
        
        return {
            "action": action,
            "content": content,
            "processed_response": processed_response,
            "code_blocks": code_blocks,
            "final_answer": final_answer,
            "tool_call_count": tool_call_count,
            "reward_result": reward_result if 'reward_result' in locals() else None
        }


async def main():
    """Main function"""
    tester = RetoolLocalTester()
    
    # Test new format
    print("Starting New Format Test...")
    await tester.test_new_format()
    
    # Test cases
    test_cases = [
        "Calculate 15 * 23 + 7",
        "Write a function to calculate the factorial of 5",
        "Generate the first 10 Fibonacci numbers",
        "What is 100 / 4?",
        "Create a Python function to find the sum of numbers from 1 to 10"
    ]
    
    # Single turn tests
    print("\n\nStarting Single Turn Tests...")
    for test_case in test_cases:
        await tester.test_single_turn(test_case)
    
    # Multi-turn tests
    print("\n\nStarting Multi-turn Tests...")
    multi_turn_conversation = [
        "Calculate 10 + 5",
        "Now write a function to calculate the factorial of that result",
        "What is the result of the factorial?"
    ]
    await tester.test_multi_turn(multi_turn_conversation)


if __name__ == "__main__":
    asyncio.run(main()) 