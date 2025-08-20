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
    
    # Match interpreter output pattern: <interpreter>output</interpreter>
    interpreter_pattern = r'<interpreter>(.*?)</interpreter>'
    interpreter_matches = re.findall(interpreter_pattern, text, re.DOTALL)
    
    # Pair code blocks with interpreter outputs
    for i, code in enumerate(code_matches):
        code_block = {
            "code": code.strip(),
            "output": (interpreter_matches[i] if i < len(interpreter_matches) 
                      else "")
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
        # Create a safe execution environment
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)
        
        # Capture print outputs (simplified)
        output_lines = []
        for line in code.split('\n'):
            if line.strip().startswith('print('):
                # Extract the expression inside print()
                match = re.search(r'print\((.*)\)', line)
                if match:
                    expr = match.group(1)
                    try:
                        result = eval(expr, {"__builtins__": {}}, local_vars)
                        output_lines.append(str(result))
                    except:
                        output_lines.append("Error evaluating expression")
        
        return '\n'.join(output_lines) if output_lines else "No output"
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
        
        # Post-process response
        processed_response = postprocess_responses(llm_response)
        print(f"\nPost-processed Response:\n{processed_response}")
        
        # Extract code blocks
        code_blocks = extract_code_blocks(processed_response)
        print(f"\nExtracted Code Blocks: {code_blocks}")
        
        # Extract final answer
        final_answer = extract_boxed_answer(processed_response)
        print(f"\nFinal Answer: {final_answer}")
        
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
        
        return {
            "user_input": user_input,
            "llm_response": llm_response,
            "processed_response": processed_response,
            "code_blocks": code_blocks,
            "code_results": code_results,
            "final_answer": final_answer
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
        test_response = """Okay, so I need to find out how many students at Dala High School are not taking any of the three classes: Math, Science, or English. The total number of students is 152. Let me see... I remember this is a problem about sets and maybe using the principle of inclusion-exclusion. Let me recall how that works.

First, the inclusion-exclusion principle for three sets says that the total number of students taking at least one of the classes is equal to the sum of the numbers in each individual class, minus the sum of the numbers in each pair of classes, plus the number in all three classes. Then, subtract that total from the overall number of students to get those not taking any of the three. 

Let me write that down step by step. Let M be the set of students taking Math, S for Science, and E for English. Then:

Total in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - |M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|

Given the numbers:

|M| = 100

|S| = 94

|E| = 57

|M ∩ S| = 73

|M ∩ E| = 24

|S ∩ E| = 27

|M ∩ S ∩ E| = 22

So plugging these into the formula:

Total = 100 + 94 + 57 - 73 - 24 - 27 + 22

Let me compute that step by step using code to ensure accuracy.

<code>
```python

M = 100
S = 94
E = 57
M_S = 73
M_E = 24
S_E = 27
M_S_E = 22

total_in_any = M + S + E - M_S - M_E - S_E + M_S_E
students_neither = 152 - total_in_any
print(students_neither)
```
</code>

<interpreter>
3
</interpreter>

The code returns 3, indicating that 3 students take none of the classes. To verify, I'll break down each category to ensure the total matches:

- Only Math and Science: 73 - 22 = 51
- Only Math and English: 24 - 22 = 2
- Only Science and English: 27 - 22 = 5
- Only Math: 100 - 51 - 2 - 22 = 25
- Only Science: 94 - 51 - 5 - 22 = 16
- Only English: 57 - 2 - 5 - 22 = 28

Summing all categories:

<code>
```python

only_M_S = 73 - 22
only_M_E = 24 - 22
only_S_E = 27 - 22

only_M = 100 - only_M_S - only_M_E - 22
only_S = 94 - only_M_S - only_S_E - 22
only_E = 57 - only_M_E - only_S_E - 22

total_verify = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22
print(total_verify)
```
</code>

<interpreter>
149
</interpreter>

The verification total is 149, so students not taking any classes are 152 - 149 = 3, confirming the initial result.

<answer>
\\boxed{3}
</answer>"""
        
        print("Original Response:")
        print(test_response)
        
        # Extract code blocks
        code_blocks = extract_code_blocks(test_response)
        print(f"\nNumber of Extracted Code Blocks: {len(code_blocks)}")
        for i, block in enumerate(code_blocks):
            print(f"\nCode Block {i+1}:")
            print(f"Code:\n{block['code']}")
            print(f"Output:\n{block['output']}")
        
        # Extract final answer
        final_answer = extract_boxed_answer(test_response)
        print(f"\nFinal Answer: {final_answer}")
        
        return {
            "code_blocks": code_blocks,
            "final_answer": final_answer
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