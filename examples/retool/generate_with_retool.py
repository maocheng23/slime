# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import re
import subprocess
import tempfile
import os
import gc
import psutil
from typing import Dict, List, Any
from contextlib import contextmanager

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError:
    raise ImportError("DeepScaler or MathDapo is not installed")
    # If import fails, use simple scoring logic
    # def get_deepscaler_rule_based_reward(response, label):
    #     return 0.0
    
    # def math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=False):
    #     return {"score": 0.0, "acc": False, "pred": ""}

RETOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 3,
    "tool_concurrency": 64,
    # Python interpreter settings
    "python_timeout": 10,  # seconds
    "python_memory_limit": "100MB",
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 512,  # MB
    "cleanup_threshold": 256,  # MB - trigger cleanup when memory exceeds this
}

SEMAPHORE = asyncio.Semaphore(RETOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    if hasattr(gc, 'collect'):
        gc.collect()


class PythonSandbox:
    """Python code sandbox, provides safe code execution environment"""
    
    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            'math', 'random', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 'statistics', 'decimal', 'fractions'
        }
    
    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety"""
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'import\s+shutil',
            r'import\s+glob',
            r'import\s+pathlib',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'compile\s*\(',
            r'execfile\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'type\s*\(',
            r'isinstance\s*\(',
            r'issubclass\s*\(',
            r'super\s*\(',
            r'property\s*\(',
            r'staticmethod\s*\(',
            r'classmethod\s*\(',
            r'__\w+__',  # double underscore methods
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"
        
        # Check imported modules
        import_pattern = r'import\s+(\w+)'
        from_pattern = r'from\s+(\w+)'
        
        imports = re.findall(import_pattern, code)
        froms = re.findall(from_pattern, code)
        
        all_imports = set(imports + froms)
        for imp in all_imports:
            if imp not in self.allowed_modules:
                return False, f"Import of '{imp}' is not allowed"
        
        return True, "Code is safe"
    
    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        
        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = temp_dir
            env['PYTHONUNBUFFERED'] = '1'
            
            yield script_path, env, temp_dir
            
        finally:
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    async def execute_code(self, code: str) -> str:
        """Execute Python code in sandbox"""
        # Check memory usage before execution
        current_memory = get_memory_usage()
        if current_memory > RETOOL_CONFIGS["max_memory_usage"]:
            cleanup_memory()
            return "Error: Memory usage too high, please try again"
        
        # Check code safety
        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"
        
        # Add necessary wrapper code with memory limits
        wrapped_code = f"""
        import sys
        import traceback
        from io import StringIO
        import resource

        # Set memory limit (100MB)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, -1))
        except:
            pass

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # User code
            {code}
            
            # Get output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Restore standard output
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Return result
            result = ""
            if stdout_output:
                result += f"Output:\\n{{stdout_output}}"
            if stderr_output:
                result += f"\\nErrors:\\n{{stderr_output}}"
            
            print(result)
            
        except Exception as e:
            # Restore standard output
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Return error information
            error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
            print(error_msg)
        """
        
        with self._create_safe_environment() as (script_path, env, temp_dir):
            # Write code to file
            with open(script_path, 'w') as f:
                f.write(wrapped_code)
            
            try:
                # Use subprocess to run code
                process = subprocess.Popen(
                    ['python3', script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True
                )
                
                # Set timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                    
                    if process.returncode == 0:
                        result = stdout.strip()
                    else:
                        result = (f"Error: Process exited with code "
                                f"{process.returncode}\n{stderr}")
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    result = (f"Error: Code execution timed out after "
                             f"{self.timeout} seconds")
                    
            except Exception as e:
                result = f"Error: Failed to execute code: {str(e)}"
            
            # Check memory usage after execution and cleanup if needed
            current_memory = get_memory_usage()
            if current_memory > RETOOL_CONFIGS["cleanup_threshold"]:
                cleanup_memory()
            
            return result


class ToolRegistry:
    """Tool registry, manages available tools"""
    
    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=RETOOL_CONFIGS["python_timeout"],
            memory_limit=RETOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        # Python code interpreter
        self.register_tool("python", {
            "name": "python",
            "description": "Execute Python code in a safe sandbox environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        })
    
    def register_tool(self, name: str, tool_spec: Dict[str, Any]):
        """Register new tool"""
        self.tools[name] = tool_spec
    
    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all tool specifications"""
        return list(self.tools.values())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool call"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        async with SEMAPHORE:
            if tool_name == "python":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"
    
    async def _execute_python(self, arguments: Dict[str, Any]) -> str:
        """Execute Python code"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"
        
        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result
# Global tool registry
tool_registry = ToolRegistry()


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


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("python", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. \
If I want to execute code, I should put the code between <code> and </code>. \
If I want to give the final answer, I should use the format 'Answer: \\boxed{answer}' or put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done





async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, ("Partial rollout is not supported for "
                                     "this function at the moment.")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt = sample.prompt
    prompt_tokens_ids = (state.tokenizer(sample.prompt, add_special_tokens=False)
                        ["input_ids"])
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    
    for turn in range(RETOOL_CONFIGS["max_turns"]):
        # Build current input
        current_input = prompt + response
        
        # Add tool specifications to prompt (if needed)
        if turn == 0:
            tool_specs = tool_registry.get_tool_specs()
            if tool_specs:
                tools_info = "Available tools:\n"
                for tool in tool_specs:
                    tools_info += f"- {tool['name']}: {tool['description']}\n"
                current_input = tools_info + "\n" + current_input
        
        payload = {
            "text": current_input,
            "sampling_params": sampling_params,
        }
        
        output = await post(url, payload, use_http2=args.use_http2)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        # Record current response tokens
        cur_response_token_ids = (state.tokenizer(cur_response, 
                                                 add_special_tokens=False)
                                 ["input_ids"])
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = (state.tokenizer(next_obs, 
                                         add_special_tokens=False)
                         ["input_ids"])
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)
        
        # Check if maximum tool call count reached
        if turn >= RETOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks
    
    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count
    
    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response
    
    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""
    
    # Get tool call count as num_turns
    num_turns = getattr(sample, 'tool_call_count', 0)
    
    # Use math_dapo compute_score function
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    
    # Encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    
    # Ensure pred is not None
    if result["pred"] is None:
        result["pred"] = ""
    
    return result 