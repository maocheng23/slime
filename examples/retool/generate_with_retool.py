# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import json
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# 导入 deepscaler 奖励模型
try:
    from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
except ImportError:
    # 如果无法导入，使用简单的评分逻辑
    def get_deepscaler_rule_based_reward(response, label):
        return 0.0

RETOOL_CONFIGS = {
    "max_turns": 5,
    "max_tool_calls": 3,
    "tool_concurrency": 64,
    # reward model
    "format_score": 0.1,
    "execution_score": 0.9,
    # tool call reward settings
    "tool_call_bonus": 0.1,  # 每次工具调用的奖励
    "max_tool_bonus": 0.3,   # 最大工具调用奖励
    # Python interpreter settings
    "python_timeout": 10,  # seconds
    "python_memory_limit": "100MB",
    "python_cpu_limit": 1,
}

SEMAPHORE = asyncio.Semaphore(RETOOL_CONFIGS["tool_concurrency"])


class PythonSandbox:
    """Python 代码沙盒，提供安全的代码执行环境"""
    
    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            'math', 'random', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 'statistics', 'decimal', 'fractions'
        }
    
    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """检查代码安全性"""
        # 检查危险操作
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
            r'__\w+__',  # 双下划线方法
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"
        
        # 检查导入的模块
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
        """创建安全的执行环境"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        
        try:
            # 创建安全的 Python 脚本
            script_path = os.path.join(temp_dir, "code.py")
            
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = temp_dir
            env['PYTHONUNBUFFERED'] = '1'
            
            yield script_path, env, temp_dir
            
        finally:
            # 清理临时目录
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    async def execute_code(self, code: str) -> str:
        """在沙盒中执行 Python 代码"""
        # 检查代码安全性
        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"
        
        # 添加必要的包装代码
        wrapped_code = f"""
import sys
import traceback
from io import StringIO

# 重定向 stdout 和 stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # 用户代码
    {code}
    
    # 获取输出
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # 恢复标准输出
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # 返回结果
    result = ""
    if stdout_output:
        result += f"Output:\\n{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result)
    
except Exception as e:
    # 恢复标准输出
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # 返回错误信息
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)
"""
        
        with self._create_safe_environment() as (script_path, env, temp_dir):
            # 写入代码到文件
            with open(script_path, 'w') as f:
                f.write(wrapped_code)
            
            try:
                # 使用 subprocess 运行代码
                process = subprocess.Popen(
                    ['python3', script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True
                )
                
                # 设置超时
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                    
                    if process.returncode == 0:
                        return stdout.strip()
                    else:
                        return f"Error: Process exited with code {process.returncode}\n{stderr}"
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    return f"Error: Code execution timed out after {self.timeout} seconds"
                    
            except Exception as e:
                return f"Error: Failed to execute code: {str(e)}"


class ToolRegistry:
    """工具注册表，管理可用的工具"""
    
    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=RETOOL_CONFIGS["python_timeout"],
            memory_limit=RETOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        # Python 代码解释器
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
        
        # 数学计算工具
        self.register_tool("calculator", {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        })
    
    def register_tool(self, name: str, tool_spec: Dict[str, Any]):
        """注册新工具"""
        self.tools[name] = tool_spec
    
    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """获取所有工具规格"""
        return list(self.tools.values())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        async with SEMAPHORE:
            if tool_name == "python":
                return await self._execute_python(arguments)
            elif tool_name == "calculator":
                return await self._execute_calculator(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"
    
    async def _execute_python(self, arguments: Dict[str, Any]) -> str:
        """执行 Python 代码"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"
        
        # 在沙盒中执行代码
        result = await self.python_sandbox.execute_code(code)
        return result
    
    async def _execute_calculator(self, arguments: Dict[str, Any]) -> str:
        """执行计算器工具"""
        try:
            expression = arguments.get("expression", "")
            # 安全地评估数学表达式
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


# 全局工具注册表
tool_registry = ToolRegistry()


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从文本中提取工具调用"""
    tool_calls = []
    
    # 匹配工具调用模式：<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            if "name" in tool_call and "arguments" in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def extract_final_answer(text: str) -> Optional[str]:
    """从文本中提取最终答案"""
    # 匹配最终答案模式：<answer>...</answer>
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def postprocess_response(resp: str) -> str:
    """后处理响应，确保标签完整性"""
    # 处理工具调用标签
    if "<tool_call>" in resp and "</tool_call>" not in resp:
        resp = resp.split("<tool_call>")[0]
    
    # 处理答案标签
    if "<answer>" in resp and "</answer>" not in resp:
        resp = resp.split("<answer>")[0]
    
    return resp


async def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """执行工具调用并返回结果"""
    if not tool_calls:
        return ""
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        
        result = await tool_registry.execute_tool(tool_name, arguments)
        results.append(f"Tool '{tool_name}' result: {result}")
    
    return "\n".join(results)


def compute_tool_call_bonus(full_text: str) -> float:
    """计算工具调用奖励"""
    tool_calls = extract_tool_calls(full_text)
    final_answer = extract_final_answer(full_text)
    
    # 基础奖励：格式正确性
    format_bonus = 0.0
    if tool_calls and final_answer:
        format_bonus += 0.1  # 有工具调用且有最终答案
    
    # 工具调用次数奖励
    tool_count_bonus = min(
        len(tool_calls) * RETOOL_CONFIGS["tool_call_bonus"],
        RETOOL_CONFIGS["max_tool_bonus"]
    )
    
    return format_bonus + tool_count_bonus


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """自定义生成函数，支持工具调用"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt = sample.prompt
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    
    for turn in range(RETOOL_CONFIGS["max_turns"]):
        # 构建当前输入
        current_input = prompt + response
        
        # 添加工具规格到prompt（如果需要）
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

        # 处理中止
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_response(cur_response)

        # 记录当前响应的token
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # 检查长度限制
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # 提取工具调用
        tool_calls = extract_tool_calls(cur_response)
        final_answer = extract_final_answer(cur_response)
        
        # 如果有最终答案，结束对话
        if final_answer is not None:
            break
        
        # 如果有工具调用，执行它们
        if tool_calls:
            tool_results = await execute_tool_calls(tool_calls)
            if tool_results:
                next_obs = f"\n\n<tool_results>\n{tool_results}\n</tool_results>\n\n"
                obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                response += next_obs
                response_token_ids += obs_tokens_ids
                loss_masks += [0] * len(obs_tokens_ids)
        
        # 检查是否达到最大工具调用次数
        if turn >= RETOOL_CONFIGS["max_tool_calls"]:
            break

    # 设置样本属性
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks
    
    # 设置状态
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """工具调用的奖励函数，使用 deepscaler 作为主要奖励模型"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # 构建完整的解决方案字符串
    solution_str = sample.prompt + sample.response
    
    # 获取真实答案
    ground_truth = sample.label.get("ground_truth", "")
    
    # 使用 deepscaler 计算基础分数
    base_score = get_deepscaler_rule_based_reward(solution_str, ground_truth)
    
    # 计算工具调用奖励
    tool_bonus = compute_tool_call_bonus(solution_str)
    
    # 组合分数：基础分数 + 工具调用奖励
    total_score = base_score + tool_bonus
    
    return total_score 