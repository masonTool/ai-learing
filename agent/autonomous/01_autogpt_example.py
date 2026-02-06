"""
AutoGPT 自主任务执行示例
=======================

AutoGPT 是最早爆红的自主型 Agent 框架，其核心理念是:
"给定一个目标，让 AI 自主思考、规划并执行，直到完成"

核心概念:
1. Goal (目标): 用户给出的最终目标
2. Thoughts (思考): AI 对当前情况的分析
3. Reasoning (推理): AI 的逻辑推理过程
4. Plan (计划): 完成目标的步骤规划
5. Criticism (反思): 对自身计划的批判性思考
6. Action (行动): 具体执行的操作

工作循环:
1. 理解目标
2. 思考当前状态
3. 制定计划
4. 执行行动（调用工具或生成代码）
5. 观察结果
6. 重复 2-5 直到完成目标

特点:
- 完全自主: 一旦启动，可以自主运行多个步骤
- 自我反思: 会评估自己的表现并调整
- 长期记忆: 可以保存和检索之前的经验
- 代码执行: 可以编写和执行代码来解决问题

适用场景: 复杂研究任务、数据分析、自动化办公、代码生成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 第一部分: 数据结构定义
# ============================================================

@dataclass
class AutoGPTThought:
    """
    AutoGPT 的思考结构
    
    每个思考周期包含:
    - thoughts: 对当前情况的分析
    - reasoning: 推理过程
    - plan: 执行计划
    - criticism: 自我批评和反思
    - speak: 对用户的回应
    """
    thoughts: str
    reasoning: str
    plan: str
    criticism: str
    speak: str


@dataclass
class AutoGPTAction:
    """
    AutoGPT 的行动结构
    
    - name: 行动名称（如搜索、写文件、执行代码等）
    - args: 行动参数
    """
    name: str
    args: Dict[str, Any]


# ============================================================
# 第二部分: 记忆系统
# ============================================================

class AutoGPTMemory:
    """
    AutoGPT 的记忆系统
    
    负责存储和检索交互历史
    """
    
    def __init__(self):
        self.short_term: List[Dict] = []
        self.max_short_term = 10
    
    def add_interaction(self, role: str, content: str):
        """添加交互记录"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)
    
    def get_context(self) -> str:
        """获取记忆上下文"""
        context = []
        for item in self.short_term[-5:]:
            context.append(f"{item['role']}: {item['content'][:150]}")
        return "\n".join(context)


# ============================================================
# 第三部分: 工具系统
# ============================================================

class AutoGPTTools:
    """AutoGPT 工具集"""
    
    def __init__(self, memory: AutoGPTMemory):
        self.memory = memory
    
    def web_search(self, query: str) -> str:
        """模拟网络搜索"""
        print(f"  [工具] 搜索: {query}")
        return f"搜索 '{query}' 的结果: 找到相关信息..."
    
    def write_file(self, file_path: str, content: str) -> str:
        """写入文件"""
        print(f"  [工具] 写入文件: {file_path}")
        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            return f"文件已写入: {file_path}"
        except Exception as e:
            return f"写入失败: {str(e)}"
    
    def read_file(self, file_path: str) -> str:
        """读取文件"""
        print(f"  [工具] 读取文件: {file_path}")
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"读取失败: {str(e)}"
    
    def execute_python(self, code: str) -> str:
        """执行 Python 代码"""
        print(f"  [工具] 执行代码")
        try:
            # 注意: 实际使用需要安全的执行环境
            result = eval(code)
            return f"执行结果: {result}"
        except Exception as e:
            return f"执行错误: {str(e)}"
    
    def task_complete(self, result: str) -> str:
        """标记任务完成"""
        return f"任务完成: {result}"


# ============================================================
# 第四部分: 核心 AutoGPT Agent
# ============================================================

class AutoGPT:
    """
    AutoGPT 核心实现
    
    实现自主任务执行的循环逻辑
    """
    
    def __init__(self, ai_name: str = "AutoGPT", llm=None):
        self.ai_name = ai_name
        self.memory = AutoGPTMemory()
        self.tools = AutoGPTTools(self.memory)
        self.max_iterations = 5
        
        # 初始化 LLM
        self.llm = llm or ChatOpenAI(
            model=config.MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.7
        )
        
        # 工具描述
        self.tools_desc = {
            "web_search": "搜索互联网获取信息",
            "write_file": "将内容写入文件",
            "read_file": "读取文件内容",
            "execute_python": "执行 Python 代码",
            "task_complete": "标记任务完成"
        }
    
    def _construct_prompt(self, goal: str) -> str:
        """
        构建系统提示
        
        这是 AutoGPT 的核心提示，指导 AI 的行为
        """
        
        tools_list = "\n".join(f"- {name}: {desc}" for name, desc in self.tools_desc.items())

        prompt = (
            f"你是 {self.ai_name}，一个自主 AI 助手。\n\n"
            f"目标: {goal}\n\n"
            "可用工具:\n"
            f"{tools_list}\n\n"
            "指令:\n"
            "1. 分析当前情况，思考如何完成目标\n"
            "2. 制定一个清晰的计划\n"
            "3. 选择合适的工具执行下一步\n"
            "4. 自我反思，确保计划合理\n"
            "5. 使用以下 JSON 格式响应:\n\n"
            "{\n"
            '    "thoughts": "当前情况的分析",\n'
            '    "reasoning": "为什么这样做",\n'
            '    "plan": "接下来的步骤",\n'
            '    "criticism": "自我反思",\n'
            '    "speak": "对用户的简要说明",\n'
            '    "action": {\n'
            '        "name": "工具名称",\n'
            '        "args": {"参数名": "参数值"}\n'
            "    }\n"
            "}\n\n"
            "重要:\n"
            "- 始终保持目标导向\n"
            "- 如果任务完成，使用 task_complete 工具\n"
            "- 不要编造信息，不确定时使用搜索工具\n"
            "- 反思你的计划是否合理高效\n"
        )
        return prompt
    
    def _execute_action(self, action_name: str, action_args: Dict) -> str:
        """
        执行工具调用
        
        参数:
            action_name: 工具名称
            action_args: 工具参数
            
        返回:
            工具执行结果
        """
        tool = getattr(self.tools, action_name, None)
        if tool:
            try:
                return tool(**action_args)
            except Exception as e:
                return f"执行出错: {str(e)}"
        return f"未知工具: {action_name}"
    
    def _get_llm_response(self, goal: str, context: str, iteration: int) -> Dict:
        """
        调用真实 LLM 获取响应
        
        使用 LangChain ChatOpenAI 调用配置的模型
        """
        
        tools_list = "\n".join(f"- {name}: {desc}" for name, desc in self.tools_desc.items())
        
        prompt = f"""你是 {self.ai_name}，一个自主 AI 助手。当前是第 {iteration + 1} 轮迭代。

目标: {goal}

历史记忆:
{context}

可用工具:
{tools_list}

指令:
1. 分析当前情况，思考如何完成目标
2. 制定一个清晰的计划
3. 选择合适的工具执行下一步
4. 自我反思，确保计划合理
5. 使用以下 JSON 格式响应:

{{
    "thoughts": "当前情况的分析",
    "reasoning": "为什么这样做",
    "plan": "接下来的步骤",
    "criticism": "自我反思",
    "speak": "对用户的简要说明",
    "action": {{
        "name": "工具名称(web_search/write_file/read_file/execute_python/task_complete)",
        "args": {{"参数名": "参数值"}}
    }}
}}

重要:
- 始终保持目标导向
- 如果任务完成，使用 task_complete 工具
- 不确定时使用 web_search 工具搜索信息
- 反思你的计划是否合理高效
- 如果已经搜索过相关信息，应该执行代码或完成任务
- 最多 3-5 轮迭代，合理安排进度

请直接输出 JSON，不要添加 markdown 代码块标记。"""

        try:
            messages = [
                SystemMessage(content="你是一个自主 AI Agent，负责分析情况并制定行动计划。只输出 JSON 格式。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # 清理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"  [LLM 调用出错: {e}]")
            # 回退到完成任务
            return {
                "thoughts": f"调用 LLM 出错: {str(e)}",
                "reasoning": "出错，终止任务",
                "plan": "结束",
                "criticism": "应该更好地处理错误",
                "speak": "遇到问题，需要人工介入",
                "action": {
                    "name": "task_complete",
                    "args": {"result": f"任务中断，原因: {str(e)}"}
                }
            }
    
    def run(self, goal: str):
        """
        运行 AutoGPT
        
        执行自主任务循环
        
        参数:
            goal: 要完成的任务目标
        """
        print("=" * 70)
        print(f"AutoGPT: {self.ai_name}")
        print("=" * 70)
        print(f"\n目标: {goal}\n")
        print("-" * 70)
        
        iteration = 0
        
        while iteration < self.max_iterations:
            print(f"\n【迭代 {iteration + 1}/{self.max_iterations}】")
            
            # 1. 获取上下文
            context = self.memory.get_context()
            
            # 2. 调用 LLM 思考和决策
            response = self._get_llm_response(goal, context, iteration)
            
            # 3. 显示思考过程
            print(f"\n思考: {response['thoughts']}")
            print(f"推理: {response['reasoning']}")
            print(f"计划: {response['plan']}")
            print(f"反思: {response['criticism']}")
            print(f"说明: {response['speak']}")
            
            # 4. 执行行动
            action = response.get('action', {})
            action_name = action.get('name')
            action_args = action.get('args', {})
            
            print(f"\n行动: {action_name}")
            result = self._execute_action(action_name, action_args)
            print(f"结果: {result}")
            
            # 5. 更新记忆
            self.memory.add_interaction("assistant", json.dumps(response, ensure_ascii=False))
            self.memory.add_interaction("system", result)
            
            # 6. 检查是否完成
            if action_name == "task_complete":
                print("\n" + "=" * 70)
                print("任务完成!")
                print("=" * 70)
                break
            
            iteration += 1
        
        if iteration >= self.max_iterations:
            print("\n达到最大迭代次数，任务终止")


# ============================================================
# 第五部分: 使用示例
# ============================================================

def main():
    """
    主函数：演示 AutoGPT 的使用
    """
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=config.MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0.7
    )
    
    # 创建 AutoGPT 实例
    autogpt = AutoGPT(ai_name="智能助手", llm=llm)
    
    # 示例 1: 简单任务
    print("\n" + "=" * 70)
    print("示例 1: 信息收集任务")
    print("=" * 70)
    autogpt.run("研究 Python 在数据科学中的应用")
    
    # 示例 2: 文件操作任务
    print("\n\n" + "=" * 70)
    print("示例 2: 文件处理任务")
    print("=" * 70)
    
    autogpt2 = AutoGPT(ai_name="文件助手", llm=llm)
    autogpt2.run("创建一个包含 Python 学习资源的文档")
    
    print("\n" + "=" * 70)
    print("AutoGPT 示例结束")
    print("=" * 70)
    
    print("""
AutoGPT 工作流程说明:

1. 目标理解
   - 解析用户给出的目标
   - 理解任务的上下文和要求

2. 思考循环 (重复执行)
   ┌─────────────────────────────┐
   │  Thoughts: 分析当前状态     │
   │  Reasoning: 逻辑推理        │
   │  Plan: 制定行动计划         │
   │  Criticism: 自我反思        │
   └──────────────┬──────────────┘
                  │
   ┌──────────────▼──────────────┐
   │  Action: 执行具体操作       │
   │  - 搜索信息                 │
   │  - 读写文件                 │
   │  - 执行代码                 │
   └──────────────┬──────────────┘
                  │
   ┌──────────────▼──────────────┐
   │  Observation: 观察结果      │
   │  更新记忆，准备下一轮        │
   └─────────────────────────────┘

3. 任务完成
   - 判断目标是否达成
   - 整理并输出最终结果

关键特点:
- 自主决策：不需要人类干预每一步
- 自我修正：通过 criticism 不断优化
- 长期运行：可以执行多步骤复杂任务
""")


if __name__ == "__main__":
    main()
