"""
BabyAGI 任务驱动型 Agent 示例
============================

BabyAGI 是一个任务驱动的自主 Agent，核心理念是:
"通过创建、优先排序和执行任务列表来完成目标"

核心概念:
1. Objective (目标): 要完成的最终目标
2. Task List (任务列表): 需要执行的任务队列
3. Task Creation (任务创建): 根据结果创建新任务
4. Task Prioritization (任务排序): 重新排序任务优先级
5. Task Execution (任务执行): 执行当前最高优先级任务

工作循环:
1. 从任务列表中取出第一个任务
2. 执行任务
3. 根据执行结果创建新任务
4. 重新排序任务列表
5. 重复直到目标达成

特点:
- 任务驱动: 一切操作都围绕任务列表展开
- 动态规划: 根据执行结果动态调整计划
- 优先级管理: 始终执行最重要的任务
- 简洁高效: 核心逻辑简单，易于理解和扩展

适用场景: 研究任务、信息收集、项目管理、待办事项自动化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 第一部分: 数据结构定义
# ============================================================

@dataclass
class Task:
    """
    任务数据结构
    
    每个任务包含:
    - task_id: 唯一标识
    - task_name: 任务名称/描述
    - priority: 优先级（数字越小优先级越高）
    - status: 状态（pending/completed/failed）
    - result: 执行结果
    - created_at: 创建时间
    - completed_at: 完成时间
    """
    task_id: int
    task_name: str
    priority: int = 0
    status: str = "pending"
    result: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


# ============================================================
# 第二部分: 任务管理器
# ============================================================

class TaskManager:
    """
    任务管理器
    
    负责任务的增删改查和排序
    """
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.task_id_counter = 1
        self.completed_tasks: List[Task] = []
    
    def add_task(self, task_name: str, priority: int = 0) -> Task:
        """
        添加新任务
        
        参数:
            task_name: 任务描述
            priority: 优先级，数字越小越优先
            
        返回:
            创建的任务对象
        """
        task = Task(
            task_id=self.task_id_counter,
            task_name=task_name,
            priority=priority
        )
        self.tasks.append(task)
        self.task_id_counter += 1
        self._sort_tasks()
        return task
    
    def get_next_task(self) -> Optional[Task]:
        """
        获取下一个要执行的任务
        
        返回优先级最高（priority 最小）的待执行任务
        """
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        if pending_tasks:
            return pending_tasks[0]
        return None
    
    def complete_task(self, task: Task, result: str):
        """
        标记任务完成
        
        参数:
            task: 要完成的任务
            result: 执行结果
        """
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now().isoformat()
        self.tasks.remove(task)
        self.completed_tasks.append(task)
    
    def _sort_tasks(self):
        """按优先级排序任务"""
        self.tasks.sort(key=lambda t: (t.priority, t.task_id))
    
    def get_task_context(self, n: int = 3) -> str:
        """
        获取最近完成的任务上下文
        
        用于指导新任务的创建
        """
        recent = self.completed_tasks[-n:] if self.completed_tasks else []
        context = []
        for task in recent:
            context.append(f"任务: {task.task_name}")
            context.append(f"结果: {task.result}")
        return "\n".join(context)
    
    def get_all_tasks_status(self) -> str:
        """获取所有任务状态摘要"""
        pending = len([t for t in self.tasks if t.status == "pending"])
        completed = len(self.completed_tasks)
        return f"待执行: {pending}, 已完成: {completed}"


# ============================================================
# 第三部分: BabyAGI 核心
# ============================================================

class BabyAGI:
    """
    BabyAGI 核心实现
    
    实现任务驱动的自主执行循环
    """
    
    def __init__(self, objective: str, llm=None):
        """
        初始化 BabyAGI
        
        参数:
            objective: 要完成的最终目标
            llm: LangChain LLM 实例
        """
        self.objective = objective
        self.task_manager = TaskManager()
        self.max_iterations = 5
        
        # 初始化 LLM
        self.llm = llm or ChatOpenAI(
            model=config.MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.7
        )
        
        # 初始化第一个任务
        self.task_manager.add_task(
            f"开始研究目标: {objective}",
            priority=1
        )
    
    def _create_new_tasks(self, current_task: Task, result: str) -> List[Dict[str, Any]]:
        """
        根据当前任务结果创建新任务
        
        调用 LLM 基于 objective 和当前结果生成新任务
        """
        
        prompt = f"""你是 BabyAGI 的任务创建助手。

总体目标: {self.objective}

刚刚完成的任务:
- 任务名称: {current_task.task_name}
- 执行结果: {result}

已完成的任务列表:
{self.task_manager.get_task_context()}

当前待完成任务数: {len([t for t in self.task_manager.tasks if t.status == 'pending'])}

请根据以上信息，创建 1-3 个新的后续任务来推进总体目标的完成。

要求:
1. 新任务必须与总体目标相关
2. 考虑刚刚完成的任务结果
3. 任务应该具体可执行
4. 合理设置优先级（1-5，数字越小优先级越高）

请以 JSON 格式输出新任务列表:
[
    {{"name": "任务1描述", "priority": 1}},
    {{"name": "任务2描述", "priority": 2}}
]

如果认为目标已基本完成，可以返回空列表 [] 或一个总结任务。

只输出 JSON，不要添加 markdown 代码块标记。"""

        try:
            messages = [
                SystemMessage(content="你是一个任务规划助手，负责创建结构化的任务列表。只输出 JSON 数组。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # 清理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            new_tasks = json.loads(content)
            
            # 验证格式
            if not isinstance(new_tasks, list):
                new_tasks = []
            
            # 确保每个任务有正确的字段
            validated_tasks = []
            for task in new_tasks:
                if isinstance(task, dict) and "name" in task:
                    validated_tasks.append({
                        "name": task["name"],
                        "priority": task.get("priority", 3)
                    })
            
            return validated_tasks
            
        except Exception as e:
            print(f"  [创建任务出错: {e}]")
            # 回退到默认任务
            return [{"name": f"继续推进: {self.objective}", "priority": 2}]
    
    def _prioritize_tasks(self):
        """
        重新排序任务优先级
        
        根据 objective 和当前上下文调整任务优先级
        """
        # 实际使用会调用 LLM 重新评估优先级
        # 这里简化为按现有 priority 排序
        self.task_manager._sort_tasks()
    
    def _execute_task(self, task: Task) -> str:
        """
        执行任务
        
        调用 LLM 实际执行任务
        
        参数:
            task: 要执行的任务
            
        返回:
            任务执行结果
        """
        print(f"  执行任务: {task.task_name}")
        
        prompt = f"""你是 BabyAGI 的任务执行助手。

总体目标: {self.objective}
当前任务: {task.task_name}

任务上下文:
{self.task_manager.get_task_context()}

请执行当前任务，并返回具体的执行结果。
结果应该:
1. 具体且有信息量
2. 与总体目标相关
3. 包含可以指导下一步行动的洞察
4. 简洁明了（100-300字）

直接输出执行结果，不需要额外解释。"""

        try:
            messages = [
                SystemMessage(content="你是一个任务执行助手，负责完成具体的任务并返回结果。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            # 截断过长的结果
            if len(result) > 500:
                result = result[:500] + "..."
            
            return result
            
        except Exception as e:
            print(f"  [执行任务出错: {e}]")
            return f"任务执行遇到错误: {str(e)[:100]}"
    
    def run(self):
        """
        运行 BabyAGI
        
        执行主循环直到完成任务或达到最大迭代次数
        """
        print("=" * 70)
        print("BabyAGI 任务驱动型 Agent")
        print("=" * 70)
        print(f"\n目标: {self.objective}\n")
        print(f"初始任务: {self.task_manager.tasks[0].task_name}\n")
        print("-" * 70)
        
        iteration = 0
        
        while iteration < self.max_iterations:
            print(f"\n【第 {iteration + 1} 轮迭代】")
            print(f"任务状态: {self.task_manager.get_all_tasks_status()}")
            
            # 1. 获取下一个任务
            current_task = self.task_manager.get_next_task()
            
            if not current_task:
                print("\n没有待执行的任务了")
                break
            
            print(f"\n当前任务 (ID: {current_task.task_id}): {current_task.task_name}")
            
            # 2. 执行任务
            result = self._execute_task(current_task)
            print(f"执行结果: {result}")
            
            # 3. 标记任务完成
            self.task_manager.complete_task(current_task, result)
            
            # 4. 创建新任务
            print("\n根据结果创建新任务...")
            new_tasks = self._create_new_tasks(current_task, result)
            
            for task_info in new_tasks:
                task = self.task_manager.add_task(
                    task_info["name"],
                    task_info["priority"]
                )
                print(f"  + 创建任务 (ID: {task.task_id}): {task.task_name} [优先级: {task.priority}]")
            
            # 5. 重新排序任务
            self.task_manager._sort_tasks()
            
            iteration += 1
            
            # 检查是否达到目标
            if self._is_objective_complete():
                print("\n目标已达成!")
                break
        
        # 输出总结
        print("\n" + "=" * 70)
        print("执行总结")
        print("=" * 70)
        print(f"总迭代次数: {iteration}")
        print(f"完成任务数: {len(self.task_manager.completed_tasks)}")
        print(f"剩余任务数: {len(self.task_manager.tasks)}")
        
        print("\n已完成任务列表:")
        for task in self.task_manager.completed_tasks:
            print(f"  ✓ {task.task_name}")
            print(f"    结果: {task.result[:50]}...")
    
    def _is_objective_complete(self) -> bool:
        """
        检查目标是否完成
        
        可以通过多种方式判断:
        1. 是否生成了总结性任务
        2. 是否达到特定的里程碑
        3. 任务列表是否为空
        """
        # 简单判断：是否有总结类任务完成
        for task in self.task_manager.completed_tasks:
            if "总结" in task.task_name or "完成" in task.task_name:
                return True
        return False


# ============================================================
# 第四部分: 高级功能示例
# ============================================================

class AdvancedBabyAGI(BabyAGI):
    """
    高级版 BabyAGI
    
    增加功能:
    - 任务依赖管理
    - 更智能的任务创建
    - 执行历史分析
    """
    
    def __init__(self, objective: str, llm=None):
        super().__init__(objective, llm)
        self.execution_history: List[Dict] = []
    
    def _execute_task(self, task: Task) -> str:
        """增强版任务执行，记录历史"""
        result = super()._execute_task(task)
        
        # 记录执行历史
        self.execution_history.append({
            "task": task.task_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        return {
            "objective": self.objective,
            "total_iterations": len(self.execution_history),
            "completed_tasks": len(self.task_manager.completed_tasks),
            "pending_tasks": len(self.task_manager.tasks),
            "success_rate": len(self.task_manager.completed_tasks) / max(len(self.execution_history), 1)
        }


# ============================================================
# 第五部分: 使用示例
# ============================================================

def main():
    """
    主函数：演示 BabyAGI 的使用
    """
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=config.MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0.7
    )
    
    print("""
BabyAGI 工作原理:

┌─────────────────────────────────────────────────────────────┐
│                        BabyAGI Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                                          │
│   │  Task List   │                                          │
│   │  [任务队列]   │                                          │
│   └──────┬───────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────┐                                      │
│   │ 1. Pull Task     │  取出最高优先级任务                  │
│   │    取出任务      │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │ 2. Execute Task  │  执行任务                            │
│   │    执行任务      │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │ 3. Create Tasks  │  根据结果创建新任务                  │
│   │    创建新任务    │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │ 4. Prioritize    │  重新排序任务优先级                  │
│   │    重新排序      │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            └──────────────► (返回起点，继续循环)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

""")
    
    # 示例 1: 研究任务
    print("\n" + "=" * 70)
    print("示例 1: 研究任务")
    print("=" * 70)
    
    agi1 = BabyAGI(objective="研究 Python 异步编程最佳实践", llm=llm)
    agi1.run()
    
    # 示例 2: 信息收集任务
    print("\n\n" + "=" * 70)
    print("示例 2: 信息收集任务")
    print("=" * 70)
    
    agi2 = BabyAGI(objective="收集 2024 年 AI 发展趋势相关信息", llm=llm)
    agi2.run()
    
    # 示例 3: 高级版本
    print("\n\n" + "=" * 70)
    print("示例 3: 高级 BabyAGI")
    print("=" * 70)
    
    agi3 = AdvancedBabyAGI(objective="设计一个微服务架构方案", llm=llm)
    agi3.run()
    
    print("\n执行摘要:")
    summary = agi3.get_execution_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("BabyAGI 示例结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
