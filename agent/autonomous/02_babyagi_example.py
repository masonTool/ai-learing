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

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime


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
    
    def __init__(self, objective: str, api_key: Optional[str] = None):
        """
        初始化 BabyAGI
        
        参数:
            objective: 要完成的最终目标
            api_key: API 密钥（实际使用）
        """
        self.objective = objective
        self.task_manager = TaskManager()
        self.max_iterations = 5
        
        # 初始化第一个任务
        self.task_manager.add_task(
            f"开始研究目标: {objective}",
            priority=1
        )
    
    def _create_new_tasks(self, current_task: Task, result: str) -> List[Dict[str, Any]]:
        """
        根据当前任务结果创建新任务
        
        这是 BabyAGI 的核心逻辑之一
        根据执行结果动态生成后续任务
        
        参数:
            current_task: 刚刚完成的任务
            result: 任务执行结果
            
        返回:
            新任务列表，每个任务包含 name 和 priority
        """
        # 模拟 LLM 创建新任务
        # 实际使用会调用 API，基于 objective 和当前结果生成新任务
        
        new_tasks = []
        
        # 根据当前任务和结果生成相关新任务
        current_task_id = current_task.task_id
        
        if "研究" in current_task.task_name:
            new_tasks.extend([
                {"name": f"深入分析目标的关键要素", "priority": 2},
                {"name": f"收集相关的背景信息", "priority": 3},
                {"name": f"整理初步发现", "priority": 4}
            ])
        elif "分析" in current_task.task_name:
            new_tasks.extend([
                {"name": f"总结分析结果", "priority": 2},
                {"name": f"识别关键模式和趋势", "priority": 3}
            ])
        elif "收集" in current_task.task_name:
            new_tasks.extend([
                {"name": f"验证收集的信息", "priority": 2},
                {"name": f"整合数据到知识库", "priority": 3}
            ])
        else:
            # 默认创建总结任务
            new_tasks.append({
                "name": f"总结关于 '{self.objective}' 的发现",
                "priority": 1
            })
        
        return new_tasks
    
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
        
        参数:
            task: 要执行的任务
            
        返回:
            任务执行结果
        """
        print(f"  执行任务: {task.task_name}")
        
        # 模拟任务执行
        # 实际使用会调用相应的工具或 API
        
        if "研究" in task.task_name:
            return f"已完成对 '{self.objective}' 的初步研究，识别出主要研究方向"
        
        elif "分析" in task.task_name:
            return f"分析完成，发现 3 个关键要素需要进一步探索"
        
        elif "收集" in task.task_name:
            return f"收集了 10 条相关信息，其中 5 条具有高价值"
        
        elif "验证" in task.task_name:
            return f"信息验证完成，确认 80% 的信息准确可靠"
        
        elif "总结" in task.task_name:
            return f"已生成关于 '{self.objective}' 的综合总结报告"
        
        else:
            return f"任务 '{task.task_name}' 执行完成"
    
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
    
    def __init__(self, objective: str, api_key: Optional[str] = None):
        super().__init__(objective, api_key)
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
    
    agi1 = BabyAGI(objective="研究 Python 异步编程最佳实践")
    agi1.run()
    
    # 示例 2: 信息收集任务
    print("\n\n" + "=" * 70)
    print("示例 2: 信息收集任务")
    print("=" * 70)
    
    agi2 = BabyAGI(objective="收集 2024 年 AI 发展趋势相关信息")
    agi2.run()
    
    # 示例 3: 高级版本
    print("\n\n" + "=" * 70)
    print("示例 3: 高级 BabyAGI")
    print("=" * 70)
    
    agi3 = AdvancedBabyAGI(objective="设计一个微服务架构方案")
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
