"""
Microsoft Semantic Kernel 示例
==============================

Semantic Kernel 是微软开源的 AI 开发 SDK，核心理念是:
"将 AI 能力集成到现有应用中"

核心概念:
1. Kernel (内核): 中央编排器，管理所有 AI 服务和插件
2. Plugins (插件): 可重用的功能模块，包含 Prompts 和 Functions
3. Planners (规划器): 自动创建执行计划的组件
4. Memory (记忆): 向量化的语义记忆系统
5. Connectors (连接器): 连接各种 AI 服务和数据源

支持语言: C#, Python, Java

特点:
- 企业级设计: 强类型、可测试、可扩展
- 多模型支持: OpenAI, Azure OpenAI, HuggingFace 等
- 灵活集成: 可以嵌入到现有应用中
- 自动规划: AI 自动创建执行计划

适用场景: 企业级 AI 应用、Microsoft 生态集成、多语言团队
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 第一部分: Semantic Kernel 核心概念
# ============================================================

@dataclass
class SKContext:
    """
    Semantic Kernel 上下文
    
    包含:
    - Variables: 变量字典
    - Memories: 语义记忆
    - Plugins: 可用插件
    - Logger: 日志记录器
    """
    variables: Dict[str, Any]
    memories: Dict[str, List[str]]
    plugins: Dict[str, 'SKPlugin']
    
    def __init__(self):
        self.variables = {}
        self.memories = {}
        self.plugins = {}
    
    def get(self, key: str, default=None):
        """获取变量值"""
        return self.variables.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置变量值"""
        self.variables[key] = value
        return self


class SKPlugin:
    """
    Semantic Kernel 插件
    
    插件是可重用的功能模块，可以包含:
    - Semantic Functions (语义函数): 基于 LLM 的函数
    - Native Functions (原生函数): 代码实现的函数
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.semantic_functions: Dict[str, 'SemanticFunction'] = {}
        self.native_functions: Dict[str, 'NativeFunction'] = {}
    
    def add_semantic_function(self, func: 'SemanticFunction'):
        """添加语义函数"""
        self.semantic_functions[func.name] = func
    
    def add_native_function(self, func: 'NativeFunction'):
        """添加原生函数"""
        self.native_functions[func.name] = func


@dataclass
class SemanticFunction:
    """
    语义函数
    
    使用自然语言定义的函数，由 LLM 执行
    """
    name: str
    prompt_template: str
    description: str
    max_tokens: int = 256
    temperature: float = 0.7
    
    def invoke(self, context: SKContext, kernel: 'SemanticKernel') -> str:
        """
        执行语义函数
        
        调用真实 LLM API
        """
        # 渲染模板
        prompt = self._render_template(context)
        
        print(f"  [SemanticFunction] {self.name}")
        print(f"  提示词: {prompt[:100]}...")
        
        # 调用真实 LLM
        try:
            llm = kernel.get_ai_service()
            messages = [
                SystemMessage(content=f"你是一个 helpful 的 AI 助手。{self.description}"),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"调用 LLM 出错: {str(e)[:100]}"
    
    def _render_template(self, context: SKContext) -> str:
        """渲染模板"""
        result = self.prompt_template
        for key, value in context.variables.items():
            placeholder = f"{{${key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


@dataclass
class NativeFunction:
    """
    原生函数
    
    使用代码实现的函数
    """
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]
    
    def invoke(self, context: SKContext) -> Any:
        """执行原生函数"""
        print(f"  [NativeFunction] {self.name}")
        return self.func(context)


# ============================================================
# 第二部分: Kernel 核心
# ============================================================

class SemanticKernel:
    """
    Semantic Kernel 核心类
    
    职责:
    1. 管理插件注册
    2. 执行函数调用
    3. 协调 AI 服务
    4. 管理记忆系统
    """
    
    def __init__(self, llm=None):
        self.plugins: Dict[str, SKPlugin] = {}
        self.ai_services: Dict[str, Any] = {}
        self.memory: Dict[str, List[str]] = {}
        
        # 初始化默认 LLM
        self._default_llm = llm or ChatOpenAI(
            model=config.MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.7
        )
    
    def get_ai_service(self) -> Any:
        """获取 AI 服务（LLM）"""
        return self._default_llm
    
    def import_plugin(self, plugin: SKPlugin):
        """
        导入插件
        
        参数:
            plugin: 要导入的插件
        """
        self.plugins[plugin.name] = plugin
        print(f"[Kernel] 导入插件: {plugin.name}")
    
    def import_plugin_from_object(self, obj: Any, plugin_name: str):
        """
        从对象导入插件
        
        自动识别带有 @sk_function 装饰器的方法
        """
        plugin = SKPlugin(plugin_name)
        
        # 遍历对象的方法
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if callable(attr) and hasattr(attr, '_sk_function'):
                func = NativeFunction(
                    name=attr_name,
                    description=getattr(attr, '_description', ''),
                    func=attr,
                    parameters=getattr(attr, '_parameters', {})
                )
                plugin.add_native_function(func)
        
        self.import_plugin(plugin)
    
    def run(self, context: Optional[SKContext] = None, *function_references) -> SKContext:
        """
        执行函数
        
        参数:
            context: 执行上下文
            function_references: 函数引用，格式 "PluginName.FunctionName"
            
        返回:
            执行后的上下文
        """
        if context is None:
            context = SKContext()
        
        for func_ref in function_references:
            plugin_name, func_name = func_ref.split('.')
            plugin = self.plugins.get(plugin_name)
            
            if not plugin:
                print(f"[Kernel] 错误: 插件 {plugin_name} 不存在")
                continue
            
            # 查找函数
            func = None
            func_type = ""
            if func_name in plugin.semantic_functions:
                func = plugin.semantic_functions[func_name]
                func_type = "semantic"
            elif func_name in plugin.native_functions:
                func = plugin.native_functions[func_name]
                func_type = "native"
            
            if not func:
                print(f"[Kernel] 错误: 函数 {func_name} 不存在")
                continue
            
            print(f"\n[Kernel] 执行: {func_ref}")
            
            # 执行函数
            if func_type == "semantic":
                result = func.invoke(context, self)
            else:
                result = func.invoke(context)
            
            # 保存结果到上下文
            context.set("input", result)
            context.set(f"{plugin_name}_{func_name}_result", result)
        
        return context
    
    def create_new_context(self) -> SKContext:
        """创建新的执行上下文"""
        return SKContext()
    
    def register_memory(self, collection_name: str, text: str):
        """注册记忆"""
        if collection_name not in self.memory:
            self.memory[collection_name] = []
        self.memory[collection_name].append(text)
    
    def search_memory(self, collection_name: str, query: str, limit: int = 3) -> List[str]:
        """搜索记忆"""
        if collection_name not in self.memory:
            return []
        # 简化的记忆搜索
        return self.memory[collection_name][:limit]


# ============================================================
# 第三部分: Planner (规划器)
# ============================================================

class Planner:
    """
    规划器
    
    根据目标自动生成执行计划
    """
    
    def __init__(self, kernel: SemanticKernel):
        self.kernel = kernel
    
    def create_plan(self, goal: str) -> 'Plan':
        """
        创建执行计划
        
        参数:
            goal: 用户目标
            
        返回:
            执行计划
        """
        print(f"\n[Planner] 为目标创建计划: {goal}")
        
        # 分析可用插件和函数
        available_functions = []
        for plugin_name, plugin in self.kernel.plugins.items():
            for func_name in list(plugin.semantic_functions.keys()) + list(plugin.native_functions.keys()):
                available_functions.append(f"{plugin_name}.{func_name}")
        
        print(f"[Planner] 可用函数: {available_functions}")
        
        # 使用 LLM 生成计划
        steps = self._generate_steps_with_llm(goal, available_functions)
        
        return Plan(goal=goal, steps=steps)
    
    def _generate_steps_with_llm(self, goal: str, available_functions: List[str]) -> List[str]:
        """使用 LLM 生成执行步骤"""
        
        if not available_functions:
            return ["GeneralPlugin.Process"]
        
        prompt = f"""你是一个任务规划助手。请为以下目标创建一个执行计划。

目标: {goal}

可用的函数:
{chr(10).join([f"- {f}" for f in available_functions])}

请分析目标，并选择最合适的函数来创建一个执行计划。
以 JSON 数组格式输出要执行的函数列表（按顺序）。

示例输出:
["WriterPlugin.Brainstorm", "WriterPlugin.Compose"]

只输出 JSON 数组，不要其他内容。"""

        try:
            llm = self.kernel.get_ai_service()
            messages = [
                SystemMessage(content="你是一个任务规划助手，负责创建执行计划。只输出 JSON 数组。"),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            content = response.content.strip()
            
            # 清理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            steps = json.loads(content)
            
            # 验证步骤是否有效
            if isinstance(steps, list) and len(steps) > 0:
                # 过滤掉不在可用函数中的步骤
                valid_steps = [s for s in steps if s in available_functions]
                if valid_steps:
                    return valid_steps
            
        except Exception as e:
            print(f"  [规划出错: {e}]")
        
        # 回退到简化规则
        return self._generate_steps_fallback(goal, available_functions)
    
    def _generate_steps_fallback(self, goal: str, available_functions: List[str]) -> List[str]:
        """回退的启发式规则生成步骤"""
        steps = []
        
        # 简化的规则引擎
        if "写" in goal or "生成" in goal or "create" in goal.lower():
            if "WriterPlugin.Brainstorm" in available_functions:
                steps.append("WriterPlugin.Brainstorm")
            if "WriterPlugin.Compose" in available_functions:
                steps.append("WriterPlugin.Compose")
            if "WriterPlugin.Translate" in available_functions:
                steps.append("WriterPlugin.Translate")
        
        if "分析" in goal or "analyze" in goal.lower():
            if "AnalysisPlugin.ExtractData" in available_functions:
                steps.append("AnalysisPlugin.ExtractData")
            if "AnalysisPlugin.Summarize" in available_functions:
                steps.append("AnalysisPlugin.Summarize")
        
        if not steps and available_functions:
            # 默认使用第一个可用函数
            steps.append(available_functions[0])
        
        return steps


@dataclass
class Plan:
    """
    执行计划
    
    包含一系列要执行的步骤
    """
    goal: str
    steps: List[str]
    
    def invoke(self, kernel: SemanticKernel, context: SKContext) -> SKContext:
        """执行计划"""
        print(f"\n[Plan] 执行计划: {self.goal}")
        print(f"步骤数: {len(self.steps)}")
        
        for i, step in enumerate(self.steps, 1):
            print(f"\n[Plan] 步骤 {i}/{len(self.steps)}: {step}")
            context = kernel.run(context, step)
        
        return context


# ============================================================
# 第四部分: 示例插件
# ============================================================

class WriterPlugin:
    """
    写作插件
    
    展示如何创建 Semantic Functions
    """
    
    def get_plugin(self) -> SKPlugin:
        """获取插件实例"""
        plugin = SKPlugin("WriterPlugin", "写作助手插件")
        
        # 头脑风暴语义函数
        brainstorm = SemanticFunction(
            name="Brainstorm",
            description="为给定主题生成创意想法",
            prompt_template="""为以下主题生成5个创意想法:
主题: {{$input}}

要求:
1. 想法要新颖独特
2. 考虑可行性
3. 列出优缺点

创意列表:""",
            max_tokens=512
        )
        
        # 写作语义函数
        compose = SemanticFunction(
            name="Compose",
            description="根据大纲撰写内容",
            prompt_template="""根据以下大纲撰写详细内容:
主题: {{$input}}
大纲: {{$outline}}

要求:
1. 内容充实，有深度
2. 逻辑清晰
3. 语言流畅

内容:""",
            max_tokens=1024
        )
        
        # 翻译语义函数
        translate = SemanticFunction(
            name="Translate",
            description="翻译内容",
            prompt_template="""将以下内容翻译成 {{$target_language}}:
{{$input}}

翻译结果:""",
            max_tokens=1024
        )
        
        plugin.add_semantic_function(brainstorm)
        plugin.add_semantic_function(compose)
        plugin.add_semantic_function(translate)
        
        return plugin


class AnalysisPlugin:
    """
    分析插件
    
    展示如何创建 Native Functions
    """
    
    def get_plugin(self) -> SKPlugin:
        """获取插件实例"""
        plugin = SKPlugin("AnalysisPlugin", "数据分析插件")
        
        # 数据提取原生函数
        def extract_data(context: SKContext):
            input_data = context.get("input", "")
            # 模拟数据提取
            return {
                "keywords": ["AI", "机器学习", "数据"],
                "sentiment": "积极",
                "entities": ["公司A", "产品B"]
            }
        
        extract_func = NativeFunction(
            name="ExtractData",
            description="从文本中提取关键数据",
            func=extract_data,
            parameters={"input": "string"}
        )
        
        # 摘要原生函数
        def summarize(context: SKContext):
            input_data = context.get("input", "")
            # 模拟摘要生成
            return f"摘要: {input_data[:50]}... (关键要点总结)"
        
        summarize_func = NativeFunction(
            name="Summarize",
            description="生成文本摘要",
            func=summarize,
            parameters={"input": "string"}
        )
        
        plugin.add_native_function(extract_func)
        plugin.add_native_function(summarize_func)
        
        return plugin


# ============================================================
# 第五部分: 使用示例
# ============================================================

def example_1_basic_usage(llm=None):
    """示例 1: 基础使用"""
    print("=" * 70)
    print("示例 1: Semantic Kernel 基础使用")
    print("=" * 70)
    
    # 创建 Kernel
    kernel = SemanticKernel(llm=llm)
    
    # 创建并导入插件
    writer_plugin = WriterPlugin().get_plugin()
    kernel.import_plugin(writer_plugin)
    
    # 创建上下文
    context = kernel.create_new_context()
    context.set("input", "人工智能在教育领域的应用")
    
    # 执行函数
    result = kernel.run(context, "WriterPlugin.Brainstorm")
    
    print(f"\n执行结果: {result.get('WriterPlugin.Brainstorm_result')}")


def example_2_chained_functions(llm=None):
    """示例 2: 链式函数调用"""
    print("\n" + "=" * 70)
    print("示例 2: 链式函数调用")
    print("=" * 70)
    
    kernel = SemanticKernel(llm=llm)
    
    # 导入多个插件
    kernel.import_plugin(WriterPlugin().get_plugin())
    kernel.import_plugin(AnalysisPlugin().get_plugin())
    
    # 创建上下文
    context = kernel.create_new_context()
    context.set("input", "机器学习技术趋势分析")
    context.set("outline", "1. 引言 2. 技术现状 3. 未来趋势 4. 结论")
    
    # 链式执行多个函数
    result = kernel.run(
        context,
        "WriterPlugin.Brainstorm",
        "AnalysisPlugin.ExtractData",
        "AnalysisPlugin.Summarize"
    )
    
    print("\n链式执行完成")
    print(f"最终结果: {result.get('input')}")


def example_3_planner(llm=None):
    """示例 3: 使用 Planner"""
    print("\n" + "=" * 70)
    print("示例 3: 使用 Planner 自动生成计划")
    print("=" * 70)
    
    kernel = SemanticKernel(llm=llm)
    
    # 导入插件
    kernel.import_plugin(WriterPlugin().get_plugin())
    kernel.import_plugin(AnalysisPlugin().get_plugin())
    
    # 创建规划器
    planner = Planner(kernel)
    
    # 创建计划
    goal = "写一篇关于机器学习的技术文章"
    plan = planner.create_plan(goal)
    
    # 执行计划
    context = kernel.create_new_context()
    context.set("input", "机器学习技术趋势")
    context.set("outline", "1. 简介 2. 主要技术 3. 应用场景 4. 未来展望")
    
    result = plan.invoke(kernel, context)
    
    print("\n计划执行完成")


def example_4_memory(llm=None):
    """示例 4: 使用记忆系统"""
    print("\n" + "=" * 70)
    print("示例 4: 使用记忆系统")
    print("=" * 70)
    
    kernel = SemanticKernel(llm=llm)
    
    # 注册记忆
    kernel.register_memory("user_preferences", "用户喜欢简洁的技术文章")
    kernel.register_memory("user_preferences", "用户对 Python 最熟悉")
    kernel.register_memory("user_preferences", "用户偏好中文内容")
    
    # 搜索记忆
    query = "用户偏好"
    memories = kernel.search_memory("user_preferences", query)
    
    print(f"搜索 '{query}' 的记忆:")
    for i, mem in enumerate(memories, 1):
        print(f"  {i}. {mem}")


def example_5_complete_app(llm=None):
    """示例 5: 完整应用"""
    print("\n" + "=" * 70)
    print("示例 5: 完整应用 - 内容生成系统")
    print("=" * 70)
    
    # 创建 Kernel
    kernel = SemanticKernel(llm=llm)
    
    # 导入插件
    kernel.import_plugin(WriterPlugin().get_plugin())
    kernel.import_plugin(AnalysisPlugin().get_plugin())
    
    # 定义任务
    tasks = [
        "为'云原生技术'主题生成创意",
        "分析一段产品描述",
        "撰写技术博客大纲"
    ]
    
    for task in tasks:
        print(f"\n任务: {task}")
        
        context = kernel.create_new_context()
        context.set("input", task)
        
        # 使用 Planner 自动规划
        planner = Planner(kernel)
        plan = planner.create_plan(task)
        
        if len(plan.steps) > 0:
            result = plan.invoke(kernel, context)
            print(f"结果: {result.get('input')}")
        else:
            print("没有可用的执行步骤")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("Semantic Kernel 示例")
    print("====================")
    print()
    print("微软开源的 AI 开发 SDK")
    print("特点: 企业级设计、多语言支持、自动规划")
    print()
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=config.MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0.7
    )
    
    # 运行示例
    example_1_basic_usage(llm=llm)
    example_2_chained_functions(llm=llm)
    example_3_planner(llm=llm)
    example_4_memory(llm=llm)
    example_5_complete_app(llm=llm)
    
    print("\n" + "=" * 70)
    print("所有示例运行完成")
    print("=" * 70)
    print()
    print("Semantic Kernel 工作流程:")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                   Semantic Kernel 架构                       │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                              │")
    print("│   User Request                                               │")
    print("│       │                                                      │")
    print("│       ▼                                                      │")
    print("│   ┌─────────────────┐                                        │")
    print("│   │     Planner     │ ← 分析问题，创建执行计划                │")
    print("│   └────────┬────────┘                                        │")
    print("│            │                                                 │")
    print("│            ▼                                                 │")
    print("│   ┌─────────────────┐     ┌─────────────────┐                │")
    print("│   │  Semantic Func  │     │  Native Func    │                │")
    print("│   │  (LLM Prompts)  │     │  (Code)         │                │")
    print("│   └────────┬────────┘     └────────┬────────┘                │")
    print("│            │                        │                        │")
    print("│            └────────┬───────────────┘                        │")
    print("│                     ▼                                        │")
    print("│            ┌─────────────────┐                               │")
    print("│            │     Kernel      │ ← 协调执行，管理上下文         │")
    print("│            └────────┬────────┘                               │")
    print("│                     │                                        │")
    print("│                     ▼                                        │")
    print("│            ┌─────────────────┐                               │")
    print("│            │    Memory/      │                               │")
    print("│            │    Connectors   │                               │")
    print("│            └─────────────────┘                               │")
    print("│                                                              │")
    print("└─────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
