"""
LangGraph 工作流示例
==================

LangGraph 是 LangChain 的扩展，用于构建复杂的多步骤工作流。
与简单 Agent 不同，LangGraph 允许:
1. 显式定义状态 (State) - 工作流中的数据流转
2. 构建图结构 (Graph) - 节点和边的连接关系
3. 支持循环 (Cycles) - 可以构建循环执行流程
4. 条件分支 (Conditional Edges) - 根据条件选择不同路径

核心概念:
- StateGraph: 状态图，定义了工作流的结构
- Node: 节点，执行具体任务的函数
- Edge: 边，定义节点间的流转关系
- State: 状态，在工作流节点间传递的数据

适用场景: 复杂业务流程、多步骤审批、循环优化任务
"""

from langchain_core.messages.base import BaseMessage
from langchain_openai.chat_models.base import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


# ============================================================
# 第一部分: 状态定义
# ============================================================

class AgentState(TypedDict):
    """
    定义工作流的状态结构
    
    TypedDict 是 Python 的类型提示，用于定义字典的结构
    这个状态会在工作流的所有节点间传递和更新
    
    字段说明:
    - messages: 消息列表，记录对话历史
    - next_step: 下一步要执行的节点名称
    - iteration_count: 迭代计数，防止无限循环
    - final_answer: 最终答案
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    """
    Annotated 是类型注解，operator.add 表示这个字段使用加法合并
    当多个节点更新 messages 时，新消息会追加到列表末尾
    """
    next_step: str
    iteration_count: int
    final_answer: str


# ============================================================
# 第二部分: 工具定义
# ============================================================

@tool
def search_web(query: str) -> str:
    """
    模拟网页搜索工具
    实际使用时可以替换为真实的搜索 API
    """
    # 这里返回模拟数据，实际应调用搜索引擎
    mock_results = {
        "天气": "今天北京天气晴朗，气温 15-25°C",
        "股票": "上证指数今日上涨 1.2%，收于 3050 点",
        "新闻": "AI 技术在医疗领域取得重大突破"
    }
    return mock_results.get(query, f"搜索 '{query}' 的结果: 找到相关信息...")


@tool
def calculate(expression: str) -> str:
    """
    计算工具
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算错误，请检查表达式"


# 工具列表
tools = [search_web, calculate]


# ============================================================
# 第三部分: 节点函数定义
# ============================================================

def agent_node(state: AgentState) -> AgentState:
    """
    Agent 节点: 决定下一步行动
    
    这是工作流的"大脑"，分析当前状态并决定:
    1. 是否需要调用工具
    2. 还是可以直接给出答案
    
    参数 state: 当前工作流状态
    返回: 更新后的状态
    """
    print("\n[Agent Node] 正在分析请求...")
    
    # 初始化 LLM（从根目录 config.py 读取）
    llm: ChatOpenAI = ChatOpenAI(
        model=config.MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # 获取最后一条用户消息
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # 构建系统提示
    system_prompt = (
        "你是一个智能助手，可以决定是否需要使用工具。\n"
        "如果需要搜索或计算，请使用相应工具。\n"
        "如果可以直接回答，请直接给出答案。\n\n"
        f"当前迭代次数: {state['iteration_count']}"
    )
    
    messages = [
        ("system", system_prompt),
        ("human", last_message)
    ]
    
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    
    # 检查是否需要调用工具
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # 需要调用工具
        print(f"  -> 决定调用工具: {response.tool_calls[0]['name']}")
        return {
            "messages": [response],
            "next_step": "tools",
            "iteration_count": state["iteration_count"] + 1,
            "final_answer": ""
        }
    else:
        # 直接给出答案
        print(f"  -> 直接给出答案")
        return {
            "messages": [response],
            "next_step": "end",
            "iteration_count": state["iteration_count"],
            "final_answer": response.content
        }


def tool_node(state: AgentState) -> AgentState:
    """
    工具执行节点: 执行 Agent 决定的工具
    
    这个节点接收 Agent 的工具调用请求，
    实际执行相应的工具函数，并返回结果
    """
    print("\n[Tool Node] 执行工具...")
    
    # 获取最后一条消息的 tool_calls
    last_message: BaseMessage = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls'):
        return state
    
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"  -> 执行: {tool_name}({tool_args})")
        
        # 找到并执行对应的工具
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                tool_results.append(result)
                print(f"  <- 结果: {result}")
                break
    
    # 创建工具结果消息
    tool_message = AIMessage(content=f"工具执行结果: {', '.join(tool_results)}")
    
    return {
        "messages": [tool_message],
        "next_step": "agent",  # 返回给 Agent 继续处理
        "iteration_count": state["iteration_count"],
        "final_answer": ""
    }


def should_continue(state: AgentState) -> str:
    """
    条件判断函数: 决定工作流走向
    
    这是一个条件边函数，根据状态决定下一步:
    - 如果达到最大迭代次数，结束工作流
    - 如果 next_step 是 "end"，结束工作流
    - 否则继续执行
    """
    if state["iteration_count"] >= 3:
        print("\n[决策] 达到最大迭代次数，结束工作流")
        return "end"
    
    if state["next_step"] == "end":
        print("\n[决策] 任务完成，结束工作流")
        return "end"
    
    return state["next_step"]


# ============================================================
# 第四部分: 构建工作流图
# ============================================================

def create_workflow():
    """
    创建工作流图
    
    这个函数构建一个有状态的工作流:
    
    ┌─────────┐
    │  agent  │◄────┐
    └────┬────┘     │
         │          │ (需要更多处理)
    ┌────▼────┐     │
    │  tools  │─────┘
    └─────────┘
         │
         │ (完成)
    ┌────▼────┐
    │   end   │
    └─────────┘
    """
    
    # 1. 创建状态图
    workflow = StateGraph(AgentState)
    
    # 2. 添加节点
    # 每个节点是一个处理函数
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 3. 设置入口点
    workflow.set_entry_point("agent")
    
    # 4. 添加条件边
    # 从 agent 节点出发，根据 should_continue 的结果决定去向
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # 如果需要工具，去 tools 节点
            "end": END         # 如果完成，结束工作流
        }
    )
    
    # 5. 添加普通边
    # tools 节点执行后，总是返回 agent 节点
    workflow.add_edge("tools", "agent")
    
    # 6. 编译工作流
    app = workflow.compile()
    
    return app


# ============================================================
# 第五部分: 使用示例
# ============================================================

def main():
    """
    演示 LangGraph 工作流的使用
    """
    print("=" * 70)
    print("LangGraph 工作流示例")
    print("=" * 70)
    
    # 创建工作流
    workflow = create_workflow()
    
    # 示例 1: 需要搜索的问题
    print("\n【示例 1】需要工具调用")
    print("-" * 50)
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content="今天北京的天气怎么样？")],
        "next_step": "agent",
        "iteration_count": 0,
        "final_answer": ""
    }
    
    # 运行工作流
    # 工作流会自动处理循环，直到达到结束条件
    result = workflow.invoke(initial_state)
    
    print(f"\n最终答案: {result['final_answer']}")
    print(f"总迭代次数: {result['iteration_count']}")
    
    # 示例 2: 可以直接回答的问题
    print("\n\n【示例 2】直接回答")
    print("-" * 50)
    
    initial_state2 = {
        "messages": [HumanMessage(content="你好，请介绍一下你自己")],
        "next_step": "agent",
        "iteration_count": 0,
        "final_answer": ""
    }
    
    result2 = workflow.invoke(initial_state2)
    print(f"\n最终答案: {result2['final_answer']}")
    
    print("\n" + "=" * 70)
    print("示例结束")
    print("=" * 70)
    
    # 可视化工作流（需要 graphviz）
    print("\n工作流图结构:")
    print("""
    ┌─────────┐      需要工具      ┌─────────┐
    │  agent  │───────────────────►│  tools  │
    └────┬────┘                   └────┬────┘
         │                             │
         │ 直接回答                    │ 执行后返回
         ▼                             │
    ┌─────────┐◄──────────────────────┘
    │   end   │
    └─────────┘
    """)


if __name__ == "__main__":
    main()
