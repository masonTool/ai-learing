"""
LangChain 基础 Agent 示例
========================

LangChain 是最流行的 Python Agent 框架，核心理念是"链式调用"(Chains)
和"工具使用"(Tools)。

核心概念:
1. Tools: Agent 可以调用的外部工具（如搜索、计算、API等）
2. Prompt: 指导 Agent 如何思考和行动的指令模板
3. Memory: 保存对话历史，让 Agent 有上下文记忆
4. AgentExecutor: 执行 Agent 的运行器，管理思考和行动的循环

适用场景: 快速构建单 Agent 应用，工具调用型任务
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import os


# ============================================================
# 第一部分: 配置和工具定义
# ============================================================

# 设置 API Key（请替换为你的实际 API Key）
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. 定义工具 (Tools)
# 工具是 Agent 可以调用的外部能力，每个工具包含:
# - name: 工具名称，Agent 通过名称识别调用哪个工具
# - func: 实际执行的函数
# - description: 工具描述，告诉 Agent 这个工具什么时候用

search_tool = DuckDuckGoSearchRun()

def calculator(expression: str) -> str:
    """
    安全计算器工具
    使用 eval 计算数学表达式，但限制可用的函数和变量
    """
    try:
        # 只允许基本数学运算，防止代码注入
        allowed_names = {"abs": abs, "max": max, "min": min, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具列表
# Agent 会根据用户输入和工具描述，决定调用哪个工具
tools = [
    Tool(
        name="web_search",
        func=search_tool.run,
        description="""
        当需要搜索互联网获取最新信息时使用此工具。
        输入应该是搜索查询字符串。
        适用于: 查询新闻、获取实时数据、搜索事实信息等。
        """
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="""
        当需要进行数学计算时使用此工具。
        输入应该是数学表达式，如 '2 + 2' 或 '100 * 0.15'。
        适用于: 加减乘除、百分比计算等。
        """
    )
]


# ============================================================
# 第二部分: Agent 创建
# ============================================================

def create_basic_agent():
    """
    创建一个基础的 ReAct Agent
    
    ReAct (Reasoning + Acting) 是 LangChain 的核心模式:
    - Thought: Agent 先思考需要做什么
    - Action: 然后执行具体动作（调用工具）
    - Observation: 观察工具返回的结果
    - Final Answer: 综合所有信息给出最终答案
    
    这个过程会循环执行，直到 Agent 认为任务完成
    """
    
    # 1. 初始化 LLM (大语言模型)
    # temperature=0 让输出更确定，适合需要精确工具调用的场景
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        # api_key="your-api-key"  # 或从环境变量读取
    )
    
    # 2. 创建记忆组件
    # ConversationBufferMemory 保存完整的对话历史
    # 这样 Agent 能记住之前的对话内容，实现多轮交互
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 在 prompt 中使用的变量名
        return_messages=True  # 返回消息对象列表而不是字符串
    )
    
    # 3. 定义 ReAct Prompt 模板
    # 这是指导 Agent 行为的系统指令
    template = """你是一个有用的 AI 助手，可以使用工具来帮助用户。

可用工具:
{tools}

工具名称: {tool_names}

使用以下格式:

Question: 用户的问题
Thought: 我应该如何解决这个问题
Action: 要使用的工具名称（必须是 [{tool_names}] 之一）
Action Input: 传递给工具的输入
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 给用户的最终回答

记住:
1. 每次只能使用一个工具
2. 必须严格按照格式输出
3. 如果不需要工具，直接给出 Final Answer

之前的对话历史:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # 4. 创建 Agent
    # create_react_agent 使用 ReAct 推理模式
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # 5. 创建 Agent 执行器
    # AgentExecutor 负责运行 Agent 的思考和行动循环
    # verbose=True 会打印详细的执行过程，便于调试
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 打印详细执行过程
        max_iterations=10,  # 最多执行 10 步，防止无限循环
        handle_parsing_errors=True  # 处理解析错误
    )
    
    return agent_executor


# ============================================================
# 第三部分: 使用示例
# ============================================================

def main():
    """
    主函数演示如何使用 Agent
    """
    print("=" * 60)
    print("LangChain 基础 Agent 示例")
    print("=" * 60)
    
    # 创建 Agent
    agent = create_basic_agent()
    
    # 示例 1: 需要搜索的问题
    print("\n【示例 1】搜索问题")
    print("-" * 40)
    query1 = "2024年诺贝尔物理学奖的获得者是谁？"
    print(f"用户: {query1}")
    
    # invoke 方法触发 Agent 执行
    # Agent 会:
    # 1. 分析问题，决定使用 web_search 工具
    # 2. 调用搜索工具获取信息
    # 3. 基于搜索结果给出答案
    response1 = agent.invoke({"input": query1})
    print(f"Agent: {response1['output']}")
    
    # 示例 2: 需要计算的问题
    print("\n【示例 2】计算问题")
    print("-" * 40)
    query2 = "如果我有 1500 元，买 3 件每件 299 元的商品，还剩多少钱？"
    print(f"用户: {query2}")
    
    # Agent 会:
    # 1. 分析问题需要计算
    # 2. 调用 calculator 工具: 1500 - (3 * 299)
    # 3. 基于计算结果给出答案
    response2 = agent.invoke({"input": query2})
    print(f"Agent: {response2['output']}")
    
    # 示例 3: 展示记忆功能
    print("\n【示例 3】多轮对话（展示记忆）")
    print("-" * 40)
    query3 = "我刚才问了什么问题？"
    print(f"用户: {query3}")
    
    # 由于设置了 Memory，Agent 能记住之前的对话
    response3 = agent.invoke({"input": query3})
    print(f"Agent: {response3['output']}")
    
    print("\n" + "=" * 60)
    print("示例结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
