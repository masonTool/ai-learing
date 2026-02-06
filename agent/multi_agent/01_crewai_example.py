"""
CrewAI 多 Agent 协作示例
=======================

CrewAI 是一个专注于多 Agent 协作的框架，灵感来自于企业团队组织。
核心理念是让多个具有不同角色的 Agent 像一个团队一样协作完成任务。

核心概念:
1. Agent (智能体): 具有特定角色、目标和背景的个体
2. Task (任务): 分配给 Agent 的具体工作
3. Crew (团队): 多个 Agent 和任务的集合
4. Process (流程): 任务执行的方式（顺序或并行）

角色定义要素:
- Role: 角色名称（如"研究员"、"作家"）
- Goal: 工作目标（要完成什么）
- Backstory: 背景故事（塑造 Agent 的行为风格）

适用场景: 内容创作、研究报告、软件开发、业务流程自动化
"""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


# ============================================================
# 第一部分: 创建不同角色的 Agent
# ============================================================

def create_agents():
    """
    创建三个不同角色的 Agent，模拟一个内容创作团队
    
    团队结构:
    1. 研究员 (Researcher): 负责收集和分析信息
    2. 作家 (Writer): 负责撰写内容
    3. 编辑 (Editor): 负责审核和润色
    """
    
    # 共享的 LLM 配置
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # ========== Agent 1: 研究员 ==========
    researcher = Agent(
        role="高级研究员",
        goal="深入研究主题，收集全面、准确的信息，为写作团队提供扎实的知识基础",
        backstory=(
            "你是一位经验丰富的研究员，在学术和商业研究领域工作了 15 年。"
            "你擅长从多个角度分析问题，善于发现关键信息和数据。"
            "你注重事实准确性，总是引用可靠的信息来源。"
            "你的研究风格严谨细致，能够为团队提供高质量的研究报告。"
        ),
        verbose=True,  # 打印详细的思考过程
        allow_delegation=False,  # 不允许委派任务给其他 Agent
        llm=llm
    )

    # ========== Agent 2: 作家 ==========
    writer = Agent(
        role="内容作家",
        goal="基于研究员提供的信息，创作引人入胜、结构清晰的文章内容",
        backstory=(
            "你是一位才华横溢的作家，曾为多家知名杂志和网站撰稿。"
            "你擅长将复杂的概念转化为通俗易懂的文字。"
            "你的写作风格生动有趣，善于讲故事，能够吸引读者的注意力。"
            "你注重文章的结构和逻辑，确保内容连贯流畅。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # ========== Agent 3: 编辑 ==========
    editor = Agent(
        role="内容编辑",
        goal="审核文章质量，确保内容准确、语言流畅、符合发布标准",
        backstory=(
            "你是一位资深编辑，在出版行业工作了 20 年。"
            "你对文字有着敏锐的洞察力，能够快速发现并修正问题。"
            "你注重细节，关注语法、拼写、标点等基础要素。"
            "同时你也关注文章的整体结构和表达效果，确保内容高质量。"
            "你以严格但公正著称，是团队质量的守护者。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return researcher, writer, editor


# ============================================================
# 第二部分: 定义任务
# ============================================================

def create_tasks(researcher, writer, editor, topic="人工智能在医疗领域的应用"):
    """
    为团队创建任务流程
    
    每个任务都有:
    - description: 任务描述（告诉 Agent 要做什么）
    - expected_output: 期望输出（告诉 Agent 输出格式）
    - agent: 执行任务的 Agent
    - context: 依赖的其他任务输出（可选）
    """
    
    # ========== 任务 1: 研究 ==========
    research_task = Task(
        description=f"""
        对以下主题进行深入研究：{topic}
        
        要求:
        1. 收集该主题的背景信息和发展历程
        2. 识别主要的发展趋势和关键数据
        3. 分析当前面临的挑战和机遇
        4. 提供至少 5 个关键要点，每个要点都要有具体例子支撑
        5. 列出相关的统计数据或研究报告
        
        输出格式:
        - 研究摘要（200字以内）
        - 关键发现（分点列出）
        - 数据支撑（具体数字和来源）
        """,
        expected_output="一份详细的研究报告，包含关键发现、数据支撑和结构化信息",
        agent=researcher
    )
    
    # ========== 任务 2: 写作（依赖研究结果）==========
    writing_task = Task(
        description=f"""
        基于研究员提供的资料，撰写一篇关于"{topic}"的文章。
        
        要求:
        1. 文章长度 800-1000 字
        2. 包含引人入胜的开头
        3. 正文分为 3-4 个段落，每段一个主题
        4. 使用研究员提供的数据和案例
        5. 结尾要有总结和展望
        6. 语言风格：专业但易懂，适合大众读者
        
        注意: 请使用上下文中的研究报告作为信息来源。
        """,
        expected_output="一篇结构完整、内容丰富的文章，800-1000字",
        agent=writer,
        context=[research_task]  # 依赖研究任务的输出
    )
    
    # ========== 任务 3: 编辑（依赖写作结果）==========
    editing_task = Task(
        description="""
        审核并润色作家提交的文章。
        
        审核要点:
        1. 内容准确性: 检查事实和数据是否正确
        2. 语言质量: 修正语法错误、拼写错误、标点问题
        3. 结构优化: 确保段落过渡自然，逻辑清晰
        4. 风格统一: 保持一致的语气和风格
        5. 字数检查: 确保在要求范围内
        
        输出:
        - 修改后的文章
        - 修改说明（列出主要改动）
        - 质量评分（1-10分）
        """,
        expected_output="修改后的最终文章、修改说明和质量评分",
        agent=editor,
        context=[writing_task]  # 依赖写作任务的输出
    )
    
    return [research_task, writing_task, editing_task]


# ============================================================
# 第三部分: 组装和执行团队
# ============================================================

def run_crew(topic="人工智能在医疗领域的应用"):
    """
    创建并运行 Crew（团队）
    
    流程说明:
    1. Crew 按照任务顺序执行（因为使用了 Sequential 流程）
    2. 每个任务的输出会自动传递给依赖它的下一个任务
    3. Agent 之间通过任务输出协作
    """
    
    print("=" * 70)
    print("CrewAI 多 Agent 协作示例")
    print("=" * 70)
    print(f"\n主题: {topic}")
    print("\n团队配置:")
    print("  - 研究员: 负责信息收集和分析")
    print("  - 作家: 负责内容创作")
    print("  - 编辑: 负责审核和润色")
    print("\n" + "-" * 70)
    
    # 1. 创建 Agent
    researcher, writer, editor = create_agents()
    
    # 2. 创建任务
    tasks = create_tasks(researcher, writer, editor, topic)
    
    # 3. 创建 Crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        process=Process.sequential,  # 顺序执行
        verbose=True  # 详细输出
    )
    
    # 4. 执行
    print("\n开始执行...\n")
    result = crew.kickoff()
    
    return result


# ============================================================
# 第四部分: 其他协作模式示例
# ============================================================

def parallel_example():
    """
    并行执行示例
    
    当任务之间没有依赖关系时，可以同时执行多个 Agent
    例如: 同时分析多个文档、并行处理多个数据源
    """
    print("\n" + "=" * 70)
    print("并行执行示例")
    print("=" * 70)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 创建三个独立的分析 Agent
    agents = []
    tasks = []
    
    for i, aspect in enumerate(["技术角度", "市场角度", "社会影响"]):
        agent = Agent(
            role=f"{aspect}分析师",
            goal=f"从{aspect}分析主题",
            backstory=f"你是{aspect}分析专家",
            llm=llm
        )
        
        task = Task(
            description=f"从{aspect}分析'电动汽车的发展趋势'",
            expected_output=f"{aspect}分析报告",
            agent=agent
        )
        
        agents.append(agent)
        tasks.append(task)
    
    # 并行执行
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.parallel  # 并行执行
    )
    
    # 注意: 实际使用时需要有返回结果合并机制
    print("三个 Agent 同时进行分析...")
    return crew.kickoff()


def hierarchical_example():
    """
    层级协作示例
    
    设置 manager_agent 作为协调者，动态分配任务
    适合任务分配不明确、需要动态调整的场景
    """
    print("\n" + "=" * 70)
    print("层级协作示例")
    print("=" * 70)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 创建项目经理 Agent
    manager = Agent(
        role="项目经理",
        goal="协调团队完成项目",
        backstory="你是经验丰富的项目经理，善于分配任务和协调资源",
        llm=llm,
        allow_delegation=True  # 允许委派任务
    )
    
    # 创建工作人员
    developer = Agent(
        role="开发工程师",
        goal="完成技术任务",
        backstory="你是全栈开发工程师",
        llm=llm
    )
    
    designer = Agent(
        role="UI设计师",
        goal="完成设计任务",
        backstory="你是资深 UI/UX 设计师",
        llm=llm
    )
    
    # 定义任务
    tasks = [
        Task(
            description="开发一个用户登录功能",
            expected_output="登录功能的代码",
            agent=developer
        ),
        Task(
            description="设计登录页面 UI",
            expected_output="UI 设计稿描述",
            agent=designer
        )
    ]
    
    # 使用层级流程，manager 负责协调
    crew = Crew(
        agents=[manager, developer, designer],
        tasks=tasks,
        process=Process.hierarchical,  # 层级流程
        manager_agent=manager
    )
    
    return crew.kickoff()


# ============================================================
# 第五部分: 主函数
# ============================================================

def main():
    """
    主函数：运行 CrewAI 示例
    """
    # 运行顺序流程示例
    result = run_crew("人工智能在教育领域的应用")
    
    print("\n" + "=" * 70)
    print("最终输出")
    print("=" * 70)
    print(result)
    
    # 可以取消注释运行其他示例
    # parallel_example()
    # hierarchical_example()


if __name__ == "__main__":
    main()
