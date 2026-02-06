"""
Microsoft AutoGen 多 Agent 对话示例
==================================

AutoGen 是微软开源的多 Agent 框架，核心思想是:
"通过对话(Conversation)让多个 Agent 协作解决问题"

核心概念:
1. ConversableAgent: 可对话的 Agent，能发送和接收消息
2. UserProxyAgent: 用户代理，代表人类用户参与对话
3. AssistantAgent: AI 助手，执行任务的 Agent
4. GroupChat: 群聊，多个 Agent 参与的对话组

特点:
- 基于对话的协作: Agent 通过自然语言交流
- 代码执行: Assistant 可以生成代码，UserProxy 可以执行代码
- 人机协作: 可以在关键节点让人类介入
- 灵活的角色定义: 可以定义任意角色的 Agent

适用场景: 代码生成与调试、复杂问题求解、多轮对话任务
"""

import autogen
from typing import Dict, List


# ============================================================
# 第一部分: 基础配置
# ============================================================

def get_llm_config():
    """
    配置 LLM 参数
    
    AutoGen 支持多种后端，包括 OpenAI、Azure 等
    """
    return {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "api_key": "your-api-key-here",  # 请替换为你的 API Key
                # "base_url": "https://api.openai.com/v1",  # 可选：自定义 API 地址
            }
        ],
        "temperature": 0.7,
        "timeout": 120,
    }


# ============================================================
# 第二部分: 基础对话示例
# ============================================================

def basic_conversation_example():
    """
    基础对话示例
    
    最简单的 AutoGen 使用场景:
    - 一个用户代理 (UserProxyAgent)
    - 一个 AI 助手 (AssistantAgent)
    - 两者进行一对一对话
    """
    print("=" * 70)
    print("基础对话示例")
    print("=" * 70)
    
    llm_config = get_llm_config()
    
    # ========== 创建 AssistantAgent ==========
    # AI 助手 Agent，负责回答问题、生成代码等
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=(
            "你是一个有帮助的 AI 助手。\n\n"
            "你的职责:\n"
            "1. 回答用户的问题\n"
            "2. 如果需要写代码，使用 Python\n"
            "3. 代码要简洁、有注释、易于理解\n"
            '4. 如果任务完成，回复 "TERMINATE"'
        )
    )

    # ========== 创建 UserProxyAgent ==========
    # 用户代理 Agent，代表人类用户
    # 可以执行代码、调用函数等
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # 不询问人类输入
        max_consecutive_auto_reply=10,  # 最大自动回复次数
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        # 代码执行配置
        code_execution_config={
            "work_dir": "coding",  # 代码执行的工作目录
            "use_docker": False,   # 不使用 Docker（生产环境建议开启）
        },
    )

    # ========== 开始对话 ==========
    # initiate_chat 启动对话
    # message: 用户发送的第一条消息
    message = (
        "请帮我写一个 Python 函数，计算斐波那契数列的前 n 个数，"
        "并测试 n=10 的情况。要求:\n"
        "1. 使用递归方式实现\n"
        "2. 处理边界情况（n <= 0）\n"
        "3. 添加函数文档字符串"
    )
    
    print(f"\n用户: {message}\n")
    
    # 开始对话
    # user_proxy 发起对话，assistant 回应
    # 对话会持续进行，直到满足终止条件
    user_proxy.initiate_chat(
        assistant,
        message=message,
    )
    
    return user_proxy, assistant


# ============================================================
# 第三部分: 群聊示例 (GroupChat)
# ============================================================

def group_chat_example():
    """
    群聊示例
    
    多个 Agent 参与的群组对话，适合复杂的协作场景。
    
    场景: 软件开发团队
    - ProductManager: 产品经理，提需求
    - Developer: 开发工程师，写代码
    - CodeReviewer: 代码审查员，审查代码
    - Tester: 测试工程师，测试代码
    """
    print("\n" + "=" * 70)
    print("群聊示例 - 软件开发团队")
    print("=" * 70)
    
    llm_config = get_llm_config()
    
    # ========== 定义各个角色的 Agent ==========
    
    # 产品经理
    product_manager = autogen.AssistantAgent(
        name="product_manager",
        llm_config=llm_config,
        system_message="""你是产品经理。

职责:
1. 明确产品需求，确保需求清晰可执行
2. 回答开发过程中的需求疑问
3. 验收最终成果
4. 如果任务完成，说 "TERMINATE"

沟通风格: 简洁明了，关注用户体验和业务价值。
        """
    )
    
    # 开发工程师
    developer = autogen.AssistantAgent(
        name="developer",
        llm_config=llm_config,
        system_message="""你是高级 Python 开发工程师。

职责:
1. 根据需求编写高质量代码
2. 使用最佳实践和设计模式
3. 代码要有清晰的注释
4. 遵循 PEP8 规范

技能: Python, 数据结构, 算法, 软件设计
        """
    )
    
    # 代码审查员
    code_reviewer = autogen.AssistantAgent(
        name="code_reviewer",
        llm_config=llm_config,
        system_message="""你是资深代码审查员。

职责:
1. 审查代码质量和规范性
2. 检查潜在的错误和安全隐患
3. 提出改进建议
4. 确保代码符合最佳实践

审查重点: 代码风格、逻辑正确性、性能、安全性
        """
    )
    
    # 测试工程师
    tester = autogen.AssistantAgent(
        name="tester",
        llm_config=llm_config,
        system_message="""你是测试工程师。

职责:
1. 设计测试用例
2. 执行测试并记录结果
3. 报告发现的 bug
4. 验证修复是否有效

测试方法: 单元测试、边界测试、异常测试
        """
    )
    
    # 用户代理（人类代表）
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config={
            "work_dir": "group_coding",
            "use_docker": False,
        },
    )
    
    # ========== 创建群聊 ==========
    
    # GroupChat: 管理多个 Agent 的对话
    groupchat = autogen.GroupChat(
        agents=[user_proxy, product_manager, developer, code_reviewer, tester],
        messages=[],  # 初始为空消息列表
        max_round=20,  # 最多 20 轮对话
    )
    
    # GroupChatManager: 管理群聊的消息流转
    # 决定哪个 Agent 在什么时候发言
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="""你是群聊管理员。

职责:
1. 根据对话上下文，选择下一个发言的 Agent
2. 确保讨论有序进行
3. 在讨论偏离主题时引导回正轨
4. 当任务完成时结束对话

选发言者的原则:
- 产品经理先发言，明确需求
- 开发者负责实现
- 审查员在代码完成后审查
- 测试员在代码审查后测试
        """
    )
    
    # ========== 启动群聊 ==========
    
    task = """
    团队任务: 开发一个用户注册系统
    
    需求:
    1. 用户输入用户名、邮箱、密码
    2. 验证输入的有效性
       - 用户名: 3-20 个字符，只能包含字母数字下划线
       - 邮箱: 有效的邮箱格式
       - 密码: 至少 8 位，包含字母和数字
    3. 检查用户名和邮箱是否已存在（模拟）
    4. 返回注册成功或失败的信息
    
    工作流程:
    1. 产品经理明确需求
    2. 开发者编写代码
    3. 审查员审查代码
    4. 测试员设计并执行测试用例
    5. 产品经理验收
    
    开始吧！
    """
    
    print(f"\n任务: {task}\n")
    print("开始群聊...\n")
    
    # 用户代理发起群聊
    user_proxy.initiate_chat(
        manager,
        message=task,
    )
    
    return groupchat


# ============================================================
# 第四部分: 自定义 Agent 行为
# ============================================================

def custom_agent_example():
    """
    自定义 Agent 行为示例
    
    展示如何:
    1. 自定义回复函数
    2. 注册自定义工具/函数
    3. 控制对话流程
    """
    print("\n" + "=" * 70)
    print("自定义 Agent 行为示例")
    print("=" * 70)
    
    llm_config = get_llm_config()
    
    # ========== 注册自定义函数 ==========
    
    # 定义可被 Agent 调用的函数
    def search_database(query: str) -> str:
        """
        模拟数据库查询
        实际应用中可以连接真实数据库
        """
        # 模拟数据库
        mock_db = {
            "用户": "共有 10000 名注册用户",
            "订单": "本月订单量 5000 单",
            "产品": "在售产品 200 款"
        }
        return mock_db.get(query, f"未找到 '{query}' 的相关信息")
    
    def calculate_statistics(data: List[float]) -> Dict:
        """
        计算统计信息
        """
        if not data:
            return {"error": "数据为空"}
        
        import statistics
        return {
            "count": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "stdev": statistics.stdev(data) if len(data) > 1 else 0
        }
    
    # ========== 创建带自定义函数的 Agent ==========
    
    # 使用 function_map 注册函数
    assistant = autogen.AssistantAgent(
        name="data_analyst",
        llm_config={
            **llm_config,
            "functions": [
                {
                    "name": "search_database",
                    "description": "查询数据库获取信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "查询关键词"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "calculate_statistics",
                    "description": "计算数据集的统计信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "数值列表"
                            }
                        },
                        "required": ["data"]
                    }
                }
            ]
        },
        system_message="""你是数据分析助手。

你可以:
1. 使用 search_database 查询数据库
2. 使用 calculate_statistics 计算统计数据

工作流程:
1. 先查询需要的数据
2. 然后进行统计分析
3. 最后给出洞察和建议

如果任务完成，回复 "TERMINATE"。
        """
    )
    
    # 用户代理，注册函数实现
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        function_map={
            "search_database": search_database,
            "calculate_statistics": calculate_statistics
        }
    )
    
    # ========== 开始对话 ==========
    
    message = """
    请帮我:
    1. 查询一下我们有多少用户
    2. 计算以下数据的统计信息: [23, 45, 67, 89, 34, 56, 78, 90, 12, 34]
    3. 给出简要分析
    """
    
    print(f"\n用户: {message}\n")
    
    user_proxy.initiate_chat(
        assistant,
        message=message
    )
    
    return user_proxy, assistant


# ============================================================
# 第五部分: 主函数
# ============================================================

def main():
    """
    主函数：运行 AutoGen 示例
    
    可以取消注释不同的示例来体验不同功能
    """
    # 示例 1: 基础对话
    print("\n" + "=" * 70)
    print("AutoGen 多 Agent 对话示例")
    print("=" * 70)
    
    # 运行基础示例
    basic_conversation_example()
    
    # 运行群聊示例（需要更多 Token）
    # group_chat_example()
    
    # 运行自定义 Agent 示例
    # custom_agent_example()
    
    print("\n" + "=" * 70)
    print("示例结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
