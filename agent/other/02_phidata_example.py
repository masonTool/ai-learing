"""
Phidata AI 助手示例
==================

Phidata 是一个轻量级的 AI 助手框架，核心理念是:
"用简单的代码构建功能强大的 AI 助手"

核心概念:
1. Assistant (助手): 基础的 AI 助手
2. LLM (大语言模型): 支持 OpenAI、Claude、本地模型等
3. Tools (工具): 助手可以调用的功能
4. Knowledge (知识库): 助手可以访问的文档
5. Storage (存储): 保存对话历史

特点:
- 简洁易用: API 设计简单直观
- 快速开发: 几行代码就能构建助手
- 灵活扩展: 易于添加自定义工具
- 生产就绪: 内置监控、日志、缓存

适用场景: 快速原型开发、轻量级应用、嵌入式助手
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


# ============================================================
# 第一部分: Phidata 核心概念
# ============================================================

@dataclass
class Message:
    """消息结构"""
    role: str  # user / assistant / system
    content: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class Tool:
    """
    工具基类
    
    Phidata 中工具是助手能力的扩展
    """
    def __init__(self, name: str, description: str, func=None):
        self.name = name
        self.description = description
        self.func = func
    
    def run(self, **kwargs) -> str:
        """执行工具"""
        if self.func:
            return self.func(**kwargs)
        return "工具执行完成"


class KnowledgeBase:
    """
    知识库
    
    存储助手可以访问的知识
    """
    def __init__(self):
        self.documents: Dict[str, str] = {}
    
    def add_document(self, name: str, content: str):
        """添加文档"""
        self.documents[name] = content
    
    def search(self, query: str) -> List[Dict]:
        """搜索知识库"""
        results = []
        query_lower = query.lower()
        for name, content in self.documents.items():
            if query_lower in content.lower() or query_lower in name.lower():
                results.append({
                    "name": name,
                    "content": content[:200] + "..."
                })
        return results


# ============================================================
# 第二部分: Phidata Assistant 实现
# ============================================================

class PhiAssistant:
    """
    Phidata 风格的 AI 助手
    
    这是一个简化实现，展示核心概念
    """
    
    def __init__(
        self,
        name: str = "助手",
        model: str = "gpt-4o-mini",
        instructions: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        show_tool_calls: bool = True
    ):
        """
        初始化助手
        
        参数:
            name: 助手名称
            model: 使用的模型
            instructions: 系统指令
            tools: 工具列表
            knowledge_base: 知识库
            show_tool_calls: 是否显示工具调用过程
        """
        self.name = name
        self.model = model
        self.instructions = instructions or "你是一个有帮助的 AI 助手"
        self.tools = tools or []
        self.knowledge_base = knowledge_base
        self.show_tool_calls = show_tool_calls
        
        # 对话历史
        self.messages: List[Message] = []
        
        # 运行统计
        self.run_count = 0
        self.tool_calls_count = 0
    
    def add_tool(self, tool: Tool):
        """添加工具"""
        self.tools.append(tool)
    
    def chat(self, message: str) -> str:
        """
        与助手对话
        
        参数:
            message: 用户消息
            
        返回:
            助手回复
        """
        # 记录用户消息
        self.messages.append(Message(role="user", content=message))
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 决定是否调用工具
        if self.tools and self._should_use_tool(message):
            response = self._handle_with_tools(message, system_prompt)
        else:
            response = self._generate_response(message, system_prompt)
        
        # 记录助手回复
        self.messages.append(Message(role="assistant", content=response))
        self.run_count += 1
        
        return response
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        prompt = f"{self.instructions}\n\n"
        
        # 添加工具信息
        if self.tools:
            prompt += "你有以下工具可以使用:\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            prompt += "\n"
        
        # 添加知识库信息
        if self.knowledge_base:
            prompt += "你可以访问知识库来回答相关问题。\n\n"
        
        return prompt
    
    def _should_use_tool(self, message: str) -> bool:
        """判断是否应该使用工具"""
        # 简单启发式判断
        tool_keywords = [
            "搜索", "计算", "查询", "查", "搜索", "找",
            "search", "calculate", "look up", "find"
        ]
        return any(kw in message.lower() for kw in tool_keywords)
    
    def _handle_with_tools(self, message: str, system_prompt: str) -> str:
        """使用工具处理"""
        # 选择合适的工具（简化版）
        selected_tool = None
        for tool in self.tools:
            if tool.name.lower() in message.lower():
                selected_tool = tool
                break
        
        # 如果没有匹配，使用第一个工具
        if not selected_tool and self.tools:
            selected_tool = self.tools[0]
        
        if selected_tool:
            if self.show_tool_calls:
                print(f"  [工具调用] {selected_tool.name}")
            
            # 执行工具
            tool_result = selected_tool.run(query=message)
            self.tool_calls_count += 1
            
            if self.show_tool_calls:
                print(f"  [工具结果] {tool_result[:100]}...")
            
            # 基于工具结果生成回复
            return f"根据查询结果: {tool_result}\n\n这是我找到的关于您提问的答案。"
        
        return self._generate_response(message, system_prompt)
    
    def _generate_response(self, message: str, system_prompt: str) -> str:
        """
        生成回复
        
        实际使用会调用 LLM API
        这里使用简化的模拟回复
        """
        # 检查知识库
        if self.knowledge_base:
            kb_results = self.knowledge_base.search(message)
            if kb_results:
                context = "\n".join([r["content"] for r in kb_results[:2]])
                return f"根据我的知识库:\n{context}\n\n希望这能帮到您！"
        
        # 默认回复
        return f"您好！我是 {self.name}。我收到了您的问题: '{message}'。我可以帮您解答各种问题。"
    
    def get_chat_history(self) -> List[Dict]:
        """获取对话历史"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.messages
        ]
    
    def print_chat_history(self):
        """打印对话历史"""
        print(f"\n{'='*50}")
        print(f"对话历史 ({len(self.messages)} 条消息)")
        print(f"{'='*50}")
        for msg in self.messages:
            print(f"\n[{msg.role.upper()}] {msg.timestamp}")
            print(f"{msg.content}")


# ============================================================
# 第三部分: 使用示例
# ============================================================

def example_1_basic_assistant():
    """示例 1: 基础助手"""
    print("\n" + "=" * 70)
    print("示例 1: 基础助手")
    print("=" * 70)
    
    # 创建基础助手
    assistant = PhiAssistant(
        name="基础助手",
        instructions="你是一个友好的 AI 助手，乐于助人。"
    )
    
    # 对话
    messages = [
        "你好！",
        "今天天气怎么样？",
        "你能帮我做什么？"
    ]
    
    for msg in messages:
        print(f"\n用户: {msg}")
        response = assistant.chat(msg)
        print(f"助手: {response}")
    
    # 显示统计
    print(f"\n统计: 运行 {assistant.run_count} 次")


def example_2_assistant_with_tools():
    """示例 2: 带工具的助手"""
    print("\n" + "=" * 70)
    print("示例 2: 带工具的助手")
    print("=" * 70)
    
    # 定义工具函数
    def search_web(query: str) -> str:
        """模拟网络搜索"""
        return f"搜索 '{query}' 的结果: Python 是最流行的编程语言之一。"
    
    def calculate(expression: str) -> str:
        """计算工具"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except:
            return "计算失败"
    
    def get_time() -> str:
        """获取当前时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建工具
    tools = [
        Tool("search", "搜索互联网信息", search_web),
        Tool("calculate", "执行数学计算", calculate),
        Tool("time", "获取当前时间", lambda: get_time())
    ]
    
    # 创建带工具的助手
    assistant = PhiAssistant(
        name="工具助手",
        instructions="你是一个有帮助的助手，可以使用工具来解决问题。",
        tools=tools,
        show_tool_calls=True
    )
    
    # 对话
    messages = [
        "搜索一下 Python 的信息",
        "计算 123 * 456",
        "现在几点了？"
    ]
    
    for msg in messages:
        print(f"\n用户: {msg}")
        response = assistant.chat(msg)
        print(f"助手: {response}")
    
    print(f"\n统计: 运行 {assistant.run_count} 次, 工具调用 {assistant.tool_calls_count} 次")


def example_3_assistant_with_knowledge():
    """示例 3: 带知识库的助手"""
    print("\n" + "=" * 70)
    print("示例 3: 带知识库的助手")
    print("=" * 70)
    
    # 创建知识库
    kb = KnowledgeBase()
    kb.add_document(
        "公司产品介绍",
        """
        我们公司的主要产品是智能助手平台。
        功能包括: 自然语言对话、任务自动化、数据分析。
        定价: 基础版免费，专业版每月 $29，企业版需联系销售。
        """
    )
    kb.add_document(
        "技术支持",
        """
        技术支持邮箱: support@company.com
        工作时间: 周一至周五 9:00-18:00
        响应时间: 一般问题 24 小时内回复，紧急问题 2 小时内回复。
        """
    )
    
    # 创建带知识库的助手
    assistant = PhiAssistant(
        name="客服助手",
        instructions="你是公司的客服助手，基于知识库回答用户问题。",
        knowledge_base=kb
    )
    
    # 对话
    messages = [
        "你们公司有什么产品？",
        "怎么联系技术支持？",
        "产品多少钱？"
    ]
    
    for msg in messages:
        print(f"\n用户: {msg}")
        response = assistant.chat(msg)
        print(f"助手: {response}")


def example_4_data_analyst_assistant():
    """示例 4: 数据分析师助手"""
    print("\n" + "=" * 70)
    print("示例 4: 数据分析师助手")
    print("=" * 70)
    
    # 模拟数据
    sample_data = {
        "sales": [100, 150, 200, 180, 220, 250, 300],
        "dates": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    }
    
    # 数据分析工具
    def analyze_data(metric: str) -> str:
        """分析数据"""
        if metric == "sales":
            total = sum(sample_data["sales"])
            avg = total / len(sample_data["sales"])
            max_val = max(sample_data["sales"])
            min_val = min(sample_data["sales"])
            return f"销售数据: 总计 {total}, 平均 {avg:.1f}, 最高 {max_val}, 最低 {min_val}"
        return "未知指标"
    
    def get_data_summary() -> str:
        """获取数据摘要"""
        return f"数据集包含 {len(sample_data)} 个指标，每个指标 {len(sample_data['sales'])} 条记录"
    
    tools = [
        Tool("analyze", "分析指定指标", analyze_data),
        Tool("summary", "获取数据摘要", get_data_summary)
    ]
    
    # 创建数据分析师助手
    assistant = PhiAssistant(
        name="数据分析师",
        instructions="你是一个专业的数据分析师，帮助用户理解和分析数据。",
        tools=tools
    )
    
    # 对话
    messages = [
        "数据概览",
        "分析销售数据",
        "有什么建议？"
    ]
    
    for msg in messages:
        print(f"\n用户: {msg}")
        response = assistant.chat(msg)
        print(f"分析师: {response}")
    
    # 打印对话历史
    assistant.print_chat_history()


# ============================================================
# 第五部分: 主函数
# ============================================================

def main():
    """主函数"""
    print("=" * 70)
    print("Phidata AI 助手示例")
    print("=" * 70)
    
    # 运行示例
    example_1_basic_assistant()
    example_2_assistant_with_tools()
    example_3_assistant_with_knowledge()
    example_4_data_analyst_assistant()
    
    print("\n" + "=" * 70)
    print("所有示例结束")
    print("=" * 70)
    
    print("""
Phidata 核心特点:

┌─────────────────────────────────────────────────────────────┐
│                   Phidata 架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────┐                  │
│   │           Assistant                  │                  │
│   │  ┌─────────┐  ┌─────────┐          │                  │
│   │  │   LLM   │  │  Tools  │          │                  │
│   │  └─────────┘  └─────────┘          │                  │
│   │  ┌─────────┐  ┌─────────┐          │                  │
│   │  │Knowledge│  │ Storage │          │                  │
│   │  └─────────┘  └─────────┘          │                  │
│   └─────────────────────────────────────┘                  │
│                                                             │
│  特点:                                                       │
│  - 简洁的 API 设计                                          │
│  - 易于添加自定义工具                                        │
│  - 内置知识库支持                                           │
│  - 对话历史管理                                             │
│  - 生产就绪特性                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

适用场景:
- 快速原型开发
- 轻量级 AI 应用
- 嵌入式助手
- 客服机器人
""")


if __name__ == "__main__":
    main()
