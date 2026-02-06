"""
Haystack Agent 示例
==================

Haystack 是一个专注于 NLP 和搜索的 Python 框架，核心理念是:
"构建强大的搜索和问答系统"

核心概念:
1. Pipeline (管道): 定义数据处理流程
2. Component (组件): 可复用的处理单元
3. Document Store (文档存储): 存储和检索文档
4. Retriever (检索器): 从文档存储中检索信息
5. Agent (代理): 自主决策的问答系统

特点:
- 模块化设计: 组件可自由组合
- 多模型支持: 支持各种 LLM 和 Embedding 模型
- 生产就绪: 企业级性能和扩展性
- 专注搜索: 在 RAG 场景表现出色

适用场景: 文档问答、企业搜索、知识库系统、RAG 应用
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


# ============================================================
# 第一部分: 数据模型
# ============================================================

@dataclass
class Document:
    """
    Haystack Document
    
    文档的基本单元，包含内容和元数据
    """
    id: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"doc_{datetime.now().timestamp()}"


@dataclass
class Answer:
    """
    答案对象
    
    包含回答内容和来源信息
    """
    answer: str
    query: str
    documents: List[Document] = field(default_factory=list)
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 第二部分: 文档存储
# ============================================================

class InMemoryDocumentStore:
    """
    内存文档存储
    
    实际使用中会使用:
    - ElasticsearchDocumentStore
    - PineconeDocumentStore
    - WeaviateDocumentStore
    - QdrantDocumentStore
    """
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.indexed = False
    
    def write_documents(self, documents: List[Document]):
        """写入文档"""
        for doc in documents:
            self.documents[doc.id] = doc
        print(f"[DocumentStore] 写入 {len(documents)} 个文档")
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)
    
    def get_all_documents(self) -> List[Document]:
        """获取所有文档"""
        return list(self.documents.values())
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        简单搜索实现
        
        实际使用会使用向量相似度搜索
        """
        results = []
        query_lower = query.lower()
        
        for doc in self.documents.values():
            # 简单的关键词匹配
            if query_lower in doc.content.lower():
                results.append(doc)
        
        return results[:top_k]


# ============================================================
# 第三部分: 组件 (Components)
# ============================================================

class Component:
    """
    组件基类
    
    所有 Haystack 组件的基类
    """
    
    def __init__(self, name: str):
        self.name = name
        self.inputs = []
        self.outputs = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """执行组件"""
        raise NotImplementedError


class Retriever(Component):
    """
    检索器
    
    从文档存储中检索相关文档
    """
    
    def __init__(self, document_store: InMemoryDocumentStore, top_k: int = 5):
        super().__init__("Retriever")
        self.document_store = document_store
        self.top_k = top_k
    
    def run(self, query: str) -> Dict[str, Any]:
        """执行检索"""
        print(f"[Retriever] 检索: '{query}'")
        
        documents = self.document_store.search(query, top_k=self.top_k)
        
        print(f"[Retriever] 找到 {len(documents)} 个相关文档")
        for i, doc in enumerate(documents, 1):
            preview = doc.content[:50] + "..." if len(doc.content) > 50 else doc.content
            print(f"  [{i}] {preview}")
        
        return {"documents": documents}


class PromptNode(Component):
    """
    提示节点
    
    使用 LLM 生成回答
    """
    
    def __init__(self, model_name: str = "default", default_prompt_template: Optional[str] = None):
        super().__init__("PromptNode")
        self.model_name = model_name
        self.default_prompt_template = default_prompt_template or (
            "基于以下文档回答问题:\n\n"
            "文档:\n"
            "{% for doc in documents %}\n"
            "{{ doc.content }}\n"
            "{% endfor %}\n\n"
            "问题: {{ query }}\n\n"
            "回答:"
        )
    
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """生成回答"""
        print(f"[PromptNode] 基于 {len(documents)} 个文档生成回答")
        
        # 构建提示词
        prompt = self._build_prompt(query, documents)
        
        # 模拟 LLM 调用
        answer = self._generate_answer(prompt, query, documents)
        
        return {
            "answers": [Answer(answer=answer, query=query, documents=documents)]
        }
    
    def _build_prompt(self, query: str, documents: List[Document]) -> str:
        """构建提示词"""
        doc_text = "\n".join([f"- {doc.content[:200]}" for doc in documents])
        return f"""基于以下信息回答问题:

{doc_text}

问题: {query}

请提供准确、简洁的回答:"""
    
    def _generate_answer(self, prompt: str, query: str, documents: List[Document]) -> str:
        """生成回答（模拟）"""
        # 实际实现会调用 LLM API
        return f"基于检索到的 {len(documents)} 个文档，{query} 的答案是：这是一个相关的重要信息..."


class Reader(Component):
    """
    阅读器
    
    从文档中提取答案
    """
    
    def __init__(self, model_name: str = "default"):
        super().__init__("Reader")
        self.model_name = model_name
    
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """从文档中提取答案"""
        print(f"[Reader] 从 {len(documents)} 个文档中提取答案")
        
        answers = []
        for doc in documents:
            # 简化的答案提取
            if any(keyword in doc.content.lower() for keyword in query.lower().split()):
                answer_text = f"根据文档 '{doc.id[:20]}...': 找到相关信息"
                answers.append(Answer(
                    answer=answer_text,
                    query=query,
                    documents=[doc],
                    score=0.8
                ))
        
        if not answers:
            answers.append(Answer(
                answer="未找到确切答案",
                query=query,
                documents=documents[:1],
                score=0.0
            ))
        
        return {"answers": answers}


# ============================================================
# 第四部分: Pipeline (管道)
# ============================================================

class Pipeline:
    """
    Haystack Pipeline
    
    将多个组件连接成处理流程
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.components: Dict[str, Component] = {}
        self.connections: List[tuple] = []
    
    def add_node(self, name: str, component: Component, inputs: List[str] = None):
        """
        添加节点
        
        参数:
            name: 节点名称
            component: 组件实例
            inputs: 输入节点列表
        """
        self.components[name] = component
        if inputs:
            for input_node in inputs:
                self.connections.append((input_node, name))
    
    def run(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """
        执行管道
        
        参数:
            query: 查询字符串
            debug: 是否输出调试信息
            
        返回:
            执行结果
        """
        print(f"\n[Pipeline] 执行: {self.name}")
        print(f"[Pipeline] 查询: '{query}'")
        
        # 简化的执行逻辑
        # 实际实现会按拓扑排序执行组件
        
        results = {"query": query}
        
        # 顺序执行所有组件
        for name, component in self.components.items():
            print(f"\n[Pipeline] 执行组件: {name}")
            
            # 准备输入
            kwargs = self._prepare_input(name, results)
            
            # 执行组件
            output = component.run(**kwargs)
            
            # 保存结果
            results[name] = output
        
        return results
    
    def _prepare_input(self, node_name: str, results: Dict) -> Dict:
        """准备组件输入"""
        # 简化的输入准备
        # 实际实现会根据连接关系传递数据
        
        kwargs = {}
        
        if "query" in results:
            kwargs["query"] = results["query"]
        
        # 传递前一个组件的输出
        for key, value in results.items():
            if key != "query":
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k not in kwargs:
                            kwargs[k] = v
        
        return kwargs
    
    def draw(self) -> str:
        """绘制管道结构"""
        diagram = f"Pipeline: {self.name}\n"
        diagram += "=" * 50 + "\n"
        
        for i, (name, component) in enumerate(self.components.items(), 1):
            diagram += f"[{i}] {name}: {component.name}\n"
        
        return diagram


# ============================================================
# 第五部分: Agent (代理)
# ============================================================

class HaystackAgent:
    """
    Haystack Agent
    
    基于工具调用的自主 Agent
    """
    
    def __init__(self, prompt_node: PromptNode, tools: List[Callable] = None, max_steps: int = 5):
        self.prompt_node = prompt_node
        self.tools = tools or []
        self.max_steps = max_steps
        self.memory: List[Dict] = []
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        运行 Agent
        
        Agent 会自主决定:
        1. 是否需要使用工具
        2. 使用哪个工具
        3. 何时给出最终答案
        """
        print(f"\n[Agent] 开始处理查询: '{query}'")
        print(f"[Agent] 可用工具: {[tool.__name__ for tool in self.tools]}")
        
        step = 0
        final_answer = None
        
        while step < self.max_steps:
            step += 1
            print(f"\n[Agent] 步骤 {step}/{self.max_steps}")
            
            # 决定下一步行动
            action = self._decide_action(query)
            
            if action["type"] == "final_answer":
                final_answer = action["answer"]
                break
            elif action["type"] == "tool_call":
                result = self._execute_tool(action["tool_name"], action["tool_input"])
                self.memory.append({
                    "step": step,
                    "action": action,
                    "result": result
                })
                print(f"[Agent] 工具结果: {result}")
        
        if final_answer is None:
            final_answer = self._generate_answer(query)
        
        return {
            "query": query,
            "answer": final_answer,
            "steps": step,
            "memory": self.memory
        }
    
    def _decide_action(self, query: str) -> Dict:
        """决定下一步行动"""
        # 简化的决策逻辑
        # 实际实现会使用 LLM 做决策
        
        if "搜索" in query or "查找" in query or "search" in query.lower():
            return {
                "type": "tool_call",
                "tool_name": "search",
                "tool_input": query
            }
        
        # 默认给出答案
        return {
            "type": "final_answer",
            "answer": f"关于 '{query}' 的答案是：这是一个复杂的问题，需要进一步研究..."
        }
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> Any:
        """执行工具"""
        for tool in self.tools:
            if tool.__name__ == tool_name or tool_name in tool.__name__:
                return tool(tool_input)
        return f"工具 {tool_name} 未找到"
    
    def _generate_answer(self, query: str) -> str:
        """生成最终答案"""
        return f"基于我的分析，{query} 的答案是：这是一个值得关注的话题..."


# ============================================================
# 第六部分: 使用示例
# ============================================================

def example_1_basic_qa():
    """示例 1: 基础问答系统"""
    print("=" * 70)
    print("示例 1: Haystack 基础问答系统")
    print("=" * 70)
    
    # 准备文档
    documents = [
        Document(
            id="doc1",
            content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
            meta={"category": "programming"}
        ),
        Document(
            id="doc2",
            content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            meta={"category": "ai"}
        ),
        Document(
            id="doc3",
            content="Docker 是一个开源的容器化平台，用于开发、部署和运行应用。",
            meta={"category": "devops"}
        ),
        Document(
            id="doc4",
            content="Python 在数据科学和机器学习领域非常流行，拥有丰富的库。",
            meta={"category": "programming"}
        )
    ]
    
    # 创建文档存储
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(documents)
    
    # 创建检索器
    retriever = Retriever(doc_store, top_k=2)
    
    # 创建阅读器
    reader = Reader()
    
    # 创建管道
    pipeline = Pipeline("基础问答管道")
    pipeline.add_node("retriever", retriever, inputs=["Query"])
    pipeline.add_node("reader", reader, inputs=["retriever"])
    
    # 执行查询
    query = "Python 是什么？"
    results = pipeline.run(query)
    
    print(f"\n查询结果:")
    if "reader" in results:
        answers = results["reader"].get("answers", [])
        for i, ans in enumerate(answers, 1):
            print(f"  [{i}] {ans.answer} (score: {ans.score})")


def example_2_rag_pipeline():
    """示例 2: RAG 管道"""
    print("\n" + "=" * 70)
    print("示例 2: RAG (检索增强生成) 管道")
    print("=" * 70)
    
    # 准备文档
    documents = [
        Document(
            id="product1",
            content="""
            智能助手 Pro 是一款企业级 AI 助手，具有以下特点：
            - 支持多轮对话
            - 集成知识库
            - 支持自定义工具
            价格：$99/月/用户
            """,
            meta={"type": "product", "name": "智能助手 Pro"}
        ),
        Document(
            id="product2",
            content="""
            数据分析平台提供实时数据分析和可视化功能：
            - 支持多种数据源
            - 拖拽式报表设计
            - AI 驱动的洞察
            价格：$199/月/用户
            """,
            meta={"type": "product", "name": "数据分析平台"}
        ),
    ]
    
    # 创建文档存储和组件
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(documents)
    
    retriever = Retriever(doc_store, top_k=2)
    
    # 使用 PromptNode 生成回答
    prompt_template = """
基于以下产品信息回答用户问题：

{% for doc in documents %}
产品: {{ doc.meta.name }}
{{ doc.content }}
{% endfor %}

用户问题: {{ query }}

请提供准确、有帮助的回答。如果信息不足，请说明。
"""
    prompt_node = PromptNode(default_prompt_template=prompt_template)
    
    # 创建管道
    rag_pipeline = Pipeline("RAG 管道")
    rag_pipeline.add_node("retriever", retriever)
    rag_pipeline.add_node("generator", prompt_node)
    
    # 查询
    queries = [
        "智能助手 Pro 多少钱？",
        "数据分析平台有什么功能？"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        results = rag_pipeline.run(query)
        
        if "generator" in results:
            answers = results["generator"].get("answers", [])
            for ans in answers:
                print(f"助手: {ans.answer}")


def example_3_agent_with_tools():
    """示例 3: 带工具的 Agent"""
    print("\n" + "=" * 70)
    print("示例 3: Haystack Agent 与工具调用")
    print("=" * 70)
    
    # 定义工具
    def search_tool(query: str) -> str:
        """搜索工具"""
        return f"搜索 '{query}' 的结果: 找到 3 个相关文档"
    
    def calculator_tool(expression: str) -> str:
        """计算器工具"""
        try:
            # 安全计算
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except:
            return "计算错误"
    
    def weather_tool(location: str) -> str:
        """天气查询工具"""
        return f"{location} 今天的天气：晴朗，25°C"
    
    # 创建 Agent
    prompt_node = PromptNode()
    agent = HaystackAgent(
        prompt_node=prompt_node,
        tools=[search_tool, calculator_tool, weather_tool],
        max_steps=3
    )
    
    # 执行任务
    tasks = [
        "搜索 Python 编程教程",
        "计算 123 + 456",
        "北京今天天气怎么样？"
    ]
    
    for task in tasks:
        result = agent.run(task)
        print(f"\n任务: {task}")
        print(f"答案: {result['answer']}")
        print(f"执行步骤: {result['steps']}")


def example_4_complex_pipeline():
    """示例 4: 复杂管道"""
    print("\n" + "=" * 70)
    print("示例 4: 复杂文档处理管道")
    print("=" * 70)
    
    # 准备大量文档
    documents = [
        Document(
            id=f"doc_{i}",
            content=f"这是第 {i} 个文档的内容，关于主题 {i % 3}...",
            meta={"topic": f"topic_{i % 3}", "index": i}
        )
        for i in range(10)
    ]
    
    # 创建存储和组件
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(documents)
    
    # 创建检索器
    retriever = Retriever(doc_store, top_k=3)
    
    # 创建阅读器
    reader = Reader()
    
    # 创建管道
    complex_pipeline = Pipeline("复杂处理管道")
    complex_pipeline.add_node("retriever", retriever)
    complex_pipeline.add_node("reader", reader)
    
    # 显示管道结构
    print("\n管道结构:")
    print(complex_pipeline.draw())
    
    # 执行
    results = complex_pipeline.run("主题 1")
    
    print("\n处理完成!")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("Haystack Agent 示例")
    print("===================")
    print()
    print("专注 NLP 和搜索的 Python 框架")
    print("特点: 模块化设计、专注 RAG、生产就绪")
    print()
    
    # 运行示例
    example_1_basic_qa()
    example_2_rag_pipeline()
    example_3_agent_with_tools()
    example_4_complex_pipeline()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成")
    print("=" * 70)
    print()
    print("Haystack 架构概览:")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                      Haystack 架构                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                              │")
    print("│   ┌─────────────────────────────────────────────────────┐    │")
    print("│   │                   Document Store                     │    │")
    print("│   │  (Elasticsearch / Pinecone / Weaviate / Qdrant)     │    │")
    print("│   └──────────────┬──────────────────────────────────────┘    │")
    print("│                  │                                            │")
    print("│   ┌──────────────▼──────────────┐                            │")
    print("│   │         Pipeline            │                            │")
    print("│   │  ┌─────────┐ ┌─────────┐   │                            │")
    print("│   │  │ Retriever│ │ Reader  │   │                            │")
    print("│   │  └────┬────┘ └────┬────┘   │                            │")
    print("│   │       └───────────┘         │                            │")
    print("│   │  ┌─────────┐ ┌─────────┐   │                            │")
    print("│   │  │PromptNode│ │  Agent  │   │                            │")
    print("│   │  └─────────┘ └─────────┘   │                            │")
    print("│   └──────────────┬──────────────┘                            │")
    print("│                  │                                            │")
    print("│   ┌──────────────▼──────────────┐                            │")
    print("│   │           Output            │                            │")
    print("│   └─────────────────────────────┘                            │")
    print("│                                                              │")
    print("└─────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
