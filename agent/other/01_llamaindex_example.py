"""
LlamaIndex RAG + Agent 示例
==========================

LlamaIndex 是一个专注于"数据检索与 LLM 结合"的框架，核心理念是:
"将外部数据高效地连接到 LLM，构建知识增强型应用"

核心概念:
1. Indexing (索引): 将数据转换为可搜索的索引
2. Querying (查询): 从索引中检索相关信息
3. RAG (Retrieval-Augmented Generation): 检索增强生成
4. Agent (代理): 结合工具的自主决策 Agent

数据索引类型:
- VectorStoreIndex: 向量存储索引，语义搜索
- TreeIndex: 树形索引，层级检索
- ListIndex: 列表索引，顺序检索
- KeywordTableIndex: 关键词表索引

特点:
- 数据连接器: 支持 100+ 数据源（文件、数据库、API等）
- 灵活的索引: 多种索引方式适应不同场景
- 强大的检索: 支持多种检索策略
- 与 Agent 结合: 可以作为 Agent 的知识库工具

适用场景: 知识库问答、文档分析、企业数据查询、RAG 应用
"""

from typing import List, Dict, Any, Optional
import os


# ============================================================
# 第一部分: 数据加载与索引
# ============================================================

def create_sample_documents():
    """
    创建示例文档
    
    实际使用中可以加载各种格式的文件:
    - PDF, Word, TXT
    - Markdown, HTML
    - JSON, CSV
    - 数据库、API 等
    """
    documents = [
        {
            "id": "doc1",
            "text": (
                "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。\n"
                "Python 的设计哲学强调代码的可读性和简洁性。\n"
                "Python 支持多种编程范式，包括面向对象、函数式和过程式编程。"
            ),
            "metadata": {"category": "编程语言", "author": "技术文档"}
        },
        {
            "id": "doc2",
            "text": (
                "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。\n"
                "常见的机器学习算法包括监督学习、无监督学习和强化学习。\n"
                "Python 是机器学习最流行的编程语言，拥有丰富的库如 scikit-learn、TensorFlow 和 PyTorch。"
            ),
            "metadata": {"category": "人工智能", "author": "技术文档"}
        },
        {
            "id": "doc3",
            "text": (
                "Docker 是一个开源的容器化平台，用于开发、部署和运行应用程序。\n"
                "容器是轻量级的、可移植的、自给自足的软件包。\n"
                "Docker 使应用程序的部署更加简单和一致。"
            ),
            "metadata": {"category": "DevOps", "author": "技术文档"}
        },
        {
            "id": "doc4",
            "text": (
                "微服务架构是一种将应用程序构建为一组小服务的方法。\n"
                "每个服务运行在自己的进程中，通过轻量级机制通信。\n"
                "微服务架构提高了系统的可扩展性和可维护性。"
            ),
            "metadata": {"category": "架构设计", "author": "技术文档"}
        }
    ]
    return documents


class SimpleVectorStore:
    """
    简化的向量存储
    
    实际 LlamaIndex 使用专业的向量数据库如:
    - Pinecone, Weaviate, Milvus
    - Chroma, FAISS
    - PostgreSQL with pgvector
    """
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict]):
        """添加文档到存储"""
        for doc in documents:
            # 简化的"嵌入"表示
            # 实际使用会调用 OpenAI 等 API 生成向量
            embedding = self._simple_embed(doc["text"])
            self.documents.append(doc)
            self.embeddings.append(embedding)
    
    def _simple_embed(self, text: str) -> List[float]:
        """
        简化的嵌入生成
        
        实际使用会调用 embedding API:
        - OpenAI: text-embedding-ada-002
        - HuggingFace: sentence-transformers
        """
        # 这里使用文本特征作为简化示例
        words = text.lower().split()
        # 创建一个简单的词频向量
        keywords = ["python", "机器学习", "docker", "微服务", "编程", "算法", "容器", "架构"]
        vector = []
        for keyword in keywords:
            count = sum(1 for word in words if keyword in word)
            vector.append(float(count))
        return vector
    
    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        语义搜索
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            最相关的文档列表
        """
        query_embedding = self._simple_embed(query)
        
        # 计算相似度（简化版余弦相似度）
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            # 简化的相似度计算
            score = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            scores.append((i, score))
        
        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx].copy()
            doc["score"] = score
            results.append(doc)
        
        return results


# ============================================================
# 第二部分: RAG 查询引擎
# ============================================================

class RAGQueryEngine:
    """
    RAG (检索增强生成) 查询引擎
    
    工作流程:
    1. 接收用户查询
    2. 从向量存储检索相关文档
    3. 将检索结果作为上下文发送给 LLM
    4. 返回生成的回答
    """
    
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
        self.retrieval_top_k = 2
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        执行 RAG 查询
        
        参数:
            question: 用户问题
            
        返回:
            包含回答和来源信息的字典
        """
        print(f"\n用户问题: {question}")
        print("-" * 50)
        
        # Step 1: 检索相关文档
        print("1. 检索相关文档...")
        retrieved_docs = self.vector_store.search(question, top_k=self.retrieval_top_k)
        
        print(f"   找到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"   [{i}] {doc['id']} (相似度: {doc['score']:.2f})")
            preview = doc['text'][:100].replace('\n', ' ')
            print(f"       预览: {preview}...")
        
        # Step 2: 构建上下文
        context = self._build_context(retrieved_docs)
        
        # Step 3: 生成回答
        print("\n2. 基于检索内容生成回答...")
        answer = self._generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc["id"] for doc in retrieved_docs],
            "context": context
        }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """构建上下文"""
        context_parts = []
        for doc in documents:
            context_parts.append(f"文档 [{doc['id']}]:\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        生成回答
        
        实际使用会调用 LLM API:
        response = openai.ChatCompletion.create(...)
        
        这里使用简化的模拟回答
        """
        # 基于上下文生成回答（简化版）
        if "python" in question.lower():
            return """基于检索到的文档，Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
它的设计哲学强调代码的可读性和简洁性，支持多种编程范式。
在机器学习领域，Python 是最流行的编程语言，拥有丰富的库。"""
        
        elif "机器学习" in question or "machine learning" in question.lower():
            return """根据检索结果，机器学习是人工智能的一个分支，使计算机能够从数据中学习。
常见的算法包括监督学习、无监督学习和强化学习。
Python 是机器学习最流行的编程语言。"""
        
        else:
            return f"""基于检索到的 {len(context.split('文档'))-1} 个相关文档，
我可以为您提供相关信息。检索到的内容涵盖多个技术主题，
包括编程语言、人工智能、容器化和架构设计。"""


# ============================================================
# 第三部分: Agent + RAG 结合
# ============================================================

class LlamaIndexAgent:
    """
    结合 RAG 的 Agent
    
    Agent 可以使用 RAG 作为工具来查询知识库
    """
    
    def __init__(self, query_engine: RAGQueryEngine):
        self.query_engine = query_engine
        self.memory = []
    
    def chat(self, message: str) -> str:
        """
        与 Agent 对话
        
        Agent 会判断是否需要查询知识库
        """
        print(f"\n用户: {message}")
        
        # 简单判断是否需要查询
        # 实际使用会由 LLM 决定
        if any(keyword in message.lower() for keyword in [
            "什么", "介绍", "是什么", "如何", "what", "how", "explain"
        ]):
            # 使用 RAG 查询
            result = self.query_engine.query(message)
            response = f"{result['answer']}\n\n参考来源: {', '.join(result['sources'])}"
        else:
            # 普通对话
            response = "您好！我可以帮您查询技术文档。请问有什么具体问题吗？"
        
        self.memory.append({"role": "user", "content": message})
        self.memory.append({"role": "assistant", "content": response})
        
        return response


# ============================================================
# 第四部分: 使用示例
# ============================================================

def main():
    """
    演示 LlamaIndex 的使用
    """
    print("=" * 70)
    print("LlamaIndex RAG + Agent 示例")
    print("=" * 70)
    
    # Step 1: 创建数据
    print("\n【步骤 1】加载文档")
    print("-" * 50)
    documents = create_sample_documents()
    print(f"加载了 {len(documents)} 个文档")
    for doc in documents:
        print(f"  - {doc['id']}: {doc['metadata']['category']}")
    
    # Step 2: 创建索引
    print("\n【步骤 2】创建向量索引")
    print("-" * 50)
    vector_store = SimpleVectorStore()
    vector_store.add_documents(documents)
    print("索引创建完成")
    
    # Step 3: 创建查询引擎
    query_engine = RAGQueryEngine(vector_store)
    
    # Step 4: RAG 查询示例
    print("\n" + "=" * 70)
    print("RAG 查询示例")
    print("=" * 70)
    
    questions = [
        "Python 是什么？",
        "机器学习有哪些类型？",
        "Docker 的主要用途是什么？",
        "微服务架构有什么优势？"
    ]
    
    for question in questions:
        result = query_engine.query(question)
        print(f"\n回答:\n{result['answer']}")
        print("\n" + "-" * 50)
    
    # Step 5: Agent 对话示例
    print("\n" + "=" * 70)
    print("Agent 对话示例")
    print("=" * 70)
    
    agent = LlamaIndexAgent(query_engine)
    
    conversations = [
        "你好！",
        "请介绍一下 Python",
        "Python 在机器学习中的作用是什么？"
    ]
    
    for msg in conversations:
        response = agent.chat(msg)
        print(f"\nAgent: {response}")
    
    print("\n" + "=" * 70)
    print("示例结束")
    print("=" * 70)
    
    print("""
LlamaIndex 工作流程:

┌──────────────────────────────────────────────────────────────┐
│                     LlamaIndex 架构                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │   数据源     │  (PDF, Word, DB, API...)                   │
│  └──────┬───────┘                                           │
│         │ 1. 加载                                           │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │   文档节点   │  (Documents / Nodes)                       │
│  └──────┬───────┘                                           │
│         │ 2. 分块/处理                                       │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │    索引      │  (VectorStore / Tree / Keyword...)         │
│  └──────┬───────┘                                           │
│         │ 3. 查询                                           │
│         ▼                                                   │
│  ┌──────────────┐     ┌──────────┐                          │
│  │  检索器      │────►│  LLM    │                           │
│  │ (Retriever)  │     │         │                           │
│  └──────────────┘     └────┬────┘                           │
│                            │ 4. 生成                         │
│                            ▼                                │
│                     ┌──────────┐                            │
│                     │  回答    │                            │
│                     └──────────┘                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

核心优势:
- 高效的语义检索
- 支持多种数据源
- 灵活的索引策略
- 与 Agent 无缝集成
""")


if __name__ == "__main__":
    main()
