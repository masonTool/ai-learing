# Agent 框架示例汇总

本目录包含了当前主流 Agent 框架的典型代码示例，每个示例都有详细的中文注释说明关键逻辑。

## 📁 目录结构

```
agent/
├── langchain/           # LangChain 生态系统
│   ├── 01_basic_agent.py       # 基础 ReAct Agent
│   └── 02_langgraph_workflow.py # 复杂工作流
│
├── multi_agent/         # 多 Agent 协作框架
│   ├── 01_crewai_example.py     # CrewAI 团队协作
│   ├── 02_autogen_example.py    # AutoGen 对话式协作
│   └── 03_metagpt_example.py    # MetaGPT 软件公司模拟
│
├── autonomous/          # 自主型 Agent
│   ├── 01_autogpt_example.py    # AutoGPT 自主任务执行
│   └── 02_babyagi_example.py    # BabyAGI 任务驱动
│
├── go_frameworks/       # Go 语言 Agent 框架
│   └── 01_trpc_agent_go_example.go  # tRPC-Agent-Go 完整示例
│
└── other/               # 其他重要框架
    ├── 01_llamaindex_example.py     # LlamaIndex RAG + Agent
    ├── 02_phidata_example.py        # Phidata 轻量级助手
    ├── 03_semantic_kernel_example.py # Microsoft Semantic Kernel
    └── 04_haystack_example.py       # Haystack NLP/搜索
```

## 🚀 快速开始

### Python 框架安装依赖

```bash
# LangChain
pip install langchain langchain-openai

# CrewAI
pip install crewai

# AutoGen
pip install pyautogen

# MetaGPT
pip install metagpt

# LlamaIndex
pip install llama-index

# Phidata
pip install phidata

# Haystack
pip install farm-haystack

# Semantic Kernel
pip install semantic-kernel
```

### Go 框架安装依赖

```bash
# tRPC-Agent-Go
go get trpc.group/trpc-go/trpc-agent-go
```

### 设置 API Key

```bash
export OPENAI_API_KEY="your-api-key"
export HUNYUAN_API_KEY="your-hunyuan-api-key"  # 腾讯混元
```

## 📚 框架对比

### Python 框架

| 框架 | 核心特点 | 适用场景 | 学习曲线 |
|------|---------|---------|---------|
| **LangChain** | 链式调用、工具集成 | 快速构建单 Agent | ⭐⭐ |
| **LangGraph** | 图结构工作流 | 复杂业务流程 | ⭐⭐⭐ |
| **CrewAI** | 角色扮演、团队协作 | 内容创作、研究 | ⭐⭐ |
| **AutoGen** | 对话式协作、代码执行 | 编程、问题求解 | ⭐⭐⭐ |
| **MetaGPT** | 模拟软件公司 | 软件开发、架构设计 | ⭐⭐⭐⭐ |
| **AutoGPT** | 完全自主执行 | 复杂研究任务 | ⭐⭐⭐ |
| **BabyAGI** | 任务列表驱动 | 项目管理、信息收集 | ⭐⭐ |
| **LlamaIndex** | RAG + Agent | 知识库问答 | ⭐⭐ |
| **Phidata** | 轻量级、简洁 | 快速原型开发 | ⭐ |
| **Semantic Kernel** | 企业级、多语言 | 企业级应用 | ⭐⭐⭐ |
| **Haystack** | 专注 NLP/搜索 | 文档问答、RAG | ⭐⭐ |

### Go 框架

| 框架 | 核心特点 | 适用场景 | 学习曲线 |
|------|---------|---------|---------|
| **tRPC-Agent-Go** | 多样化 Agent、高并发 | 微服务、高并发场景 | ⭐⭐⭐ |

## 🎯 选择指南

### 根据使用场景选择：

1. **快速原型开发**
   - Python: Phidata / LangChain
   - Go: tRPC-Agent-Go

2. **多 Agent 协作系统**
   - Python: CrewAI / AutoGen
   - Go: tRPC-Agent-Go (ChainAgent/ParallelAgent)

3. **复杂工作流**
   - Python: LangGraph / Semantic Kernel
   - Go: tRPC-Agent-Go (GraphAgent/CycleAgent)

4. **知识库应用**
   - Python: LlamaIndex / Haystack
   - Go: tRPC-Agent-Go + 向量数据库

5. **自主任务执行**
   - Python: AutoGPT / BabyAGI
   - Go: tRPC-Agent-Go (自定义逻辑)

6. **软件开发**
   - Python: MetaGPT
   - Go: tRPC-Agent-Go + Code Generation

7. **企业级应用**
   - Python: Semantic Kernel
   - Go: tRPC-Agent-Go (腾讯内部首选)

## 💡 核心概念对比

### Agent 架构模式

```
1. ReAct 模式 (LangChain)
   Thought -> Action -> Observation -> (循环)

2. 对话模式 (AutoGen)
   Agent A <-> Agent B <-> Agent C
   
3. 工作流模式 (LangGraph)
   Start -> Node A -> Node B -> Node C -> End
              ↓         ↑
              └--循环--┘

4. 任务列表模式 (BabyAGI)
   Task List -> Execute -> Create New Tasks -> Prioritize

5. 角色协作模式 (CrewAI)
   Role A --
            --> Task --> Role B --> Task --> Role C
   Role D --

6. 插件模式 (Semantic Kernel)
   Kernel -> Plugins (Semantic + Native) -> Planner

7. Pipeline 模式 (Haystack)
   Document Store -> Retriever -> Reader/PromptNode -> Output

8. tRPC-Agent-Go 模式
   User Request -> Agent Router -> Agent Execution -> Response
                    ↓
            LLM/Chain/Parallel/Cycle/Graph Agent
```

## 🌍 语言生态对比

### Python 生态
- **成熟度**: 最成熟，框架众多
- **库支持**: 丰富的 ML/NLP 库
- **适用**: 快速开发、研究原型
- **代表框架**: LangChain, CrewAI, AutoGen

### Go 生态
- **成熟度**: 新兴，但发展迅速
- **库支持**: 高性能、云原生友好
- **适用**: 生产环境、高并发服务
- **代表框架**: tRPC-Agent-Go

## 📖 学习路径建议

### Python 新手入门路线

```
Week 1: Phidata → 了解基础概念
Week 2: LangChain → 掌握工具调用
Week 3: CrewAI → 学习多 Agent 协作
Week 4: LlamaIndex → 理解 RAG
Week 5: LangGraph → 复杂工作流
Week 6: AutoGen → 对话式 Agent
Week 7: Semantic Kernel → 企业级开发
Week 8: Haystack → 搜索和问答
```

### Go 开发者路线

```
Week 1: tRPC-Agent-Go 基础 → LLMAgent, 工具调用
Week 2: 进阶类型 → ChainAgent, ParallelAgent
Week 3: 复杂场景 → CycleAgent, GraphAgent
Week 4: 生产部署 → tRPC 集成、监控
```

### 进阶路线

```
1. 深入理解 ReAct 论文
2. 实现自定义 Agent
3. 学习 Agent 评估方法
4. 探索多模态 Agent
5. 研究 Agent 安全性和对齐
6. 对比不同框架的实现差异
```

## 🔗 参考资源

### Python 框架
- [LangChain 官方文档](https://python.langchain.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [AutoGen 文档](https://microsoft.github.io/autogen/)
- [MetaGPT GitHub](https://github.com/geekan/MetaGPT)
- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [Phidata 文档](https://docs.phidata.com/)
- [Semantic Kernel 文档](https://learn.microsoft.com/semantic-kernel/)
- [Haystack 文档](https://docs.haystack.deepset.ai/)

### Go 框架
- [tRPC-Agent-Go (腾讯内部)](https://git.woa.com/trpc-go/trpc-agent-go)
- [tRPC-Go 框架](https://github.com/trpc-group/trpc-go)

## 📝 示例说明

每个示例文件都包含：

1. **框架介绍**: 核心理念、核心概念、适用场景
2. **详细注释**: 每行关键代码都有中文说明
3. **完整示例**: 可运行的完整代码
4. **结构图解**: 框架架构和流程的可视化说明

建议按以下顺序阅读：
1. 先了解框架介绍部分
2. 运行示例代码
3. 仔细阅读代码注释
4. 修改代码进行实验

## 🤝 腾讯内部推荐

### 技术选型建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| Python 服务 | LangChain + LlamaIndex | 生态最成熟 |
| Go 微服务 | tRPC-Agent-Go | 与 tRPC 生态无缝集成 |
| 多 Agent 系统 | CrewAI / AutoGen | 协作能力最强 |
| 企业级应用 | Semantic Kernel | 微软背书，企业级设计 |
| 文档问答 | Haystack | 专注 RAG，性能优秀 |

### tRPC-Agent-Go 优势

1. **与 tRPC 生态集成**: 完美兼容腾讯内部服务框架
2. **高性能**: Go 语言天生适合高并发场景
3. **多样化 Agent**: 支持 5 种 Agent 类型
4. **事件驱动**: 灵活的扩展机制
5. **腾讯混元支持**: 内置腾讯混元大模型支持

## 🔧 框架选择决策树

```
开始
│
├─ 使用 Go 语言？
│  └─ 是 → tRPC-Agent-Go
│  └─ 否 → 继续
│
├─ 需要多 Agent 协作？
│  ├─ 对话式协作 → AutoGen
│  └─ 角色式协作 → CrewAI
│
├─ 需要复杂工作流？
│  ├─ 图结构 → LangGraph
│  └─ 企业级 → Semantic Kernel
│
├─ 需要 RAG/搜索？
│  ├─ 通用 RAG → LlamaIndex
│  └─ 专注搜索 → Haystack
│
├─ 需要自主执行？
│  ├─ 完全自主 → AutoGPT
│  └─ 任务驱动 → BabyAGI
│
└─ 快速原型？
   ├─ 最轻量 → Phidata
   └─ 最通用 → LangChain
```
