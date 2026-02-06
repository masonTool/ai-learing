# Agent 框架示例汇总

本目录包含主流 Agent 框架的代码示例，每个示例都有详细的中文注释。

## 📁 目录结构

```
agent/
├── langchain/           # LangChain 生态系统
├── multi_agent/         # 多 Agent 协作框架
├── autonomous/          # 自主型 Agent
├── go_frameworks/       # Go 语言 Agent 框架
└── other/               # 其他重要框架
```

## 📚 框架对比

| 框架 | 核心特点 | 适用场景 |
|------|---------|---------|
| LangChain | 链式调用、工具集成 | 快速构建单 Agent |
| LangGraph | 图结构工作流 | 复杂业务流程 |
| CrewAI | 角色扮演、团队协作 | 内容创作、研究 |
| AutoGen | 对话式协作、代码执行 | 编程、问题求解 |
| MetaGPT | 模拟软件公司 | 软件开发 |
| AutoGPT | 完全自主执行 | 复杂研究任务 |
| BabyAGI | 任务列表驱动 | 项目管理 |
| LlamaIndex | RAG + Agent | 知识库问答 |
| Phidata | 轻量级、简洁 | 快速原型 |
| Semantic Kernel | 企业级、多语言 | 企业级应用 |
| Haystack | 专注 NLP/搜索 | 文档问答 |
| tRPC-Agent-Go | 多样化 Agent、高并发 | 微服务、生产环境 |

各框架的详细用法和代码说明，请查看对应示例文件中的注释。
