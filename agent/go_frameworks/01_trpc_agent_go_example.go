/*
tRPC-Agent-Go 完整示例
======================

tRPC-Agent-Go 是腾讯 tRPC-Go 团队开源的 Go 语言 Agent 框架，
专注于为 Go 生态提供自主 Agent 开发能力。

核心特点:
1. 多样化 Agent 类型 - 支持 LLMAgent、ChainAgent、ParallelAgent、CycleAgent、GraphAgent
2. 事件驱动架构 - 基于 Event 的通信机制
3. 工具调用 - 支持函数调用和外部工具集成
4. 子 Agent 协作 - 支持层级化的 Agent 协作
5. 与 tRPC 生态集成 - 完美兼容 tRPC 服务框架

适用场景:
- 需要高并发的 Agent 应用
- 微服务架构下的 Agent 部署
- 与现有 Go 服务集成的 AI 能力
*/

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

// ============================================================
// 第一部分: 基础 Agent 示例
// ============================================================

// BasicAgentExample 展示最基础的 LLM Agent 使用
func BasicAgentExample() {
	fmt.Println("=" + string(make([]byte, 70)))
	fmt.Println("示例 1: 基础 LLM Agent")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		创建基础 LLM Agent

		LLMAgent 是最基础的 Agent 类型，具备:
		- 大模型推理能力
		- 工具调用能力
		- 上下文记忆能力
	*/

	// 配置 Agent
	config := &agent.Config{
		Name:    "基础助手",
		Model:   "hunyuan-lite", // 腾讯混元模型
		Timeout: 30 * time.Second,
	}

	// 创建 Agent
	llmAgent, err := agent.NewLLMAgent(config)
	if err != nil {
		log.Printf("创建 Agent 失败: %v", err)
		return
	}

	// 创建上下文
	ctx := context.Background()

	// 执行对话
	response, err := llmAgent.Execute(ctx, &event.Request{
		Message: "你好，请介绍一下你自己",
	})
	if err != nil {
		log.Printf("执行失败: %v", err)
		return
	}

	fmt.Printf("Agent 回复: %s\n", response.Message)
}

// ============================================================
// 第二部分: 带工具调用的 Agent
// ============================================================

// ToolAgentExample 展示带工具调用的 Agent
func ToolAgentExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 2: 带工具调用的 Agent")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		工具调用是 Agent 的核心能力
		通过 Tools，Agent 可以:
		- 查询数据库
		- 调用外部 API
		- 执行计算
		- 操作文件系统
	*/

	// 定义工具
	calculatorTool := &tool.Tool{
		Name:        "calculator",
		Description: "执行数学计算",
		Parameters: map[string]interface{}{
			"expression": "string",
		},
		Handler: func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			expression := params["expression"].(string)
			// 实际实现会解析并计算表达式
			return fmt.Sprintf("计算结果: %s = 42", expression), nil
		},
	}

	searchTool := &tool.Tool{
		Name:        "search",
		Description: "搜索信息",
		Parameters: map[string]interface{}{
			"query": "string",
		},
		Handler: func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			query := params["query"].(string)
			return fmt.Sprintf("搜索 '%s' 的结果: 找到相关信息...", query), nil
		},
	}

	// 创建带工具的 Agent
	config := &agent.Config{
		Name:  "工具助手",
		Model: "hunyuan-lite",
		Tools: []tool.Tool{*calculatorTool, *searchTool},
	}

	llmAgent, err := agent.NewLLMAgent(config)
	if err != nil {
		log.Printf("创建 Agent 失败: %v", err)
		return
	}

	ctx := context.Background()

	// 执行需要工具调用的对话
	response, err := llmAgent.Execute(ctx, &event.Request{
		Message: "计算 123 * 456 的结果",
	})
	if err != nil {
		log.Printf("执行失败: %v", err)
		return
	}

	fmt.Printf("Agent 回复: %s\n", response.Message)
}

// ============================================================
// 第三部分: ChainAgent - 链式处理
// ============================================================

// ChainAgentExample 展示链式 Agent
func ChainAgentExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 3: ChainAgent - 链式处理")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		ChainAgent 将多个 Agent 按顺序执行
		数据流: Input -> Agent1 -> Agent2 -> Agent3 -> Output

		适用场景:
		- 数据处理流水线
		- 多阶段内容生成
		- 逐步细化的任务处理
	*/

	// 创建子 Agent
	planningAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "规划Agent",
		Model: "hunyuan-lite",
		SystemPrompt: `你是一个任务规划专家。
将用户的请求分解为清晰的执行步骤。`,
	})

	researchAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "研究Agent",
		Model: "hunyuan-lite",
		SystemPrompt: `你是一个研究专家。
基于规划步骤收集相关信息。`,
	})

	writingAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "写作Agent",
		Model: "hunyuan-lite",
		SystemPrompt: `你是一个写作专家。
基于研究结果生成最终内容。`,
	})

	// 创建 ChainAgent
	chainConfig := &agent.ChainConfig{
		Name: "内容创作链",
		Agents: []agent.Agent{
			planningAgent,
			researchAgent,
			writingAgent,
		},
		// 配置数据传递方式
		PassThrough: true, // 将前一个 Agent 的输出传递给下一个
	}

	chainAgent, err := agent.NewChainAgent(chainConfig)
	if err != nil {
		log.Printf("创建 ChainAgent 失败: %v", err)
		return
	}

	ctx := context.Background()

	response, err := chainAgent.Execute(ctx, &event.Request{
		Message: "写一篇关于 Go 语言并发的技术文章",
	})
	if err != nil {
		log.Printf("执行失败: %v", err)
		return
	}

	fmt.Printf("ChainAgent 输出:\n%s\n", response.Message)
}

// ============================================================
// 第四部分: ParallelAgent - 并行处理
// ============================================================

// ParallelAgentExample 展示并行 Agent
func ParallelAgentExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 4: ParallelAgent - 并行处理")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		ParallelAgent 同时执行多个 Agent
		执行方式: Agent1 + Agent2 + Agent3 (同时执行)

		适用场景:
		- 多维度数据分析
		- 并行信息收集
		- 多角度内容评估
	*/

	// 创建多个专业 Agent
	marketAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "市场分析Agent",
		Model:        "hunyuan-lite",
		SystemPrompt: "从市场角度分析产品的机会和挑战。",
	})

	technicalAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "技术分析Agent",
		Model:        "hunyuan-lite",
		SystemPrompt: "从技术角度评估产品的可行性和创新点。",
	})

	financeAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "财务分析Agent",
		Model:        "hunyuan-lite",
		SystemPrompt: "从财务角度分析产品的盈利模式和成本结构。",
	})

	// 创建 ParallelAgent
	parallelConfig := &agent.ParallelConfig{
		Name: "综合分析团队",
		Agents: []agent.Agent{
			marketAgent,
			technicalAgent,
			financeAgent,
		},
		// 聚合方式
		Aggregator: func(results []*event.Response) *event.Response {
			// 合并所有 Agent 的分析结果
			var combined string
			for i, result := range results {
				combined += fmt.Sprintf("\n=== %d. %s ===\n%s\n", i+1, result.AgentName, result.Message)
			}
			return &event.Response{
				Message: "综合分析报告:\n" + combined,
			}
		},
	}

	parallelAgent, err := agent.NewParallelAgent(parallelConfig)
	if err != nil {
		log.Printf("创建 ParallelAgent 失败: %v", err)
		return
	}

	ctx := context.Background()

	response, err := parallelAgent.Execute(ctx, &event.Request{
		Message: "评估一款新的 AI 助手产品",
	})
	if err != nil {
		log.Printf("执行失败: %v", err)
		return
	}

	fmt.Printf("ParallelAgent 输出:\n%s\n", response.Message)
}

// ============================================================
// 第五部分: CycleAgent - 循环迭代
// ============================================================

// CycleAgentExample 展示循环 Agent
func CycleAgentExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 5: CycleAgent - 循环迭代")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		CycleAgent 循环执行 Agent 直到满足终止条件
		执行流程:
		1. 生成器 Agent 生成内容
		2. 检查器 Agent 评估质量
		3. 如未达标，返回步骤 1
		4. 如达标，输出结果

		适用场景:
		- 代码生成与优化
		- 内容迭代改进
		- 问题求解与验证
	*/

	// 生成器 Agent
	generatorAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "生成器",
		Model:        "hunyuan-lite",
		SystemPrompt: "根据需求生成解决方案，根据反馈进行改进。",
	})

	// 检查器 Agent
	reviewerAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "检查器",
		Model:        "hunyuan-lite",
		SystemPrompt: `检查生成的内容是否满足要求。
如果不满足，指出具体问题。
如果满足，回复 "APPROVED"。`,
	})

	// 创建 CycleAgent
	cycleConfig := &agent.CycleConfig{
		Name:      "迭代优化循环",
		Generator: generatorAgent,
		Reviewer:  reviewerAgent,
		// 终止条件
		TerminationCondition: func(response *event.Response) bool {
			// 当检查器回复包含 APPROVED 时终止
			return len(response.Message) > 0 &&
				(response.Message == "APPROVED" || contains(response.Message, "APPROVED"))
		},
		MaxIterations: 5, // 最大迭代次数
	}

	cycleAgent, err := agent.NewCycleAgent(cycleConfig)
	if err != nil {
		log.Printf("创建 CycleAgent 失败: %v", err)
		return
	}

	ctx := context.Background()

	response, err := cycleAgent.Execute(ctx, &event.Request{
		Message: "设计一个高效的缓存系统架构",
	})
	if err != nil {
		log.Printf("执行失败: %v", err)
		return
	}

	fmt.Printf("CycleAgent 最终输出:\n%s\n", response.Message)
}

// ============================================================
// 第六部分: GraphAgent - 图工作流
// ============================================================

// GraphAgentExample 展示图工作流 Agent
func GraphAgentExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 6: GraphAgent - 图工作流")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		GraphAgent 基于有向图执行工作流
		特点:
		- 支持条件分支
		- 支持循环
		- 支持并行路径

		适用场景:
		- 复杂业务流程
		- 状态机实现
		- 审批流程
	*/

	// 创建节点 Agent
	inputAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "输入处理",
		Model: "hunyuan-lite",
	})

	classifyAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "分类器",
		Model: "hunyuan-lite",
		SystemPrompt: `将用户请求分类为以下类型之一:
- "技术问题"
- "业务咨询"
- "其他"`,
	})

	techAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "技术支持",
		Model:        "hunyuan-lite",
		SystemPrompt: "提供专业的技术支持。",
	})

	businessAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "业务顾问",
		Model:        "hunyuan-lite",
		SystemPrompt: "提供业务相关的咨询和建议。",
	})

	generalAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "通用助手",
		Model:        "hunyuan-lite",
		SystemPrompt: "处理一般性问题。",
	})

	// 创建 GraphAgent
	graphConfig := &agent.GraphConfig{
		Name: "智能路由系统",
		Nodes: map[string]agent.Agent{
			"input":    inputAgent,
			"classify": classifyAgent,
			"tech":     techAgent,
			"business": businessAgent,
			"general":  generalAgent,
		},
		// 定义边（连接）
		Edges: []agent.Edge{
			{From: "input", To: "classify"},
			{From: "classify", To: "tech", Condition: func(r *event.Response) bool {
				return contains(r.Message, "技术问题")
			}},
			{From: "classify", To: "business", Condition: func(r *event.Response) bool {
				return contains(r.Message, "业务咨询")
			}},
			{From: "classify", To: "general", Condition: func(r *event.Response) bool {
				return !contains(r.Message, "技术问题") && !contains(r.Message, "业务咨询")
			}},
		},
		EntryNode: "input",
	}

	graphAgent, err := agent.NewGraphAgent(graphConfig)
	if err != nil {
		log.Printf("创建 GraphAgent 失败: %v", err)
		return
	}

	ctx := context.Background()

	// 测试技术问题
	response, _ := graphAgent.Execute(ctx, &event.Request{
		Message: "如何实现一个高性能的 Redis 连接池？",
	})
	fmt.Printf("技术问题路由结果:\n%s\n\n", response.Message)

	// 测试业务问题
	response, _ = graphAgent.Execute(ctx, &event.Request{
		Message: "这个产品的商业模式是什么？",
	})
	fmt.Printf("业务问题路由结果:\n%s\n", response.Message)
}

// ============================================================
// 第七部分: 事件系统与中间件
// ============================================================

// EventSystemExample 展示事件系统
func EventSystemExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 7: 事件系统与中间件")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		tRPC-Agent-Go 基于事件驱动架构
		事件类型:
		- RequestEvent: 用户请求事件
		- ToolCallEvent: 工具调用事件
		- ResponseEvent: 响应事件
		- ErrorEvent: 错误事件

		中间件可以:
		- 记录日志
		- 监控性能
		- 修改请求/响应
		- 实现重试逻辑
	*/

	// 创建 Agent
	llmAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "事件示例Agent",
		Model: "hunyuan-lite",
	})

	// 注册事件处理器
	llmAgent.OnEvent(event.RequestEvent, func(e event.Event) {
		fmt.Printf("[事件] 收到请求: %s\n", e.GetRequest().Message)
	})

	llmAgent.OnEvent(event.ToolCallEvent, func(e event.Event) {
		fmt.Printf("[事件] 调用工具: %s\n", e.GetToolName())
	})

	llmAgent.OnEvent(event.ResponseEvent, func(e event.Event) {
		fmt.Printf("[事件] 生成响应: %s...\n", e.GetResponse().Message[:50])
	})

	// 添加中间件
	llmAgent.Use(func(next agent.Handler) agent.Handler {
		return func(ctx context.Context, req *event.Request) (*event.Response, error) {
			start := time.Now()
			fmt.Printf("[中间件] 开始处理请求: %s\n", req.Message)

			resp, err := next(ctx, req)

			duration := time.Since(start)
			fmt.Printf("[中间件] 请求处理完成, 耗时: %v\n", duration)

			return resp, err
		}
	})

	ctx := context.Background()
	llmAgent.Execute(ctx, &event.Request{
		Message: "演示事件系统",
	})
}

// ============================================================
// 第八部分: 完整应用示例
// ============================================================

// CompleteApplicationExample 展示完整应用
func CompleteApplicationExample() {
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("示例 8: 完整应用 - 智能客服系统")
	fmt.Println("=" + string(make([]byte, 70)))

	/*
		综合示例: 智能客服系统
		- 使用 GraphAgent 进行意图识别和路由
		- 使用 ChainAgent 处理复杂查询
		- 使用工具查询订单、用户信息等
		- 使用事件系统记录日志
	*/

	// 定义工具
	orderTool := &tool.Tool{
		Name:        "query_order",
		Description: "查询订单信息",
		Parameters: map[string]interface{}{
			"order_id": "string",
		},
		Handler: func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			orderID := params["order_id"].(string)
			// 模拟查询
			return map[string]interface{}{
				"order_id":   orderID,
				"status":     "已发货",
				"items":      []string{"商品A", "商品B"},
				"total":      299.99,
				"created_at": "2024-01-15",
			}, nil
		},
	}

	// 意图识别 Agent
	intentAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "意图识别",
		Model: "hunyuan-lite",
		SystemPrompt: `识别用户意图，分类为:
- ORDER_QUERY: 订单查询
- PRODUCT_INQUIRY: 产品咨询
- COMPLAINT: 投诉建议
- GENERAL: 其他问题`,
	})

	// 订单查询处理 Agent
	orderAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:  "订单处理",
		Model: "hunyuan-lite",
		Tools: []tool.Tool{*orderTool},
		SystemPrompt: "帮助用户查询订单信息，提供准确的订单状态。",
	})

	// 产品咨询 Agent
	productAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "产品咨询",
		Model:        "hunyuan-lite",
		SystemPrompt: "介绍产品特点和优势，回答产品相关问题。",
	})

	// 投诉处理 Agent
	complaintAgent, _ := agent.NewLLMAgent(&agent.Config{
		Name:         "投诉处理",
		Model:        "hunyuan-lite",
		SystemPrompt: "认真听取用户投诉，提供解决方案，记录投诉内容。",
	})

	// 创建客服图工作流
	csGraph := &agent.GraphConfig{
		Name: "智能客服系统",
		Nodes: map[string]agent.Agent{
			"intent":   intentAgent,
			"order":    orderAgent,
			"product":  productAgent,
			"complaint": complaintAgent,
		},
		Edges: []agent.Edge{
			{From: "intent", To: "order", Condition: func(r *event.Response) bool {
				return contains(r.Message, "ORDER_QUERY")
			}},
			{From: "intent", To: "product", Condition: func(r *event.Response) bool {
				return contains(r.Message, "PRODUCT_INQUIRY")
			}},
			{From: "intent", To: "complaint", Condition: func(r *event.Response) bool {
				return contains(r.Message, "COMPLAINT")
			}},
		},
		EntryNode: "intent",
	}

	customerService, _ := agent.NewGraphAgent(csGraph)

	// 测试场景
	testCases := []string{
		"我的订单 12345 到哪里了？",
		"这款手机有什么特点？",
		"我对你们的服务很不满意！",
	}

	ctx := context.Background()
	for _, tc := range testCases {
		fmt.Printf("\n用户: %s\n", tc)
		resp, err := customerService.Execute(ctx, &event.Request{Message: tc})
		if err != nil {
			fmt.Printf("错误: %v\n", err)
			continue
		}
		fmt.Printf("客服: %s\n", resp.Message)
	}
}

// ============================================================
// 工具函数
// ============================================================

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(s[:len(substr)] == substr) ||
		(s[len(s)-len(substr):] == substr) ||
		findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ============================================================
// 主函数
// ============================================================

func main() {
	fmt.Println("tRPC-Agent-Go 完整示例")
	fmt.Println("=====================")
	fmt.Println()
	fmt.Println("tRPC-Agent-Go 是腾讯开源的 Go 语言 Agent 框架")
	fmt.Println("特点: 多样化 Agent 类型、事件驱动、高并发支持")
	fmt.Println()

	// 运行所有示例
	BasicAgentExample()
	ToolAgentExample()
	ChainAgentExample()
	ParallelAgentExample()
	CycleAgentExample()
	GraphAgentExample()
	EventSystemExample()
	CompleteApplicationExample()

	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("所有示例运行完成")
	fmt.Println("=" + string(make([]byte, 70)))
	fmt.Println()
	fmt.Println("tRPC-Agent-Go 架构概览:")
	fmt.Println()
	fmt.Println("┌─────────────────────────────────────────────────────────────┐")
	fmt.Println("│                    tRPC-Agent-Go 架构                        │")
	fmt.Println("├─────────────────────────────────────────────────────────────┤")
	fmt.Println("│                                                              │")
	fmt.Println("│  ┌─────────────────────────────────────────────────────┐    │")
	fmt.Println("│  │                   Agent 类型                         │    │")
	fmt.Println("│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │")
	fmt.Println("│  │  │ LLM     │ │ Chain   │ │Parallel │ │ Cycle   │   │    │")
	fmt.Println("│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │   │    │")
	fmt.Println("│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │    │")
	fmt.Println("│  │  ┌─────────┐ ┌─────────┐                           │    │")
	fmt.Println("│  │  │ Graph   │ │ Custom  │                           │    │")
	fmt.Println("│  │  │ Agent   │ │ Agent   │                           │    │")
	fmt.Println("│  │  └─────────┘ └─────────┘                           │    │")
	fmt.Println("│  └─────────────────────────────────────────────────────┘    │")
	fmt.Println("│                                                              │")
	fmt.Println("│  ┌─────────────────────────────────────────────────────┐    │")
	fmt.Println("│  │                   核心组件                           │    │")
	fmt.Println("│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │")
	fmt.Println("│  │  │  Event   │  │  Tool    │  │ Context  │          │    │")
	fmt.Println("│  │  │  System  │  │  Registry│  │ Manager  │          │    │")
	fmt.Println("│  │  └──────────┘  └──────────┘  └──────────┘          │    │")
	fmt.Println("│  └─────────────────────────────────────────────────────┘    │")
	fmt.Println("│                                                              │")
	fmt.Println("│  ┌─────────────────────────────────────────────────────┐    │")
	fmt.Println("│  │                   扩展能力                           │    │")
	fmt.Println("│  │  中间件 │ 钩子 │ 自定义Agent │ tRPC集成              │    │")
	fmt.Println("│  └─────────────────────────────────────────────────────┘    │")
	fmt.Println("│                                                              │")
	fmt.Println("└─────────────────────────────────────────────────────────────┘")
	fmt.Println()
	fmt.Println("适用场景:")
	fmt.Println("- 高并发 Agent 应用")
	fmt.Println("- 微服务架构下的 AI 能力")
	fmt.Println("- 复杂工作流编排")
	fmt.Println("- 多 Agent 协作系统")
}
