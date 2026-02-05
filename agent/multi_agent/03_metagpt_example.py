"""
MetaGPT 多 Agent 软件公司示例
============================

MetaGPT 是一个独特的多 Agent 框架，其核心理念是:
"模拟一个完整的软件公司组织架构来解决复杂问题"

核心概念:
1. Role (角色): 模拟公司中的职位，如产品经理、架构师、工程师等
2. Action (动作): 角色可以执行的具体操作
3. Environment (环境): Agent 交互的上下文
4. Message (消息): Agent 间的通信方式

公司角色体系:
- ProductManager: 产品经理，负责需求分析和产品定义
- Architect: 架构师，负责技术架构设计
- ProjectManager: 项目经理，负责任务分解和调度
- Engineer: 工程师，负责代码实现
- QaEngineer: 测试工程师，负责质量保证

特点:
- 标准化流程 (SOP): 每个角色都有标准操作流程
- 结构化输出: 每个角色产出标准格式的文档
- 文档驱动: 通过文档在不同角色间传递信息
- 自组织: 角色之间自主协调完成任务

适用场景: 软件开发、系统架构设计、复杂项目规划
"""

import asyncio
from typing import Optional
from metagpt.actions import Action, UserRequirement
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.environment import Environment
from metagpt.logs import logger


# ============================================================
# 第一部分: 基础概念 - Action
# ============================================================

class WriteProductRequirement(Action):
    """
    动作: 编写产品需求文档 (PRD)
    
    在 MetaGPT 中，Action 是角色可以执行的具体操作。
    每个 Action 继承自 Action 基类，需要实现 run 方法。
    
    输入: 用户需求
    输出: PRD 文档
    """
    
    # Action 的名称
    name: str = "WriteProductRequirement"
    
    # Action 的上下文描述
    context: Optional[str] = None
    
    async def run(self, requirement: str) -> str:
        """
        执行编写 PRD 的动作
        
        参数:
            requirement: 原始需求描述
            
        返回:
            PRD 文档内容
        """
        # 实际使用时会调用 LLM 生成 PRD
        # 这里使用模拟数据展示结构
        
        prd = f"""
# 产品需求文档 (PRD)

## 1. 需求概述
{requirement}

## 2. 用户故事
- 作为用户，我希望能快速完成任务
- 作为管理员，我希望能管理系统设置

## 3. 功能列表
### 3.1 核心功能
- 用户注册与登录
- 个人信息管理
- 数据展示与操作

### 3.2 扩展功能
- 通知系统
- 数据分析

## 4. 非功能需求
- 性能: 页面加载时间 < 2秒
- 安全: 数据加密存储
- 可用性: 支持 99.9% 在线时间

## 5. 界面原型
[待设计]

## 6. 验收标准
- 所有功能按需求实现
- 通过测试用例
- 代码审查通过
"""
        return prd


class WriteTechnicalDesign(Action):
    """
    动作: 编写技术设计文档
    
    架构师根据 PRD 输出技术设计文档
    """
    
    name: str = "WriteTechnicalDesign"
    
    async def run(self, prd: str) -> str:
        """
        根据 PRD 创建技术设计文档
        
        参数:
            prd: 产品需求文档
            
        返回:
            技术设计文档
        """
        
        design = """
# 技术设计文档

## 1. 架构概览
- 架构风格: 微服务架构
- 技术栈: Python + FastAPI + PostgreSQL + Redis
- 部署: Docker + Kubernetes

## 2. 系统架构图
```
┌─────────────┐
│   前端      │  (React/Vue)
└──────┬──────┘
       │ HTTP/REST
┌──────▼──────┐
│  API 网关   │  (Nginx/Kong)
└──────┬──────┘
       │
┌──────▼──────┐     ┌──────────┐
│  业务服务   │────►│  数据库  │
└─────────────┘     └──────────┘
```

## 3. 模块设计

### 3.1 用户模块
- 用户注册/登录 API
- JWT Token 认证
- 用户信息管理

### 3.2 数据模块
- CRUD 操作
- 数据验证
- 分页查询

## 4. 数据库设计

### 4.1 用户表 (users)
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 其他表...

## 5. API 设计

### 5.1 用户注册
- POST /api/v1/auth/register
- Request: {username, email, password}
- Response: {user_id, token}

### 5.2 用户登录
- POST /api/v1/auth/login
- Request: {email, password}
- Response: {token, user_info}

## 6. 安全设计
- 密码加密存储 (bcrypt)
- API 限流
- 输入验证和 SQL 注入防护

## 7. 性能考虑
- 数据库索引优化
- Redis 缓存热点数据
- 异步处理耗时操作
"""
        return design


class WriteCode(Action):
    """
    动作: 编写代码
    
    工程师根据技术设计文档编写代码
    """
    
    name: str = "WriteCode"
    
    async def run(self, design: str) -> str:
        """
        根据技术设计编写代码
        
        参数:
            design: 技术设计文档
            
        返回:
            代码实现
        """
        
        code = '''
# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import bcrypt
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="用户管理系统", version="1.0.0")
security = HTTPBearer()

# 数据模型
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

# 模拟数据库
users_db = {}
id_counter = 1

# JWT 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_token(user_id: int) -> str:
    """创建 JWT Token"""
    expire = datetime.utcnow() + timedelta(hours=24)
    payload = {"user_id": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 JWT Token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="无效的 Token")

@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user: UserRegister):
    """用户注册"""
    global id_counter
    
    # 检查邮箱是否已存在
    if any(u["email"] == user.email for u in users_db.values()):
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    
    # 密码加密
    password_hash = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    
    # 创建用户
    new_user = {
        "id": id_counter,
        "username": user.username,
        "email": user.email,
        "password_hash": password_hash,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    users_db[id_counter] = new_user
    id_counter += 1
    
    return UserResponse(**{k: v for k, v in new_user.items() if k != "password_hash"})

@app.post("/api/v1/auth/login")
async def login(credentials: UserLogin):
    """用户登录"""
    # 查找用户
    user = next((u for u in users_db.values() if u["email"] == credentials.email), None)
    
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    
    # 验证密码
    if not bcrypt.checkpw(credentials.password.encode(), user["password_hash"]):
        raise HTTPException(status_code=401, detail="密码错误")
    
    # 生成 Token
    token = create_token(user["id"])
    
    return {
        "token": token,
        "user": {k: v for k, v in user.items() if k != "password_hash"}
    }

@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user(token_data: dict = Depends(verify_token)):
    """获取当前用户信息"""
    user_id = token_data["user_id"]
    user = users_db.get(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return UserResponse(**{k: v for k, v in user.items() if k != "password_hash"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return code


class WriteTestCases(Action):
    """
    动作: 编写测试用例
    
    测试工程师编写测试代码
    """
    
    name: str = "WriteTestCases"
    
    async def run(self, code: str) -> str:
        """
        编写测试用例
        
        参数:
            code: 源代码
            
        返回:
            测试代码
        """
        
        tests = '''
# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app, users_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_db():
    """重置数据库"""
    users_db.clear()
    yield
    users_db.clear()

class TestUserRegistration:
    """用户注册测试"""
    
    def test_register_success(self):
        """测试正常注册"""
        response = client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
    
    def test_register_duplicate_email(self):
        """测试重复邮箱"""
        # 先注册一个用户
        client.post("/api/v1/auth/register", json={
            "username": "user1",
            "email": "same@example.com",
            "password": "password123"
        })
        
        # 再用相同邮箱注册
        response = client.post("/api/v1/auth/register", json={
            "username": "user2",
            "email": "same@example.com",
            "password": "password123"
        })
        assert response.status_code == 400
        assert "邮箱已被注册" in response.json()["detail"]
    
    def test_register_invalid_email(self):
        """测试无效邮箱"""
        response = client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "invalid-email",
            "password": "password123"
        })
        assert response.status_code == 422  # 验证错误

class TestUserLogin:
    """用户登录测试"""
    
    def test_login_success(self):
        """测试正常登录"""
        # 先注册
        client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "login@example.com",
            "password": "password123"
        })
        
        # 再登录
        response = client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "password123"
        })
        assert response.status_code == 200
        assert "token" in response.json()
    
    def test_login_wrong_password(self):
        """测试错误密码"""
        # 先注册
        client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "wrong@example.com",
            "password": "password123"
        })
        
        # 用错误密码登录
        response = client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

class TestUserProfile:
    """用户信息测试"""
    
    def test_get_profile(self):
        """测试获取用户信息"""
        # 注册并登录
        register_response = client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "profile@example.com",
            "password": "password123"
        })
        
        login_response = client.post("/api/v1/auth/login", json={
            "email": "profile@example.com",
            "password": "password123"
        })
        token = login_response.json()["token"]
        
        # 获取用户信息
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["email"] == "profile@example.com"
    
    def test_get_profile_no_token(self):
        """测试无 Token 访问"""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 403  # 未提供认证

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        return tests


# ============================================================
# 第二部分: 定义角色
# ============================================================

class ProductManager(Role):
    """
    角色: 产品经理
    
    职责:
    1. 理解用户需求
    2. 编写产品需求文档 (PRD)
    3. 定义验收标准
    """
    
    name: str = "Alice"  # Agent 的名字
    profile: str = "产品经理"  # 角色描述
    goal: str = "创建清晰、完整的产品需求文档"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置角色可以执行的动作
        self.set_actions([WriteProductRequirement])
        # 设置角色观察的消息类型
        self._watch([UserRequirement])


class Architect(Role):
    """
    角色: 架构师
    
    职责:
    1. 根据 PRD 设计技术架构
    2. 编写技术设计文档
    3. 定义数据库结构和 API
    """
    
    name: str = "Bob"
    profile: str = "架构师"
    goal: str = "设计可扩展、高性能的技术架构"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteTechnicalDesign])


class Engineer(Role):
    """
    角色: 工程师
    
    职责:
    1. 根据技术设计编写代码
    2. 实现业务功能
    3. 编写单元测试
    """
    
    name: str = "Charlie"
    profile: str = "开发工程师"
    goal: str = "编写高质量、可维护的代码"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteCode])


class QaEngineer(Role):
    """
    角色: 测试工程师
    
    职责:
    1. 根据需求和代码编写测试用例
    2. 执行测试
    3. 报告和跟踪缺陷
    """
    
    name: str = "David"
    profile: str = "测试工程师"
    goal: str = "确保软件质量，发现并报告缺陷"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteTestCases])


# ============================================================
# 第三部分: 运行示例
# ============================================================

async def run_metagpt_example():
    """
    运行 MetaGPT 示例
    
    模拟一个完整的软件开发流程:
    1. 产品经理分析需求并输出 PRD
    2. 架构师根据 PRD 设计技术方案
    3. 工程师根据设计编写代码
    4. 测试工程师编写测试用例
    """
    
    print("=" * 70)
    print("MetaGPT 多 Agent 软件公司示例")
    print("=" * 70)
    print("\n模拟场景: 开发一个用户管理系统")
    print("\n团队配置:")
    print("  - Alice: 产品经理")
    print("  - Bob: 架构师")
    print("  - Charlie: 开发工程师")
    print("  - David: 测试工程师")
    print("\n" + "-" * 70)
    
    # 创建环境
    env = Environment()
    
    # 创建角色
    pm = ProductManager()
    architect = Architect()
    engineer = Engineer()
    qa = QaEngineer()
    
    # 添加角色到环境
    env.add_roles([pm, architect, engineer, qa])
    
    # 发布需求
    requirement = "开发一个用户管理系统，包含用户注册、登录、信息管理功能"
    print(f"\n【用户需求】{requirement}\n")
    
    # 创建需求消息
    msg = Message(
        content=requirement,
        role="User",
        cause_by=UserRequirement
    )
    
    # 发布消息到环境
    env.publish_message(msg)
    
    # 模拟执行流程
    # 注意: 这里为了演示目的，直接调用 Action
    # 实际 MetaGPT 会自动处理消息传递和角色协调
    
    print("\n" + "=" * 70)
    print("Step 1: 产品经理编写 PRD")
    print("=" * 70)
    
    prd_action = WriteProductRequirement()
    prd = await prd_action.run(requirement)
    print(prd)
    
    print("\n" + "=" * 70)
    print("Step 2: 架构师设计技术方案")
    print("=" * 70)
    
    design_action = WriteTechnicalDesign()
    design = await design_action.run(prd)
    print(design)
    
    print("\n" + "=" * 70)
    print("Step 3: 开发工程师编写代码")
    print("=" * 70)
    
    code_action = WriteCode()
    code = await code_action.run(design)
    print(code)
    
    print("\n" + "=" * 70)
    print("Step 4: 测试工程师编写测试")
    print("=" * 70)
    
    test_action = WriteTestCases()
    tests = await test_action.run(code)
    print(tests)
    
    print("\n" + "=" * 70)
    print("软件开发流程完成!")
    print("=" * 70)
    
    return {
        "prd": prd,
        "design": design,
        "code": code,
        "tests": tests
    }


# ============================================================
# 第四部分: 主函数
# ============================================================

def main():
    """
    主函数
    
    注意: MetaGPT 需要特定的配置文件和环境
    这里的示例展示了核心概念和结构
    """
    print("\n注意: 运行完整示例需要安装 MetaGPT 并配置环境")
    print("pip install metagpt\n")
    
    # 运行示例
    try:
        result = asyncio.run(run_metagpt_example())
        
        # 保存结果到文件
        import os
        os.makedirs("output", exist_ok=True)
        
        with open("output/prd.md", "w") as f:
            f.write(result["prd"])
        
        with open("output/design.md", "w") as f:
            f.write(result["design"])
        
        with open("output/main.py", "w") as f:
            f.write(result["code"])
        
        with open("output/test_main.py", "w") as f:
            f.write(result["tests"])
        
        print("\n输出文件已保存到 output/ 目录")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("这是演示代码，展示 MetaGPT 的结构和概念")


if __name__ == "__main__":
    main()
