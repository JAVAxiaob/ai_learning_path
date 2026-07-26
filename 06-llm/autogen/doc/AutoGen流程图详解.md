# AutoGen 流程图详解

> 位置: 06-llm/autogen/doc/
> 配套文档: AutoGen-MultiAgent协作框架.md | AutoGen性能优化重难点.md | AutoGen面试题汇总.md

---

## 一、Multi-Agent 三大协作模式流程

### 1.1 群聊模式 RoundRobinGroupChat (软件开发团队示例)

```mermaid
flowchart TD
    Start[人类输入任务] --> Init[初始化Agent团队<br/>PM/架构师/程序员/测试/评审]
    Init --> Register[注册Agent角色+System Prompt+绑定Tools]
    Register --> SetTerminate[设置终止条件<br/>COMPLETE/APPROVED+PASS/MaxMsg=20]

    SetTerminate --> RoundRobin[开始轮流发言]

    subgraph 第1轮发言
        RoundRobin --> PM[1️⃣ 产品经理Agent<br/>输出结构化PRD文档]
        PM --> Arch[2️⃣ 架构师Agent<br/>根据PRD画类图API]
        Arch --> Coder[3️⃣ 程序员Agent<br/>写完整代码+CodeInterpreter执行]
        Coder --> Tester[4️⃣ 测试Agent<br/>跑pytest生成覆盖率报告]
        Tester --> Reviewer[5️⃣ 评审Agent<br/>CodeReview提Bug]
    end

    Reviewer --> CheckTerm{触发终止条件?}
    CheckTerm -->|未触发 COMPLETE还没说| RoundRobin2[进入下一轮发言]
    RoundRobin2 --> PM2[产品经理修改PRD]
    PM2 --> Arch2[架构师调整方案]
    Arch2 --> Coder2[程序员修复Bug]
    Coder2 --> Tester2[重跑测试]
    Tester2 --> Reviewer2[评审说 PASS ✓]

    Reviewer2 --> Coder3[程序员最后输出 COMPLETE]
    Coder3 --> Triggered[COMPLETE + APPROVED + PASS 三条件全满足]
    Triggered -->|终止条件满足!| HumanProxy{需要人类确认?}
    HumanProxy -->|自动确认| Output[输出最终项目代码+文档]
    HumanProxy -->|转人工审核| User[返回给用户判断]
```

### 1.2 顺序工作流 SequentialWorkflow (确定性流水线)

```mermaid
sequenceDiagram
    participant S1 as Step1: 需求文档生成
    participant S2 as Step2: 架构设计
    participant S3 as Step3: 代码实现
    participant S4 as Step4: 单元测试
    participant S5 as Step5: CodeReview循环

    Note over S1: 输入: 用户原始需求描述
    S1->>S1: 调用LLM生成PRD.md
    S1-->>S2: 输出: PRD文档 + 验收标准

    S2->>S2: 根据PRD画Mermaid架构图
    S2->>S2: 定义RESTful接口+数据库Schema
    S2-->>S3: 输出: 设计文档 DesignDoc.md

    S3->>S3: 写FastAPI代码 models.py+routers.py
    S3->>S3: CodeInterpreter跑静态检查
    S3-->>S4: 输出: src/ 完整源码包

    S4->>S4: 写pytest测试用例 (覆盖>80%)
    S4->>S4: 自动执行测试收集Coverage
    alt 所有测试通过且达标
        S4-->>S5: PASS! 送审
    else 有失败用例 or 覆盖率低
        S4-->>S3: ❌打回改代码 发回Step3重写
    end

    S5->>S5: LLM审查Bug/性能/安全
    alt Review通过
        S5-->>Output: 输出最终交付物
    else Review有改进意见
        S5-->>S3: 🔧发回Step3按Review改
    end
```

### 1.3 等级模式 Hierarchical Swarm (老板+员工)

```mermaid
flowchart TD
    User[人类用户提复杂需求] --> Boss[CEO Boss Agent<br/>总体规划+任务拆解]

    Boss --> TaskPlan[任务规划分解<br/>拆成3个子任务A/B/C]

    subgraph 并行子任务分派
        TaskPlan --> TaskA[任务A: 市场调研用户画像]
        TaskPlan --> TaskB[任务B: 技术方案选型]
        TaskPlan --> TaskC[任务C: 预算成本ROI估算]
    end

    subgraph 员工Agent独立执行
        TaskA --> EmpMkt[市场Agent]
        EmpMkt --> ReportA[竞品分析报告/用户问卷]
        TaskB --> EmpEng[工程Agent]
        EmpEng --> ReportB[架构图/技术栈/排期]
        TaskC --> EmpFin[财务Agent]
        EmpFin --> ReportC[人力预算/服务器成本/IRR]
    end

    ReportA --> Boss
    ReportB --> Boss
    ReportC --> Boss

    Boss --> Integrate[Boss汇总三份报告<br/>生成完整商业计划书BP]
    Integrate --> Check{老板自己检查是否完整?}
    Check -->|信息充足| FinalOut[输出给用户最终BP]
    Check -->|某部分缺信息| ReAsk[重派对应员工补调研]
    ReAsk --> EmpMkt
    ReAsk --> EmpEng
    ReAsk --> EmpFin
```

---

## 二、Agent核心能力实现流程

### 2.1 Tool Calling 工具调用完整链路

```mermaid
sequenceDiagram
    participant LLM as LLM大模型
    participant System as System Prompt注入Tool Schema
    participant User as 用户问题
    participant Parser as JSON解析+Pydantic校验
    participant Retry as 重试/Fallback
    participant Executor as 函数执行器
    participant Fn as 实际Python函数@tool

    Note over System: 启动时 自动反射@tool装饰的方法→JSON Schema<br/>每个工具: {name, description, parameters:{type,properties,required}}
    System-->>LLM: 在System Prompt追加工具描述

    User-->>LLM: "帮我查北京2026年7月25号天气?"

    LLM->>LLM: 判断需要工具/直接答
    alt 需要工具调用
        LLM-->>Parser: 输出特殊JSON格式<br/>{"name":"get_weather","args":{"city":"北京","date":"2026-07-25"}}
    else 直接答
        LLM-->>User: 普通文字回答 结束流程
    end

    Parser->>Parser: JSON语法解析
    alt JSON解析失败 括号不匹配/格式错
        Parser-->>Retry: 触发JSON修复器 尝试重拼
        Retry->>LLM: "上次JSON格式错误，请重输出合法JSON" + 错误hint
        LLM-->>Parser: 重新输出JSON 最多重试2次
    end

    Parser->>Parser: Pydantic Model参数类型校验<br/>args.city:str ✓ args.date:date格式 ✓
    alt 参数校验失败 类型/必填/枚举
        Parser-->>Retry: 发参数错误回LLM
        Retry->>LLM: "参数date格式应为YYYY-MM-DD，收到: xx"
    end

    Parser-->>Executor: ✅ 合法ToolCall(name+valid_args)
    Executor-->>Fn: 实际调用 get_weather("北京","2026-07-25")
    Fn-->>Executor: 返回结果 {"temp":32,"weather":"晴"}
    Executor-->>LLM: 把函数返回值作为User消息再塞给LLM
    LLM->>LLM: 基于真实工具返回数据生成自然语言答
    LLM-->>User: "北京今天晴，32度，建议穿短袖"
```

### 2.2 Memory 记忆系统三层架构

```mermaid
flowchart TD
    subgraph L1[短期记忆 Short Term]
        ST1[SlidingWindowBuffer<br/>滑动窗口最近30条对话]
        ST2[TokenBudgetMemory<br/>按上下文Token预算128K保留尾部]
        ST3[SummaryBuffer<br/>旧对话LLM压缩成摘要 + 最新原文]
    end

    subgraph L2[长期记忆 Long Term]
        LT1[(VectorStore 向量数据库<br/>Milvus/pgvector/Chroma)]
        LT2[重要对话自动打标签存LT]
        LT3[用户画像Profile RAG<br/>用户偏好/历史订单/过敏史等结构化]
    end

    subgraph L3[共享记忆 Shared]
        SH1[Agent间共享的工作黑板Blackboard]
        SH2[群聊全员可读写的Notes文档]
    end

    UserInput[用户输入] --> RAGRetrieve[RAG检索]
    RAGRetrieve --> LT1
    RAGRetrieve --> LT3

    RAGRetrieve --> BuildContext[组装上下文]
    ST1 --> BuildContext
    ST2 --> BuildContext
    BuildContext --> BuildPrompt[组装System+Context+User+Tools]

    BuildPrompt --> LLMCall

    LLMCall -->|对话结束后| MemPolicy{记忆策略}
    MemPolicy -->|重要?| SaveLT[存长期向量库]
    MemPolicy -->|普通| SaveST[仅保留短期滑窗]
    SaveST --> ST1
    SaveLT --> LT1
    SaveLT --> LT2
```

### 2.3 Planning 规划能力三种模式

```mermaid
flowchart LR
    subgraph 模式1 ReAct边想边做[最通用]
        T1[用户复杂任务] --> R1[Thought<br/>思考: "我需要先查API文档"]
        R1 --> A1[Action: 调用search_tool("FastAPI JWT")]
        A1 --> O1[Observation: 搜索结果...]
        O1 --> R2{是否解决?}
        R2 -->|否| R1
        R2 -->|是| Finish[Final Answer]
    end

    subgraph 模式2 Plan-and-Execute[大任务先拆解]
        T2[开发学生管理系统] --> P1[LLM生成完整执行计划<br/>1.建模型 2.写接口 3.鉴权 4.测试]
        P1 --> Ex1[Step1执行]
        Ex1 --> P2{步骤1完成?}
        P2 -->|重写| Ex1
        P2 -->|OK| Ex2[Step2执行...]
        Ex2 -->|全部步骤OK| Output2[交付最终结果]
    end

    subgraph 模式3 Reflexion自我反思[最聪明最耗时]
        T3[Bug修复任务] --> Try1[第一次尝试写patch]
        Try1 --> RunTest[运行单测 → 5失败]
        RunTest --> Reflect[Self-Reflection<br/>LLM自我反思: "为什么失败? 第3行逻辑反了!"]
        Reflect --> SelfCritic[自我批评+改进思路写进System Prompt]
        SelfCritic --> Try2[第二次尝试 换思路]
        Try2 --> Run2[测 → 2失败]
        Run2 --> MaxIter{迭代<max=10?}
        MaxIter -->|是| Reflect
        MaxIter -->|否| Human[转人工Handoff]
        Run2 -->|最终0失败| FinalOK[完成]
    end
```

---

## 三、软件开发多Agent团队完整执行流程

### 3.1 从需求到代码完整闭环

```mermaid
flowchart TD
    Input[用户: "做FastAPI学生管理系统 要求JWT+增删改查+pytest>80%"]

    Input --> PM[产品经理Agent]
    PM -->|写出| PRD[PRD.md<br/>背景/目标/UserStory×5/验收标准]
    PRD -->|含| Stories["US1: 学生POST创建<br/>US2: GET列表分页<br/>US3: JWT登录<br/>US4: PUT/PATCH更新<br/>US5: DELETE删除"]

    PM --> SayApproved[PM说: APPROVED 需求确认]

    SayApproved --> Arch[架构师Agent]
    Arch -->|输出| DesignDoc[DesignDoc.md]
    DesignDoc -->|包含| ArchStruct["三层架构: models/schemas/routers<br/>技术: FastAPI+SQLAlchemy+SQLite+PyJWT"]
    ArchStruct --> DBSchema["DB Schema: users(id,username,pass_hash)<br/>students(id,name,age,email,create_at)"]
    ArchStruct --> APIs[RESTful: 7个API路径+请求响应样例]

    Arch --> Say2[架构师说: APPROVED 设计完成]

    Say2 --> Coder[程序员Agent + CodeInterpreterTool]
    Coder --> File1[生成main.py应用入口]
    Coder --> File2[生成models.py SQLAlchemy模型]
    Coder --> File3[生成schemas.py Pydantic校验]
    Coder --> File4[生成routers/下auth.py students.py]
    Coder --> File5[生成requirements.txt]

    Coder --> RunPip[虚拟环境pip install依赖]
    RunPip --> StartServer[uvicorn跑起本地服务:8000]
    StartServer --> APITest[CodeInterpreter调用curl测试API]

    APITest --> CoderFix{测试失败}
    CoderFix -->|401未授权| FixAuth[修复JWT中间件逻辑]
    FixAuth --> APITest

    Coder --> Say3[程序员说: COMPLETE ✓ 代码+自测通过]

    Say3 --> Tester[测试Agent]
    Tester --> TestCases[生成test_students.py 30个用例]
    TestCases --> RunTests[pytest --cov=app 测覆盖率]
    RunTests --> CoverageReport[生成Coverage报告: 83% ✓ >80%]

    Tester --> Say4[测试Agent说: PASS 全绿覆盖率83%]

    Say4 --> Reviewer[代码审查官]
    Reviewer --> CheckList[审查: SQL注入/XSS/并发/PII/错误处理]
    CheckList --> Issues[Issue1: 用户密码hash强度弱 用bcrypt<br/>Issue2: /docs公开 产品环境要关]
    Issues --> LowRisk[风险中低 建议优化 不阻塞]
    Reviewer --> Say5[Reviewer说: PASS 可上线 + 改进建议清单]

    Say3 --> TriggerTerminate
    SayApproved --> TriggerTerminate
    Say4 --> TriggerTerminate
    Say5 --> TriggerTerminate

    TriggerTerminate[条件满足: COMPLETE+APPROVED+PASS ✓ ✓ ✓] --> Output[输出最终项目压缩包<br/>代码+PRD+设计+测试报告+覆盖率]
```

---

## 四、终止条件与安全防护流程

### 4.1 五层死循环防护网

```mermaid
flowchart TD
    AgentLoop[Agent执行循环] --> L1{层1: 硬迭代上限max_iter=10}
    L1 -->|>10次| STOP1[强制终止! 返回部分结果]

    L1 -->|OK| L2{层2: EarlyStopping连续3轮无进展}
    L2 -->|输出相同/相同工具调用| STOP2[终止: Agent陷入循环]
    L2 -->|有新信息| L3{层3: 相同工具+相同参数重复≥3次}
    L3 -->|重复调用同一个API| STOP3[终止: 可能是ToolError导致]

    L3 -->|OK| L4{层4: Token消耗超过MaxBudget=100K}
    L4 -->|超| STOP4[终止: Token超预算]

    L4 -->|OK| L5{层5: 人工介入Handoff触发}
    L5 -->|Agent说"I need human help"| HUMAN[转人工后台介入]
    L5 -->|OK| NextIter[进入下一轮迭代]

    STOP1 --> Log[记录退出原因+状态]
    STOP2 --> Log
    STOP3 --> Log
    STOP4 --> Log
    HUMAN --> Log
```

### 4.2 终止条件表达式

```python
# AutoGen 支持 | 和 & 逻辑运算符组合条件
from autogen_agentchat.conditions import (
    TextMentionTermination,   # 出现某关键词就终止
    MaxMessageTermination,    # 超过N条消息
    MaxTokensTermination,     # Token预算上限
    TimeoutTermination,       # 时间上限
    HandoffTermination,       # Agent转交给人类
)

# 软件开发场景的终止条件组合:
termination = (
    # 程序员说"COMPLETE" 或者 总消息超20条 硬防
    TextMentionTermination("COMPLETE") | MaxMessageTermination(20)
) & (  # 并且 同时满足
    # 产品经理APPROVED 加 评审PASS
    TextMentionTermination("APPROVED") & TextMentionTermination("PASS")
)
# 解读: 要同时满足 (有COMPLETE或消息爆20条) 并且 (APPROVED和PASS都说了)
# 相当于: 代码写完了(COMPLETE) 且需求确认了(APPROVED) 且审查通过了(PASS)
#          或者 强行到20条了 同时前面的APPROVED和PASS也齐了
```

---

## 五、工具执行与沙箱安全流程

### 5.1 Code Interpreter执行流程

```mermaid
flowchart TD
    AgentNeed[Agent想执行代码验证] --> GenCode[生成Python代码块]
    GenCode --> StaticCheck[静态AST安全检查]
    StaticCheck --> Danger{危险操作?}
    Danger -->|import os.system<br/>subprocess/写文件/etc/| Reject[❌拒绝执行 安全风险]
    Danger -->|安全: pandas/numpy/matplotlib计算| Docker[启动Docker沙箱容器]

    Docker --> MountVol[挂载临时工作目录/workspace]
    MountVol --> NetOff[默认关闭网络! 防止下载恶意包]
    NetOff --> MemLimit[内存限制 2GB 防止fork炸弹]
    MemLimit --> CPU[CPU核数限制 2核]

    CPU --> Exec[python3 user_code.py 容器内执行]
    Exec --> OutputSTDOUT[捕获stdout/stderr/图片/文件]
    OutputSTDOUT --> Clean[强制销毁容器 不留残余]
    Clean --> ReturnResult[返回 stdout+生成图表图像给Agent]
```

### 5.2 Tool调用安全审计

```
每次工具调用都会记录的审计字段:
┌────────────────────────────────────────────┐
│ timestamp: 2026-07-25T12:30:00             │
│ agent: SeniorCoder                         │
│ tool_name: get_pII_order_by_id             │
│ args: {"order_id": 12345}                  │
│ user_has_permission: user.role∈[admin,op]  │
│ args_has_pii_leak: args.order_id合法 ✓     │
│ execution_latency_ms: 128                  │
│ return_value_length: 362字节               │
│ return_value_has_excessive_pii: 否 ✓       │
│ approved_by_policy: 是 ✓                   │
└────────────────────────────────────────────┘
```