# AutoGen 性能优化重难点解析

> 位置: 06-llm/autogen/doc/
> 配套文档: AutoGen-MultiAgent协作框架.md | AutoGen流程图详解.md | AutoGen面试题汇总.md

---

## 一、Multi-Agent性能瓶颈分析

### 1.1 Agent vs LLM调用链延迟结构 (90%项目踩的坑

```
单Agent LLM调用链路 基础流程：
用户输入 → Prompt组装 1ms
     ↓ 3ms Tokenize
LLM HTTP请求 (网络R/T 首Token 800ms
     ↓
逐Token生成 30 Tok ×50ms=1.5s
Tool调用+结果返回 25ms
总计: 约**2.4s** / 每轮

Multi-Agent 5人群聊 ×10轮= 50次LLM调用 = 120秒 = **2分钟起步!**
```

| 瓶颈 | 占比 | 原因 |
|-----|------|-----|
| LLM网络延迟+推理 | **80-90%** | 每次API调用/模型推理 |
| Tool调用开销 | 5-10% | 代码解释器/数据库/搜索API |
| 死循环重试 | 5-15% | 2-3次重复调用才对 |
| 协调/上下文拼装 | <2% | 上下文构造、tokenize、序列化 |

### 1.2 6大优化维度对照表

| 优化方向 | 手段 | 延迟减少 | 成本减少 | 准确率影响 |
|-------|-----|-------|-------|----------|
| **模型选型 | gpt-4o → gpt-4o-mini → 本地Ollama qwen | **70%** | **95%** | -5~-15% |
| **并行执行** | Tool调用async并发 | 2-5x | 0% | 0% |
| **缓存Semantic** | 相似问题命中直接返回 | 90% (命中时) | 80% | 0% |
| **Prompt压缩** | 去冗余、摘要旧对话、SlidingWindow | 30-50% 更快+省Token | 40% | 轻微+ |
| **减少Agent人数** | 5人群聊 → 3人分工 | 40% | 40% | 质量-5% |
| **SSE流式** | 首字800ms→120ms感观 不等待全部Token | 主观快5-10x体验 | 0% | 0% |

---

## 二、模型选型与成本控制

### 2.1 Agent分层路由 (Hot/Warm/Cold三层模型分级调用

```mermaid
flowchart TD
    UserTask[用户任务] --> Router{意图分类LLM Judge]

    Router -->|简单任务 闲聊/格式化/简单抽取| Mini[GPT-4o-mini / 本地Qwen-7B]
    Router -->|中等任务 代码/工具用标准问答| Mid[GPT-4o / Claude-3.5]
    Router -->|复杂任务 架构设计/复杂推理| Large[GPT-4o / Claude-3-Opus]

    Mini --> Cost1["成本: $0.15/1M入+0.6/1M出 tokens<br/>延迟: 300ms/1K tokens<br/>准确率基线: 简单任务95%"]
    Mid --> Cost2["成本: $5/1M入+$15/1M出<br/>延迟: 800ms/1K<br/>中等任务准确率=90%"]
    Large --> Cost3["成本: $75/1M入<br/>延迟: 1.8s/1K<br/>复杂任务80%→95%"]
```

代码实现：
```python
# FallbackChain 失败降级 确保可用性
async def smart_llm_call(user_prompt, task_complexity):
    models = ["gpt-4o-mini", "gpt-4o", "claude-3-opus"]
    idx = min(task_complexity//33)  # 0/1/2 选模型
    for model in models[idx:]:
        try:
            return await llm(model, prompt=user_prompt)  # 先选模型
        except RateLimitError:
            continue  # 限流降级到下一个贵模型
    raise AllModelsBusy()
```

### 2.2 Token消耗估算 (1个软件开发Agent团队)

```
1个用户请求 → 10轮对话 → 50次LLM调用
─────────────────────────────
平均每次输入 8K tokens (含历史+工具输出+System Prompt)
平均每次输出 1K tokens
─────────────────────────────
50×(8K输入 + 1K输出) = 450K tokens / 用户请求

GPT-4o成本:
  输入8K×50×$5/1M  = 400K×$5 = $2.00
  输出1K×50×$15/1M = 50K×$15  = $0.75
  合计 **$2.75 / 用户请求 ❌ 太贵！

GPT-4o-mini 80%任务替代后成本:
  40次mini: $0.06, 10次4o: $0.55 → **$0.61 / 请求 (省78%)

**进一步优化: 3级缓存50%命中 → **$0.3 / 请求
```

成本金字塔：大模型只留20%最难点任务量占80%用便宜模型

---

## 三、死循环 & 重试机制

### 3.1 死循环的5个典型根因+防护

| 死循环类型 | 现象 | 根因 | 修复代码 |
|----------|------|-----|--------|
| **JSON解析循环** | 每次输出Schema错→重试→还错 | LLM不会生成严格JSON | 加Pydantic OutputParser+JSON_Repair |
| **参数校验循环** | 每次参数都传错order_id类型 | 参数描述不清楚 | 优化Tool参数描述 Few-Shot示例 |
| **相同工具重复调用** | 连3次调用search("天气" 相同参数 | Tool返回结果LLM不理解不处理 | EarlyStopping相同参数2次→跳过 |
| **Human-in-the-Loop 卡死** | Agent一直Handoff转人工没人接 | Human Agent | 超时未响应 |
| **自我怀疑循环** | Planner→Executor→Critic: "不对"→Planner重写 | 评估标准模糊不清 | 设MaxRounds≤3 Critic轮次硬性封顶 |

#### ✅ JSON Repair 拯救50%的解析失败

```python
# 50%的LLM JSON错误是少了逗号/引号。装包就能修
import json_repair

def robust_json_parse(raw_text):
    try:
        return json.loads(raw_text)  # 先正常解析
    except:
        try:
            return json_repair.loads(raw_text)  # 90%坏JSON能修好
        except Exception as e:
            # 再不行: 提取{...}最外层大括号
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            return json_repair.loads(match.group())
```

### 3.2 五层防护网代码（面试逐行可考代码示例：

```python
class AntiDeadLoopProtector:
    def __init__(self, max_iter=10, early_stop_n=3):
        self.iter_count = 0
        self.history_calls = Counter()  # 工具调用计数
        self.max_iter = max_iter
        self.early_stop_n = early_stop_n

    def before_iter(self, agent_name, tool_call):
        # 层1: 硬迭代上限
        self.iter_count += 1
        if self.iter_count >= self.max_iter:
            raise TerminationException(f"HardStop: {self.max_iter} iterations")

        # 层2: 相同工具+相同参数 N次重复
        key = (agent_name, tool_call.name, frozenset(tool_call.args.items()))
        self.history_calls[key] += 1
        if self.history_calls[key] >= self.early_stop_n:
            raise TerminationException(f"SameToolSameArgs: {key}")

        # 层3: 输出内容相似度 (连续输出相同话)
        # 层4: Token Budget
        # 层5: 时间超时 wall_clock timeout 300s硬终止
```

---

## 四、Memory优化

### 4.1 三层Memory的Token预算控制

```
原始对话历史 100轮对话 600K Token → 直接全塞上下文 ❌ 爆128K窗
优化后三层架构:
┌───────────────────────────────────────────────┐
│ L1 滑窗最近10轮 (原始文本)      ~=  8K Token │ ← 给LLM看最鲜活对话 热记忆
├───────────────────────────────────────────────┤
│ L2 中间90轮 (LLM自动摘要总结段落)    4K Token │ ← 不丢失关键决策点
├───────────────────────────────────────────────┤
│ L3 全量旧对话向量库 (RAG检索Top4)   8K Token │ ← 需要时从向量库TopK召回
└───────────────────────────────────────────────┘
  Total 送入LLM Context总计: 20K Token ← 比原来600K省×30
```

实现滑动摘要缓冲，每满20轮触发自动摘要：
```python
# AutoGen实现 (SummaryBufferMemory实现)
class RollingSummaryMemory:
    def trigger_summarize_if_needed(self):
        # 满20轮 中间10轮总结成1段摘要
        if len(self.buffer) - self.last_k - self.window >= 20:
            middle = self.buffer[self.last_k : -self.window]
            summary = llm(f"Summarize: \n{middle}")  # 压缩10轮→100字
            self.summaries.append(summary)
            del self.buffer[self.last_k:-self.window]
```

### 4.2 Semantic Cache 语义缓存

```
相同问题 "如何加JWT鉴权?" → 语义哈希 → 向量 → 相同语义哈希
                              ↓命中→ 42. 回答返回缓存
新问题 "FastAPI怎么做JWT登录认证" → 余弦相似度>0.95 → 命中

实现：
1. 问题 Embedding 存向量数据库: Redis + pgvector
2. Top1 相似度>0.95 直接返回缓存答案
3. 不命中 → 走正常Agent流程再写回缓存

省80%常见重复问题（客服机器人场景
```

---

## 五、Tool调用优化

### 5.1 Tool调用Schema优化（Schema优化占Tool成功率 50%的LLM选错/参数Schema质量直接决定50%以上

```python
# ❌ 错误：Schema描述太模糊
@Tool
def search_wiki(keyword):
    "搜索"

# ✅ 正确：Schema描述+类型+枚举+FewShot示例（FewShot示例给例子
@Tool(description="""
根据订单号查订单详情【示例:
- 入参: order_id长整型(如12345)
- 出参: OrderInfo对象
- 错误: 404找不到抛OrderNotFoundException
""", return_value=OrderInfo)
def get_order(order_id: Annotated[long, "订单ID长整型"]) -> OrderInfo:
    return orderMapper.selectById(order_id)  # 业务代码
```

### 5.2 并发异步并发批量合并工具

```python
# ❌ 串行：3次搜索 3*800ms=2.4s
a = search("财务报告")
b = search("技术方案")
c = search("竞品分析")

# ✅ asyncio.gather 并发同时发：800ms搞定，等最慢的一个
import asyncio
results = await asyncio.gather(
    asearch("财务报告"),
    asearch("技术方案"),
    asearch("竞品分析")
)  # 并发3倍快
```

---

## 六、开发场景高并发生产优化

### 6.1 并发请求队列 + Worker Pool 线程池

```
用户请求1 ──→ ┌──────────────────┐  队列排队  BackPressure流量削峰填谷
用户请求2 ──→ │  Redis Queue  │ ──→ Worker 1 (AgentTeam1 处理)
用户请求N ──→ │  (优先级队列) │ ──→ Worker 2 (gpt-4o-mini快速通道
              └──────────────────┘ ──→ Worker N

生产环境：
- 队列：Redis/Celery/RabbitMQ
- 持久化：请求失败重试+死信队列
- 每Agent实例池：并发控制10并发/单GPU/实例限制
```

### 6.2 监控指标体系 (6大监控：

| 指标名 | 告警阈值 | 说明 |
|-------|---------|------|
| **端到端成功率 | <95%告警 | 用户请求最终成功率 |
| **每轮平均LLM调用次数 | >15告警 | 死循环嫌疑 |
| **Tool调用成功率 | <90%告警 | JSON解析/参数校验错 |
| **平均首Token延迟 | >3s告警 | 网络延迟异常 |
| **Token单请求成本 | >$1告警 | 成本异常消费 |
| **Cache命中率 | <50%告警 | 缓存没生效 |

监控面板Dashboard: Grafana + Prometheus指标：
```python
# 打点代码模板指标
metrics = Counter("agent_success_total", "Total Agent executions", labelnames=["agent_name", "task_type"])
histogram = Histogram("agent_latency_seconds", "End to end latency", buckets=[0.5, 1, 3, 5, 10, 30])
```

---

## 七、常见报错速查Checklist

| 症状 | 排查清单 (按出现频率排查概率：
1. 「LLM JSON输出JSONSchema错 → 50%+json_repair.py修
2. 「工具调用参数传错type → Schema参数加Annotated类型描述重写详细详细FewShot示例
3. 「Agent跑死循环转圈圈 → 5层死循环Protector.max_iter=10检查
4. 「上下文窗口爆ContextWindowOverflowError → SlidingWindow短摘要短上下文
5. 「成本1单请求$5+ → 模型路由Router小模型小模型+缓存
6. 「首Token首字响应慢>3s+ → 启用SSE流式+用户体验开