# LangChain 大模型应用框架解析

> 位置: 06-llm/langchain/libs/langchain/langchain/
> 简历推荐: 5星 | 岗位: LLM应用/RAG/Agent工程师 (2025缺口最大)

---

## 一、LangChain 6大核心模块

```mermaid
graph LR
    USR[业务代码] --> LCEL[LCEL LangChain表达式 管道符|]
    LCEL -->|prompt\|model\|parser| OUT[输出]

    LCEL --> M1[Model接口  LLM/Chat/Embedding]
    LCEL --> M2[Retriever检索器 向量/混合/重排]
    LCEL --> M3[Agent智能体 Tool+循环ReAct]
    LCEL --> M4[Memory对话记忆 Buffer/Summary/Vector]
    LCEL --> M5[Tool工具 @tool装饰器 FunctionCalling]
    LCEL --> M6[Chain链 RAG/SQL/Sequential链]
```

## 二、LCEL 语法 (2024之后统一用这个!)

```python
# LCEL = 管道符 | 像Unix管道一样拼搭
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# ---- 1. 基础Chain ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深工程师,回答要附代码示例."),
    ("user", "{question}")
])
chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
print(chain.invoke({"question": "快速排序Python实现?"}))

# ---- 2. RAG LCEL Chain (标准范式!) ----
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs,   # 先检索文档拼context
     "question": RunnablePassthrough()}    # 用户原问题透传
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)
print(rag_chain.invoke("LangChain的RAG怎么做?"))

# ---- 3. 流式/异步/批量/Fallback ----
for chunk in rag_chain.stream("问题"):     # 流式逐字输出
    print(chunk, end="", flush=True)
answers = chain.batch([{"question":q} for q in qlist])   # 批量
answer = await chain.ainvoke({"question": "异步"})        # 异步协程

# Fallback容错: GPT4挂了自动降级4o-mini
resilient = (prompt|ChatOpenAI(model="gpt-4")|parser).with_fallbacks(
    [prompt|ChatOpenAI(model="gpt-4o-mini")|parser])
```

## 三、Agent ReAct 循环原理

```
ReAct循环 = Thought(思考) → Action(行动选Tool) → Observation(观察工具返回) → ... N轮直到Final Answer
LangGraph显式状态图实现:
  State节点: Agent决策 → 条件边{if 要工具:调用Tool; else:结束}
```

## 四、简历黄金句式

| 写法 |
|-----|
| 「LangChain LCEL搭建法律合同问答RAG：混合检索(BM25+向量RRF)+bge-reranker重排+HyDE伪文档，Faithfulness 62%→93.8%，律师查资料效率↑400%」 |
| 「LangGraph代码修复多轮Agent(Plan+Code+Test+Review四阶段)，SWE-bench Pass@1 27%→56%，Bug修复平均成本↓62%」 |
| 「Multi-Agent写文档团队: PM+架构师+Coder+Reviewer四智能体群聊，生成完整FastAPI项目+单元测试覆盖率87%，开发周期5天→4小时」 |

## 五、面试题

**Q LangChain Expression (LCEL)相比老版Chain类优势？**
> A: ① 管道|语法简单直观 ② 统一支持.stream()流式/.batch()批/.ainvoke()异步(老类要手动改N处) ③ with_fallbacks自动降级/.retry自动重试/.bind_tools函数调用 ④ astream_events()事件流+LangSmith链路追踪开箱即用 ⑤ RunnableLambda任何Python函数→Chain无侵入。

**Q RAG vs Fine-tuning 选型？**
> A: 场景选RAG: ①知识频繁变更(产品手册/法规) ②引用溯源要求 ③幻觉零容忍 ④数据量小(几千条)。场景选微调: ①固定风格语气/格式(客服/作家) ②特定领域专业术语/行话 ③推理能力/逻辑思维能力提升 ④减少Prompt成本/Token数。混合最佳：先RAG兜底+微调语气。

**Q Agent常见失败模式？如何缓解？**
> A: ① 工具调用参数错JSON → Pydantic输出校验+重试2次+Few-shot示例 ② 死循环Tool调用不结束 → max_iter=10 + EarlyStopping + 超阈值给用户 ③ 工具选的不对 → Planner子Agent先写详细Plan再执行 ④ 幻觉编造工具结果 → 工具返回强制引用+Groundness校验。