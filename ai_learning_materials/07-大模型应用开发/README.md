# 🚀 07 - 大模型应用开发LLM 章节导览

> **⭐⭐⭐⭐⭐ 2025 AI岗位需求量最大、薪资溢价最高、进入门槛最友好的方向。无论你原有什么技术栈，都强烈推荐学这个！**
> 预计学习周期：3周 (21天) | 目标掌握度：⭐⭐⭐⭐⭐ L5精通级 (简历核心项目来源)
> 配套项目路径：`../../06-llm/langchain/` / `llama_index/` / `autogen/`

---

## 📚 本章节文件索引

| 文件名 | 内容 | 优先级 |
|-------|------|--------|
| **README.md** (本文) | 全景图谱 + 四大核心技术 + 项目组合 | ⭐⭐⭐⭐⭐ 先读 |
| **LLM基础与Prompt工程.md** | Prompt七技巧 / CoT / 结构化输出 / 微调选型对比表 | ⭐⭐⭐⭐⭐ 必学 |
| **LangChain深度解析.md** | LCEL管道语法 / RAG链 / Memory / Router + 完整代码 | ⭐⭐⭐⭐⭐ 必学 |
| **LlamaIndex专业RAG.md** ⭐ | Advanced-RAG 7招升级 / RAGAS 5指标评估 / GraphRAG | ⭐⭐⭐⭐⭐ 进阶高薪 |
| **Multi-Agent多智能体.md** | ReAct模式 / LangGraph / AutoGen 3种协作模式 | ⭐⭐⭐⭐ 加分 |
| **代码实战.md** | LangChain RAG系统 + LangGraph Agent两个完整项目 | ⭐⭐⭐⭐⭐ 必做 |
| **面试题库.md** | 100道LLM面试题 + 答案 (RAG选型对比/Agent失败模式必出) | ⭐⭐⭐⭐⭐ 必背 |
| **GitHub项目推荐.md** | 三大框架源码阅读路线 | ⭐⭐⭐ 参考 |

---

## 🗺️ LLM应用四大核心技术图谱

```mermaid
graph TD
    A[LLM应用能力全景] --> B1[🛠️ Prompt工程 根基]
    A --> B2[📚 RAG 检索增强生成 ⭐80%项目都用]
    A --> B3[🤖 Agent 工具调用/自主规划 2025爆发]
    A --> B4[🔧 Fine-tuning 微调 LoRA/QLoRA 定制化]

    B1 --> B1a[Zero/Few-shot示例]
    B1 --> B1b[CoT思维链: "Let's think step by step"]
    B1 --> B1c[结构化输出 JSON Schema/Pydantic]
    B1 --> B1d[角色人设 System Prompt]
    B1 --> B1e[安全 提示注入防护]

    B2 --> B2a[基础Naive-RAG: 切块→向量→检索→LLM]
    B2 --> B2b[⭐Advanced-RAG 7招升级: 语义切块/Query改写/重排/RRF融合...]
    B2 --> B2c[⭐Modular-RAG: 评估优化循环 RAGAS自动调参]
    B2 --> B2d[GraphRAG: 知识图谱+向量 多跳推理]

    B3 --> B3a[⭐ReAct循环: Thought→Action→Observation]
    B3 --> B3b[Tool Function Calling 参数校验+重试]
    B3 --> B3c[LangGraph显式状态机]
    B3 --> B3d[Multi-Agent群聊/流水线/等级制]

    B4 --> B4a[SFT有监督微调: 领域指令数据/人设语气]
    B4 --> B4b[LoRA/QLoRA: 只训A×B小矩阵, 70B→几百MB]
    B4 --> B4c[DPO偏好对齐: 替代RLHF]
```

---

## ⚡ 核心1：RAG 从入门到高薪 (Naive → Advanced 7招升级)

### 📊 RAG 发展三阶段对比

| 阶段 | 技术组合 | 准确率水平 | 简历写法价值 |
|-----|---------|-----------|-------------|
| 👶 **Naive RAG** | 固定大小Token切块(500/50重叠) + TopK向量召回 + Stuff塞上下文 | 60%~70% | ❌ 不建议写：太烂大街了 |
| 🧑 **Advanced RAG** ⭐ | ⭐语义切块/SentenceWindow + ⭐HyDE伪文档/多查询生成 + ⭐混合检索(BM25+向量 RRF融合) + ⭐BGE-reranker重排(Top50→Top4) + ⭐LongContextReorder头尾优化 | 80%~92% | ✅⭐ 写这个：足够秒杀80%候选人 |
| 👑 **Modular RAG** | 上面 + RAGAS 5指标自动评估 + Self-RAG自我反思检索 + CRAG检索正确性校验 + Query Router路由(简单问题不检索省Token) | 90%~97% | ⭐⭐⭐ 写这个：直通大厂高薪 |

> 💰 **简历黄金句式 (照着改就行)**：
> `「LlamaIndex搭建企业级金融合同知识库RAG：SentenceWindow 3句话切块+BM25向量RRF融合+HyDE查询增强+bge-reranker-v2-m3精排+Lost in the Middle 头尾重排，覆盖12年研报800万份，RAGAS Faithfulness从62%→93.8%，Top3命中率从62%→88%，律师检索从2小时→3分钟，月均Token成本$18K→$7K」`

### 🔍 RAG的完整7步执行流程图 (面试照着画)

```mermaid
flowchart LR
    U[用户问题] --> 1[1.Query查询增强改写<br>HyDE/StepBack/多查询生成]
    1 --> 2[2.混合检索召回<br>BM25关键词+向量相似度 RRF融合 TopK=50]
    2 --> 3[3.精排Reranker<br>Cohere/bge-reranker Top50→Top4]
    3 --> 4[4.上下文重排序<br>LongContextReorder 放最相关在头尾]
    4 --> 5[5.Prompt组装<br>Context + 问题 + 输出格式要求]
    5 --> 6[6.LLM生成<br>流式SSE输出]
    6 --> 7[7.后处理+Groundness校验<br>答案溯源引用+幻觉检测]
    7 --> R[最终答案 + 引用来源文档]
```

---

## 🎮 核心2：Agent 智能体 (2025最热门增长点)

### ReAct循环 = 智能体的大脑 (面试标准答案)

```
【用户问题】："2024年第三季度阿里巴巴的净利润同比增长率？财务报告里找"

Thought 1: 我需要先找到阿里巴巴2024Q3财务报告。
            应该调用 「WebSearch工具」 搜索关键词。
 Action 1: tool=web_search, params={"query": "阿里巴巴 2024 Q3 财务报告 净利润"}
Observ 1: 搜到了PDF链接: https://.../2024Q3.pdf，里面有2024Q3净利润387亿，2023Q3净利润308亿。

Thought 2: 我现在拿到了两年的净利润数据。需要用 「计算器工具」 算同比增长率。
            公式: (387 - 308) / 308 × 100%
 Action 2: tool=calculator, params={"expr": "(387-308)/308*100"}
Observ 2: 计算结果 = 25.65%

Thought 3: 已经得到所有需要的信息。可以给用户最终回答了。
 FinalA: "2024年Q3阿里巴巴净利润同比增长率约25.65% (387亿→308亿)，
          来源: 2024Q3未经审计财报第15页利润表。"
```

### ⚠️ Agent 4大失败模式 (面试必问：怎么防止无限循环?)

| 失败模式 | 现象 | 缓解方案 |
|---------|------|---------|
| 🔄 死循环 | 重复调用相同工具相同参数 | max_iter=10硬终止 + 连续3次同工具EarlyStop |
| ❌ 参数格式错 | Tool Calling JSON解析失败占50%！ | Pydantic Schema校验 + 重试2次 + Few-shot示例 |
| 🎲 工具选错 | 应该用计算器选了网页搜索 | Planner先显式写Plan再执行 / RouterAgent分类 |
| 🤥 幻觉编造观察结果 | 不调用工具，直接编"工具返回xxx" | 强制严格模式：必须等真实Tool执行完，不接受LLM编造 |

> 💡 LangGraph 解决80%以上问题：把Agent从"LLM自由发挥"改成 **显式状态机 + 条件边**，每个节点做什么、边怎么流转都是你代码写死的。Agent失败率降低一个数量级！

---

## 🔧 核心3：Prompt工程 七大必会技巧 (每面都考)

| # | 技巧 | 模板示例 | 适用场景 |
|---|------|---------|---------|
| 1 | **人设注入 System Prompt** | `你是10年经验的Java架构师，请用Spring Boot 3最佳实践回答` | 任何专业场景 |
| 2 | **Few-shot示例** | 先给3个"Question→Answer"对，再给第4个Question让它学着输出格式 | 结构化输出/特殊格式/分类标注 |
| 3 | **CoT 思维链** | `Let's think step by step. 先列出思路，再给出最终答案。` | 数学/推理/规划类问题，准确率+20~30% |
| 4 | **结构化JSON输出** | `严格按如下JSON Schema返回，不要任何多余文字：{"items": [{"name": str, "score": float 0-1}]}` | 接API/数据库入库，解析不报错 |
| 5 | **Self-Check 自我校验** | `输出答案前，先自己检查3个方面：①事实正确？②有没有遗漏要点？③格式符合？` | 重要场景，幻觉减少50% |
| 6 | **角色反转 / 反思** | `作为面试官，请针对你的回答提3个尖锐问题，再逐一回答。` | 生成高质量内容/技术文档 |
| 7 | **分步骤拆分** | `先做A得到结果X。基于X再做B。最后基于B结果给出答案。` | 复杂长任务，每一步检查减少累计错误 |

---

## 🤷‍♂️ 核心4：RAG vs 微调 选型灵魂十问 (必考题！)

| 考虑维度 | 选RAG | 选微调 (LoRA) | 两个都要 ✅ 生产最佳实践 |
|---------|-------|---------------|---------------------|
| 知识更新频率 | 高 (每天/每周变 产品手册/法规) ✅ | 低 (固定知识 季度更新) | 平时RAG动态更新+季度微调语气 |
| 引用溯源要求 | 有 (金融/医疗/法律必须给来源) ✅ | ❌ 微调学了，但说不出哪条来的 | RAG给溯源+微调语气 |
| 数据量小 (<1000条) | ✅ 直接RAG 零成本 | ❌ 数据少微调过拟合 | RAG先试试 |
| 风格语气固定 (客服/作家) | ❌ RAG语气不可控 | ✅ 微调语气/格式/行话 | ✅ 先调语气风格 + RAG填事实 |
| 推理逻辑能力提升 | ❌ RAG塞资料不提升逻辑 | ✅ 思维链微调逻辑提升 | ✅ 微调CoT能力+RAG填充知识 |
| 上线延迟/成本 | ⚠️ 每次检索+拼长上下文贵+慢 | ✅ 直接生成短Prompt省Token | 选微调如果成本敏感 |
| 幻觉零容忍 | ✅ RAG强制看资料 幻觉↓80% | ⚠️ 还是会瞎编 | RAG为主，辅助Groundness校验 |
| 部署复杂度 | 低 (向量库+LLM) | 高 (GPU训练+实验+评估) | 先RAG跑通，之后看ROI要不要微调 |

---

## 🎯 章节结业标准 (达到直接投简历)

- [ ] 能说出从 Naive-RAG → Advanced-RAG的**至少5个具体升级技术点**
- [ ] 能画出RAG 7步流程 + 每步的作用
- [ ] 能写出LangChain LCEL 完整RAG Pipeline 代码 (含重排/流式)
- [ ] 能解释ReAct模式 + Agent 4种失败模式的缓解方案
- [ ] 能说出 RAG vs 微调 选型对比 至少5个维度
- [ ] 能独立搭出一个完整RAG系统：文档入库+检索+SSE流式前端
- [ ] 知道RAGAS 5个核心评估指标：Faithfulness/Relevancy/ContextP/ContextR 各是什么意思
- [ ] 面试题库正确率 ≥ 80%