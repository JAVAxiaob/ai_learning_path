# LlamaIndex RAG性能优化重难点解析

> 位置: 06-llm/llama_index/doc/
> 配套文档: LlamaIndex-RAG索引引擎详解.md | LlamaIndex流程图详解.md | LlamaIndex面试题汇总.md

---

## 一、Advanced-RAG 7大优化手段详细说明

### 1.1 切块优化Chunking (占RAG质量 30%权重)

| 切块算法 | 适用文档 | 准确率 | 速度 |
|---------|---------|-------|-----|
| Token固定分块 | 通用 | ⭐⭐ | 最快 |
| SentenceWindow窗口 | 句子密集/法律 | ⭐⭐⭐⭐ | 快 |
| Semantic语义切块 | 语义段落清晰/论文 | ⭐⭐⭐⭐⭐ | 慢×3(要算Embedding) |
| Hierarchical父子Chunk | 书籍/长报告 | ⭐⭐⭐⭐⭐ | 中 |

> 🏆 最佳实践：语义切块 > 句子窗口 > 固定Token切块。小数据先固定chunk快速baseline再迭代

**语义切块代码：**
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
# 断点相似度阈值：相邻两句向量余弦<0.85则断开
parser = SemanticSplitterNodeParser(
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    breakpoint_percentile_threshold=85,  # 85百分位=中等敏感度
    buffer_size=3
)
documents = parser.get_nodes_from_documents(docs)
```

---

### 1.2 查询优化 Query Transform (质量 +10-20%)

| 技术 | 原理 | 最佳场景 |
|-----|------|---------|
| **HyDE 伪文档检索** | 让LLM先"猜"一个答案D'，用D'去搜 而不是用Query搜 | 开放域问答/技术问题 比原Query强很多 |
| StepBack抽象 | Q→更高层抽象问题先答 背景知识+具体Q | 日期/财报/具体事实类 |
| QueryRewrite重写 | 用户口语化问句→正式书面问句 | 对话口语/Chat场景 |
| SubQuestion子问题 | 复杂Q拆成N个子问题各自检索 | 跨文档/多步骤推理 |

```python
from llama_index.core.query_engine import TransformQueryEngine, SubQuestionQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE
hyde = HyDEQueryTransform(include_original=True)  # 原Query+假Query双路并行
hybrid_engine = TransformQueryEngine(vector_query_engine, hyde)
```

---

### 1.3 混合检索 + 重排 (质量+15%)

```
质量三步排序：
Step 1 召回 (宽):  BM25关键词 + 向量语义 + RRF融合 → Top50 候选
          召回率Recall要>98% 别漏正例！
Step 2 元数据过滤Filter: 时间/部门/权限先过滤掉无关
Step 3 精排 (准):  Reranker BGE-M3/ColBERT → Top4 送LLM
          精度Precision 保证给LLM的都是真相关
```

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.retrievers.bm25 import BM25Retriever

# 双路召回
vector_ret = VectorIndexRetriever(index, similarity_top_k=50)
bm25_ret = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=50)

# RRF融合
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor import LongContextReorder

# Reranker精排 50→4
colbert = ColbertRerank(
    top_n=4, model="colbert-ir/colbertv2.0", keep_retrieval_score=True
)

# 解决Lost-in-the-Middle: 把最相关的放Prompt的头和尾！
reorder = LongContextReorder()

query_engine = RetrieverQueryEngine.from_args(
    vector_ret, node_postprocessors=[colbert, reorder]
)
```

---

### 1.4 Lost in the Middle 现象与修复

现象：LLM注意力U型分布 → 开头/结尾内容看的特别清楚，中间塞进去的内容容易直接忽略 → TopK中间放最重要的文档反而没看到

缓解三招：
1. **LongContextReorder后置处理器**：LlamaIndex内置，把最高分的文档轮流放头/尾，避免堆中间
2. **降低TopK**：K=8塞太满，不如K=4精排后的。用精排少而精 > 堆一堆垃圾文档
3. **Map-Reduce式合成**：不是全塞一个Prompt，每个文档单独回答子问题再Reduce汇总

---

## 二、评估体系 (面试必考RAGAS 5指标)

### 2.1 各指标含义详细对照

| 指标 | 全称 | 范围 | 检查的是 | 常见低分原因 |
|-----|------|-----|---------|-----------|
| **Faithfulness忠实度** | - | 0-1 越高越好 | 答案有没有幻觉？每句能不能在Context找到证据 | 上下文太少/检索漏了 |
| Answer Relevancy | - | 0-1 | 有没有答非所问跑题？ | Prompt模板有问题/LLM发散 |
| Context Precision | - | 0-1 | 检索TopK中，真正相关的是不是排在前面 | 向量模型选的差 |
| Context Recall | - | 0-1 | 答案用到的证据，检索返回的Context覆盖率够不够 | Chunk切太大/太小漏了 |
| Context Entity Recall | - | 0-1 | 答案的实体/数字/专有名词，是不是都在Context里有 | Metadata过滤掉了正确文档 |

> 🏆 **及格线**: Faithfulness≥90%, AnswerRelevancy≥85%

### 2.2 RAGAS自动打分代码

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)

# 准备评估集 至少100道黄金问答对
eval_dataset = Dataset.from_dict({
    "question": ["2024Q3营收同比增长?", "退货政策是什么?"...],  # 100道
    "ground_truth": ["同比+23%", "30天无理由"...],  # 专家标答
})

# 跑RAG系统生成答案和Context
result = evaluate(eval_dataset, metrics=[
    faithfulness, answer_relevancy, context_precision, context_recall
])
print(result)
# {'faithfulness': 0.9345, 'answer_relevancy': 0.8912, ... }
```

---

## 三、成本与速度优化

### 3.1 Embedding模型选型

| 模型 | 向量维度 | 中文效果 | 成本 (1M Tok) | 速度 |
|-----|--------|---------|-------------|-----|
| bge-small-zh-v1.5 | 512 | ⭐⭐⭐⭐⭐ | **本地免费** ⭐首选 | GPU快 |
| text-embedding-3-small | 1536 | ⭐⭐⭐⭐ | $0.02 便宜 | API |
| text-embedding-ada-002 | 1536 | ⭐⭐⭐ | $0.10 | 老版不推荐 |
| bge-m3 | 1024 | ⭐⭐⭐⭐⭐ 多语言最强 | 本地免费 | 略慢 |

> 🏆 中文项目：本地部署bge-small-zh-v1.5，成本=0，效果比ada-002中文好3-5点

### 3.2 大模型成本 (生成阶段)

场景：1000次查询/日，平均每查询上下文8K Token + 1K输出

| 模型 | 每日成本$ | 月成本$ | 质量 |
|-----|---------|--------|-----|
| gpt-4o-mini | $0.24 | $7.2 | ⭐⭐⭐⭐ 客服场景够 |
| gpt-4o | $8.4 | $252 | ⭐⭐⭐⭐⭐ 复杂推理 |
| Qwen2-72B本地部署 | **$0 硬件折旧** | - | ⭐⭐⭐⭐ |

---

## 四、向量数据库索引选型

| 索引 | 适用数据量 | Recall@5% | 内存 | 构建速度 |
|-----|----------|----------|-----|--------|
| HNSW | 1M以下 | 99% | 较大(×2向量) | 快 O(N log N) | ⭐默认首选 |
| IVFFlat | 1M-1亿 | 95-98% | 中 | 中 | 超大规模才用 |
| IVF-PQ 乘积量化 | >1亿 | 90-95% | ×1/16极小 | 慢 | VLDB亿级向量 |
| DiskANN | 超大规模 | 98% | 内存极小SSD | 中 | 最新技术 |

HNSW参数调优 (pgvector示例)：
```sql
-- m=16 邻居数(越大Recall高内存大), ef_construction=64 构建搜索深度
CREATE INDEX ON docs USING hnsw (emb vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
-- 查询时ef_search越大越准越慢
SET hnsw.ef_search = 128;
```