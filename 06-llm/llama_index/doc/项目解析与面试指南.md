# LlamaIndex 专业级RAG框架解析

> 位置: 06-llm/llama_index/llama-index-core/llama_index/
> 简历推荐: 5星 | 岗位: RAG/知识库工程师 (专精RAG一件事做到极致)

---

## 一、LlamaIndex 专而精 vs LangChain 大而全

```
LangChain = 通用工具台: 什么都能做 (Agent/Tool/Chat/RAG百宝箱)
LlamaIndex = 精密RAG瑞士军刀: 只做RAG, 把每一步优化做到天花板
LlamaIndex RAG特色:
├── Connectors 200+种: Google Drive/Notion/GitHub/Confluence/微信聊天/SharePoint...  一键连
├── Chunking切块: SentenceWindow/语义切块/Markdown结构化切块/Hierarchical分层
├── Index索引: Vector向量 / Summary摘要树 / Tree树 / 关键词 / KnowledgeGraph知识图谱
├── QueryEngine查询引擎: Router路由 / SubQuestion子问题拆解 / SQL2Text / Pandas表格问答
├── RAG优化7件套: HyDE/混合检索BM25+向量/重排Reranker/父文档/Metadata过滤/QueryTransform/Self-RAG
└── Evaluation: RAGAS 5指标+ LLM-as-Judge 自动打分 (Faithfulness/Relevancy/...)
```

## 二、高级RAG 7招实战 (面试必背!)

```python
# ===== 完整高级RAG流水线示例 =====
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    Settings, get_response_synthesizer
)
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser
from llama_index.core.retrievers import AutoMergingRetriever, RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# ---- 1. 切块优化: 语义切块+句子窗口 (比硬切Token强!) ----
parser = SentenceWindowNodeParser.from_defaults(
    window_size=3, window_metadata_key="window", original_text_metadata_key="original"
)
# 或语义切块: 语义断点切, 不硬切在句子中间
semantic_parser = SemanticSplitterNodeParser(embed_model=Settings.embed_model)

# ---- 2. 混合检索 + 重排 + 父文档检索粗排Top50 → 精排Top4 ----
vector_retriever = index.as_retriever(similarity_top_k=50)
# Colbert/BGE Reranker把50篇精排到4篇给LLM
reranker = ColbertRerank(top_n=4, model="colbert-ir/colbertv2.0", keep_retrieval_score=True)

# ---- 3. HyDE 查询增强: 先让LLM写一篇"假答案"去检索 比原问题强! ----
hyde = HyDEQueryTransform(include_original=True)  # 原问题+假问题双路并行检索

# ---- 4. 组装完整RAG Pipeline ----
query_engine = RetrieverQueryEngine.from_args(
    vector_retriever,
    response_synthesizer=get_response_synthesizer(response_mode="compact"),
    node_postprocessors=[reranker],  # 重排后置处理
)
# 外层包HyDE查询增强
final_engine = TransformQueryEngine(query_engine, hyde)
ans = final_engine.query("2024Q3财报中的营收同比增长率是多少?")
```

## 三、RAG质量评估 (RAGAS 5指标)

```
核心4指标 (面试必须答出每个的含义):
1. Faithfulness 答案忠实度: 答案中每个statement都能在检索上下文里找到证据吗? → LLM判分 越高越好
2. Answer Relevancy 答案相关性: 答案回答了用户的问题吗? 有没有跑题? → 越高越好
3. Context Precision 上下文精度: 检索返回TopK文档中, 真正相关的排在前面吗? → 排序越前越好
4. Context Recall 上下文召回率: 生成答案用到的所有证据点, 都在检索返回的文档里吗? → 覆盖率越高越好
5. Context Entity Recall 实体召回: 答案里的实体 都在检索文档里吗? → 防幻觉加指标
```

## 四、简历黄金句式

| 写法 |
|-----|
| 「LlamaIndex金融研报知识库：HyDE+混合检索BM25向量RRF融合+bge-reranker-v2-m3重排+SentenceWindow切块，覆盖12年研报800万份，RAGAS Faithfulness从62%→93.8%，分析师研报检索2h→3分钟」 |
| 「SubQuestionQueryEngine复杂问题拆解：多文档跨文件聚合问答，单文档正确率79%→跨文档复杂问答正确率91%」 |
| 「知识图谱+向量混合索引：Neo4j实体关系GraphRAG，多跳推理复杂问题准确率比纯向量RAG提升28%」 |

## 五、面试题

**Q Naive-RAG天花板低，怎么升级到Advanced-RAG？**
> A: 从7方面升级：① 切块(语义/SentenceWindow/父文档小Chunk召回大Chunk送LLM) ②查询增强(HyDE伪文档/Query2Doc/StepBack/子问题拆解) ③ 检索(混合BM25+向量RRF融合/Metadata结构化先过滤再向量) ④ 重排(Reranker Cohere/bge-reranker-v2-m3 Top50→Top4) ⑤分层(Hierarchical/HyDE父子Chunk) ⑥生成(Self-RAG/CRAG自我反思校验/Web搜索兜底补) ⑦缓存(Semantic Cache相似问题直接返回旧答案)。

**Q GraphRAG vs 向量RAG区别？适用什么场景？**
> A: 纯向量RAG只能做局部语义匹配，多跳推理(A→B→C→D问D和A关系)直接垮。GraphRAG：先LLM抽实体关系三元组存Neo4j→问题转Cypher图查询→路径+证据一起给LLM。适用复杂关系问题(供应链/医学因果/人物关系网络)，缺点是构建成本高(抽实体不准会GIGO)。

**Q RAG中"Lost in the Middle"现象？怎么缓解？**
> A: LLM注意力U型曲线：对Prompt开头和结尾内容特别敏感，中间内容容易忽略→TopK检索中间位置的文档被"看不见"。缓解：① 重排Reranker后最相关文档故意放中间以外的位置(头/尾) ② 降低TopK不要塞太满 ③ 用LongContextReorder PostProcessor把最相关的放头和尾。