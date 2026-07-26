# LlamaIndex RAG流程图详解

> 位置: 06-llm/llama_index/doc/
> 配套文档: LlamaIndex-RAG索引引擎详解.md | LlamaIndex性能优化重难点.md | LlamaIndex面试题汇总.md

---

## 一、Naive-RAG 标准流程图 (三步经典)

```mermaid
flowchart TD
    subgraph Offline离线数据准备
        A1[源数据: PDF/Word/网页/Notion/DB]
        A1 --> A2[SimpleDirectoryReader/Connector 200+种]
        A2 --> A3[切分 Chunking: TokenTextSplitter<br/>默认Chunk=500 Token, 重叠=100]
        A3 --> A4[向量化 Embedding Model<br/>bge-small/text-embedding-3-small]
        A4 --> A5[(VectorStore向量数据库<br/>pgvector/Milvus/Chroma/Weaviate)]
    end

    subgraph Online在线查询
        B1[用户问题Query]
        B1 --> B2[Query也向量化]
        B2 --> B3[向量相似度检索TopK=4<br/>余弦距离/L2内积]
        B3 --> B4[拼Prompt模板<br/>Context: {Top4文档内容}<br/>Query: {用户问题}]
        B4 --> B5[LLM大模型生成答案]
        B5 --> B6[回答用户 + 附引用来源Chunk]
    end

    A5 --> B3
```

---

## 二、Advanced-RAG 七件套全流程优化

```mermaid
flowchart TD
    Start[用户问题Q] --> Q1[步骤1 Query Transform查询变换]

    Q1 --> HyDE["🔀 HyDE伪文档<br/>LLM先生成一篇假设答案D'<br/>用D'的向量去检索"]
    Q1 --> StepBack["↩️ Step-Back抽象<br/>原始问题 → 更高层抽象问题<br/>例: Q23财报营收→先问'2023整体经营情况'"]
    Q1 --> SubQ["🔨 SubQuestion拆解<br/>复杂问题拆多个子问题独立查"]

    HyDE --> R1
    StepBack --> R1
    SubQ --> R1[步骤2. 混合检索]

    R1 --> BM25["📚 BM25关键词检索<br/>专有名词/数字/代码精确匹配"]
    R1 --> Dense["🧮 Dense向量检索<br/>语义模糊匹配"]
    R1 --> KG["🕸️ GraphRAG知识图谱<br/>多跳推理Cypher查路径"]

    BM25 --> Fusion
    Dense --> Fusion["步骤3. RRF融合排序<br/>Reciprocal Rank Fusion<br/>1/(k+rank)加权融合各路结果Top50候选"]
    KG --> Fusion

    Fusion --> Filter[步骤4. Metadata元数据过滤<br/>先过滤时间/部门/文档类型<br/>缩小范围再算向量]

    Filter --> Rerank[步骤5. Reranker精排<br/>ColBERT/bge-reranker-v2-m3<br/>粗排50→精排4]

    Rerank --> Reorder[步骤6. LongContextReorder<br/>Lost-in-the-Middle现象修正<br/>最相关的移到Prompt头+尾]

    Reorder --> SelfRAG[步骤7. Self-RAG/Corrective-RAG自我校验<br/>🤖生成答案 → 置信度打分<br/>→ ❌ 低置信 → 重检索/加WebSearch兜底补知识]

    SelfRAG --> Output[✅ 最终答案 + 引用证据链 + 自我校验分]
```

---

## 三、切块Chunking算法对比流程

```mermaid
flowchart LR
    subgraph 方案A 固定Token切分[简单但切坏语义]
        FA[PDF文档] --> SPLIT1["TokenTextSplitter<br/>硬切每500Token<br/>不管句子/段落边界"]
        SPLIT1 --> BAD1["⚠️ 问题: 句子中间断<br/>'苹果公司宣布 今年Q3营收'<br/>→ Chunk1: '苹果公司宣布 今'<br/>→ Chunk2: '年Q3营收' 语义切断!"]
    end

    subgraph 方案B 句子窗口切分[LlamaIndex特色]
        FB --> SW["SentenceWindowNodeParser<br/>window_size=3"]
        SW --> SWRetrieve["召回时命中1个句子"]
        SWRetrieve --> SWExpand["展开窗口: 命中句+前后3句 → 给LLM看完整上下文5-7句"]
    end

    subgraph 方案C 语义切块[最优切块]
        FC --> SEMANTIC["SemanticSplitterNodeParser<br/>用Embedding相似度切"]
        SEMANTIC --> COMPARE1[相邻句向量余弦相似度<阈值→断]
        COMPARE1 --> GOOD["✅ 语义段落边界自然断开<br/>不会切断完整意思"]
    end

    subgraph 方案D 父文档切块[召回细粒度 提供粗粒度]
        FD --> PARSER["HierarchicalParentChildNodeParser<br/>子Chunk小粒度(128T)精搜"]
        PARSER --> RECALL["子Chunk命中 向量相似度准"]
        RECALL --> GET_PARENT["给LLM展开父Chunk(512T)大上下文"]
    end
```

---

## 四、RAGAS质量评估5指标流程

```mermaid
flowchart TD
    subgraph 输入
        Q[用户问题集 100道] --> PIPE[RAG流水线]
        PIPE --> C[检索到Top4 Context]
        PIPE --> A[LLM生成答案]
    end

    subgraph 评估5指标
        C --> F1[1. Faithfulness 忠实度]
        A --> F1
        F1 --> LLMEVAL1["LLM-as-Judge 判每句<br/>答案中的Statement<br/>能否在Context中找到证据？<br/>✅答案无幻觉 ⭐最重要指标"]

        A --> F2[2. Answer Relevancy 答案相关性]
        Q --> F2
        F2 --> LLMEVAL2["答案有没有答非所问？<br/>是不是在回答用户原问题？"]

        C --> F3[3. Context Precision 精度]
        LLMEVAL3["检索返回TopK文档中<br/>真正相关的排在越前越好？<br/>Top1就对了才是好系统"]

        C --> F4[4. Context Recall 召回率]
        A --> F4
        LLMEVAL4["生成答案用到的所有证据点<br/>是不是都在检索Context里？<br/>漏证据答案就可能靠幻觉补"]

        A --> F5[5. Context Entity Recall 实体召回]
        C --> F5
        LLMEVAL5["答案中提到的所有实体<br/>(公司名/数字/人名)<br/>是不是都在检索文档中出现过？"]
    end

    subgraph 输出评估报告
        LLMEVAL1 --> REPORT[RAGAS评估报告]
        LLMEVAL2 --> REPORT
        LLMEVAL3 --> REPORT
        LLMEVAL4 --> REPORT
        LLMEVAL5 --> REPORT
        REPORT --> SCORE["综合得分：Faith 95% + Rel 90%<br/>= 加权得分0.93<br/>建议：优化切块+Rerank"]
    end
```

---

## 五、Router Query Engine 路由流程

```mermaid
flowchart TD
    UserQ[用户问题] --> Router[Router LLM分类器]

    Router -->|"问具体产品价格/订单数据<br/>→ SQL结构化数据"| SQL["SQLQueryEngine<br/>自然语言→Cypher/SQL → 查数据库 → 格式化回答"]

    Router -->|"问PDF文档内部知识<br/>→ 语义相似"| VECTOR[VectorIndexQueryEngine<br/>标准向量TopK]

    Router -->|"问公司整体战略/年度总结<br/>→ 需要全文档摘要级"| SUMMARY[SummaryIndexTreeSummarize<br/>Tree结构逐层合并摘要]

    Router -->|"复杂问题跨多个文档<br/>→ 子问题拆解"| SubQEngine[SubQuestionQueryEngine<br/>Q→Q1+Q2+Q3独立查询<br/>再汇总答案]

    SQL --> Final
    VECTOR --> Final[组装Final Response + 引用来源]
    SUMMARY --> Final
    SubQEngine --> Final
```

---

## 六、索引构建全流程

```mermaid
flowchart LR
    Input[Documents 1000 PDF] --> LOAD[LlamaParse/Unstructured解析]

    LOAD --> NODES[NodeParser切分为Nodes<br/>每个Node=一个Chunk+Metadata]

    NODES --> EMBED[Batch向量化: bge-m3 批量并行]

    EMBED --> INDEX_TYPE{选什么索引?}

    INDEX_TYPE -->|"90%场景"| VSTORE[(VectorStoreIndex向量索引<br/>HNSW/IVF 余弦相似度检索)]
    INDEX_TYPE -->|"全局总结/报告摘要"| SUMTREE[(SummaryIndex摘要树索引)]
    INDEX_TYPE -->|"关键词精确查询"| KW[(KeywordTable关键词倒排索引)]
    INDEX_TYPE -->|"多跳推理/实体关系"| GRAPH[(KnowledgeGraphIndex<br/>LLM抽三元组→Neo4j存图)]

    VSTORE --> PERSIST["StorageContext.to_disk()<br/>持久化到磁盘/远程"]
    SUMTREE --> PERSIST
    KW --> PERSIST
    GRAPH --> PERSIST
```