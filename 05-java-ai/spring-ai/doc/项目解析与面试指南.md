# Spring AI 后端LLM服务解析

> 位置: 05-java-ai/spring-ai/
> 简历推荐: 5星 | 岗位: Java后端(转AI首选! 缺口极大)

---

## 一、Spring AI 四大核心抽象 (可插拔!)

```mermaid
graph TD
    BIZ[你的业务代码] --> CC[ChatClient Fluent流式API]
    CC --> C1[ChatModel接口 聊天大模型]
    CC --> E1[EmbeddingModel接口 向量化]
    CC --> V1[VectorStore接口 向量数据库]
    CC --> T1[Tool接口 Function Calling工具调用]

    C1 --> OpenAI[OpenAI/Azure/智谱/通义/文心/Ollama本地/Claude/Gemini]
    E1 --> OpenAIEmb[OpenAI/Ollama/文心/bge本地MiniLM]
    V1 --> PG[(pgvector / Milvus / Pinecone / Redis-Search / Chroma)]
    T1 --> Weather[自定义@Tool方法 自动反射调用]

    Prompt[.st Prompt模板系统 + BeanOutputConverter自动转Java对象]
    Advisors[AOP切面: 日志/安全/压缩/ChatMemory对话记忆]
```

## 二、核心代码

### RAG 检索增强生成 (7步全流程)

```java
// ====== 1. application.yml 配置 ======
spring:
  ai:
    openai: api-key: ${OPENAI_API_KEY}
    vectorstore.pgvector: index-type: hnsw, dimensions: 1536

// ====== 2. RAG Service 核心 ======
@Service
public class RagService {
    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    // --- 离线: 文档入库 ---
    public void ingest(String pdfPath) {
        var docs = new TokenTextSplitter(500, 100)  // 切500token 重叠100
            .apply(new PagePdfDocumentReader(pdfPath).read());
        vectorStore.add(docs);  // 自动: doc→Embedding→存pgvector
    }

    // --- 在线: 检索+回答 ---
    public String answer(String question) {
        // Step1: 问题向量→TopK=4最相似文档
        List<Document> docs = vectorStore.similaritySearch(
            SearchRequest.query(question).withTopK(4).withSimilarityThreshold(0.7));
        // Step2: 拼入System Prompt {context}占位符
        String context = docs.stream().map(Document::getContent).collect(joining("\n---\n"));
        // Step3: 大模型生成
        return chatClient.prompt()
            .system(s->s.param("context", context).text("基于参考资料回答，不知道就说不知道。资料:{context}"))
            .user(question).call().content();
    }
}
```

### Function Calling 工具调用 (让LLM调你的Java代码!)

```java
@Component
public class OrderTools {
    @Tool("根据订单号查订单详情,参数订单ID")
    public OrderInfo getOrderById(@P("订单ID,Long类型") Long orderId) {
        return orderMapper.selectById(orderId);  // 你的业务SQL
    }
}

// Controller注册Tool
@RestController @RequestMapping("/ai")
public class AiController {
    private final ChatClient chatClient;
    public AiController(ChatClient.Builder b, OrderTools tools) {
        this.chatClient = b.defaultTools(tools)  // 注册!
            .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory(),"sess1")).build();
    }
    @GetMapping("/chat")
    public String chat(String q) {
        // 大模型自动判断: 若需要订单信息→反射调用getOrderById→把结果回传模型生成最终答!
        return chatClient.prompt().user(q).call().content();
    }
}
```

> Function Calling 原理 (面试必须会!): 大模型不是直接执行Java！①Spring把@Tool方法→JSON Schema给System Prompt ②模型需要工具时生成特殊JSON `{name:getOrderById,args:{123}}` ③Spring拦截JSON反射调你的方法拿返回值 ④返回值作为User消息再发给模型 ⑤模型基于真实数据继续推理。是**多轮对话协议**不是模型执行代码！

## 三、简历黄金句式

| 写法 |
|-----|
| 「Spring AI + pgvector搭建企业知识库RAG：18万份文档向量化+混合检索+bge-reranker重排，RAGAS Faithfulness 62%→93.8%，客服工单人力↓47%」 |
| 「Function Calling销售助手Agent：4个工具(订单/库存/客户/物流)自动意图识别调用，销售查询工单↓47%，客单价+12.3%」 |
| 「SSE流式Chat体验：ChatClient.stream()+ServerSentEvent+前端EventSource，首字响应820ms，完整回答比同步等待用户满意度↑40%」 |

## 四、面试题

**Q ChatModel vs 原生HTTP调用OpenAI的好处？**
> A: ① Advisor/AOP切面: 对话记忆、内容安全、重试策略一键加 ② 多模型可切换：开发Ollama本地/生产OpenAI/合规私有化通义，换依赖不改代码 ③ 内置Prompt模板+BeanOutputConverter自动转Java对象(不用手写JSON解析) ④ 统一抽象: 10+LLM厂商零切换成本。

**Q pgvector 索引: IVFFlat vs HNSW 原理+选型？**
> A: IVFFlat: K-Means聚类分nlist桶→先查最近桶再桶内暴力。内存小查询快，Recall略低(参数nprobe控制)。HNSW: 多层跳表近似邻居图，Recall接近精确，查询更稳定但构建慢内存大。默认选HNSW，亿级+超大规模再考虑IVFFlat+PQ乘积量化。

**Q RAG优化7种高级手段？**
> A: ① 语义切块Semantic Chunking ②HyDE伪文档扩展(先让LLM写假答案去检索) ③混合检索BM25+向量RRF融合 ④重排Reranker Top50→Top4 ⑤父文档检索(小Chunk召回大Chunk送LLM) ⑥Self-RAG/Corrective-RAG自检 ⑦Semantic Cache相似问题直接用旧答案。