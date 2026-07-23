# 11-05 Java后端AI服务解析 (Spring AI + DJL)

> 📂 项目: `spring-ai/` `djl-demo/` `DeepLearningExamples/`
> ⭐ 简历推荐: ⭐⭐⭐⭐⭐ (Java后端转AI第一选择!) | 🎯 岗位: Java后端(AI业务方向)、MLOps工程师、AI系统工程师

---

## 一、Java AI生态全景

### 1.1 Java做AI的三大核心理由

```
① 企业级生产: 绝大多数公司后端是Java微服务/Spring Cloud → 把AI模型直接跑在现有Java服务里,
   零跨语言调用成本, 不用单独搭Python服务, 不用维护RPC/gRPC/HTTP接口。
   
② 性能&稳定: JVM性能媲美C++(JIT后), 稳定的多线程/GC/监控, 企业级中间件(Redis/Kafka/HBase)
   天然集成, 不用Python GIL问题。

③ 人才复用: 公司已有Java团队, 不用专门招AI工程化小组, 原有团队学习DJL/Spring AI即可落地。
```

### 1.2 Java主流AI框架对比

| 框架 | 厂商 | 核心能力 | 推荐场景 |
|-----|------|---------|---------|
| **Spring AI** | Spring官方 | LLM接入、RAG、Function Calling、Embedding、Vector Store抽象 | LLM应用开发 (RAG/Agent/Chatbot) |
| **DJL (Deep Java Library)** | AWS亚马逊 | 纯Java深度学习推理，支持Pytorch/TF/ONNX引擎 | 模型推理微服务、CV/NLP模型Java化 |
| **ONNX Runtime Java** | Microsoft | ONNX格式模型跨引擎推理 | 部署ONNX模型 |
| **Tribuo** | Oracle | ML训练+推理全流程 (GBDT/SVM/线性模型) | Java内置机器学习 |
| **TensorFlow Java** | Google | TensorFlow模型推理+训练 | TF生态用户 |

---

## 二、DJL (Deep Java Library) 深度解析

### 2.1 DJL架构: Engine统一抽象

```
DJL核心设计 = 一套API抽象 + 多个Engine后端 (Strategy模式)
┌───────────────────────────────────────────────────────┐
│                   DJL API (djl-api)                   │  ← 你只写一次代码
│     Model / Predictor / Translator / NDArray          │
├───────────────┬───────────────┬───────────────────────┤
│ PyTorch Engine│  MXNet Engine │ TensorFlow/ONNX Engine │  ← 运行时切换引擎
├───────────────┴───────────────┴───────────────────────┤
│           native .so/.dll  (自动下载!)                  │  ← DJL自动帮你下载对应平台
└───────────────────────────────────────────────────────┘
```

> 🔥 **DJL最牛特性**: Maven加依赖就可以跑PyTorch模型！不用自己装libtorch/不用写JNI/不用处理跨平台。DJL自动根据OS下载CPU/GPU版native包到缓存里。

### 2.2 DJL经典项目结构

```
05-java-ai/djl-demo/
├── getting-started/                  入门: 手写数字识别(MNIST)
├── inference/                        推理类应用
│   ├── image_classification/         ResNet图片分类 (猫/狗/ImageNet)
│   ├── object_detection/             YOLO/SSD物体检测
│   ├── face_landmark/                人脸关键点
│   ├── pose_estimation/              人体姿态
│   ├── bert_qa/                      BERT问答 (SQuAD)
│   ├── sentiment_analysis/           情感分析
│   └── word_embedding/               词向量相似度
├── training/                         Java端训练 (用引擎训练)
│   └── transfer_learning_on_cifar10/ CIFAR10迁移学习
├── huggingface/                      HuggingFace模型一键加载
│   ├── hf_nlp_classification/        BERT文本分类
│   └── hf_pipeline/                  NLP pipeline直接用
├── spring-boot/                      ← Spring Boot + DJL 微服务 (生产模板!)
├── multi-thread-inference/           高并发压测性能调优
├── kafka/                            Kafka消费数据 + DJL流式推理
└── android/                          DJL跑在安卓上
```

### 2.3 DJL核心代码: ResNet图片分类微服务

```java
// ================ 1. Spring Boot DJL服务 (生产级) ================
@Configuration
public class DjlConfig {
    // 模型启动时单例加载 (不要每次请求加载!)
    @Bean
    public ZooModel<Image, Classifications> resnet50Model() throws IOException, ModelException, TranslateException {
        Criteria<Image, Classifications> criteria = Criteria.builder()
            .setTypes(Image.class, Classifications.class)
            .optArtifactId("resnet")               // ModelZoo内的模型名
            .optGroupId("ai.djl.zoo")
            .optFilter("layers", "50")              // ResNet 50层
            .optFilter("flavor", "v1")
            .optProgress(new ProgressBar())
            .build();
        return criteria.loadModel();
    }
}

// ================ 2. Service层: 线程池+对象池 ================
@Service
public class ImageClassificationService {
    private final Predictor<Image, Classifications> predictor;

    // Predictor不是线程安全! 每个线程1个/Pool, 不要全局共享
    public ImageClassificationService(ZooModel<Image, Classifications> model) {
        this.predictor = model.newPredictor();  // Predictor含缓存, 复用性能翻倍
    }

    public Classifications predict(byte[] imageBytes) throws Exception {
        Image img = ImageFactory.getInstance().fromInputStream(new ByteArrayInputStream(imageBytes));
        return predictor.predict(img);   // DJL自动做前处理+推理+后处理
    }
}

// ================ 3. Controller层 ================
@RestController
@RequestMapping("/api/v1/vision")
public class VisionController {
    @PostMapping("/classify")
    public R<List<Classifications.Classification>> classify(@RequestParam("file") MultipartFile file) {
        long t0 = System.currentTimeMillis();
        Classifications result = imageService.predict(file.getBytes());
        long t = System.currentTimeMillis() - t0;
        return R.ok(result.topK(5)).msg("推理耗时: " + t + "ms");
    }
}
```

### 2.4 DJL性能调优 (高并发场景面试题!)

| 优化点 | 做法 | 效果 |
|-------|-----|------|
| **Predictor复用** | 不要每次predict new一个Predictor，用对象池/ThreadLocal | 性能×5~10 |
| **Engine后端选择** | CPU用ONNX Runtime (ORT), GPU用PyTorch Engine | CPU速度×2~3 |
| **多线程池** | `optDevice(Device.cpu(4))` 指定CPU核数; GPU用批处理 | CPU利用率↑60% |
| **Batch推理** | predictor.batchPredict(List<Input>) → 批量一次算 | GPU吞吐×3~5 |
| **NDManager内存管理** | 显式try-with-resources关闭NDArray，避免堆外内存OOM | Full GC次数↓80% |
| **并行流水线** | Decode→Preprocess→Inference→Postprocess 4个阶段用Disruptor/BlockingQueue | 高并发吞吐×N |

> ✍️ **简历写法**: 「基于DJL搭建企业级图像分类微服务 (Spring Boot + PyTorch Engine CUDA11.7)；通过Predictor池化+Batch推理+Disruptor流水线四层优化，单卡A10吞吐量从18 QPS提升至143 QPS (8x)，P99延迟53ms，生产无故障稳定运行180天+」

---

## 三、Spring AI (Spring官方LLM框架) 深度解析

### 3.1 Spring AI 核心抽象 (一张图记住!)

```
项目: 05-java-ai/spring-ai/
Spring AI = 4大核心接口抽象 + 多种实现 (可插拔)
┌──────────────────────────────────────────────────────────┐
│                    你的业务代码                           │
└───────────────┬──────────────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────────────┐
    │         ChatClient (新API)       │  ← Fluent流式接口
    │  chatClient.prompt("你好").call() │
    └──────────────────────────────────┘
┌──────────────────┬───────────────────────┬─────────────────────┐
│ ChatModel接口     │ EmbeddingModel接口    │ VectorStore接口      │
│ (聊天大模型)      │ (Embedding向量化)     │ (向量数据库)         │
├──────────────────┼───────────────────────┼─────────────────────┤
│ OpenAI           │ OpenAiEmbedding       │ SimpleVectorStore   │
│ Azure OpenAI     │ OllamaEmbedding       │ pgvector            │
│ Ollama (本地)     │ VertexAiEmbedding     │ Milvus / Pinecone   │
│ 智谱AI/通义/文心  │ 文心/智谱 Embedding   │ Redis Stack Search  │
│ Anthropic Claude │ bge-m3 embedding      │ Chroma / Qdrant     │
│ Google Gemini    │ MiniLM 本地           │ Weaviate            │
└──────────────────┴───────────────────────┴─────────────────────┘
                         ▲
                         │
              ┌────────────────────┐
              │ Prompt 模板系统    │  → .st 文件, 变量占位 {topic}
              │ Output Converter   │  → BeanOutputConverter 自动转Java对象
              │ Function Calling   │  → @Tool注解 调用Java方法
              │ RAG: Retrieval     │  → Query → Embedding → SearchVector → 拼Prompt
              │ Advisors (AOP)     │  → 日志/安全/压缩/记忆 切面
              └────────────────────┘
```

### 3.2 核心代码1: ChatClient + Function Calling

```java
// ========== 1. application.yml ==========
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4o-mini
          temperature: 0.7
    vectorstore:
      pgvector:
        index-type: hnsw
        dimensions: 1536

// ========== 2. Function Calling: 用@Tool让大模型调用Java方法 ==========
@Component
public class WeatherTools {
    @Tool("获取指定城市的实时天气, 参数是城市名, 比如 '北京'")
    public WeatherInfo getWeatherByCity(@P("城市名称,中文") String cityName) {
        // 这里写真实逻辑: 调第三方天气API
        return weatherApi.call(cityName);
    }
}

// ========== 3. ChatClient + Tools 调用 ==========
@RestController
@RequestMapping("/ai")
public class ChatController {
    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder builder, WeatherTools weatherTools) {
        this.chatClient = builder
            .defaultTools(weatherTools)    // 注册Tool, 大模型需要时自动调用
            .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory(), "session-1"))
            .build();
    }

    @GetMapping("/chat")
    public String chat(@RequestParam String question) {
        // 大模型会自动判断: 如果问题要"天气"→ 调用Java getWeatherByCity() → 拼结果再返回
        return chatClient.prompt().user(question).call().content();
    }
}
```

### 3.3 核心代码2: RAG (检索增强生成) 7步走

```java
// RAG = 向量化入库 + 检索增强 两部分
@Service
public class RagService {
    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    // ======= 离线部分: 文档入库 =======
    public void ingestDocument(String filePath) {
        var documentReader = new PagePdfDocumentReader(filePath);  // PDF阅读器
        var splitter = new TokenTextSplitter(500, 100);           // 切分: 500token/块, 重叠100
        var docs = splitter.apply(documentReader.read());
        // 内部自动: doc → EmbeddingModel → 向量 → 存pgvector
        vectorStore.add(docs);
    }

    // ======= 在线部分: 检索+生成 =======
    @GetMapping("/rag")
    public String ragAnswer(String userQuery) {
        // Step1: 用户问题Embedding → 向量库TopK=4相似文档
        List<Document> relevantDocs = vectorStore.similaritySearch(
            SearchRequest.query(userQuery).withTopK(4).withSimilarityThreshold(0.7)
        );
        // Step2: 把检索到的文档拼进System Prompt的{{context}}里
        String systemPrompt = """
            你是一名企业知识库助手。请仅基于下面提供的参考资料回答用户问题。
            如果参考资料里没有答案,请诚实回答"我不了解这个问题"。
            参考资料: {context}
            """;
        String docsStr = relevantDocs.stream().map(Document::getContent).collect(Collectors.joining("\n---\n"));
        // Step3: 发大模型 + 返回回答
        return chatClient.prompt()
            .system(s -> s.param("context", docsStr))
            .user(userQuery).call().content();
    }
}
```

```mermaid
flowchart LR
    subgraph 离线 文档入库
        PDF[PDF/Word/网页/Markdown] --> SPLIT[TokenTextSplitter<br/>切500token重叠100]
        SPLIT --> EMB[EmbeddingModel<br/>每块转1536维向量]
        EMB --> DB[(pgvector / Milvus 向量库)]
    end
    subgraph 在线 RAG问答
        Q[用户问题] --> E2[Embedding问题成向量]
        E2 --> VS[向量相似检索 TopK=4]
        DB --> VS
        VS --> CONTEXT[检索文档拼入Prompt<br/>作为{context}上下文]
        CONTEXT --> LLM[ChatModel 大模型]
        Q --> LLM
        LLM --> A[最终回答!]
    end
```

---

## 四、简历亮点 + 面试题

### ✍️ 简历句式 (JavaAI方向)

| 方向 | 简历写法 (含量化!) |
|-----|-----------------|
| DJL推理微服务 | 「基于DJL+Spring Boot搭建商品图片质检服务：ResNet50模型，A10 GPU通过Predictor池化+Batch 8推理，吞吐量18 QPS→143 QPS (7.9×)，P99延迟53ms，月均自动拦截违规图片32万张」 |
| Spring AI RAG | 「Spring AI + pgvector搭建企业内部知识库RAG系统：向量化18万份文档 (PDF/Word/Markdown)，文档检索命中率94.7%，大模型回答Hallucination率从38%降至8.2%，员工问题解决平均耗时从2小时→8分钟」 |
| Function Calling | 「基于Spring AI Function Calling实现销售助手Agent：接入订单/库存/客户/物流4个API工具，自动意图识别并调用，销售查询类工单减少47%，客单价提升12.3%」 |
| 流式响应 | 「SSE流式输出实现Chat体验: ChatClient.stream() + ServerSentEvent + 前端EventSource，首字响应延迟820ms，完整返回9.3s，比传统同步等待体验提升40%用户满意度」 |
| 向量库优化 | 「pgvector 向量索引调优: HNSW(m=16, ef_construction=200) 替换ivfflat，1800万向量查询延迟750ms→12ms，Recall@4从83%→94.8%，CPU占用率↓55%」 |

### 🎯 高频面试题

**Q1: DJL中的Model和Predictor有什么区别？Predictor是线程安全的吗？**
> A: Model = 模型实例(内存里的权重+计算图)，重量级、线程安全、全局单例加载1次即可。Predictor = Model创建的推理会话，内部缓存中间计算结果，**不是线程安全的**！生产不能所有线程共享一个Predictor，必须用对象池(Pool)或ThreadLocal各线程一个。Predictor复用比每次new性能提升5~10倍。

**Q2: Spring AI中RAG的完整流程是什么？有哪些优化手段？**
> A: 完整流程：①文档切分(TokenSplitter/SemanticChunker) → ②Embedding → ③入向量库 → ④查询:用户问题Embedding → ⑤相似度检索TopK → ⑥Prompt拼接Context → ⑦大模型生成。优化手段：切分(语义切块+Metadata过滤)、检索(混合检索: 向量+关键词BM25重排Reranker、HyDE伪文档扩展、Reciprocal Rank Fusion)、生成(去重/摘要压缩/引用溯源)、缓存(问题+答案Semantic Cache)。

**Q3: Function Calling / Tool Use 原理是什么？为什么大模型能"调用"Java方法？**
> A: 大模型不是真的直接调用你的方法！而是：①注册Tool时，框架把你的@Tool方法名/参数/描述转成JSON Schema发给大模型的System Prompt；②模型推理时，如果判断需要工具 → 模型生成一个特殊JSON格式 {name:"getWeather", args:{city:"北京"}} (而不是正常回答)；③Spring AI在客户端拦截到这个JSON → 反射调用你的Java方法拿到返回值；④把返回值再作为User消息发回给模型；⑤模型基于工具返回的真实数据继续推理，生成最终答案。是一种**多轮对话协议**，不是模型执行代码！

**Q4: RAG中如何缓解"幻觉"(Hallucination)？**
> A: ① 检索优化：提高TopK召回、混合检索(BM25+向量)、Reranker(如bge-reranker-v2-m3)精排；② Prompt工程：明确要求"基于参考资料回答，没有就说不知道"，要求带引用编号；③ 后处理：答案和原文做相似度校验，不相关段落打标删除；④ 事实校验(Citation Grounded Generation)：强制模型每个回答片段对应一个引用源块；⑤ 多路投票：多模型/多Chunk版本独立回答，RAG Fusion取一致答案。

---

**下一篇**: 👉 [11-06 LLM应用与MLOps解析](11-06_LLM应用与MLOps解析.md)