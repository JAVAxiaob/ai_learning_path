# LangChain4j 快速入门指南

> 位置: 05-java-ai-service/langchain4j/doc/
> 配套: LangChain4j面试题汇总-核心25题.md
> 对比: SpringAI-LLM应用集成指南.md (05-java-ai/spring-ai/doc/)

---

## 📦 1. Maven依赖配置

### 1.1 BOM统一版本管理（推荐）
```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-bom</artifactId>
      <version>1.12.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <!-- 核心模块：AiServices接口 + @Tool + OutputParser -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j</artifactId>
  </dependency>

  <!-- 选1：OpenAI GPT-4o/Mini -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-open-ai</artifactId>
  </dependency>

  <!-- 选2：本地Ollama Qwen2/LLaMA -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-ollama</artifactId>
  </dependency>

  <!-- 选3：智谱AI Kimi/GLM-4 -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-zhipu-ai</artifactId>
  </dependency>

  <!-- 向量库选1：Pgvector PostgreSQL向量插件 -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-pgvector</artifactId>
  </dependency>

  <!-- ✨ 新手推荐：Easy RAG 一键配置5行代码搭完RAG -->
  <dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-easy-rag</artifactId>
  </dependency>
</dependencies>
```

---

## 🚀 2. 第一个AiServices程序（10行Hello World）

```java
// Step 1: 声明AI服务接口（类似MyBatis @Mapper）
interface Assistant {
    @SystemMessage("你是幽默的Java老司机，用程序员段子回答问题，结尾加个emoji")
    String chat(String question); // LangChain4j自动生成JDK动态代理实现类
}

// Step 2: 创建并使用
public class HelloLangChain4j {
    public static void main(String[] args) {
        Assistant assistant = AiServices.builder(Assistant.class)
            .chatLanguageModel(OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o-mini")
                .temperature(0.7)
                .build())
            .build();

        String answer = assistant.chat("Java开发35岁后真的会失业吗？");
        System.out.println(answer);
        // 输出："兄弟别慌~ 35岁失业的是只会写CRUD的，会LangChain4j的叫架构师👨‍💻..."
    }
}
```

---

## 🧩 3. 核心7大注解速查表

| 注解 | 位置 | 作用 | 示例 |
|---|---|---|---|
| `@SystemMessage` | 方法/类 | 系统人设提示词，支持SpEL `{{var}}` | `@SystemMessage("你是{{role}}，公司：{{company}}")` |
| `@UserMessage` | 方法 | 用户消息模板，默认第一个参数就是用户问题 | `@UserMessage("分析以下订单：{{order}}，用户补充：{{q}}")` |
| `@V("name")` | 方法参数 | 注入模板变量值，替换`{{name}}` | `ask(@V("role") String role, @V("q") String q)` |
| `@MemoryId` | 方法参数 | 指定该参数为ChatMemory分桶主键，独立会话隔离 | `chat(String q, @MemoryId Long userId)` |
| `@Tool` | 普通Java方法 | 声明为LLM可调用工具 | `@Tool("查询订单物流")` |
| `@P` | @Tool方法参数 | 工具参数四要素：name/description/required/defaultValue | `@P(value="订单ID", defaultValue="8848") Long orderId` |
| `@Moderate` | AiServices方法 | 自动内容审核：输入输出双重合规检查 | `@Moderate(onInputViolation=BLOCK)` |

---

## 🔧 4. 工具调用完整示例（电商订单查询）

```java
// Step 1: 定义工具类（普通Spring Bean加@Tool注解）
@Component
public class OrderTools {
    @Autowired OrderMapper orderMapper;
    @Autowired LogisticsApi logisticsApi;

    @Tool({
        "根据订单ID查询订单完整详情和物流轨迹",
        "用户问：'我订单到哪了/发货没/多少钱'必须调用此工具",
        "禁止猜测订单状态，一律用工具查询真实数据"
    })
    public OrderFullInfo queryOrderByOrderId(
        @P(description = "订单ID，Long正整数，示例8848", required = true)
        Long orderId,
        @P(description = "返回详情级别：SIMPLE/FULL", required = false, defaultValue = "FULL")
        String detailLevel
    ) {
        Order order = orderMapper.selectById(orderId);
        LogisticsInfo logistics = logisticsApi.queryLatest(order.getLogisticsNo());
        return new OrderFullInfo(order, logistics, detailLevel);
    }
}
```

```java
// Step 2: AiServices接口中注册工具
interface ECommerceAssistant {
    @SystemMessage("你是京东金牌客服，回答中必须包含'亲'字~ 订单相关问题请先调用queryOrderByOrderId工具")
    String chat(String userQuestion,
                @MemoryId Long customerId,
                Object tools); // ✅ 将工具Bean作为参数传入！LangChain4j自动扫描@Tool
}

// Step 3: 使用
ECommerceAssistant ai = AiServices.builder(ECommerceAssistant.class)
    .chatLanguageModel(chatModel)
    .chatMemoryProvider(memId -> MessageWindowChatMemory.withMaxMessages(30))
    .build();
// 调用：工具自动被扫描并注册
String ans = ai.chat("我订单8848到哪了？", 10086L, orderTools);
```

---

## 🗄️ 5. RAG知识库示例（Pgvector）

### 5.1 文档入库（PDF导入）
```java
@Service
public class KnowledgeBaseService {
    @Autowired PgvectorEmbeddingStore vectorStore;
    @Autowired EmbeddingModel bgeZhEmbedding; // 中文Embedding模型BGE-M3

    public void importPdf(Path pdfPath) {
        // 1. 加载+解析PDF（自动用PDFBox）
        Document doc = FileSystemDocumentLoader.loadDocument(pdfPath,
            new ApachePdfBoxDocumentParser());
        // 2. 切块：递归字符分割300字符/50重叠
        List<TextSegment> segments = DocumentSplitters.recursive(300, 50).apply(doc);
        // 3. 批量Embedding + 入库Pgvector
        vectorStore.addAll(segments.stream()
            .map(seg -> EmbeddingMatch.from(
                bgeZhEmbedding.embed(seg.text()),
                seg.text(),
                seg.metadata().toMap()
            )).toList());
    }
}
```

### 5.2 检索增强生成（RAG查询）
```java
ContentRetriever ragRetriever = ContentRetriever.builder()
    .embeddingStore(vectorStore)
    .embeddingModel(bgeZhEmbedding)
    .maxResults(20)           // 粗排Top20
    .minScore(0.7f)           // 相似度阈值
    .reranker(JinaReranker.withApiKey(jinaKey)  // ✅ 加Reranker精排
              .withModel("jina-reranker-v2-base-multilingual"))
    .rerankMaxResults(4)      // 精排Top4给LLM看
    .build();

RagAssistant ai = AiServices.builder(RagAssistant.class)
    .chatLanguageModel(chatModel)
    .contentRetriever(ragRetriever) // ✅ 一行=自动RAG增强！检索内容自动拼System Message
    .build();
```

---

## 💡 6. 和Spring AI的写法对比（相同功能不同代码风格）

| 功能 | LangChain4j 接口注解写法 | Spring AI 流式Builder写法 |
|---|---|---|
| 声明AI服务 | `interface Assistant { String chat(String q); }` + AiServices.create() | `@Bean ChatClient client = ChatClient.builder(model)...build();` |
| 系统提示词 | 接口方法上`@SystemMessage("...")` | client.defaultSystem("...") |
| 会话记忆 | 方法参数加`@MemoryId Long userId` | Advisor链加`ChatMemoryAdvisor(redis, sessionId)` |
| 工具调用 | 方法参数传`Object tools` Bean，扫描@Tool | client.defaultTools(toolBean1, toolBean2) |
| RAG增强 | AiServices.builder().contentRetriever(retriever) | Advisor链加`QuestionAnswerAdvisor(vectorStore)` |
| 结构化输出 | 接口返回`List<POJO>`自动选OutputParser | client.prompt().user(q).call().entity(List<POJO>.class) |

---

## ⚡ 7. 生产环境建议清单

- ✅ **LLM选模型路由**：GPT-4o-mini解决90%问题，复杂场景才上GPT-4o，成本省35倍
- ✅ **ChatMemory选TokenWindow**：精确按Token控制，别用MessageWindow（消息少但Token可能爆Context）
- ✅ **工具>10个必加ToolSearchStrategy**：Vector检索Top5相关工具才挂LLM，省Token+提准确率
- ✅ **必加Guardrails护栏**：PII脱敏+Prompt注入检测，合规必过项
- ✅ **分布式场景用RedisChatMemory**：不要用内存InMemory，K8s多Pod不共享内存

