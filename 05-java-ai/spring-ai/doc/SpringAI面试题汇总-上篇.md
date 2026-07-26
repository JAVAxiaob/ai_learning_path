# Spring AI面试题汇总-上篇 (Q1-Q20 带详细标准答案)

> 位置: 05-java-ai/spring-ai/doc/
> 配套文档: SpringAI-LLM应用集成指南.md | SpringAI流程图详解.md | SpringAI性能优化重难点.md
> 下篇 (Q21-Q40): Tool Calling & 工程实践 + 生产部署性能

---

## 一、基础架构题 (Q1-Q10)

---

### Q1. Spring AI 四大抽象接口: ChatModel/EmbeddingModel/VectorStore/Tool 解耦思想是什么？换LLM要改几行代码？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义/核心解耦思想
Spring AI借鉴Spring Data/Cloud的抽象哲学，用"面向抽象编程，而非具体实现"的解耦模式，把Java代码与底层具体AI技术栈完全解耦。四大核心接口构成Spring AI的支柱：

| 四大接口 | 职责 | 核心方法 | 内置实现数 |
|---|---|---|---|
| **ChatModel** | 对话/文本生成大模型抽象 | ChatResponse call(Prompt) | 12+ (OpenAI/Ollama/智谱/百炼/Minimax/豆包/通义/文心/Anthropic/AWS Bedrock/Google Vertex/HuggingFace) |
| **EmbeddingModel** | 文本→float[]向量抽象(RAG检索用) | EmbeddingResponse embed(String) | 10+ (OpenAI/BGE/M3E/智谱/百炼/Ollama) |
| **VectorStore** | 向量CRUD+相似度搜索抽象 | add(List) / similaritySearch(query) | 10+ (Pgvector/Milvus/Qdrant/Redis/Chroma/Pinecone/Weaviate/Elasticsearch) |
| **Tool / @Tool** | Function Calling工具函数抽象(Java方法暴露给LLM调用) | 反射自动生成JSON Schema + 自动调用 | 无限制，任意Java Bean方法可注册 |

#### 2. 耦合 vs 解耦对比表：
| 维度 | 硬编码写OpenAI API(耦合) | Spring AI抽象(解耦) |
|---|---|---|
| 换LLM代码改动量 | 改几百行(换HttpClient/JSON/鉴权) | ✅ **改2行**: pom换provider JAR + yml改api-key/base-url |
| 单元测试mock | 手写MockServer几百行 | ✅ new DummyChatModel("固定回复") 一行搞定 |
| 向量库选型 | 硬编码Pgvector JDBC | ✅ 换VectorStore Bean，检索代码零动 |

#### 3. 代码示例 - 切换LLM Java零改动：
pom.xml:
```xml
<!-- 用OpenAI GPT-4o -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
<!-- 换Ollama本地Qwen2-7B，只换上面这一个dependency -->
```

Controller业务代码（只注入抽象接口，零Vendor锁定）：
```java
@RestController
public class ChatController {
    private final ChatModel chatModel;  // ✅ 注入ChatModel接口，不是具体类
    public ChatController(ChatModel chatModel) { this.chatModel = chatModel; }
    @GetMapping("/ask")
    public String ask(String q) { return chatModel.call(q); }  // ✅ 通用API
}
```

#### 4. 常见面试坑：
- 🔴 坑①：直接注入OpenAiChatModel具体类→失去解耦意义，换LLM全项目重构
- 🔴 坑②：以为换LLM真零成本→不同模型System Prompt/Token限制/Function Calling细节不同需适配
- ✨ 加分：除四大接口还有ImageModel/AudioTranscriptionModel/AudioSpeechModel共7大模型抽象

---

### Q2. Spring AI vs LangChain4j 选型？Java RAG两大框架对比 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 两大框架背景
| 维度 | Spring AI (Pivotal官方) | LangChain4j (社区开源) |
|---|---|---|
| 开发团队 | VMWare/Spring官方，和Spring Boot同维护者 | 社区独立团队(前RedHat工程师)，Java版LangChain |
| 设计哲学 | Spring生态延伸，自动配置优先，约定>配置 | 纯LLM SDK，框架无关，任何Java SE/Quarkus/Micronaut可用 |
| 成熟度 | 2024年GA，1.0+稳定，背靠Spring 20年工程积累 | 2023年发布，0.3x+版本，社区活跃度高 |

#### 2. 功能矩阵详细对比：
| 功能点 | Spring AI | LangChain4j | 胜方 |
|---|---|---|---|
| Spring Boot Starter自动装配 | ✅✅✅ 官方一级支持 | ⚠️ 第三方starter，手写一半配置 | Spring AI |
| RAG组件丰富度 | ⭐ 标准7种Chunking，中规中矩 | ✅✅ 20+文档Loader/10种Chunking/高级管道 | LangChain4j |
| Function Calling工具调用 | ✅ @Tool AOP风格简单 | ✅✅ @Tool+多模态+参数校验更细 | LangChain4j微胜 |
| Agent推理(ReAct/Plan) | ⚠️ 仅基础ReAct | ✅✅ ReAct/ToolCalling/Router全家桶 | LangChain4j大胜 |
| 聊天记忆Memory | ✅ 7种+Advisor组合 | ✅✅ 15+种含TokenWindow/压缩记忆 | LangChain4j |
| 可观测性/监控 | ✅✅ Micrometer自动Prometheus打点 | ⚠️ 自己写Interceptor | Spring AI大胜 |
| 生产企业级(重试/熔断/限流) | ✅✅ 无缝Spring Retry/Resilience4j | ⚠️ 只有简单重试 | Spring AI大胜 |
| 微服务生态融合(Security/Cloud) | ✅✅✅ Spring天然打通 | ❌ 完全手写集成 | Spring AI大胜 |
| 社区Star(2025年) | ~5.5k | ~15k | LangChain4j |

#### 3. 场景选型黄金决策树：
```
场景判断
├─ 已经在用Spring Boot微服务栈 + 有Spring经验 + 要监控/重试/鉴权全打通
│   → ✅✅✅ 选 Spring AI（一栈式，不引入第二套生态）
│
├─ 全新AI应用 + 非Spring栈(Quarkus等) + 复杂Agent/高级RAG + Serverless
│   → ✅✅✅ 选 LangChain4j（纯SDK，无框架依赖）
│
└─ 极致复杂 + 万级并发 + 多模态工作流
    → ✅ 组合用：Spring AI做外层微服务基建 + LangChain4j做内层AI编排
```

#### 4. 高频面试追问：
- 追问1：冲突吗？能不能一起用？→ 完全不冲突！业界很多项目这么搭。
- 追问2：长期投谁？→ Spring AI！背靠Spring巨头，2.0功能追上后LangChain4j会边缘化。

---

### Q3. ChatClient Bean怎么注入？什么场景用ChatClientBuilder多客户端？(⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：ChatClient是流式Builder风格高级对话客户端
ChatModel是低级API(每次传Prompt)，ChatClient封装流式API+Advisor链+工具注册+记忆管理，99%场景推荐用ChatClient。

#### 2. 三种注入方式：
| 方式 | 代码 | 场景 |
|---|---|---|
| ①自动配置默认Bean | @Autowired ChatClient c | 简单全局单LLM |
| ②⭐@Bean自定义(最常用) | 下方代码 | 指定System Prompt/工具/记忆 |
| ③ChatClientBuilder动态构建 | 控制器里动态new | 多租户/多模型 |

#### 3. 最常用生产级@Bean示例：
```java
@Configuration
public class SpringAiConfig {
    @Bean
    public ChatClient customerServiceChatClient(
            ChatModel chatModel, VectorStore kb, ChatMemory redisMemory) {
        return ChatClient.builder(chatModel)
            .defaultSystem("你是某电商金牌客服【小助】，亲切专业，结尾加亲亲~")
            .defaultAdvisors(
                new LoggingAdvisor(),
                new ChatMemoryAdvisor(redisMemory),
                new QuestionAnswerAdvisor(kb, SearchRequest.builder().topK(20).build()),
                new ContentSafeAdvisor(PIIFilter.EMAIL_PHONE_IDCARD)
            )
            .defaultTools(orderTools)
            .build();
    }
}
```

#### 4. 多客户端4大典型场景：
| 场景 | 多客户端差异 | 为什么不能单客户端 |
|---|---|---|
| 🔸多角色 | hrChatClient vs itChatClient(知识库/工具/人设完全不同) | 切角色手动传Prompt易串 |
| 🔸多模型分级 | simple(GPT4o-mini) vs complex(GPT4o) vs local(Ollama) | 成本差35倍！按复杂度路由 |
| 🔸多租户隔离 | TenantA→SchemaA，TenantB→SchemaB | 不隔离=A查到B文档(GDPR事故💥) |
| 🔸特殊参数 | 文案t=0.9 vs 代码t=0.1 vs SQL t=0 | 每次传Options易忘 |

#### 5. 面试坑：
- 🔴 坑①：ChatClient单例改Prompt→多线程污染！→ 用链式构建新请求：`.prompt().user().call()`
- 🔴 坑②：每次请求new ChatClient→重复注册Advisor浪费→固定配置做Spring单例

---

### Q4. System Prompt模板设计规范: role/objective/constraints/examples四段式黄金写法 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：System Prompt是LLM的人设剧本，占回答质量权重40%
90%人写的烂Prompt只有一行"你是客服助手"→等于没写！工业界四段式黄金结构：

```
①【ROLE角色】：专业身份+人格+语气
   "你是某电商金牌售后客服，工号A1039，5年经验，语气亲切像邻家姐姐，结尾加🤗"

②【OBJECTIVE任务目标】：做什么？不做什么？边界？
   "处理退货退款咨询，禁止推荐商品/查竞品价格，超权限引导转人工"

③【CONSTRAINTS行为约束N条】：✨最重要！越具体越好
   "1. 严格基于知识库，禁止编造政策
    2. 普通商品7天无理由，生鲜定制不退货
    3. 退款原路返回支付账户
    4. 用户激动先道歉再解决
    5. 30-200字简洁
    6. 来源用[1][2]标注"

④【EXAMPLES示例2-3个】：Few-Shot秒对齐风格
   用户:"手机刚买能退吗？"
   你:"亲亲~当然可以🤗 手机属普通商品，签收7天内包装完好无人为损坏都可以无理由退货哒！生成退货地址吗？"
```

#### 2. 烂Prompt vs 四段式黄金Prompt对比：
| 维度 | 烂Prompt(90%人写) | 四段式黄金 | 提升 |
|---|---|---|---|
| 政策一致率 | 55%乱编 | 95%+ | **+40%** |
| 人设稳定度 | 30%忽冷忽热 | 92%+ | **+62%** |
| 越权回答率 | 48% | 3% | **-45%** |
| 用户满意度 | 3.2/5 | 4.6/5 | **+44%** |

#### 3. 常见面试追问：
- System Prompt太长被忘中间→Lost-in-the-Middle修复：约束条件放**最开头+最结尾**各一份，U型注意力头尾记得住
- 示例写几个？→分类2-3个，格式1个足矣，太多占Token反降效

---

### Q5. @Tool注解的FunctionCalling原理: Java方法→JSON Schema→LLM返回→反射调用→再送LLM全过程 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. Function Calling 5步核心流转：
```
用户问:"我订单8848的物流到哪了？"
│
▼ Step1 应用启动时 @Tool→JSON Schema自动生成（反射提取）
       → {name:"queryLogistics", description:"查询物流轨迹", parameters:{orderId:Long示例8848}}
       → 打包进 /v1/chat/completions 的 tools参数发LLM
│
▼ Step2 LLM智能判断→返回tool_calls指令(不生成回答)
       → {finish_reason:"tool_calls", function:{name:"queryLogistics", arguments:'{"orderId":8848}'}}
│
▼ Step3 Spring解析→Jackson转参数→反射调用Java方法
       → Method.invoke(orderTools, 8848L) → 返回LogisticsInfo("顺丰","运输中","上海分拨","明天送达")
│
▼ Step4 工具结果作为role=tool消息→再送LLM二次请求（把真实数据塞回去）
│
▼ Step5 LLM基于真实数据生成自然语言回答用户
       → "亲亲~8848号订单顺丰运输中📦，已到上海分拨中心，预计明天下午送达哟~"
```

#### 2. 完美@Tool方法（三要素齐全，调用成功率95%+）：
```java
@Component  // ✅ 必须Spring Bean
public class OrderTools {
    /** ✨ 黄金三要素：方法描述 + @P每个参数描述 + 返回值描述 */
    @Tool(
        value = "根据订单ID查询物流轨迹和预计送达时间",
        returnDescription = "LogisticsInfo(company/status/location/eta)"
    )
    public LogisticsInfo queryLogisticsByOrderId(
        @P(value = "订单ID，Long正整数，示例：8848", required = true)
        Long orderId
    ) {
        Order order = orderMapper.selectById(orderId);
        if (order == null) return LogisticsInfo.notFound("订单不存在");
        return logisticsApi.queryRealTime(order.getTrackingNumber());
    }
}
```

#### 3. 常见面试坑（90%中2个以上）：
| 坑 | 坏代码 | 后果 | 修复 |
|---|---|---|---|
| 漏@Tool方法描述 | @Tool void cancelOrder(Long id) | LLM不知道干啥 | 写用途+返回+场景 |
| 漏@P参数描述 | @P Long id | "订单123"解析成id=1 | 类型+语义+2个示例 |
| @Tool加private方法 | private void xxx() | Spring扫不到 | 必须public |
| 没权限校验 | 谁都能cancelOrder(-1L) | 💥水平越权！ | getCurrentUserId()==order.getUserId() |
| 返回String乱格式 | return "\n物流：顺丰\n" | LLM二次解析不稳 | 返回POJO结构化JSON |

---

### Q6. Advisor AOP切面体系: LoggingAdvisor/ChatMemoryAdvisor/ContentSafeAdvisor责任链顺序 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：Advisor是LLM调用链路的横切关注点
AOP责任链模式，一个请求流过：日志→脱敏→记忆→RAG→限流→LLM→出过滤→存记忆→审计→监控

```
[用户输入]
  ▼ 1 LoggingAdvisor ←【入站最先】记录原始请求+TraceId→ES
  ▼ 2 ContentSafeAdvisor-IN ←【PII脱敏】手机号/身份证/Token→***
  ▼ 3 ChatMemoryAdvisor-LOAD ←【记忆载入】sessionId→Redis取历史N轮
  ▼ 4 QuestionAnswerAdvisor ←【RAG检索】Top50+Reranker→Top4→塞context
  ▼ 5 PromptTemplateAdvisor ←【模板渲染】{question}/{context}/{user_name}填充
  ▼ 6 RateLimitAdvisor ←【限流】令牌桶10QPS超了抛429
   ┌──────────────────────────────┐
   │      LLM 实际调用ChatModel   │
   └──────────────────────────────┘
  ▼ 7 ContentSafeAdvisor-OUT ←【出站过滤】内部IP/密码过滤
  ▼ 8 ChatMemoryAdvisor-SAVE ←【记忆保存】Q&A加回Redis超长Summary压缩
  ▼ 9 LoggingAdvisor-OUT ←【出站记录】Token消耗+耗时→Prometheus
  ▼ 10 MetricAdvisor ←【最后】打点cost/p95/retry/tool成功率
[用户响应]
```

#### 2. 核心Advisor对比表：
| Advisor | 功能 | order()建议位置 | 关键配置 |
|---|---|---|---|
| LoggingAdvisor | 全链路日志→ES/Mongo | 1入站 + 倒数2出站 | level=INFO maskHeaders=["Authorization"] |
| ContentSafeAdvisor | PII双向过滤（手机/邮箱/身份证/银行卡/内网IP） | 2入站(先脱敏) + 7出站(LLM胡言过滤) | type=ALL / 自定义正则黑名单 |
| ChatMemoryAdvisor | 会话保存/加载上下文补全 | 4入站(先载入历史再RAG) + 8出站 | memoryType=InMemory/Redis/JDBC retrieveSize=最近10条 |
| QuestionAnswerAdvisor | RAG检索核心：相似搜索+TopK+Template注入 | 5入站 | vectorStore reranker=JinaReranker topK=20→4 |
| RetryAdvisor | 429/5xx/网络超时指数退避重试 | 紧挨LLM调用外层 | maxAttempts=5 backoff=200ms |
| RateLimitAdvisor | 用户ID/IP/API Key QPS限流防刷 | 3入站(被限流别浪费后面资源) | bucket=100/分钟 |
| TokenBudgetAdvisor | 超窗口自动截断旧对话 | 6入站(记忆后最长易爆) | budget=28000 strategy=TRUNCATE_OLDEST |

#### 3. 面试坑：
- 🔴 顺序错误①：RAG在记忆之前→用户问题缺上下文，检索文档完全不对！必须先补历史再检索
- 🔴 顺序错误②：ContentSafe只放入站→LLM被注入吐数据库密码→出站也必须加！
- ✨ 加分：自定义Advisor实现ChatClientAdvisor接口，重写aroundCall()，和Spring AOP ProceedingJoinPoint模式完全一样。

---

### Q7. Prompt Template怎么传参？{context}/{question}占位符示例 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 三种传参方式对比：
| 传参方式 | 代码 | 场景 |
|---|---|---|
| 链式键值对(最常用) | `.param("k1",v1).param("k2",v2)` | 参数少手写方便 |
| Map传参 | `.param(Map.of("k","v"))` | 参数多(>3个)动态来 |
| POJO传参 | `.params(userDto)` | 参数多结构化 |

#### 2. 完整RAG QA示例：
Controller代码：
```java
@GetMapping("/qa")
public String qa(String question, @RequestHeader("X-User-Name") String userName) {
    // Step1 RAG检索Top20文档
    List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(question).withTopK(20));
    // 拼带编号的参考资料字符串，塞进{context}
    String contextStr = docs.stream()
        .map(doc -> String.format("[%d] %s", doc.getMetadata().get("chunk_id"), doc.getContent()))
        .collect(Collectors.joining("\n---\n"));
    // Step2 链式传参渲染占位符
    return chatClient.prompt()
        .system(qaSystemPromptResource)  // resources/prompts/rag-qa-system.st
        .user("""
            用户{user_name}问：{question}
            参考下面知识库回答，没有就说"未找到相关信息"，结尾加来源[n]。
            【知识库】{context}
            """)
        .param("user_name", userName)  // ✨ 大括号名字必须完全匹配！
        .param("question", question)
        .param("context", contextStr)
        .call().content();
}
```

#### 3. 面试坑：
- 🔴 坑①占位符名字大小写/拼写不一致→模板{userName}传参.param("username",x)→空字符串→LLM抽风！
- 🔴 坑②特殊字符{}传参(比如正则)→被当占位符边界→转义{{}}
- 🔴 坑③{context}直接传List<Document>→渲染[Document@1a2b3c4d]对象地址！手动转成结构化字符串

---

### Q8. ChatModel支持哪些LLM? OpenAI/Ollama/智谱/百炼/Minimax接入方法 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 内置Provider完整列表(2025年Spring AI 1.2.x)：
| 分类 | Provider | Maven Starter | 场景 |
|---|---|---|---|
| 🔴国际大厂 | OpenAI(GPT系列) | spring-ai-openai-spring-boot-starter | 生产主力效果最好 |
| | Anthropic Claude | spring-ai-anthropic-spring-boot-starter | 长上下文/法律/写作 |
| | Google Gemini/Vertex AI | spring-ai-vertex-ai-spring-boot-starter | 多模态/谷歌云用户 |
| | Azure OpenAI | spring-ai-azure-openai-spring-boot-starter | 国内可直连/企业合规 |
| | AWS Bedrock | spring-ai-bedrock-spring-boot-starter | 多国数据驻留合规 |
| 🟠国内大厂 | 阿里百炼/通义千问 | spring-ai-dashscope-spring-boot-starter | 阿里系国内合规 |
| | 百度文心千帆 | spring-ai-qianfan-spring-boot-starter | 百度系国内合规 |
| | 字节豆包火山方舟 | spring-ai-volcengine-spring-boot-starter | 字节系国内合规 |
| | 智谱AI(GLM系列) | spring-ai-zhipuai-spring-boot-starter | 国内性价比王 |
| | MiniMax | spring-ai-minimax-spring-boot-starter | 视频语音多模态 |
| 🟢开源本地 | Ollama(几乎所有开源模型) | spring-ai-ollama-spring-boot-starter | 本地开发/离线/合规 |
| | HuggingFace本地推理 | spring-ai-huggingface-spring-boot-starter | 小模型CPU推理 |

#### 2. 智谱GLM-4-Plus接入示例（国内性价比王）：
pom.xml:
```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-zhipuai-spring-boot-starter</artifactId>
</dependency>
```
application.yml:
```yaml
spring:
  ai:
    zhipuai:
      api-key: ${ZHIPU_KEY}
      chat: options: { model: glm-4-plus, temperature: 0.2 }
```
✅ Java代码完全不动！ChatModel接口注入继续用。

#### 3. 面试加分：没有Starter的小众LLM(LM Studio/本地vLLM/OpenRouter)怎么办？
→ 所有兼容OpenAI API格式的直接复用OpenAI Starter改base-url就行：
```yaml
spring:
  ai:
    openai:
      api-key: "EMPTY"
      base-url: http://localhost:8000/v1  # 本地vLLM启动的OpenAI兼容接口
      chat: options: { model: qwen2-72b-awq }
```

---

### Q9. StructuredOutputConverter: LLM输出→Java对象(POJO)自动反序列化怎么用 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：解决"LLM输出自然语言字符串→Java解析成对象费劲"的痛点
自动做3件事：① POJO转JSON Schema System Prompt约束 ② 发LLM ③ Jackson转POJO+Bean Validation校验+失败重试

#### 2. 完整代码示例：自然语言→结构化订单对象
POJO定义（@Description+Validation=@LLM看的约束）：
```java
@Description("电商订单查询结果，字段严格按描述填写。输出纯JSON，不要解释、反引号、Markdown。直接{开头}结尾")
public record OrderQueryResult(
    @NotNull(message = "orderId不能空")          // ✅ Bean Validation自动校验
    @Description("订单ID 1-9位正整数，无前缀")
    Long orderId,
    
    @NotBlank
    @Pattern(regexp = "^(待付款|待发货|运输中|已签收|已取消|退款中)$")  // ✅ 枚举正则限制
    @Description("订单状态枚举，只能是括号内6个：(待付款|待发货|运输中|已签收|已取消|退款中)")
    String status,
    
    @Positive
    @Description("订单金额元，保留2位小数，正数")
    BigDecimal amount,
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Description("下单时间北京时间，格式严格yyyy-MM-dd HH:mm:ss，示例2025-01-15 15:30:00")
    LocalDateTime createTime
) {}
```

Controller一行调用：
```java
@GetMapping("/order/parse")
public OrderQueryResult parseOrder(String text) {
    // text = "帮我看看上周三下午3点那5000块的订单到哪了？顺丰那个..."
    return chatClient.prompt()
        .user("从自然语言中提取订单信息：" + text)
        .call(OrderQueryResult.class);  // ✨ 魔法行：自动Schema+校验+重试
}
```
→ 直接返回强类型Java对象，不用写正则/JSON解析！

#### 3. 常见失败场景+解决：
| 失败场景 | 原因 | 修复 |
|---|---|---|
| LLM输出"好的我来查询订单..."前缀不是JSON | 约束弱 | @Description里加"**必须只输出JSON，不要任何前后解释**" |
| 字段名不匹配(order_id vs orderId) | 命名习惯不同 | @JsonProperty("order_id")别名 |
| 枚举值乱写"已发货"不在列表 | 没明确列 | @Pattern+Description里把所有值完整列出来 |
| 日期格式乱ISO8601 | LLM默认ISO | @JsonFormat+描述写严格格式+示例 |

---

### Q10. Ollama本地部署Qwen2-7B，Spring AI零代码切换？对比GPT-4o效果/速度/成本 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. Ollama本地5分钟零代码切换步骤：
```bash
Step 1 装Ollama：https://ollama.com/download 双击10秒
Step 2 拉模型：ollama pull qwen2:7b-instruct  (4.7GB 推荐16G内存+)
       8G内存机器：ollama pull qwen2:1.5b-instruct (1GB效果凑合能跑)
       4090玩家：ollama pull qwen2:72b-instruct-q4_0 (40GB 逼近GPT-4)
Step 3 启动后台自动运行：ollama serve  (默认http://localhost:11434)
Step 4 Spring AI 改pom+yml，Java代码零改动！
```

#### 2. pom+yml切换对比：
pom.xml二选一：
```xml
<!-- 模式A 生产GPT-4o -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
<!-- 🔁 切本地Ollama：注释上面打开下面，Java代码零改动！
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-ollama-spring-boot-starter</artifactId>
</dependency> -->
```
application.yml切换：
```yaml
spring:
  ai:
    openai:  # 生产模式
      api-key: ${OPENAI_KEY}
      base-url: https://api.openai.com
      chat: options: { model: gpt-4o-mini, temperature: 0.2 }
    # 🔁 切Ollama时注释上面打开下面：
    # ollama:
    #   base-url: http://localhost:11434
    #   chat: options: { model: qwen2:7b-instruct, num_ctx: 8192, temperature: 0.2 }
```
✅ 用@Profile("dev"/"prod")+多yml文件，dev自动Ollama省Token钱，prod自动GPT。

#### 3. 📊 本地Qwen2-7B vs GPT-4o全维度对比表（面试重点）：
| 维度 | Ollama Qwen2-7B本地 | GPT-4o-mini API | GPT-4o API |
|---|---|---|---|
| **单次成本** | ✅✅✅ 0元！电费忽略 | $0.15+$0.6/1M Token | $2.5+$10/1M Token |
| 每月100万次请求 | ≈¥50电费折旧 | ≈¥1,200 | ≈¥38,000 |
| **数据隐私** | ✅✅✅ 数据不出本机/内网（医疗/金融/政府刚需！） | ❌ 发美国OpenAI服务器 | ❌ 同左 |
| 中文任务表现 | ✅✅ 国内政策/常识/成语比GPT-4o-mini好 | ⭕ OK但网络流行语偏差 | ✅✅✅ 最好 |
| 英文/代码 | ⭕ 7B有限，代码偶尔Bug | ✅✅ 好 | ✅✅✅ 顶尖 |
| 复杂推理数学 | ⭕ 简单OK，复杂易错题 | ✅✅ 中等稳 | ✅✅✅ 奥数级别 |
| 延迟8K入+1K出 | 3090: 3-6s / CPU i7: 15-25s | ✅✅ 1-2s | ✅ 2-4s |
| 离线可用性 | ✅✅✅ 断网/专网/船舶军工也能用 | ❌ 必须公网 | ❌ |
| 适合场景 | ✅ 本地开发0成本<br>✅ 内网数据敏感合规<br>✅ 简单FAQ客服 | ✅ 主力生产90%场景首选 | ✅ 10%复杂场景VIP高价值 |

#### 4. 业界黄金组合：
```
┌──────────────────────────────────────────────┐
│ 生产3级模型路由                               │
│ ① 开发环境100%走Ollama本地 0成本              │
│ ② 生产90%简单请求→GPT-4o-mini 低成本           │
│ ③ 生产9%复杂推理→GPT-4o 高效果                 │
│ ④ 生产1%降级兜底→Ollama内网部署(OpenAI挂了)    │
└──────────────────────────────────────────────┘
```
#### 5. Ollama坑：
- 🔴 默认num_ctx=2048，用户输入>2000字直接截断胡言乱语！→yml强制`num_ctx: 8192`，Qwen2-7B最大支持32768。

---

## 二、RAG深入题 (Q11-Q20)

---

### Q11. 7种Chunking切块算法: Token/SentenceWindow/Semantic/ParentChild 准确率排序 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：Chunking是RAG召回率天花板基石（占RAG效果权重30%）
把长文档切成小块给Embedding，切不好→检索漏→回答错。7种主流算法对比：

| 算法 | 原理 | 准确率 | 速度 | 适用场景 |
|---|---|---|---|---|
| ✅ **RecursiveCharacterTextSplitter(默认) | 按\n\n→\n→。→,→ 递归切，重叠20% | ⭐⭐⭐ | ✅✅✅极快 | 通用场景90%首选！万金油默认 |
| ✨ **SentenceWindowRetriever 句子窗口** | 以句子为单位切，检索时带回前后3句上下文 | ⭐⭐⭐⭐ | ✅✅快 | 法律/合同/论文（句子边界清晰 |
| ✨✨ **SemanticSplitter 语义切块** | Embedding相似度突变点做断点，语义不跨块 | ⭐⭐⭐⭐⭐ 最准 | ⚠️慢(要算Embedding) | 知识库/百科/复杂长文最佳 |
| 👨‍👩‍👧 **ParentDocument 父子块** | 小块(256token)检索+父块(2048token)给LLM | ⭐⭐⭐⭐ | ✅✅中 | 要精细定位+完整上下文 |
| 📚 **Markdown/HTML按标题结构切块** | 按#/##/### 标题层级切 | ⭐⭐⭐⭐ | ✅✅✅极快 | Markdown/HTML/confluence/wiki等结构化文档 |
| 📄 **Token固定切** | 严格按LLM Token数切，不考虑语义边界 | ⭐⭐ | ✅✅✅极快 | 只对Token数敏感严格限制的场景 |
| 🔗 **AgenticChunking(递归+Agent分块** | LLM自己判断怎么切+加元数据 | ⭐⭐⭐⭐⭐(理论最准) | ❌极慢(每块都调LLM | 超高质量小数据集(1000份以内) |

#### 2. 工业界准确率性能排序（准确率从高到低）：
AgenticChunking > SemanticSplitter ≈ ParentChild > SentenceWindow > 结构切 > RecursiveCharacter > 固定Token切

速度排序（速度从快到慢）：
Token固定 > Recursive > 结构切 > SentenceWindow > ParentChild > Semantic >> Agentic

#### 3. Spring AI SentenceWindow示例代码：
```java
// SentenceWindow拆分器：检索时自动带回每句前后3句上下文
DocumentSplitter splitter = SentenceWindowSplitter.builder()
    .windowSize(3)            // 句子前后各带3句
    .build();
List<Document> chunks = splitter.apply(List.of(wholeDoc));
// 效果：检索到第N句 → LLM拿到的是N-3到N+3句共7句，上下文不丢失!
```

#### 4. 面试高频追问：切块大小怎么定？
→ 经验公式：**Chunk大小 = 模型上下文长度的 1/30 ~ 1/20。
→ 7B模型8K上下文→256-512token；70B/商用模型32K→1024-2048token。
→ 重叠Overlap=Chunk大小的10-20%（边界信息不丢）。

---

### Q12. Pgvector三大索引: HNSW vs IVFFlat vs Exact 选型？500万向量场景为什么选HNSW？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 三大索引全方位对比表：

| 维度 | HNSW ⭐生产默认首选 | IVFFlat+PQ 超大规模 | Exact 精确线性扫描 |
|---|---|---|---|
| 原理 | 多层跳表随机近邻图 | K-Means聚类分桶→先查最近桶→桶内暴力+PQ编码压缩 | 全量计算每个向量距离 |
| 构建时间(500万1536维) | 慢10-30分钟(一次性) | 中3-10分钟 | 0(不需要构建) |
| 查询延迟p95 | ✅✅✅ 1-5ms极快 | ✅✅ 5-15ms | ❌ 100-5000ms极慢 |
| 内存占用(500万) | 较大 ~60GB | ✅✅✅ 极小PQ64→1GB(64倍) | 大 30GB |
| Recall@50(召回率) | ✅✅✅ 99%+接近精确 | ⭐ 90-95%(调nprobe提) | ✅✅✅ 100%(金标准) |
| 插入性能(增量更新) | ✅ 动态增删快O(logN) | ⚠️ 插新点桶分布变→重训中心点，周期reindex | ✅ 无影响 |
| 适合数据规模 | ✅ 10万-1亿向量 | ✅✅✅ 亿级+超大规模 | ❌ <1万小规模基准 |

#### 2. 500万向量场景为什么选HNSW？→三大理由：
1. **延迟达标**：在线业务要求p95<10ms内响应，HNSW 1-5ms完爆其他；
2. **召回够**：99%+接近精确扫描，用户体感无差别；
3. **增量友好**：新文档随时插不用reindex，IVFFlat插入就要周期性重训聚类中心点。

#### 3. HNSW生产级调优SQL（90%教程漏了ef_search设置！）：
建表时：
```sql
CREATE INDEX docs_embedding_idx ON docs USING hnsw (embedding vector_cosine_ops)
WITH (m = 16,            -- 每层邻居数 越大Recall高内存大:12-48
      ef_construction = 128);  -- 构建搜索深度，越大索引越准构建越慢
```
**查询前必须SET！默认40太小，Recall掉20%💥：
```sql
SET hnsw.ef_search = 256;  -- 在线查询搜索深度128-256
SELECT content, 1-(embedding <=> $1) cosine_sim
FROM docs
WHERE 1-(embedding <=> $1) > 0.7  -- 相似度阈值
ORDER BY embedding <=> $1
LIMIT 50;
```

---

### Q13. ef_construction vs ef_search 参数调优？为什么90%教程忽略设置导致召回率掉20% (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 两个参数定义+对比：
| 参数 | 生效时机 | 作用 | 调整影响 | 默认值 | 生产推荐 |
|---|---|---|---|---|---|
| ef_construction | 🔧 仅**建索引时**一次性用 | 建HNSW图时每个点找多少邻居候选 | 越大→索引越准→构建越慢 内存略大 | 16 | 128（线上稳定） |
| ef_search | 🚀 **每次查询**设置！ | 查询时搜索路径深度，遍历多少候选再取TopK | 越大→召回越高→查询越慢 | ❌40太小了！ | 256（线上查询必须手动SET） |

#### 2. 为什么90%教程漏了→召回率掉20%？
→ 教程只写建表CREATE INDEX不提查询SET ef_search，大家跑Demo默认40→线上500万条→Recall直接从98%掉到75%！💥 用户体感就是答非所问。

#### 3. 调参曲线参考表：
| 数据量 | ef_construction | ef_search | 典型Recall@50 |
|---|---|---|---|
| 10万 | 64 | 128 | 97% |
| 100万 | 128 | 256 | 96% |
| 500万 | 128 | 256 | 95% |
| 1000万 | 160 | 512 | 94% |
| 5000万+ | 160 | 512 | 90%+ |

#### 4. Spring Boot里怎么全局设置不用每次SQL SET？
Spring Boot Data配置DataSource后加一个Connection级配置，每次连接自动初始化：
```java
@Configuration
public class PgvectorConfig {
    @Bean
    public CommandLineRunner setEfSearch(DataSource ds) {
        return args -> {
            try (Connection c = ds.getConnection()) {
                try (var s = c.createStatement()) {
                    s.execute("SET hnsw.ef_search = 256");
                    s.execute("SET hnsw.ef_search = 256");
                }
            }
        };
    }
}
```
或者HikariCP连接池用`connection-init-sql: SET hnsw.ef_search=256`
→ 每个新连接初始化跑一次，不用每次查询都SET。

---

### Q14. 混合检索实现: 向量相似度 + BM25关键词 + RRF融合 分数加权公式 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 纯向量 vs 纯关键词 vs 混合检索效果对比：
| 检索方式 | 语义匹配(同义词/缩写/改写 | 关键词精确匹配(专有名词/数字/人名) | 综合Recall |
|---|---|---|---|
| 纯向量相似度 | ✅✅✅ "如何退手机/退定金→都能命中"退货" | ❌ "订单号8848"搜不到包含8848精确 | 88% |
| 纯BM25关键词 | ❌ "手机想退钱"和"退货政策"字完全不匹配 | ✅✅✅ 数字/专有名词/型号精确 | 75% |
| ✅混合检索 | ✅✅✅ | ✅✅✅ | **95%+业界最佳实践** |

#### 2. RRF (Reciprocal Rank Fusion 倒数排名融合) 黄金公式：
```
分别用向量和BM25各出Top100结果，
对每个文档doc：
RRF_Score(doc) = Σ 1 / (k + rank_i)
                    i=两种检索方式
k=60是论文最优常数(防止第一名不被拉太狠)
最后按RRF_Score降序取Top50→送Reranker
```

#### 3. Spring AI 混合检索实现代码：
```java
// 1) BM25关键词检索（用PostgreSQL原生tsvector+ts_rank）
List<Document> bm25Docs = jdbcTemplate.query(
    "SELECT id, content, ts_rank(to_tsvector(content), plainto_tsquery(?) as score
     FROM docs WHERE to_tsvector(content) @@ plainto_tsquery(?)
     ORDER BY score DESC LIMIT 100",
    (rs, i) -> new Document(rs.getString("content")),
    question, question
);

// 2) 向量相似度Top100
List<Document> vecDocs = vectorStore.similaritySearch(SearchRequest.query(question).withTopK(100));

// 3) RRF融合 (k=60)
Map<String, Double> rrfScores = new HashMap<>();
for (int i=0; i<bm25Docs.size(); i++) mergeScore(rrfScores, bm25Docs.get(i), i+1));
for (int i=0; i<vecDocs.size();  i++) mergeScore(rrfScores, vecDocs.get(i),  i+1);

// 辅助：1/(k+rank) 累加
private void mergeScore(Map<String,Double> m, Document doc, int rank) {
    String key = doc.getId();
    double add = 1.0 / (60 + rank);
    m.merge(key, add, Double::sum);
}

// 4) 按RRF分数降序取Top50送去Reranker精排
List<Document> fusedTop50 = rrfScores.entrySet().stream()
    .sorted(Map.Entry.<String,Double>comparingByValue().reversed())
    .limit(50)
    .map(e -> docMap.get(e.getKey()))
    .toList();
```
→ 单独向量→混合RRF一般**Recall +8-12%提升**。

#### 4. 面试追问：RRF k取什么值？为什么不用0？
→ k=60是微软Bing搜索论文经典值，防止单一检索方式第一名直接被拉爆，其他结果完全没机会。如果k=0，第一名分数就是1/0无限大，后面永远追不上。

---

### Q15. Reranker精排50→Top4原理: ColBERT vs Jina-Reranker vs BGE-Reranker (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. Reranker是RAG准确率最后一公里（召回(50个候选文档→精排→给LLM看最重要Top4）
向量相似度是"粗排"，Reranker是"精排"：CrossEncoder把**Question和Document交叉注意力计算，算真实语义匹配度→Top4差10-15%提升。

#### 2. 三大主流Reranker模型对比表：
| 模型 | 参数量 | 中文效果 | 英文效果 | 50条单条耗时 | 适用场景 |
|---|---|---|---|---|---|
| ✅ **Jina-Reranker-v2-zh | 800M | ⭐⭐⭐⭐⭐中文最佳 | ⭐⭐⭐⭐ | 3090:15ms/条 | 国内中文场景首选 |
| ✅ BGE-Reranker-v2-large | 560M | ⭐⭐⭐⭐中文其次 | ⭐⭐⭐⭐⭐ | 3090:10ms/条 | 中英文混合/开源首选 |
| ✨ ColBERTv2 (Late Interaction) | 110M | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅✅ 2ms极快(预存倒排) | 高并发>10QPS延迟敏感场景 |
| Cohere Rerank-3 | - | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | API调用按需付费 | 英文商业付费API |
| 无Reranker直接向量Top4 | - | - | - | 0ms | 简单Demo/POC/10QPS以下 |

#### 3. Spring AI Reranker代码示例：
```java
// Pgvector粗排Top50
List<Document> top50 = vectorStore.similaritySearch(SearchRequest.query(q).withTopK(50));

// ✨ Jina Reranker精排50→Top4（+12%准确率）
JinaRerankModel reranker = JinaRerankModel.builder()
    .apiKey(System.getenv("JINA_API_KEY"))
    .model("jina-reranker-v2-base-multilingual")
    .topN(4)          // ← 只返回最相关4个给LLM！别给太多Lost-in-the-Middle
    .build();

RerankResponse resp = reranker.call(new RerankRequest(q, top50));
List<Document> top4 = resp.getResults();  // ✅ 精排后的4块丢给LLM上下文
```

#### 4. 面试追问：为什么50→4不给50全给LLM？→3大原因：
1. **Lost-in-the-Middle 中间文档30%信息LLM注意不到；
2. **上下文成本：4块约4×500字≈2000字省Token钱；
3. **噪声：相关性倒数十几个块反而误导LLM，信噪比下降。
→ 经验最佳实践：粗排Top50+Reranker→Top3-6，不多给。

---

### Q16. HyDE查询优化: 什么用假答案去搜比真问题搜更好？什么场景不能用？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. HyDE (Hypothetical Document Embeddings 假设文档嵌入) 核心思想：
用户自然语言问题和知识库Chunk文档的**向量空间分布不一样**！
- 用户问题：口语化短、疑问、句子零散结构
- 文档Chunk：书面化长、陈述、完整段落结构
→ 即使语义相同，向量相似度也拉不近！💥

**HyDE巧妙两跳解决：**
```
Step 1 让LLM不参考知识库，"假装"写一个假答案（假设的理想文档片段）
        用户问:"年假有几天？" → LLM假造: "根据公司员工手册5.2条，正式员工入职满1年享受5天带薪年假，工龄每增1年加1天，封顶15天。病假需提前24小时申请..."
Step 2 拿这个假答案的向量去搜知识库！
        假答案是陈述语气，和知识库Chunk风格一致→分布对齐→相似度更高
        → 召回率+5-10%提升
```

#### 2. HyDE vs 普通Query检索效果对比：
| 指标 | 普通问题检索 | HyDE假答案检索 | 提升 |
|---|---|---|---|
| Recall@50 | 82% | 92% | **+10%** |
| 最终QA准确率 | 75% | 87% | **+12%** |
| 典型适用场景FAQ/政策/技术文档/法律法规 | 一般不明显 | 提升极大 | FAQ类提升有限→重复度高 |

#### 3. Spring AI HyDE代码实现：
```java
@Component
public class HydeRetriever {
    // 双LLM：轻量Mini专门写假答案省成本，GPT4o做最终回答
    private final ChatModel miniLlm;  
    private final VectorStore vectorStore;

    public List<Document> hydeSearch(String question, int topK) {
        // Step 1 让LLM生成假设的理想答案（假装它已经在知识库找到了）
        String hydePrompt = """
            你是HR政策文档专家，写一段200字左右的"如果手册里有答案那段话"
            风格要和正式员工手册完全一致，专业陈述句式。
            不需要正确回答，只要写出假设的政策片段。
            用户问题：%s
            假设文档片段：
            """.formatted(question);
        String hypotheticalDoc = miniLlm.call(hydePrompt);  // 伪造的假文档
        
        // Step 2 拿假文档去搜知识库！（+includeOriginal两路融合最好）
        SearchRequest req1 = SearchRequest.query(question).withTopK(topK);       // 原Query
        SearchRequest req2 = SearchRequest.query(hypotheticalDoc).withTopK(topK);// 假Doc
        List<Document> results1 = vectorStore.similaritySearch(req1);
        List<Document> results2 = vectorStore.similaritySearch(req2);
        
        // Step 3 RRF融合两路结果
        return RrfFusion.merge(results1, results2, topK);
    }
}
```
→ includeOriginal=true两路搜是最佳实践，HyDE单独搜会有偏移问题。

#### 4. 什么场景**绝对不能用HyDE**？→反效果！
| ❌ 禁用HyDE场景 | 为什么不能用 |
|---|---|
| ① 事实精确查询（"订单8848金额多少？"） | LLM假答案会写假数字→搜出来完全错的文档 |
| ② 数字/日期/ID/型号等精确检索 | 假答案胡编数字→相似度偏十万八千里 |
| ③ 知识库很小<100条 | 直接搜就100%命中，加HyDE浪费Token还偏 |
| ④ 超开放问题（"人生意义是什么？"无固定答案） | 假答案偏得离谱，反而误导 |
| ⑤ LLM本身很弱7B小模型幻觉多 | 假答案错得离谱→搜出来更错 |

---

### Q17. Lost-in-the-Middle现象: LLM注意力U型怎么缓解？LongContextReorder后置处理器 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：LLM注意力天然U型分布（开头结尾记得住，中间忘掉30%）
学术论文《Lost in the Middle: How Language Models Use Long Contexts》实证：
LLM对上下文开头和结尾的信息**召回率90%+，但中间段信息召回率骤降到60-70%！💥**
→ 向量检索按相似度排序返回的Top4，最相关的块大概率排在中间→LLM没看到→答非所问。

```
LLM注意力分布图（横轴=上下文位置，纵轴=信息召回率）
召回率100%┃█                    █
          ┃█                    █
           90%┃ █                  █ 
          ┃  █                █
           70%┃   █            █   ← 中间直接掉30%
          ┃    ██████████████
           0% ┗━━━━━━━━━━━━━━━━━━━━━━→ 位置
              开头              中间             结尾
```

#### 2. 三大缓解方案组合拳（工业界标准做法）：
| 方案 | 原理 | 效果 |
|---|---|---|
| 1️⃣ **LongContextReorder重排处理器** | 把最相关的Chunk放**第1位+最后1位**，次相关塞中间 | ✅✅✅ 最有效+5-8% |
| 2️⃣ **System Prompt约束头尾重复** | 最重要的规则在System Prompt的**开头+结尾各写一次** | ✅✅ +3-5% |
| 3️⃣ **减少TopK数量不要贪多** | 别塞20个给LLM，Reranker后给Top3-6刚好 | ✅✅ 质量提升+省Token |

#### 3. Spring AI LongContextReorder代码：
```java
List<Document> top50 = vectorStore.similaritySearch(SearchRequest.query(q).withTopK(50));
List<Document> top4 = jinaReranker.rerank(q, top50, 4);  // 精排后Top4：按相关性降序[A,B,C,D]
// A最相关！但是直接[A,B,C,D]塞上下文，A在头部记得住，B C中间丢30%，D尾部记得住
// ✨ 重排后顺序 [A, C, D, B] → A最相关放第1位(头)，B次相关放最后1位(尾)！CD塞中间
List<Document> reordered = LongContextReorderProcessor.INSTANCE.reorder(top4);
// reordered 顺序：[第1名相关性, 第4, 第3, 第2名相关性]
// → LLM头尾都记住最重要的信息了！
```

**重排前后QA准确率对比：**
| 设置 | QA准确率 |
|---|---|
| Top4直接按相关度顺序[A最相关,B,C,D] | 78% |
| ✅ LongContextReorder重排 | **87%** |
| **+9%提升** | |

#### 4. 面试追问：注意力为什么是U型？
→ 语言学+深度学习联合解释：① 首因效应(Primacy Effect)序列开头进入模型时无前面KV缓存干扰，注意力最"清醒"；② 近因效应(Recency Effect)序列结尾就在输出前刚过完，注意力还热；③ 中间段被前面的KV挤、被后面的KV覆盖→注意力稀释。Transformer的因果Mask导致后面的token能Attend到前面但注意力被稀释。

---

### Q18. Metadata先过滤 vs 后过滤顺序: 100万条数据先过滤部门到2万条 再算向量快多少？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 定义：Metadata过滤+向量相似度联合查询的两种策略
RAG系统每个文档有元数据：`部门dept、权限role、文档类型type、上传时间time、客户ID tenantId`

```
用户(财务部小Q)问:"报销政策？"
    ↓
┌───────────────────────────────────────────────────────┐
│ 策略A 【先过滤后检索】⭐推荐！生产99%场景                 │
│  Step1 WHERE dept='财务部' AND deleted=false             │
│        → 从100万条 → 先硬过滤到2万条财务相关文档          │
│  Step2 仅对这2万条做 HNSW 相似度Top50                    │
│   → 耗时：过滤1ms + 向量算1ms ≈ 2ms ✅极快               │
├───────────────────────────────────────────────────────┤
│ 策略B 【先检索后过滤】❌错误！生产灾难                     │
│  Step1 全表100万条做HNSW向量相似度Top500                   │
│  Step2 从这500条里WHERE dept='财务部'→可能最后剩3条！      │
│   → 耗时：5ms + 结果里99%是其他部门→召回率极差💥           │
│   → 风险：财务部文档在100万条里只排Top500之后！直接漏掉     │
└───────────────────────────────────────────────────────┘
```

#### 2. 📊 性能+准确率 先过滤vs后过滤 对比（100万条，HNSW索引）：
| 维度 | ✅ 先过滤(WHERE+相似度) | ❌ 后过滤(相似度+WHERE) | 先过滤提升 |
|---|---|---|---|
| **查询延迟** | ~1-3ms | ~5-20ms | ✅ 快5-10倍 |
| **召回率Recall** | ✅✅✅ 95%+（在对的池子里搜不丢） | ⭐ 30-70%（Top500里可能根本没财务部文档） | **+25-65%** 💥天差地别 |
| 内存/IO占用 | 只扫过滤后的小部分数据 | 全索引扫描+TopK排序 | 小10倍+ |

#### 3. Pgvector Spring Data SQL实现 先过滤：
```sql
-- ✅ 生产级WHERE子句先Metadata硬过滤！
SELECT content, 1-(embedding <=> $1) AS sim
FROM docs
WHERE 
  tenant_id = $2        -- 多租户：先限定客户
  AND dept = $3         -- 部门：小Q只能看财务部
  AND role::jsonb ?| array['finance_staff']  -- RBAC权限
  AND created_at > NOW()-INTERVAL '2 years'   -- 时间：只看2年内
  AND NOT deleted      -- 软删除
ORDER BY embedding <=> $1
LIMIT 50;
```
→ WHERE条件把100万→2万，仅在2万里做向量排序。

#### 4. 面试坑：Pgvector的索引能不能用到Metadata过滤？
→ 能不能走索引？→ 可以！**组合索引**：`(dept, tenant_id)`的btree索引先用，再走HNSW相似索引→两步都快。别把所有条件放向量索引（HNSW不支持btree过滤的混合索引，要分开建）。

---

### Q19. 向量维度: 768/1024/1536/3072 选高维度还是低维度？准确率+存储+延迟权衡 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. 主流Embedding模型维度全景对比：
| 模型 | 维度d | 中文MTEB得分 | 单向量存储(binary) | 500万条存储 | 单次HNSW搜索延迟 |
|---|---|---|---|---|---|
| M3E-Base / BGE-Base | **768** | 62分 | 3KB(768×4B) | 15GB | 1ms |
| M3E-Large / BGE-Large | **1024** | 67分 | 4KB | 20GB | 1.5ms |
| OpenAI text-embedding-3-large | **3072** | 71分 | 12KB | 60GB | 4ms |
| OpenAI text-embedding-ada-002(旧) | 1536 | 60分 | 6KB | 30GB | 2ms |
| Jina-embedding-v3 | **1024** | 68分中文最好 | 4KB | 20GB | 1.5ms |

#### 2. 📊 维度×准确率×成本权衡三难抉择：
```
        维度越高 → 准确率↑ 但是：存储↑x倍 延迟↑x倍 成本↑x倍
        维度越低 → 省钱快 但是：准确率↓
```

**场景选型黄金经验值：**
| 场景 | 推荐维度 | 理由 |
|---|---|---|
| ✅ 通用知识库/中文为主(大多数) | **1024** | BGE-Large/M3E-Large性价比甜点位：67分，20GB/500万条 |
| 中文语义简单FAQ/检索任务 | **768** | 62分，省一半存储，15GB够用了 |
| 超大亿级向量+成本敏感 | 768+PQ64乘积量化压缩→**64字节** | 量化掉12倍，15GB→1.2GB，Recall-5%可接受 |
| 英文/多语言+要求最高准确率 | 3072 (OpenAI Large) | 71分最高，但贵3倍慢3倍 |
| 必须用OpenAI API生态 | 1024(text-3-small推荐，比ada便宜+小) | ada-002的1536被淘汰了，3代更好 |

#### 3. 面试追问：能不能把1536维降成768维用？
→ 可以！用**PCA主成分分析**线性降维，1536→768，Recall仅损失1-2%，存储直接减半。训练集上用sklearn PCA fit后保留前768主成分→768维向量，离线全量转一次就行。

---

### Q20. 多租户Multi-Tenant RAG权限控制: 每个用户只能查自己部门文档，怎么正确实现？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 5级多租户权限控制（从弱到强，生产第4级起）

| 级别 | 实现方式 | 隔离性 | 性能 | 成本 | 适合场景 |
|---|---|---|---|---|---|
| ①应用层过滤(最差❌) | 应用代码查出来后for循环if(tenantId==用户tenant) 丢 | ❌无！黑客改前端参数看全库💥 | 慢(全查完丢90%) | 低 | Demo/玩具 |
| ②Metadata WHERE过滤(⭐标准做法) | 每个doc存tenant_id/dept，SQL WHERE tenant_id=$user.tenant先过滤 | ✅逻辑隔离(强) | ✅快 | ✅最低 | 90%标准SaaS |
| ③Schema隔离(PG Schema per tenant) | 每个租户独立Schema+独立docs表+独立HNSW索引 | ✅✅更强隔离 | ✅快 | 中(Schema×N维护) | SaaS大客户定制 |
| ④Database隔离(独立PG库/Docker) | 每个租户独立数据库甚至独立容器 | ✅✅✅物理隔离最强 | ✅快 | 高(运维成本N库) | 金融/政府强合规 |
| ⑤独立向量库Milvus分collection | 每个租户独立Collection/Partition | ✅✅逻辑分区隔离 | ✅快 | 高 | Milvus集群用户 |

#### 2. ✅ 级别②标准生产方案Spring Boot实现（90%场景）：
**① 文档入库时强制带上租户+部门+权限Metadata：**
```java
// Document.getMetadata() 必须硬塞租户信息，别信前端传的！
void addDoc(Long currentTenantId, String currentDept, byte[] pdfBytes) {
    List<Document> chunks = pdfSplitter.split(parsePdf(pdfBytes));
    for (Document chunk : chunks) {
        // ✅ 强制覆盖写入，防止恶意伪造
        chunk.getMetadata().put("tenantId", currentTenantId);  
        chunk.getMetadata().put("dept", currentDept);
        chunk.getMetadata().put("uploaderUid", SecurityUtil.getUid());
        chunk.getMetadata().put("uploadTime", Instant.now());
    }
    vectorStore.add(chunks);
}
```

**② 检索时WHERE子句(先过滤)条件带当前用户权限Context：**
```java
@GetMapping("/search")
public List<Document> search(String q) {
    // ✅ 从SecurityContext拿当前用户信息，不用前端传
    UserDetails me = SecurityUtil.currentUser();
    SearchRequest req = SearchRequest.query(q)
        .withTopK(50)
        // ✅ Filter表达式：先过滤用户能看的，再向量相似
        .withFilter(new FilterExpressionBuilder()
            .eq("tenantId", me.getTenantId())  // 1️⃣必加：租户硬隔离！
            .in("dept", me.getVisibleDepts())  // 2️⃣部门可见范围
            .build());
    return vectorStore.similaritySearch(req);
}
```

#### 3. 常见面试坑（90%SaaS踩过）：
| 坑 | 后果 | 修复 |
|---|---|---|
| 🔴 tenantId从前端query参数传 | 💥黑客传别人的tenantId直接看数据 | 从SecurityContext/Token拿，永不信前端 |
| 🔴 没给tenant_id单独B树索引 | WHERE过滤慢 全表扫100万 | CREATE INDEX idx_docs_tenant ON docs(tenant_id) |
| 🔴 向量相似度TopN之后才过滤tenantId → Lost in 50条里没用户部门的，直接漏掉！ | 召回率掉50%+ | 先WHERE Metadata过滤，再算相似度(级别②) |
| 🔴 租户共用索引但少部分租户数据量100倍于其他 | 大租户的索引独占HNSW缓存，小租户慢抖 | 用分区表PARTITION BY tenant_id + HNSW分分区 |
| 🔴 员工离职后权限没变 | 继续看原部门文档→泄露 | RBAC动态权限 + 每次搜实时查user_roles表 |

---
> 📌 上篇 Q1-Q20完成（基础架构10题+RAG深入10题）。下篇Q21-Q40：Tool Calling 8题+生产部署10题 详细标准答案。
