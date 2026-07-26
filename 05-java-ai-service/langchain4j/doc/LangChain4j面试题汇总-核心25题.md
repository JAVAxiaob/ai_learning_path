# LangChain4j 面试题汇总（核心25题 带详细标准答案）

> 位置: 05-java-ai-service/langchain4j/doc/
> 配套: Spring AI面试题汇总(上下篇40题) | DJL面试题汇总(25题)
> 核心考点: Spring AI vs LangChain4j对比 | AiServices接口驱动 | @Tool/@P注解 | Guardrails | EasyRAG | Agentic Patterns

---

## 一、架构与核心概念 (Q1-Q10)

---

### Q1. LangChain4j vs Spring AI 核心对比15项：Java生态两大RAG框架怎么选？(⭐⭐⭐⭐⭐必考！)

**【标准答案】**

#### 1. 核心架构设计理念15项全方位对比（面试必考表格）：

| 对比维度 | LangChain4j (Java原生社区驱动) | Spring AI (Spring官方) | 胜出 |
|---|---|---|---|
| **设计哲学** | ✅ **接口+注解(AiServices)** 声明式开发，像写MyBatis Mapper | ✅ **流式Builder(ChatClient)** 函数式API，类似WebClient | 平手，看团队风格 |
| **Spring Boot整合** | ✅ langchain4j-spring-boot-starter(可选) | ✅✅✅ 官方深度整合Starter全家桶，配置属性自动装配 | **Spring AI胜** |
| **Quarkus支持** | ✅✅✅ 官方Quarkus扩展原生，GraalVM Native AOT | ❌ 完全不支持 | **LangChain4j胜** |
| **纯Java SE无框架** | ✅ 100%支持，零依赖直接用 | ⚠️ 强依赖Spring容器环境，无Spring很难用 | **LangChain4j胜** |
| **LLM厂商数** | ✅ 35+家(OpenAI/智谱/百炼/Kimi/本地) | ✅ 20+家(主要大厂，长尾少) | LangChain4j略胜 |
| **向量库数量** | ✅✅✅ **45+向量库** (Pgvector/Milvus/Qdrant/Chroma/Redis/ES/ArcadeDB/Coherence/Jvector...) | ✅ 15+家(主流有，小众少) | **LangChain4j大胜** |
| **AiServices接口** | ✅ 独有核心，@SystemMessage/@UserMessage/MemoryId写接口即可 | ❌ 无此概念，需手写ChatClient调用 | **LangChain4j独有** |
| **结构化输出体系** | ✅✅✅ PojoOutputParser/Enum/List/Set/BigDecimal共20+种解析器 | ✅ StructuredOutputConverter支持POJO | LangChain4j更全 |
| **工具调用@Tool** | ✅ @Tool + @P(name/description/required/defaultValue)四要素全 | ✅ @Tool + @P(三要素，无defaultValue) | LangChain4j@P功能更多 |
| **护栏Guardrails** | ✅✅✅ InputGuardrails/OutputGuardrails独立模块内置 | ❌ 无内置，需ContentSafeAdvisor自己写 | **LangChain4j胜** |
| **Easy RAG模块** | ✅ langchain4j-easy-rag一行代码搭RAG，零配置 | ❌ 无，需自己组合VectorStore/Splitter | **LangChain4j胜** |
| **智能体Agentic** | ✅ ReAct/Planner/Orchestration/MCP多智能体独立模块 | ⚠️ 基础Tool Calling，复杂Agent模式需自己写 | **LangChain4j胜** |
| **流式输出** | ✅ Flux/TokenStream.onNext/onComplete/onError | ✅ Flux<String> SSE | 平手 |
| **社区成熟度** | ⭐⭐⭐ GitHub Stars 7k+，Java社区独立项目 | ⭐⭐⭐⭐ Spring官方背书，Spring生态流量大 | 平手 |
| **生产稳定性** | ✅ 1.0正式版发布，API稳定 | ✅ 1.0刚发布，Spring背书稳定 | 平手 |

#### 2. 🎯 选型决策树（面试被问"我们项目选哪个"按这个说）：
```
你是什么项目？
  ├─ ① 已有Spring Boot 3.x技术栈，纯后端团队 → ✅ 选Spring AI！Starter开箱即用，团队学习成本最低
  ├─ ② Quarkus/Native微服务，启动快内存小场景 → ✅ 选LangChain4j！唯一支持GraalVM AOT原生
  ├─ ③ 纯Java SE桌面程序/无Spring/旧Spring 5.x → ✅ 选LangChain4j！零容器依赖直接用
  ├─ ④ 需要非常多向量库(40+)/小众模型(通义/DeepSeek/阶跃星辰...) → ✅ 选LangChain4j！集成全
  ├─ ⑤ 复杂Agent多智能体协作/Guardrails合规 → ✅ 选LangChain4j！高级特性多
  └─ ⑥ 大厂Spring生态/已有Spring Cloud微服务全家桶 → ✅ 选Spring AI！集成Gateway/Feign/Config最顺
```

#### 3. 代码风格直观对比（同样功能：客服+订单查询工具）：
```java
// ✅ LangChain4j写法：声明式接口，像MyBatis Mapper。3行=全部代码
public interface CustomerServiceAi {
    @SystemMessage("你是京东金牌客服小京，语气亲切专业，按员工手册回答")
    @UserMessage("用户问题：{{it}}。用知识库+订单工具回答")
    String chat(String userQuestion, @MemoryId Long userId, @Tool Object orderTools);
}
// 调用：直接注入接口 → ai.chat("我订单8848到哪了", 10086L, tools);
```

```java
// ✅ Spring AI写法：流式Builder API，像WebClient。灵活但代码多
@Bean ChatClient customerChatClient(ChatModel m, VectorStore v) {
    return ChatClient.builder(m)
        .defaultSystem("你是京东金牌客服小京...")
        .defaultAdvisors(new QuestionAnswerAdvisor(v), new ChatMemoryAdvisor(redis))
        .defaultTools(orderTools)
        .build();
}
// 调用：注入ChatClient → client.prompt().user("我订单8848哪了").call().content();
```

---

### Q2. AiServices核心设计：为什么用接口+注解写AI服务？JDK动态代理+MethodInterceptor底层原理 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. AiServices = JDK动态代理 + 注解驱动 + 自动装配
```
开发写个Interface → AiServices.create(Class)生成代理对象 → 调方法自动：
  1. 解析@SystemMessage/@UserMessage的SpEL模板 {{变量}}
  2. 组装ChatMemory（根据@MemoryId找对应会话）
  3. 扫描@Tool注解的参数→自动转ToolSpecification→传给LLM
  4. 调用LLM Chat API，自动解析工具调用+反射执行@Tool方法
  5. 根据返回值类型自动用OutputParser（String→POJO→List→Enum）
  6. 把对话写回ChatMemory
```
→ 完全像MyBatis @Mapper注解写SQL：**接口没有实现类，但调方法就执行完整LLM链路。**

#### 2. AiServices完整使用示例（7大注解全）：
```java
// ✅ 核心注解7件套：1个@SystemMessage + 1@UserMessage + 1@MemoryId + 1@V(变量) + 工具+输出
public interface HrAssistantAi {

    // ✨ @SystemMessage: 写在接口方法上，SpEL模板支持{{变量}}注入
    @SystemMessage("""
        你是HR助手{{hrName}}，公司：{{companyName}}。
        严格遵守手册：试用期3个月，年假5-15天，病假需医院证明。不知道就转HRBP分机8080。
        """)
    @UserMessage("员工{{employeeName}}问：{{question}}。结合员工信息：{{empInfo}}，用工具查完准确回答")
    // ✨ 返回值自动用PojoOutputParser转POJO，不用手写JSON解析！
    HrAnswer askQuestion(
        @V("hrName") String hrName,             // ✨ @V: 注入模板变量值
        @V("companyName") String company,
        @V("employeeName") String empName,
        @V("question") String q,
        @V("empInfo") EmployeeProfile info,      // 对象自动JSON序列化进{{模板}}
        @MemoryId Long empId,                    // ✨ @MemoryId：按员工ID独立ChatMemory
        Object tools);                            // ✨ 工具类实例，扫描其中@Tool方法
}
```

```java
// 调用代码：没有实现类！AiServices.create()动态代理生成
HrAssistantAi ai = AiServices.builder(HrAssistantAi.class)
    .chatLanguageModel(openAiChatModel)         // 绑定LLM
    .chatMemoryProvider(memId -> MessageWindowChatMemory.withMaxMessages(20)) // 按MemoryId分内存
    .tools(new HrTools(), new AttendanceTools())// 全局工具（或者方法参数传也行）
    .build();

// ✅ 调方法，一行=完整LLM+记忆+工具+结构化输出
HrAnswer ans = ai.askQuestion("HR小王", "字节跳动", "张三", "我年假有几天？", 
                              empProfile, 10086L /*MemoryId*/);
```

#### 3. 底层实现原理（面试加分问原理）：
→ **JDK动态代理**：`AiServices.create()` → `Proxy.newProxyInstance(classLoader, interfaces, handler)`
→ **InvocationHandler.invoke()** 里干这6件事：
```
1. 取Method上@SystemMessage/@UserMessage注解值，SpEL解析{{变量}}替换成参数值
2. 找参数中@MemoryId→从ChatMemoryProvider取对应记忆→合并System/User/历史Message
3. 找所有@Tool参数→反射扫描方法上@Tool→自动生成ToolSpecification JSON Schema
4. 调ChatModel.generate()，如果LLM返回tool_calls：
   → 解析参数→反射.invoke(toolInstance, args)→结果回喂LLM→loop直到无tool_calls
5. 最终返回内容：返回类型是String直接返回，POJO用PojoOutputParser，List用PojoListOutputParser
6. 把本轮User+Assistant消息add进ChatMemory持久化
```

---

### Q3. @Tool注解 + @P注解四要素完整写法：LangChain4j比Spring AI多了什么？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. @Tool/@P黄金四要素 + LangChain4j独有特性：
LangChain4j @P比Spring AI **多了1个defaultValue默认值特性** → 解决LLM漏传参数90%问题。
```java
@Component
public class PaymentTools {

    @Tool(
        name = "query_user_payment_last_30_days",  // ① 工具名（可选，默认方法名）
        value = {                                  // ② 工具描述数组（支持多行，更清晰！）
            "查询用户近30天支付宝/微信/银行卡支付流水账单",
            "当用户问：花了多少钱/账单明细/消费记录/都买了什么，必须调用此工具",
            "不要猜测金额，必须用工具查询真实数据"
        },
        returnBehavior = ReturnBehavior.TO_LLM,     // 返回给LLM(默认)，可选DIRECT直接给用户
        searchBehavior = SearchBehavior.ALWAYS_VISIBLE  // ✅独有：SEARCHABLE(默认用工具检索找) / ALWAYS_VISIBLE
    )
    List<PaymentRecord> queryRecentPayments(
        // ✨ @P四要素：name(参数名给LLM看) + description(描述) + required + defaultValue【独有！】
        @P(name = "userId",
           description = "用户ID Long正整数，示例10086，不能为0或负",
           required = true)                    // ③ required：JSON Schema的required数组
        Long userId,

        @P(description = "支付状态过滤：ALL/SUCCESS/REFUND/PARTIAL", required = false)
        String status,

        @P(description = "返回条数正整数1-100，超过100截断",
           defaultValue = "20")                // ✨④ 独有defaultValue！LLM不传就用20，不用null判断
        Integer limit
    ) {
        // 不用判断limit == null，有defaultValue兜底永远是整数
        return paymentMapper.selectRecent(userId, status, limit);
    }
}
```

#### 2. LangChain4j @Tool vs Spring AI @Tool 细节对比：
| 特性 | LangChain4j @Tool + @P | Spring AI @Tool + @P |
|---|---|---|
| 参数名重写`@P(name="xxx")` | ✅ 支持（解决无-parameters编译参数arg0问题） | ❌ 不支持，默认参数名 |
| 参数默认值`@P(defaultValue)` | ✅ ✅✅ 独有！不传就用默认，防止NPE | ❌ 无，只能自己判null |
| 工具描述多行长文本 | ✅ `String[] value()`数组多行写 | ✅ `String value()` 单字符串 |
| Tool返回策略ReturnBehavior | ✅ TO_LLM / DIRECT(直接给用户跳LLM) | ❌ 无，永远回LLM |
| Tool可搜索性SearchBehavior | ✅ SEARCHABLE工具向量检索 / ALWAYS_VISIBLE | ❌ 无，工具全挂LLM上 |
| Tool元数据metadata() | ✅ 支持Anthropic缓存提示等特殊元数据 | ❌ 无 |
| @ToolMemoryId注解参数 | ✅ 方法参数注入记忆ID，工具内访问上下文 | ❌ 无 |

#### 3. 面试坑：为什么必须写`defaultValue`？
→ LLM调用工具漏传参数率约15-30%，特别optional参数。Spring AI只能自己写`if (limit == null) limit = 20;`每个参数都要判一遍。LangChain4j`defaultValue="20"`→编译器保证不进null，代码干净。

---

### Q4. ChatMemoryProvider按MemoryId分桶：1000用户如何隔离不串会话？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 核心接口设计：AiServices的ChatMemory + ChatMemoryProvider（比Spring AI更灵活）
```java
// 两种绑定ChatMemory方式：
// 方式A 【固定单会话】 全局绑定同一个ChatMemory → 所有调用共享（单用户桌面端）
AiServices.builder(Cls.class).chatMemory(MessageWindowChatMemory.withMaxMessages(20))

// 方式B 【分桶隔离 生产SaaS用⭐】ChatMemoryProvider<ID> → 按@MemoryId动态创建/查找每个用户的独立记忆
// → Function<MemoryId, ChatMemory> 进来一个ID，返回对应ChatMemory实例
AiServices.builder(Cls.class)
    .chatMemoryProvider(memId -> {  // memId = 方法上@MemoryId注解参数传的值
        // 每个用户Redis Hash Key = "ai:mem:" + memId
        return RedisChatMemory.builder(memId)
            .maxMessages(20).redisTemplate(redis).ttl(Days7).build();
    })
```

#### 2. 完整示例：按租户+用户二级隔离（SaaS多租户）
```java
// ① 接口方法@MemoryId支持传组合主键（推荐用Record封装）
public interface SaasCustomerAi {
    @SystemMessage("你是SaaS客服助手，客户：{{tenant.name}}")
    String chat(@V("question") String q,
                @MemoryId TenantUserId id);  // ✨ 自定义MemoryId对象：tenantId+userId组合
}
// 自定义MemoryId：必须正确equals+hashCode！record自动生成，推荐用record
public record TenantUserId(Long tenantId, Long userId) {}
```

```java
// ② ChatMemoryProvider按组合Key取Redis分桶
SaaS_AI = AiServices.builder(SaasCustomerAi.class)
    .chatMemoryProvider(tenantUserId -> {
        String key = "saas:mem:%d:%d".formatted(tenantUserId.tenantId(), tenantUserId.userId());
        return RedisChatMemory.builder()
            .redisTemplate(redisTemplate)
            .key(key)
            .maxMessages(30)
            .ttl(Duration.ofDays(14))
            .build();
    })
    .chatLanguageModel(chatModel)
    .build();
```

```java
// ③ 调用：租户100的用户8848 → 绝对不会和租户200的用户8848串会话！
TenantUserId id1 = new TenantUserId(100L, 8848L);
TenantUserId id2 = new TenantUserId(200L, 8848L);  // 同userId不同tenant
ai1.chat("我订单8848", id1);  // key = saas:mem:100:8848
ai2.chat("我订单8848", id2);  // key = saas:mem:200:8848 → 不同桶完全隔离
```

#### 3. LangChain4j内置3种ChatMemory对比（面试必考选型）：
| ChatMemory类型 | 算法 | Token控制方式 | 适用场景 | 推荐度 |
|---|---|---|---|---|
| `MessageWindowChatMemory` | 固定消息条数N | 保留最近N条消息（N=20约8-12KToken） | 简单场景，对话轮次少 | ✅ 新手首选，简单不易错 |
| ✨`TokenWindowChatMemory` | 按Token数滑动窗口 | TikTokens精确计数，保留最近≤MaxToken条（如28000Token） | 生产推荐！最准确，不会意外爆Context | ✅✅✅ 生产首选 |
| `SingleSlotChatMemoryStore` | 单条存储(存key-value) | 只存结构化提取槽位，不存全文 | 表单填写/实体提取场景 | ⭐ 特殊场景 |

---

### Q5. 结构化输出OutputParser体系：20+种解析器 vs Spring AI单一StructuredOutput (⭐⭐⭐⭐)

**【标准答案】**

#### 1. LangChain4j OutputParser 20+种全景图：
```
基础类型解析（9种）：String/Integer/Long/Double/Float/Boolean/BigDecimal/BigInteger/Byte
日期时间解析（4种）：Date/LocalDate/LocalTime/LocalDateTime
枚举解析（4种）：Enum / EnumList / EnumSet / EnumCollection
POJO结构化（4种）：Pojo / PojoList / PojoSet / PojoCollection
集合容器（3种）：StringList/StringSet/StringCollection
```

#### 2. 结构化输出自动推断代码：
AiServices会**根据方法返回值类型自动选择解析器**，不用手动指定！这是最大的便利性：
```java
public interface DataExtractorAi {
    // 返回List<OrderItem> → 自动选PojoListOutputParser！不用配置
    @UserMessage("从下面订单邮件提取所有订单项。邮件：\n{{email}}")
    List<OrderItem> extractOrderItems(@V("email") String emailText);

    // 返回BigDecimal → 自动选BigDecimalOutputParser
    @UserMessage("算一下订单总金额。订单项JSON：{{items}}")
    BigDecimal calculateTotal(@V("items") List<OrderItem> items);

    // 返回Enum → 自动选EnumOutputParser，只允许枚举范围内的值！LLM输出不在枚举内重试
    @UserMessage("判断这封邮件的紧急程度：LOW/MEDIUM/HIGH/URGENT。邮件：{{text}}")
    UrgencyLevel classifyUrgency(@V("text") String emailText);

    // 返回Map<String, Object> → 自动JSON转Map
    @UserMessage("提取简历信息为JSON：姓名/电话/工作年限/技能列表。简历：{{resume}}")
    Map<String, Object> extractResumeInfo(@V("resume") String resume);
}
```

#### 3. PojoOutputParser + Bean Validation 2次重试校验：
LangChain4j自动集成Jakarta Validation（@NotNull/@Size/@Pattern等注解），解析完后自动校验：
```java
public class Resume {
    @NotNull(message = "姓名不能为空")
    @Size(min = 2, max = 20)
    String name;

    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号格式错误")
    String phone;

    @Min(value = 0, message = "工作年限不能负") @Max(50)
    Integer yearsOfExperience;

    @Size(max = 50)
    List<String> skills;
}
// 第一次LLM输出{name:null, phone:"123"} → 校验不通过 → 自动把错误信息回喂LLM修正
// → 最多重试2次，成功率从70%升到98%+
```

---

### Q6. Guardrails护栏模块：Input/Output独立内容安全怎么实现？比Spring AI手写Advisor方便在哪？(⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. LangChain4j Guardrails = 内置InputGuardrails + OutputGuardrails
Spring AI没有内置，得自己写Advisor判断。LangChain4j官方模块开箱即用。

```java
// ① 注册Guardrails
InputGuardrails inputGuard = InputGuardrails.builder()
    // 1️⃣ PII检测：手机号/身份证/银行卡自动脱敏
    .add(new PiiDetectionAndMaskingGuard(Phone, IdCard, BankCard, Email))
    // 2️⃣ 关键词黑名单：政治/色情/暴力
    .add(new KeywordBlacklistGuard(List.of("抽奖", "刷单", "博彩")))
    // 3️⃣ Prompt注入检测：Jailbreak/忽略之前的指令
    .add(new PromptInjectionDetectionGuard(OPENAI_MODERATION_ENDPOINT))
    // 4️⃣ 长度限制：超长5000字符直接拒
    .add(new MaxLengthGuard(5000, "请分段提问，单次不超过5000字"))
    .build();

OutputGuardrails outputGuard = OutputGuardrails.builder()
    // 1️⃣ 输出不能含PII：防止LLM把记忆中的手机号复述出来
    .add(new PiiDetectionAndMaskingGuard())
    // 2️⃣ 事实校验（RAG场景）：没引用到知识库内容的输出加警告
    .add(new HallucinationDetectionGuard(knowledgeEmbedding, threshold=0.7f))
    // 3️⃣ 禁止输出的关键词（竞品/脏话/内部机密）
    .add(new KeywordBlacklistGuard(List.of("竞品A", "竞品B", "薪资")))
    .build();
```

```java
// ② 绑定到AiServices上 → 全局生效，不用每个接口写
HrAssistantAi ai = AiServices.builder(HrAssistantAi.class)
    .chatLanguageModel(chatModel)
    .inputGuardrails(inputGuard)    // 用户输入进来先过4层护栏
    .outputGuardrails(outputGuard)  // LLM输出给用户前过3层护栏
    .onInputGuardrailFailure(ctx -> { // 触发护栏时自定义返回：不用抛异常给前端
        log.warn("输入护栏触发 {} userId={}", ctx.getRuleName(), ctx.getUserId());
        return GuardrailResult.blocked("提问包含敏感内容，请修改后重试。触发规则："+ctx.getRuleName());
    })
    .build();
```

#### 2. Guardrails vs 手写Advisor（Spring AI方式）对比：
| 维度 | LangChain4j内置Guardrails | Spring AI 手写ContentSafeAdvisor |
|---|---|---|
| PII检测 | ✅ 内置PiiDetectionGuard，正则+类型库现成 | ❌ 自己写正则Pattern |
| Prompt注入检测 | ✅ 集成OpenAI Moderation/Judge0模型 | ❌ 自己买模型/写规则 |
| 幻觉检测输出 | ✅ HallucinationDetectionGuard 向量相似度校验 | ❌ 自己调Embedding比较 |
| 重试机制 | ✅ GuardrailResult.retry("把输出改掉") | ❌ 自己写循环重调 |
| 失败处理 | ✅ onFailure回调统一处理 | ❌ 每个Advisor写Try-catch |
| 开发量 | 20行配置=全部护栏 | 300-500行自己写每个规则 |

---

### Q7. Easy RAG模块：5行代码搭完完整RAG系统怎么做到的？(⭐⭐⭐⭐)

**【标准答案】**

#### 1. langchain4j-easy-rag 5行代码完整版（开箱即用，不用自己装配Splitter/Embedding/Store）：
pom.xml加依赖：
```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-easy-rag</artifactId> <!-- ✨ 一站式，传递性引入所有必需依赖 -->
</dependency>
```
```java
// 🚀 EasyRAG 5行搞定：文档加载→切块→Embedding→入库→检索→回答一条龙
import static dev.langchain4j.rag.easy.EasyRAG.*;

// 1️⃣ 文档导入：递归扫目录下所有PDF/Docx/TXT/Markdown
ingestAllDocumentsRecursively(Paths.get("./docs/company-manual"));

// 2️⃣ 直接问答：自动调OpenAI text-embedding-3-small + InMemoryEmbeddingStore + RecursiveSplitter
String answer = chat("年假政策怎么规定的？");
System.out.println(answer); // → 根据手册回答，100%开箱不用任何配置
```

#### 2. EasyRAG默认配置（可覆盖自定义）：
| 组件 | 默认值 | 自定义覆盖方式 |
|---|---|---|
| LLM | OpenAI GPT-4o-mini（env OPENAI_API_KEY） | EasyRAG.chatModel(yourCustomModel) |
| Embedding模型 | OpenAI text-embedding-3-small | EasyRAG.embeddingModel(BGE_SMALL_ZH) |
| 文档切块 | RecursiveDocumentSplitter 300字/50字重叠 | EasyRAG.documentSplitter(自定义) |
| 向量库 | InMemoryEmbeddingStore(内存，重启丢) | EasyRAG.embeddingStore(PgvectorEmbeddingStore(...)) |
| 文档解析 | AutoDetectParser(PDFBox+POI+Tika) | EasyRAG.documentParser(自定义) |
| 检索参数 | Top-K=10，无重排器 | EasyRAG.contentRetriever(ContentRetriever.builder()...) |

#### 3. 生产级EasyRAG配置（换Pgvector+BGE中文Embedding，10行搞定）：
```java
// ✅ 从Demo模式升级生产模式：替换3个组件即可，其他不变
EasyRAG.customize()
    .chatLanguageModel(OllamaChatModel.builder().baseUrl("http://gpu:11434").modelName("qwen2:7b").build())
    .embeddingModel(new BgeSmallZhV15EmbeddingModel()) // 本地中文Embedding省API成本
    .embeddingStore(PgvectorEmbeddingStore.builder()  // 持久化向量数据库
        .dataSource(dataSource).table("docs_embeddings").dimension(384).build())
    .contentRetriever(ContentRetriever.builder()
        .embeddingStore(store).embeddingModel(model).maxResults(20).minScore(0.7f)
        .reranker(new JinaReranker("jina-reranker-v2-base-multilingual")) // 加Reranker精排Top4
        .build())
    .applyConfiguration(); // → 其他代码完全不变！chat()直接用
```

---

### Q8. ToolSearchStrategy工具检索：当有200+工具时全挂LLM上爆Context怎么办？向量检索选相关Top5工具 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. 工具太多的问题：
当系统有>20个@Tool时（大公司HR系统50个工具，电商100+），全挂LLM每次都传200个工具的JSON Schema→**浪费10K+Token，LLM工具选择准确率暴跌到<40%，根本选不对工具**💥

#### 2. LangChain4j独有ToolSearchStrategy（Spring AI没有！面试独有点）：
```
解决：200个工具不全挂LLM上，先用问题Embedding去搜工具的描述向量，
      召回Top5最相关的工具，这5个才挂给LLM调用！
→ Token省90%，工具选择准确率从<40%→>92%✨
```

**代码实现：VectorToolSearchStrategy（向量检索工具）**
```java
// 工具检索配置：把200个工具的描述向量化存EmbeddingStore
VectorToolSearchStrategy toolSearch = VectorToolSearchStrategy.builder()
    .embeddingModel(bgeZhEmbedding)      // 用BGE中文Embedding算工具描述向量
    .embeddingStore(InMemoryEmbeddingStore.forToolSpecs())
    .maxToolsToInclude(5)                // ✨ 最多取5个最相关工具给LLM
    .minScore(0.65f)                     // 相似度<0.65的工具直接不挂
    .addAllTools(hrTools, payrollTools, attendanceTools, erpTools, ...) // 注册200工具
    .build();

// 绑到AiServices上
HrSuperAi ai = AiServices.builder(HrSuperAi.class)
    .chatLanguageModel(chatModel)
    .toolSearchStrategy(toolSearch)      // ✨ 启用工具检索！不全都给LLM
    .build();
```

#### 3. 200个工具场景3种策略效果对比：
| 策略 | LLM看到工具数 | 每请求Token消耗 | 工具选择准确率 |
|---|---|---|---|
| ❌ 全量挂载（Spring AI默认）| 200个 | +12K Token（浪费12元每1000次）| 35-40% |
| ✨ SimpleToolSearchStrategy（关键词匹配）| Top5相关 | +400 Token | 78% |
| ✅✅✅ VectorToolSearchStrategy（向量语义匹配⭐生产）| Top5最相关 | +400 Token | **92-95%** |

---

### Q9. Agentic Patterns智能体模式：ReAct / Planner / Orchestrator 三种模式分别用在哪？(⭐⭐⭐⭐)

**【标准答案】**

#### 1. LangChain4j Agentic三大模式对比（Spring AI仅有基础Tool Calling）：
| 模式 | 核心思想 | 循环步骤 | 适用场景 |
|---|---|---|---|
| ① **ReAct Agent** 最常用⭐ | Reason(思考) + Act(行动) 交替循环 | Thought→Action→Observation→Thought→...→Final Answer | 通用90%场景：客服/问答/搜索+工具 |
| ② **Planner + Executor** | 先列完整执行计划Plan→再一步步Executor执行 | Plan(生成N步任务列表)→Executor逐个执行→修正计划→完成 | 多步骤复杂任务："订机票+酒店+接送机+发邮件" |
| ③ **Orchestrator Multi-Agent** | 调度多个子Agent协作（专家各司其职）| BossAgent分派→HRAgent/FinanceAgent/OrderAgent分别执行→汇总 | 超级复杂多领域："员工入职全流程"（HR+IT+财务+行政4Agent协作） |

#### 2. ReAct Agent最简代码（langchain4j-agentic）：
```java
ReActAgent agent = ReActAgent.builder()
    .chatLanguageModel(gpt4o)
    .tools(searchTool, orderQueryTool, paymentRefundTool, emailTool)
    .chatMemory(TokenWindowChatMemory.withMaxTokens(28000))
    .maxIterations(10)    // 最多思考-行动10轮，防死循环
    .toolExecutionHandler(new LoggingToolHandler())
    .build();
// 用户："把订单8848退了，退款邮件发我" → ReAct自动循环：
// Thought1: 我先查订单状态→Action: queryOrder(8848)→Observation: 未发货可退
// Thought2: 状态正常可退→Action: refundOrder(8848)→Observation: 退款成功¥299
// Thought3: 现在发邮件→Action: sendEmail→Observation: 邮件已发送
// Thought4: 所有步骤完成，总结结果给用户
AgentResult result = agent.run("把我订单8848退了，退款确认邮件发我邮箱");
```

---

### Q10. Moderate内容审核注解：@Moderate在接口方法上自动调Moderation API，拒绝违规输入输出 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. @Moderate注解使用（一行注解=内容安全，不用Guardrails模块的重量级配置）：
```java
public interface SafeChatAi {
    // ✨ @Moderate：输入+输出 自动调用OpenAI Moderation API检测
    // Category = hate/sexual/violence/self-harm 六大分类
    @Moderate(
        categoriesToCheck = { HATE, VIOLENCE, SELF_HARM, SEXUAL }, // 要检测的类别
        onInputViolation = @ViolationBehavior(
            action = BLOCK,    // BLOCK拒答 / WARN给警告 / LOG只记录不拦截
            message = "你的提问包含违规内容，请遵守社区规范"
        ),
        onOutputViolation = @ViolationBehavior(action = BLOCK, message = "抱歉，无法生成此内容")
    )
    @SystemMessage("你是健康生活助手，只回答合法合规问题")
    String safeChat(String userText, @MemoryId Long userId);
}
```
→ 比Spring AI自己写Advisor少写100行代码，注解驱动零侵入。

---

## 二、生产部署 & 高级特性 (Q11-Q25)

---

### Q11. Quarkus + LangChain4j Native AOT编译：启动5ms vs Spring Boot的1.5秒，省90%内存 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. GraalVM Native Image性能对比（同机器同代码RAG程序）：
| 指标 | Spring Boot 3.x + Spring AI (JIT) | Quarkus 3.x + LangChain4j (Native AOT) | 提升倍数 |
|---|---|---|---|
| **启动时间** | 1500ms (1.5秒) | ✅ **5ms** | 快300倍💥 |
| **内存占用RSS** | 450MB | ✅ **45MB** | 省90%内存💥 |
| 冷启动首请求 | 2500ms | ✅ 20ms | 快125倍 |
| 打包镜像大小 | 280MB (JRE+依赖) | ✅ 80MB | 省71%体积 |
| 容器副本成本 | 450MB × 100Pod = 45GB内存 | 45MB × 100Pod = 4.5GB | 省40GB=月省¥3000云资源 |
| K8s HPA快速扩容 | 1.5秒+容器启动=3-5秒就绪 | 100ms内就绪=秒级扩容 | |

#### 2. Quarkus LangChain4j Starter依赖：
```xml
<dependency>
    <groupId>io.quarkiverse.langchain4j</groupId>
    <artifactId>quarkus-langchain4j-openai</artifactId> <!-- Quarkus官方扩展 -->
</dependency>
```
```java
// Quarkus下AiServices变成CDI Bean，直接@Inject，不用AiServices.create()
@RegisterAiService // ✨ Quarkus专用注解，自动生成Bean注册CDI容器
@SystemMessage("你是Quarkus原生助手")
public interface QuarkusAi {
    @UserMessage("{{q}}")
    String chat(String q, @MemoryId String sessionId);
}
// 业务代码直接注入：
@Inject QuarkusAi ai;
```

→ 面试加分点：**LangChain4j是Java生态唯一官方支持Quarkus/Micronaut/Helidon等多框架的LLM SDK，Spring AI只绑Spring**。云原生Serverless场景（冷启动要快）选LangChain4j+Quarkus Native是天作之合。

---

### Q12. MCP Model Context Protocol：LangChain4j-agentic-mcp对接外部工具服务器 (⭐⭐⭐⭐)

**【标准答案】**

#### 1. MCP是什么？→ LLM的USB-C接口协议，标准化调用外部工具/资源/提示词
MCP客户端可以连MCP服务器（本地CLI/本地工具/远端SaaS），LangChain4j 1.12+原生支持。

#### 2. LangChain4j MCP代码示例：
```java
// ① 启动MCP客户端，连接一个本地文件系统MCP服务器和GitHub MCP服务器
McpClient mcpClient = McpClient.builder()
    .transport(new StdioMcpTransport("npx", "@modelcontextprotocol/server-filesystem", "./docs/"))
    .transport(new SseMcpTransport("https://mcp.github.com", ghToken))
    .build();
// ② 创建Agent时把MCP Client所有工具全部自动注册进去
Agent agent = ReActAgent.builder()
    .chatLanguageModel(gpt4o)
    .tools(mcpClient.getAllAvailableTools())  // ✨ 自动把远端MCP工具转成@Tool可用
    .build();
agent.run("统计docs目录下所有PDF的总页数，然后发Issue到GitHub/langchain4j说明文档整理完成");
```
→ MCP未来是LLM互操作的HTTP协议级标准，LangChain4j领跑支持，面试提这个=前沿加分。

---

### Q13. Token限流：AITokenRateLimiter每分钟1000Token，Bucket4j令牌桶+Redis分布式限流 (⭐⭐⭐⭐⭐)

**【标准答案】**

#### 1. Token限流 vs 请求限流：LLM用Token限流更合理（1个长请求=100个短请求Token）
```java
// ✅ 按Token计费限流（和OpenAI账单对齐，防刷Token恶意消费）
AiTokenRateLimiter rateLimiter = AiTokenRateLimiter.builder()
    .bucket4j()
    .redisStandalone(redisClient)  // 分布式：多Pod共享同一令牌桶
    // 三档限流（用户/租户/全局）
    .addLimit(BucketKey.userId(), Bandwidth.classic(50_000, Refill.greedy(50_000, Duration.ofMinutes(1)))) // 单用户50K/分
    .addLimit(BucketKey.tenantId(), Bandwidth.classic(500_000, Refill.greedy(500_000, Duration.ofMinutes(1)))) // 租户500K/分
    .addLimit(BucketKey.global(), Bandwidth.classic(5_000_000, Refill.intervally(5_000_000, Duration.ofMinutes(1)))) // 全局5M/分
    .onLimitExceeded(ctx -> new RetryAfterResult(Duration.ofSeconds(ctx.getWaitTimeSeconds()),
            "当前请求较多，预计"+ctx.getWaitTimeSeconds()+"秒后重试"))
    .build();
```

---

### Q14. 观测性Observability：Micrometer + OpenTelemetry 全链路自动埋点 AiServicesListener (⭐⭐⭐⭐)

**【标准答案】**

#### 1. AiServicesListener 7个回调钩子（全链路观测）：
```java
@Bean
AiServiceListener metricsListener() {
    return new AiServiceListener() {
        Long start;
        public void onRequest(AiServiceRequestIssuedEvent e) {  // ① 请求发出
            start = System.nanoTime();
            Metrics.counter("ai.requests.total").tag("method", e.methodName()).increment();
        }
        public void onResponse(AiServiceResponseReceivedEvent e) {  // ② 响应收到
            double sec = (System.nanoTime() - start) / 1e9;
            Metrics.timer("ai.response.latency").tag("model", e.model()).record(sec);
            Usage u = e.usage();
            Metrics.counter("ai.tokens.input").increment(u.inputTokens());
            Metrics.counter("ai.tokens.output").increment(u.outputTokens());
        }
        public void onToolExecution(ToolExecutedEvent e) {} // ③工具执行
        public void onGuardrailFail() {} // ④护栏触发
        public void onError(AiServiceErrorEvent e) { // ⑤错误
            Metrics.counter("ai.errors").tag("type", e.exceptionClass()).increment();
        }
        public void onStart() {} public void onComplete() {} // ⑥开始/⑦结束
    };
}
```

---

### Q15-Q25：11道高频加分题
> 位置：LangChain4j面试题 第二部分（后续补充，详见仓库）

---

## 📋 附：面试考点命中率预测表

| 问题 | 出现概率 | 分值权重 | 掌握程度自检 |
|---|---|---|---|
| Q1 Spring AI vs L4J 15项对比 | 90%面试必考 | 20分 | □不熟 □能说5项 □能说10项+选型树 |
| Q2 AiServices接口原理 | 75% | 15分 | □不熟 □会用 □懂JDK代理6步 |
| Q3 @Tool/@P四要素写法 | 70%+代码题 | 15分 | □不熟 □会写 □对比Spring AI差异 |
| Q4 ChatMemoryProvider隔离 | 60% | 10分 | □不熟 □能做用户隔离 □能做多租户 |
| Q5 OutputParser 20种 | 45% | 8分 | □不熟 □知道POJO/List □知道Validation重试 |
| Q6 Guardrails护栏 | 40%大厂常问 | 8分 | □不熟 □知道功能 □能说输入输出各3层 |
| Q7 Easy RAG | 30% | 5分 | □不熟 □知道5行代码 □知道生产配置 |
| Q8 ToolSearchStrategy | 35%加分题 | 5分 | □不熟 □知道为什么用 □能说准确率提升数据 |
| Q11 Quarkus Native性能 | 40%云原生场景 | 10分 | □不熟 □知道快 □能对比3项数据+选型 |

