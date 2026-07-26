# Spring AI 性能优化重难点解析

> 位置: 05-java-ai/spring-ai/doc/
> 配套文档: SpringAI-LLM应用集成指南.md | SpringAI流程图详解.md | SpringAI面试题汇总.md

---

## 一、Pgvector 三大索引选型对比

### 1.1 IVFFlat vs HNSW 原理+选型 (高频面试题)

```
场景: 500万条Chunk, 向量维度=1536

┌──────────────────────────────────────────────────────────────┐
│ 方案A: HNSW  (默认⭐推荐)                                      │
│ 原理: 多层跳表随机近邻图  16邻居 × 256搜索深度                  │
│ 构建时间: 慢 10-30分钟 (离线构建一次性)                         │
│ 查询延迟: 极快 1-5ms p95                                        │
│ 内存: 较大 向量×2字节 500万×2×1536×4B= 60GB内存                 │
│ Recall@5: 99%+  几乎接近精确线性扫描                            │
│ 适合: 百万-千万级 在线业务 对延迟敏感                            │
├──────────────────────────────────────────────────────────────┤
│ 方案B: IVFFlat + PQ乘积量化 (超大规模VLDB)                      │
│ 原理: K-Means聚16384个中心点桶 → 先查最近桶 → 桶内暴力+编码压缩   │
│ 内存: 极小  PQ64×压缩 ×64倍 60GB→1GB 巨大优势                   │
│ Recall@5: 90-95% (调nprobe参数可再提)                            │
│ 适合: 亿级+向量 成本敏感 容忍轻微掉点                            │
├──────────────────────────────────────────────────────────────┤
│ 方案C: Exact 精确线性扫描 (精确)                                 │
│ 延迟: 100-5000ms 极慢 ⚠️小规模1万条以内用                         │
│ 适合: 基准测分/召回率金标准                                      │
└──────────────────────────────────────────────────────────────┘
```

#### HNSW 生产级调优SQL

```sql
-- 第一步: 建表时选 HNSW 索引
CREATE INDEX docs_embedding_idx
ON docs USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,                 -- 邻居数 越大Recall越高内存越大: 12-48
    ef_construction = 128   -- 构建搜索深度: 越大构建越慢索引越准
);

-- 第二步: 在线查询 设置查询搜索深度
SET hnsw.ef_search = 256;   -- 默认40 ⚠️ 改成128-256 在线查询才准!
SELECT content, 1 - (embedding <=> $1) as cosine_sim
FROM docs
WHERE 1 - (embedding <=> $1) > 0.7  -- 相似度阈值过滤
ORDER BY embedding <=> $1
LIMIT 50;
```

> 🔴 面试陷阱90%: 很多教程默认ef_search=40太小了！Recall掉20%，必须查之前SET一下。

---

## 二、RAG质量优化 7件套 (Spring版)

```
RAG准确率 62% → 94%的路径:
┌───────────────────────────────────────────────────────────┐
│1. ✅ Chunk切块优化: 语义切块/SentenceWindow            +8%  │
│2. ✅ 混合检索 BM25+向量 RRF融合                           +5%  │
│3. ✅ Reranker精排 jina-reranker-v2 Top50→Top4            +6%  │
│4. ✅ HyDE查询增强 伪文档检索                                +5%  │
│5. ✅ 元数据先过滤后向量 减少搜索域                           +2%  │
│6. ✅ Lost-in-the-Middle重排 把最相关文档放头尾             +3%  │
│7. ✅ Self-RAG低置信重检索/WebSearch兜底补                  +3%  │
└───────────────────────────────────────────────────────────┘
```

---

## 三、Function Calling 常见坑位5大问题 + 解决

### 3.1 JSON Schema写的烂 调用成功率30%→90%

```java
// ❌ 烂代码: Schema描述为空 不写参数含义
@Tool
public OrderInfo queryOrder(String id) { ... }

// ✅ 黄金写法: 3要素全描述@Tool描述 + @P参数说明 + 类型+示例
@Tool(value = "根据订单号查询订单详情表",
      returnDescription = "OrderInfo对象 含物流状态/金额/创建时间")
public OrderInfo getOrderById(
    @P(value = "订单ID Long类型 示例: 12345678", required = true)
    Long orderId
) {
    return orderMapper.selectById(orderId);
}
```

### 3.2 循环调用/返回结果太长/参数错误三层防护

```java
// 注册重试 2次 + 结果截断 + 参数校验
List<ToolCallback> callbacks = Arrays.asList(
    new RetryToolCallback(2, RateLimitExceededException.class), // 限流重2次
    new ToolResultLengthCallback(2000)  // 工具返回>2K字符就截断摘要
);
ChatClient client = builder.defaultTools(orderTools).toolCallbacks(callbacks).build();
```

---

## 四、性能&成本优化

### 4.1 SSE流式比同步好10倍体验

```java
// ❌ 同步 首字800ms 用户等到死 整段8秒才出完 体验-50分
@GetMapping("/chat")
public String chat(String q) {
    return chatClient.prompt().user(q).call().content();
}

// ✅ SSE流式 首字120ms 用户立刻看到 体感快5-10倍
@GetMapping(value="/chat/stream", produces=MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> chatStream(String q) {
    return chatClient.prompt().user(q).stream()
        .content()
        .map(token -> "data:" + token + "\n\n")
        .concatWithValues("data:[DONE]\n\n");
}
```

### 4.2 成本: GPT-4o-mini做主力 GPT-4o只留复杂场景

```
场景: 日均1000次请求 8K上下文+1K输出
────────────────────────────────
GPT-4o:    $8.4/天  $252/月
GPT-4o-mini:$0.24/天 $7.2/月

💡 实际项目：路由判断90%用mini，10%复杂问题才用4o → 月$27 省×10倍
```

```java
// 三级模型路由: 简单→Mini 中等→GPT4o 复杂→Opus
RouterFunction<ChatClient> router = RouterFunctions.<ChatClient>route()
    .taskComplexity(LOW, miniClient)
    .taskComplexity(MEDIUM, gpt4oClient)
    .taskComplexity(HIGH, opusClient)
    .build();
```

### 4.3 Redis Semantic Cache 节省80%重复Token

```
同样问题用户问了100次？走缓存不调用LLM:
问题 → Embedding向量 → Redis VSS搜索 >0.95相似度 → 命中旧答案返回
实现: Spring Cache + RedisSearch vector similarity
命中率: FAQ/客服场景可达 70-90%
```

---

## 五、生产级稳定性8大保障

| 保障 | 配置 | 指标监控 |
|-----|------|---------|
| 重试 | Spring Retry: 429限流→指数退避 2^x秒×5次 | 重试次数告警 |
| 熔断 | Resilience4j: 错误率>50%熔断→降级Ollama本地 | 熔断状态Grafana |
| 限流 | Bucket4j令牌桶: 100QPS/API Key | 限流拒绝次数 |
| 超时 | 连接10s/首字30s/读60s 超时就断 | p50/p95/p99延迟 |
| 监控 | Micrometer: token消耗/耗时/错误打点到Prometheus | Token成本日曲线 |
| 记忆 | MessageChatMemoryAdvisor: InMemory→Redis分布式 | 会话长度分布 |
| 审计 | LogAdvisor: 全链路日志存ES 敏感数据Mask脱敏 | Tool调用审计 |
| 降级 | 3级降级: GPT4o→Mini→Ollama本地部署兜底 | 降级次数/时长 |

---

## 六、错误排查清单Top 10

| 症状 | 排查项 | 概率 |
|-----|-------|-----|
| RAG回答答非所问 | 1.Chunk切错？2.Template填错context占位符？3.ef_search太小检索不准？ | 70% |
| Function Calling不执行 | 1.@Tool没被Spring扫描到？2.defaultTools()没注册？3.方法名/参数名写错 | 60% |
| 流式响应乱码断行 | produces="text/event-stream"漏？SSE格式data:\n\n错漏 | 50% |
| pgvector查询超级慢>1s | 没建索引？ef_search设太大？topk太高？ | 40% |
| 部署到Linux中文乱码 | 启动参数-Dfile.encoding=UTF-8 / LANG=en_US.UTF-8 | 30% |
| 每次都重新加载Embedding | EmbeddingModel Bean scope错误 没单例 | 25% |
| 高并发下ChatMemory串数据 | InMemoryChatMemory→RedisChatMemory分sessionId隔离 | 20% |
| 生成SQL注入风险 | ChatClient加ContentSafeAdvisor + @Tool里加SQL参数化校验 | 20% |
| 同样问题答案忽对忽错 | temperature设到0.2以下 别0.7+ 增加seed固定随机种子 | 15% |
| 成本超预算$100/天 | 查advisor日志是不是重复调用Tool / Context塞太多无用Chunk | 10% |