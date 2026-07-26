# Spring AI面试题汇总 (40题)

> 位置: 05-java-ai/spring-ai/doc/
> 配套文档: SpringAI-LLM应用集成指南.md | SpringAI流程图详解.md | SpringAI性能优化重难点.md

---

## 一、基础架构题（10题）

### Q1. Spring AI 四大抽象接口: ChatModel/EmbeddingModel/VectorStore/Tool 解耦思想是什么？换LLM要改几行代码？

### Q2. Spring AI vs LangChain4j 选型？Java RAG框架对比

### Q3. ChatClient Bean怎么注入？什么场景用ChatClientBuilder多客户端？

### Q4. System Prompt模板设计规范: role/objective/constraints/examples四段式黄金写法

### Q5. @Tool注解的FunctionCalling原理: Java方法→JSON Schema→LLM返回→反射调用→再送LLM全过程

### Q6. Advisor AOP切面体系: LoggingAdvisor/ChatMemoryAdvisor/ContentSafeAdvisor责任链顺序

### Q7. Prompt Template怎么传参？{context}/{question}占位符示例

### Q8. ChatModel支持哪些LLM? OpenAI/Ollama/智谱/百炼/Minimax接入方法

### Q9. StructuredOutputConverter: LLM输出→Java对象(POJO/Pydantic)自动反序列化怎么用

### Q10. Ollama本地部署Qwen2-7B，Spring AI怎么零代码切换？对比GPT-4o效果/速度/成本

---

## 二、RAG深入题（12题）

### Q11. 7种Chunking切块算法: Token/SentenceWindow/Semantic/ParentChild 准确率排序

### Q12. Pgvector三大索引: HNSW vs IVFFlat vs Exact 选型？500万向量场景为什么选HNSW？

### Q13. ef_construction vs ef_search 参数调优？为什么90%教程忽略设置导致召回率掉20%

### Q14. 混合检索实现: 向量相似度 + BM25关键词 + RRF融合 分数加权公式

### Q15. Reranker精排50→Top4原理: ColBERT vs Jina-Reranker vs BGE-Reranker

### Q16. HyDE查询优化: 什么用假答案去搜比真问题搜更好？什么场景不能用？

### Q17. Lost-in-the-Middle现象: LLM注意力U型怎么缓解？LongContextReorder后置处理器

### Q18. Metadata先过滤 vs 后过滤顺序: 100万条数据先过滤部门到2万条 再算向量快多少

### Q19. 向量维度: 768/1024/1536/3072 选高维度还是低维度？准确率+存储+延迟权衡

### Q20. 多租户Multi-Tenant RAG权限控制: 每个用户只能查自己部门文档，怎么正确实现？

### Q21. 增量索引更新: 新文档上传怎么加到Pgvector？全量重建 vs 增量插入

### Q22. 距离函数: Cosine余弦 vs L2欧氏 vs 内积(Inner Product)选型对比

---

## 三、Tool Calling & 工程实践（8题）

### Q23. @Tool注解的方法要写哪些三要素描述才能让LLM调用成功率从30%升到90%？

### Q24. Tool Calling死循环: 同一工具同参数连续调用3次，5层防护网代码写

### Q25. Tool返回结果太长(>2万Token): 结果截断摘要 vs LLM压缩 vs 上下文丢失 权衡

### Q26. 自定义Advisor怎么实现: TokenBudgetAdvisor超过8K Token自动截断旧对话

### Q27. ChatMemory状态隔离: 高并发下UserA的会话历史跑到UserB场景，RedisChatMemory分布式怎么实现

### Q28. SSE流式 vs WebSocket: 首字120ms vs 同步8秒体验差距，生产实现注意

### Q29. 结构化输出: LLM输出JSON转POJO，字段校验失败Retry2次逻辑怎么写

### Q30. 多模态: 图片+文字输入Spring AI实现？图片URL→Base64→GPT-4o多模态识别

---

## 四、生产部署&性能（10题）

### Q31. 生产Spring AI系统稳定性8保障: 重试/熔断/限流/超时/监控/记忆/审计/降级

### Q32. 成本控制: GPT-4o $8.4 vs GPT-4o-mini $0.24 天成本差35倍，Router分级路由怎么写

### Q33. Redis Semantic Cache语义缓存: 相似度>0.95命中，FAQ场景节省多少Token

### Q34. Prometheus监控指标6个: Token总量/LLM延迟/Tool成功率/重试次数/熔断状态/缓存命中率

### Q35. 模型三级降级Fallback: GPT-4o限流→4o-mini→本地Ollama降级策略代码

### Q36. PII数据脱敏: 用户手机号/身份证/订单号要传给LLM前的内容安全过滤怎么实现

### Q37. 批处理向量化入库: 10万PDF文档，并行ThreadPool+批量Embedding加速技巧

### Q38. 微服务拆分: AI服务 vs 业务服务 边界怎么画？Feign调用AI服务隔离故障

### Q39. K8s部署HPA扩容依据: GPU利用率 vs 请求并发数 vs Token生成速率 扩缩容指标

### Q40. 可观测性全链路追踪: MDC/TraceId串起 Spring MVC → AI调用 → LLM API → VectorDB查询