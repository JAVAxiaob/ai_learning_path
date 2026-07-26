# LangChain 面试题汇总

> 位置: 06-llm/langchain/libs/langchain/langchain/

---

## 一、基础概念

### Q1: LangChain是什么？核心价值是什么？

**A**: LangChain是一个大模型应用开发框架，核心价值是提供工具链整合能力，让开发者可以快速构建RAG、Agent等复杂应用。

### Q2: LCEL是什么？相比老版Chain有什么优势？

**A**: LangChain Expression Language，管道符语法。优势：
- 简洁直观的|语法
- 统一支持stream/batch/async
- 内置fallback/retry机制
- 更好的可观测性

### Q3: LangChain有哪些核心模块？

**A**: 
- Model: LLM/Chat/Embedding接口
- Retriever: 检索器
- Agent: 智能体
- Memory: 对话记忆
- Tool: 工具调用
- Chain: 链

---

## 二、RAG相关

### Q4: RAG的基本流程是什么？

**A**: 
1. 文档加载→切分→嵌入→存储到向量数据库
2. 用户提问→检索相关文档
3. 构建Prompt（问题+上下文）
4. LLM生成回答

### Q5: RAG有哪些优化策略？

**A**: 
- 混合检索（BM25+向量）
- 重排（CrossEncoder/Reranker）
- 伪文档（HyDE）
- 自适应检索

### Q6: RAG vs Fine-tuning怎么选？

**A**: 
- RAG适合：知识频繁变更、需要溯源、数据量小
- Fine-tuning适合：固定风格、专业领域、需要推理能力

---

## 三、Agent相关

### Q7: Agent的ReAct循环是什么？

**A**: Thought（思考）→ Action（调用工具）→ Observation（观察结果）→ 重复直到完成

### Q8: Agent常见失败模式有哪些？

**A**: 
- 工具调用参数错误
- 无限循环
- 工具选择错误
- 幻觉

### Q9: 如何优化Agent性能？

**A**: 
- 设置最大迭代次数
- 使用Planner子Agent
- 添加输出校验
- Few-shot示例

---

## 四、工程实践

### Q10: 如何实现模型热更新？

**A**: 
- 使用环境变量配置模型
- 动态加载配置
- 支持模型切换

### Q11: 如何处理API限流？

**A**: 
- 使用retry机制
- 添加请求间隔
- 使用fallback模型

### Q12: 如何实现可观测性？

**A**: 
- 使用LangSmith追踪
- 添加自定义回调
- 记录调用日志

---

## 五、性能优化

### Q13: 如何减少Token消耗？

**A**: 
- 启用缓存
- 限制上下文长度
- 使用更高效的Prompt

### Q14: 如何优化检索性能？

**A**: 
- 使用更快的嵌入模型
- 优化向量数据库索引
- 减少检索数量

### Q15: 如何实现异步处理？

**A**: 
- 使用ainvoke/astream
- 异步Retriever
- 异步工具调用

---

## 六、架构设计

### Q16: 生产环境如何部署LangChain应用？

**A**: 
- 使用FastAPI/Flask封装API
- Docker容器化
- Kubernetes编排
- 负载均衡

### Q17: 如何保证系统稳定性？

**A**: 
- 健康检查
- 熔断降级
- 监控告警
- 日志记录

### Q18: 多租户场景如何设计？

**A**: 
- 隔离向量数据库
- 独立API密钥
- 资源配额限制