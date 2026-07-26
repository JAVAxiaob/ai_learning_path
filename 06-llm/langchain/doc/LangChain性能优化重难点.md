# LangChain 性能优化重难点分析

> 位置: 06-llm/langchain/libs/langchain/langchain/

---

## 一、性能优化策略

### 1. 检索优化

```python
# 使用混合检索
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(docs)
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### 2. 缓存优化

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# 相同查询会被缓存
result1 = chain.invoke({"question": "相同问题"})
result2 = chain.invoke({"question": "相同问题"})  # 命中缓存
```

### 3. 流式输出

```python
# 流式响应，减少等待时间
for chunk in chain.stream({"question": "长问题"}):
    print(chunk, end="", flush=True)
```

### 4. Fallback机制

```python
# 主模型失败自动降级
resilient_chain = chain.with_fallbacks([backup_chain])
result = resilient_chain.invoke({"question": "..."})
```

### 5. 批量处理

```python
# 批量调用，减少API请求次数
questions = ["Q1", "Q2", "Q3"]
results = chain.batch([{"question": q} for q in questions])
```

## 二、常见坑点

### 坑1：RAG检索质量差

**现象**：返回不相关的文档

**原因**：
- 嵌入模型选择不当
- 文档切分不合理
- 检索参数设置不当

**解决方案**：
```python
# 使用更好的嵌入模型
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="bge-large-en-v1.5")

# 优化检索参数
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "score_threshold": 0.7}
)
```

### 坑2：Agent无限循环

**现象**：Agent反复调用同一个工具

**原因**：
- 工具描述不清
- 缺少终止条件
- 推理能力不足

**解决方案**：
```python
# 设置最大迭代次数
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    early_stopping_method="generate"
)
```

### 坑3：Token消耗过大

**现象**：API费用过高

**原因**：
- 上下文窗口过大
- 重复调用
- 未启用缓存

**解决方案**：
```python
# 启用缓存
set_llm_cache(InMemoryCache())

# 限制上下文长度
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("user", "{question}")
]).partial(context="{context[:4000]}")  # 限制context长度
```

### 坑4：异步调用问题

**现象**：异步调用报错

**原因**：
- 部分组件不支持异步
- 线程安全问题

**解决方案**：
```python
# 使用异步支持的组件
result = await chain.ainvoke({"question": "..."})
```

## 三、性能监控

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"question": "..."})
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
```