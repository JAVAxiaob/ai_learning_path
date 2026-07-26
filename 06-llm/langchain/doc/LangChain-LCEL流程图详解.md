# LangChain LCEL 流程图详解

> 位置: 06-llm/langchain/libs/langchain/langchain/

---

## 一、LCEL架构全景图

```mermaid
graph TD
    A[用户输入] --> B[LCEL Pipeline]
    B --> C[PromptTemplate]
    C --> D[LLM/ChatModel]
    D --> E[OutputParser]
    E --> F[输出结果]
    
    B --> G[Retriever]
    G --> H[VectorStore]
    H --> I[Document]
    G --> C
    
    B --> J[Tool]
    J --> K[外部API]
    D --> J
    J --> D
    
    B --> L[Memory]
    L --> C
```

## 二、LCEL管道符流程

```mermaid
flowchart LR
    A[input] --> B["RunnablePassthrough()"]
    B --> C["PromptTemplate"]
    C --> D["ChatOpenAI()"]
    D --> E["StrOutputParser()"]
    E --> F[output]
    
    style A fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

```python
# LCEL基础示例
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    ("user", "{question}")
])

chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# 执行
result = chain.invoke({"question": "Hello!"})
```

## 三、RAG标准流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant Chain as RAG Chain
    participant Retriever as Retriever
    participant VectorDB as VectorStore
    participant LLM as ChatOpenAI
    
    User->>Chain: {"question": "LangChain是什么?"}
    Chain->>Retriever: 检索相关文档
    Retriever->>VectorDB: similarity_search(question)
    VectorDB-->>Retriever: 返回文档列表
    Retriever-->>Chain: 格式化文档
    Chain->>Chain: 构建Prompt
    Chain->>LLM: prompt + context
    LLM-->>Chain: 回答
    Chain-->>User: 最终答案
```

```python
# RAG LCEL实现
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)
```

## 四、Agent ReAct循环

```mermaid
graph TD
    A[开始] --> B[思考: 分析问题]
    B --> C{需要工具?}
    C -->|是| D[调用工具]
    D --> E[获取观察结果]
    E --> B
    C -->|否| F[输出最终答案]
    F --> G[结束]
```

```python
# Agent实现
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return "搜索结果..."

agent = create_openai_tools_agent(ChatOpenAI(), [search], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search])

result = agent_executor.invoke({"input": "今天天气怎么样?"})
```

## 五、多模态流程

```mermaid
graph TD
    A[图片输入] --> B[图像理解]
    B --> C[生成描述]
    C --> D[文本问答]
    D --> E[输出回答]
    
    F[文本输入] --> D
```

```python
# 多模态示例
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "描述这张图片:"},
        {"type": "image_url", "image_url": "https://example.com/image.jpg"}
    ]
)

result = ChatOpenAI(model="gpt-4o").invoke([message])
```