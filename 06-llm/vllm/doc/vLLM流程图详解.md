# vLLM 推理引擎流程图详解

> 位置: 06-llm/vllm/doc/
> 配套文档: vLLM高吞吐推理引擎.md | vLLM性能优化重难点.md | vLLM面试题汇总.md

---

## 一、vLLM PagedAttention 核心内存管理流程图

```mermaid
flowchart TD
    subgraph GPU显存问题现状 为什么HuggingFace原生慢
        PROBLEM["传统Transformers推理显存浪费<br/>KV Cache每个请求独立大Tensor<br/>BERT-base seq=4096 显存浪费×4 浪费75%!"]
        PROBLEM --> WHY["问题: 内部碎片! 请求长度差异大<br/>ReqA用200 Token剩下200空着浪费<br/>ReqB 4000Token刚好占满"]
    end

    subgraph 解法: PagedAttention = OS 虚拟内存页机制 翻版
        OS["操作系统虚拟内存页思路<br/>物理内存切4KB固定大小Page<br/>进程逻辑地址空间连续<br/>→ 映射到离散物理Page! 零碎片!"]

        OS --> COPY1["✅ 同样思路应用到LLM KV Cache!"]
        COPY1 --> PAGE["把KV Cache 切成固定大小的 KV Block<br/>每个Block存 K,V tokens=16个"]

        PAGE --> TABLE["🔑 每个Request维护 Block Table 页表<br/>req_id → [Block#3, Block#7, Block#1, Block#5...]<br/>逻辑上连续的tokens → 映射到显存中任意离散物理KV Block!"]
    end

    subgraph 显存池 KV Cache Pool
        POOL["GPU显存初始化时 预先切好<br/>N 个空闲 KV Block 放到 FreeList 链表"]
        FREE[("FreeList空闲链表<br/>[1,2,4,5,6,8,9,...]未用")]
        ALLOC[("已分配<br/>Block#3给Req1<br/>Block#7给Req1...")]
        POOL --> FREE
    end

    subgraph 动态分配回收 无碎片
        REQNEW[新请求来了 256 tokens]
        REQNEW --> NEED["256 / 每Block16 = 需要16个Blocks"]
        NEED --> GET["从FreeList取16个空闲Blocks"]
        FREE -- "取16" --> GET
        GET --> ADD["把Block号填到请求的Block Table页表"]
        ADD --> ALLOC

        COMPLETE["推理完的请求 释放KV占用"]
        COMPLETE --> RETURN["把它的Blocks全还回FreeList链表"]
        RETURN --> FREE
    end

    TABLE --> CALC["⭐Attention计算时<br/>按Block Table查找 拼出完整KV<br/>Block之间可以任意顺序/位置 → 照样正确算Attention!"]
```

---

## 二、Continuous Batching 动态批处理 时序图

```mermaid
sequenceDiagram
    participant Client1 as 用户1 请求A 512字输出长
    participant Client2 as 用户2 请求B 128字输出短
    participant Client3 as 用户3 请求C 2048超长输出
    participant SB as vLLM Scheduler调度器
    participant GPU as GPU Worker推理

    Note over GPU: 时间片1 = Step 0 (解码Step=解码1个Token)

    Client1->>SB: 加入请求A 输入Prompt 长=100
    SB->>GPU: Prefill Phase A: 编码A的100个输入Token<br/>Prefill计算密集 一次算完输入KV缓存

    loop Step 1-200 每Step 12ms 并行解所有请求
        GPU->>GPU: Decode Step Batch={A,B} (同时跑!)<br/>Fused MultiQueryAttention 一次性解2个请求的下1Token<br/>vLLM 把不同长度请求都塞进同一次GPU Kernel Launch

        Note over Client2,SB: ⏰ 第Step 10时 用户B请求进来了
        Client2->>SB: 请求B到达
        SB->>SB: ✅ Continuous Batching: 不等当前Batch解完<br/>Step结束就立刻把B拼进下一次Batch!
        SB->>GPU: Step 11: 先Prefill B输入(100Token)<br/>紧接着把B加入Decode Batch

        GPU-->>Client1: A已输出128Tok 实时Stream推送
        GPU-->>Client2: Step12开始 B也开始输出Token 立即响应!

        Note over SB: ⏰ Step 50 请求B已输出128 → B结束! (输出到EOS)
        GPU->>SB: B complete
        SB->>SB: 立刻回收B的KV Blocks到FreeList
        SB->>SB: B腾出来的Batch空位 → 下Step立刻让新请求C挤进来!

        Client3->>SB: (刚到) 请求C
        SB->>GPU: Step 51 Batch={A,C} 2个请求继续并行解码
    end

    GPU-->>Client3: C在Step52就开始有输出了! 不用等A完才处理!
    GPU-->>Client1: A最后Step200完成 总耗时200×12ms=2.4s
    GPU-->>Client3: C最后Step2050完成 但A从Step51就开始被C一起同跑
```

---

## 三、vLLM 推理服务端整体架构

```mermaid
flowchart TD
    subgraph 用户入口层
        API[OpenAI兼容API<br/>/v1/chat/completions]
        UI[WebUI 前端Llama3 Chat]
        SDK[Python SDK / cURL / LangChain 接入]
    end

    subgraph vLLM 核心 APIServer Uvicorn/FastAPI
        API --> HTTP[FastAPI异步服务<br/>uvicorn workers=4]
        SDK --> HTTP
        UI --> HTTP

        HTTP --> EngineCore["LLMEngine 核心引擎 单例"]
    end

    subgraph Scheduler调度调度器核心三策
        EngineCore --> SCHED{"Scheduler调度策略"}
        SCHED --> FCFS["FCFS 先来先服务 默认"]
        SCHED --> PRIORITY["优先级调度 付费用户VIP插队"]
        SCHED --> LO["LengthAware 短请求优先插队 降低Avg等待时间"]
    end

    subgraph GPU Worker 实际跑模型
        SCHED --> WORKER1["GPU Worker GPU:0<br/>Llama3-70B TP=4"]
        SCHED --> WORKER2["GPU Worker GPU:1"]
        SCHED --> WORKER3["GPU Worker GPU:2"]
        SCHED --> WORKER4["GPU Worker GPU:3"]

        subgraph PagedAttention内存池
            WORKER1 --> KMEM["KV Cache Page池 预分配Blocks"]
            WORKER1 --> MODEL["模型权重 FP8/FP16/BF16"]
            WORKER1 --> KERNEL["Fused Custom CUDA Kernels<br/>PagedAttention / RMSNorm / RoPE"]
        end
    end
```

---

## 四、Prefill vs Decode 两阶段流水线

```mermaid
flowchart LR
    subgraph Prefill阶段 输入Prompt
        INPUT["用户输入<br/>'你好，请写一首关于春天的诗' → 40 Tokens"]
        INPUT --> ENCODE["Prefill 编码阶段<br/>一次性输入所有Prompt tokens<br/>Q×K^T 整个矩阵一次算完"]
        ENCODE --> KV["生成 KV Cache<br/>存在 Paged KV Blocks 中<br/>Block Table 记住位置"]
        KV --> PROB["得到第一个输出Token的分布<br/>argmax选出第1个输出字"]
    end

    subgraph Decode阶段 一个一个Token蹦
        PROB --> T1["第1Token输出<br/>'春' 1个Token"]
        T1 --> INSERT["把Token追加输入 KV Block末尾<br/>Block不够了从FreeList取新的"]
        INSERT --> NEXT["Decode计算: 只有1个新Token Q<br/>和所有历史KV(41个Token)算Attention"]
        NEXT --> T2["输出第2Token<br/>'风'"]
        T2 --> LOOP{... 循环 Decode 120次 ...}
        LOOP --> EOS["遇到<｜end｜> 停止Token<br/>一首120字的诗写完 共120次Decode"]
    end
```

---

## 五、张量并行 TP=4 多卡协作时序

```mermaid
sequenceDiagram
    participant SCHED as Scheduler调度层
    participant GPU0 as GPU:0 TP Rank0
    participant GPU1 as GPU:1 TP Rank1
    participant GPU2 as GPU:2 TP Rank2
    participant GPU3 as GPU:3 TP Rank3

    Note over GPU0,GPU3: Linear层按列(输出维度)或行(输入维度)切片

    SCHED->>GPU0: Linear(4096→4096) 输入x [B,1,4096]
    SCHED->>GPU1: 同x
    SCHED->>GPU2: 同x
    SCHED->>GPU3: 同x 广播同一份输入给4张卡

    par 4卡并行算 各算1/4权重列
        GPU0->>GPU0: y0 = x @ W[:,0:1024] → [B,1,1024]
        GPU1->>GPU1: y1 = x @ W[:,1024:2048]
        GPU2->>GPU2: y2 = x @ W[:,2048:3072]
        GPU3->>GPU3: y3 = x @ W[:,3072:4096]
    end

    Note over GPU0,GPU3: NVLink AllGather 卡间拼结果
    GPU0-->GPU1: y0传
    GPU1-->GPU2: y1传
    GPU2-->GPU3: y2传
    GPU3-->GPU0: y3传
    Note over GPU0,GPU3: 每张卡最终拿到完整 y = concat(y0,y1,y2,y3) = [B,1,4096]

    Note over GPU0: AllReduce求和Attention输出
    Note over GPU0,GPU3: 每Decode Step循环 NVLink高速互联≈900GB/s 延迟<10μs
```