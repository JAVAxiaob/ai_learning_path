# vLLM 性能优化重难点解析

> 位置: 06-llm/vllm/doc/
> 配套文档: vLLM高吞吐推理引擎.md | vLLM流程图详解.md | vLLM面试题汇总.md

---

## 一、PagedAttention vs 传统KV Cache 显存对比

### 1.1 显存利用率提升 实际项目中的数据

```
场景: Llama3-70B A100 80GB 4卡, BlockSize=16, 并发请求=64

┌────────────────────────────────────────────────────────────────┐
│ 传统HuggingFace Transformers推理:                                 │
│   每个请求预分配 max_seq_len × 2 × H × D = 4K×2×128×128B        │
│   = 128MB/请求 预留, 但实际平均只用 512Tokens=16MB                │
│   64请求 × 128MB = 8.2GB 预分配                                 │
│   实际只用了: 64×16MB = 1GB, 浪费 = 7.2GB = 88% 显存白瞎了 ❌    │
├────────────────────────────────────────────────────────────────┤
│ ✅ vLLM PagedAttention:                                          │
│   KV Cache切成固定16Token Block, 用多少申请多少, 无预分配浪费      │
│   同样64请求平均512Token / Block16 = 32 Blocks/请求               │
│   64×32=2048 Blocks × 512KB/Block = 1GB 实际占用                 │
│   FreeList 剩余可用Blocks = 7GB 可给更多请求!                     │
│   ✅ 利用率: 传统的 1/6 → 到vLLM 95%+ 满载                        │
│   可同时处理请求数: ×5-8倍, 吞吐翻倍!                             │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Block Size 怎么调最优?

| Block Size | 显存利用率 | 计算效率 | 推荐场景 |
|------------|-----------|---------|---------|
| 8 | 最高 碎片少 | 略低 | 小模型7B/13B |
| **16 ⭐默认 | 90%+ | 高平衡 | 通用7B-70B |
| 32 | 略低(碎片多) | 最高 (Kernel更规整) | 长上下文128K |

---

## 二、启动参数黄金配置 (Llama3-70B A100 80GB生产)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/Llama-3-70B-Instruct \
    \
    # ====== 1. 分布式 张量并行 ======
    --tensor-parallel-size 4 \           # 70B至少4×A100, 权重每张卡17.5GB
    \
    # ====== 2. 精度 FP8最优 (Hopper架构H100专用) ======
    --dtype float16 \                     # A100: bfloat16首选
    --quantization fp8 \                  # H100开FP8 = 速度×2 + 显存×2
    \
    # ====== 3. 上下文长度 ======
    --max-model-len 8192 \                # ⚠️ 别开太大128K 吃显存巨多
    --gpu-memory-utilization 0.95 \       # 95%显存都给KV Cache池 (默认90%)
    \
    # ====== 4. 调度参数 (⭐最影响吞吐QPS) ======
    --max-num-seqs 512 \                  # 同时并发处理请求数量 (H100 2048, A100=512)
    --max-num-batched-tokens 32768 \      # 单次Decode Batch总Token数上限
    \
    # ====== 5. 连续批处理 调优 ======
    --scheduler-priority-strategy fcfs \  # 公平队列 付费场景改priority
    --enable-chunked-prefill \            # 长输入拆Chunks 混合Decode跑 防卡顿
    --chunked-prefill-size 2048 \
    \
    # ====== 6. 服务端 ======
    --host 0.0.0.0 --port 8000 \
    --api-key sk-production-key \
    --served-model-name gpt-4o-mini-clone \  # 改个名字冒充OpenAI 接口代码零修改
    --trust-remote-code \
    --disable-log-requests                  # 生产关掉每请求console日志
```

---

## 三、量化选型: AWQ vs GPTQ vs FP8 vs FP16

| 量化方法 | 精度损失 | 速度比FP16 | 显存 | 支持架构 | 场景 |
|---------|---------|-----------|-----|---------|-----|
| FP32 baseline | 0% | 1x | 100% | All | 只用来验证精度 |
| BF16/FP16 | 0-0.3% | 1x基线 | ×0.5 = 70B=140GB | A100+/3090+ | 生产首选最高精度 |
| **FP8 动态量化 ⭐** | 0.3-1% | **2x** | ×0.25 = 70B→70GB | H100 Hopper只有 | H100必开 2倍速度 |
| AWQ 4bit权重量化 | 1-2% | 1.5x | ×0.125=35GB | A100+ | 显存不够塞模型下 |
| GPTQ 4bit | 1-2.5% | 1.3x | ×0.125=35GB | 全架构 | AWQ不行的老模型用 |
| BitsAndBytes 4bit | 2-4% | 0.5x 慢! | ×0.125 | 全 | 别用生产 只适合推理调试 |

> 🏆 **推荐方案**: H100=FP8, A100/4090=BF16优先→不够才AWQ-4bit

---

## 四、Continuous Batching 与 Chunked Prefill 详解

### 4.1 为什么 Chunked Prefill 能解决长尾阻塞问题?

```
问题: 1个长Prompt用户(输入8K字)Prefill 200ms, 阻塞住200个短请求的Decode
    → 所有用户都卡200ms 等这个人Prefill完 → P99延迟飙升200ms

✅ 解法 Chunked Prefill:
长Prefill切成4块 每块2048Tokens, 每块Prefill用50ms
每块Prefill结束后, 中间插空跑一下其他200个Decode请求
时间线变成:
[Prefill Chunk1:50ms] → [Decode ALL 200: 12ms] →
[Prefill Chunk2:50ms] → [Decode ALL 200: 12ms] →
[Prefill Chunk3:50ms] → [Decode ALL 200: 12ms] →
[Prefill Chunk4:50ms] → [Decode ALL 200: 12ms]

结果:
传统方式: 200用户等待总延迟 = 200+12 = 212ms ❌
Chunked:  200用户等待P99 = 50+12 = 62ms ✅ 降3.5倍!
代价: 长Prefill用户总耗时从200ms → 248ms 略涨一点点(值得)
```

---

## 五、vLLM vs TGI vs TensorRT-LLM vs SGLang 推理引擎对比

| 引擎 | 吞吐基准 A100 7B=8K context | 延迟 | 生态 | 上手难度 | 生产推荐 |
|-----|--------------------------|-----|-----|---------|---------|
| HuggingFace原生 | 100 tok/s 基线 | 中 | 100%兼容 | 1星 | ❌ 千万别直接生产 |
| Text Generation Inference (TGI) | 600 tok/s ×6 | 低 | OpenAI兼容 | 2星 | 老项目可维持 |
| **vLLM ⭐** | **2,500 tok/s ×25** | **最低** | 98%OpenAI兼容好 | 2星 | **生产首选第一名🥇** |
| TensorRT-LLM (英伟达亲儿子) | 3,000 tok/s ×30最快 | 中 | 非OpenAI要自开发 | 4星 难 | 极致性能+Triton服务 |
| SGLang 新秀 | 2,800 tok/s ×28 | 最低 | RadixAttention+FunctionCalling强 | 2星 | FunctionCall多选SGLang |

---

## 六、常见错误排查清单Top 10

| 症状/报错 | 排查方案 | 出现频率 |
|----------|---------|---------|
| CUDA OOM 显存不够 | 1. 先降--max-model-len 4K试试<br/>2. 降--gpu-memory-utilization 0.9→0.85<br/>3. 升tensor-parallel-size多卡分 | 90% |
| 比预期慢很多 QPS只有一半 | 1. dtype错float32了→bfloat16/fp16<br/>2. --max-num-seqs默认256→升512/1024<br/>3. disable_log_requests减少打印 | 70% |
| 并发一高就返回503 | 调大--max-num-seqs + 前端加退避重试 | 60% |
| NVLink AllReduce 超慢/带宽低 | 检查NCCL_P2P_DISABLE=0 / nvidia-smi topo看NVLink有没有连上 | 40% |
| 长Prompt输入P99延迟飙升5s+ | 开--enable-chunked-prefill + --chunked-prefill-size=2048 | 35% |
| 相同请求每次输出差异大 | temperature=0 关闭随机 + seed固定 | 25% |
| TP多卡权重加载卡死/超时 | 预下载权重到本地磁盘 --download-dir避免网络慢 | 20% |
| Function Calling 格式乱 | 升级vLLM≥0.4.3 + --enable-auto-tool-call | 20% |
| KV Cache命中率低/浪费严重 | 降--block-size 32→16 碎片更少 | 10% |
| 系统吞吐量抖动忽高忽低 | 关闭--enforce-eager, 保持CUDA Graph优化 | 10% |