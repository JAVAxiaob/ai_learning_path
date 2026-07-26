# vLLM高吞吐推理面试题汇总 (30题)

> 位置: 06-llm/vllm/doc/
> 配套文档: vLLM高吞吐推理引擎.md | vLLM流程图详解.md | vLLM性能优化重难点.md

---

## 一、PagedAttention核心原理（8题）

### Q1. 操作系统虚拟内存页机制 × KV Cache结合 = PagedAttention! 页表/Block/FreeList 三个关键结构设计

### Q2. KV Cache内部碎片问题: 传统Transformers推理为什么显存浪费高达75%？PagedAttention怎么解决

### Q3. Block Size 8/16/32 调参依据? 小模型/长上下文/大模型各选多少

### Q4. PagedAttention 计算时KV在离散物理Block不连续，CUDA Kernel怎么能正确算？指针跳跃访问的Kernel写法思路

### Q5. Block Table页表每请求维护: 实际物理Block号 + 请求逻辑Token位置怎么映射

### Q6. GPU显存利用率: 从传统HuggingFace 20% → vLLM 95% 的核心3设计 预分配池 + 按用分配 + 无碎片回收

### Q7. PagedAttention v1 vs v2 改进点? v2支持前缀共享/分页表/多维度Block

### Q8. RadixAttention (SGLang) 相比PagedAttention的优势：前缀树缓存/多轮对话共享Prefix部分KV

---

## 二、调度与批处理（7题）

### Q9. Continuous Batching 动态批处理 vs 静态Batch 等待一批全完才下一批：为什么vLLM平均等待低10倍

### Q10. Chunked Prefill分块预填充: 8K长Prompt Prefill 200ms阻塞200个Decode，怎么切成4块插空跑把P99从212ms降到62ms

### Q11. Scheduler调度三策略: FCFS公平/优先级VIP插队/LengthAware短任务优先 适用业务场景

### Q12. --max-num-seqs参数: 并发请求数设置多少最合适？A100=512 H100=2048 显存+计算权衡公式

### Q13. 投机采样Speculative Decoding: 小模型草稿猜5Token, 大模型并行验证，速度×2-3的零精度损失加速原理

### Q14. Prefill算力密集型 vs Decode访存密集型: 为什么两个阶段瓶颈完全不同（计算 vs 显存带宽）

### Q15. CUDA Graph优化: 相同形状批量请求为什么开Graph省30%Kernel Launch CPU开销

---

## 三、分布式 & 量化（7题）

### Q16. 张量并行 TP vs 流水线并行 PP 区别？70B为什么选张量并行TP=4 切Linear层不切Layers

### Q17. TP=4 AllReduce AllGather的NVLink带宽要求？PCIe 4.0 x16=32GB/s vs NVLink 4=900GB/s 为什么NVLink必要

### Q18. 量化4选1: FP8 / AWQ-4bit / GPTQ / BitsAndBytes 精度×速度×显存×硬件要求对比表

### Q19. FP8精度原理: H100 FP8 TensorCore 原生支持 1.5-2倍加速 vs BF16 转换时ScalingFactor缩放因子怎么避免溢出

### Q20. AWQ 4bit Activation-aware Weight Quantization 为什么比GPTQ好？激活值分布敏感位保留原理

### Q21. LoRA 多个适配器 同时服务多租户vLLM --enable-lora怎么做到几乎零额外开销切换Adapter

### Q22. KV Cache量化 FP8/INT4: 权重已经量化了, KV Cache再量化4bit又省一半显存的方法

---

## 四、对比 & 生产部署（8题）

### Q23. vLLM vs TGI vs TensorRT-LLM vs SGLang 推理引擎四选 吞吐/延迟/生态/上手难度对比表

### Q24. 生产监控6大必看指标: GPU利用率/排队请求数/P99每Token延迟/吞吐tok/s/显存碎片率/KV Cache命中率

### Q25. K8s HPA自动扩缩容: 根据GPU利用率90%阈值 还是 队列等待数 扩更准?

### Q26. 多模型vLLM单实例? 还是多vLLM实例每GPU一模型? 7B+13B混合部署资源隔离方案

### Q27. Function Calling工具调用支持vLLM ≥0.4.3版本: 自动JSON格式校验和参数重解析怎么接入

### Q28. vLLM降级容错: GPU挂了1张卡 健康检查K8s自动杀Pod重建 怎么避免请求丢失

### Q29. 压测工具: vllm-benchmark / Locust 自定义脚本 模拟真实流量分布(短/中/长请求7:2:1)怎么测真实P99延迟

### Q30. OpenAI兼容API无缝切换: 怎么改1行代码让项目从api.openai.com → vLLM自部署BaseURL 成本从GPT-4o-mini $0.15降到本地$0