# vLLM 大模型推理引擎解析

> 位置: 08-mlops/vllm/
> 简历推荐: 5星 | 岗位: LLM推理优化/大模型平台/MLOps工程师

---

## 一、核心技术: PagedAttention

```mermaid
graph TD
    subgraph 传统KV缓存: 不同请求seq不同 连续分配→大量Padding碎片
        T[传统问题] --> T1[GPU显存碎片率50%+]
        T --> T2[显存利用率<40%]
    end
    subgraph vLLM PagedAttention = 操作系统虚拟内存搬到GPU!
        V1[把每个请求KV Cache切固定Block=16token]
        V1 --> V2[Block不要求物理连续! 用BlockTable逻辑映射]
        V2 --> V3[缺页分配 用完归还内存池]
        V3 --> V4[显存利用率飙升至 90%~95%! + 连续批处理CB]
        V4 --> R[同一张A100 吞吐2-10倍!]
    end
```

> PagedAttention 是vLLM作者的神作，现在所有主流推理框架(TensorRT-LLM/SGLang/llamafile)都抄了这个设计。

## 二、生产部署一键命令

```bash
# 1. 单卡部署 Llama3.1-8B (OpenAI 100%兼容API, LangChain改base_url直接用!)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --served-model-name llama3-8b \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --host 0.0.0.0 --port 8000

# curl http://localhost:8000/v1/chat/completions  body完全和OpenAI格式一致

# 2. 双卡张量并行TP=2跑 70B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2

# 3. AWQ INT4量化部署 (省显存 更大KV Cache!)
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3-8b-instruct-awq \
    --quantization awq \
    --max-model-len 32768
```

## 三、性能优化7招

| 手段 | 原理 | 提升 |
|-----|-----|-----|
| **AWQ/GPTQ INT4量化** | 权重4bit存 | 显存÷2.2, 速度×1.5 |
| **Chunked Prefill** | 长prefill切成小块插入decode间隙 | 延迟抖动↓80% |
| **Continuous Batching** | 迭代级别调度,一请求完立刻补新请求 | 吞吐×2~5 |
| **Speculative Decoding** | 小模型草稿(快)→大模型验证(准) | 解码×2~3加速 |
| **Prefix Caching** | 共享系统Prompt缓存KV | 多轮对话复用×4 |
| **FP8 KV Cache** | KV从FP16→FP8精度几乎不降 | 显存÷2, 上下文加倍 |
| **LoRA Adapters热切换** | 多租户场景一份权重+N个LoRA | 多任务显存↓N倍 |

## 四、简历黄金句式

| 写法 |
|-----|
| 「vLLM+PagedAttention部署Llama3.1-70B：双卡A100 TP=2 + AWQ 4bit量化+Chunked Prefill，支持32K上下文，吞吐64→218 req/s (3.4×)，P99延迟2.1s，月GPU成本↓38%」 |
| 「大模型推理服务平台：vLLM+K8s HPA自动扩缩+GPU节点池，SLA 99.95%，日均调用量4200万次，GPU平均利用率72%，QPS峰值5600」 |
| 「推理成本优化：FP8 KV+前缀缓存+投机解码(Llama 8B草稿+70B验证)，平均每1K Token成本从$0.012→$0.0043 (↓64%)」 |

## 五、面试题

**Q Continuous Batching (Orchestration级别批处理) vs 静态Batching区别？**
> A: 静态Batching: 攒够N个请求才一起跑(像训练)，慢请求拖垮整个批次，GPU等空闲。连续Batching: 每个迭代(tick)级别调度，任何请求生成完立刻插入新请求填充GPU，不用等慢请求。短请求不会被长请求卡住。GPU利用率从40%→90%。

**Q 投机解码(Speculative Decoding)原理？**
> A: 两步：① 草稿小模型(Draft Model, 快但不准) 一次猜γ个连续Token ② 目标大模型(Target Model, 慢但准) 并行一次验证这γ个Token。全部猜对就全接收，错了就截断到猜对的地方。平均每次前进>1个Token(不用每步都跑大模型)，吞吐×2~3，输出分布数学上严格等价大模型直接生成。

**Q KV Cache 显存怎么估算？给公式？**
> A: KV Cache显存 = 2 × n_layers × n_tokens × d_model × bytes_per_value。举例: Llama3-8B, 32层, d=4096, 1个用户16K上下文, FP16(2字节): 2 × 32 × 16384 × 4096 × 2字节 = 8.59 GB。8个并行用户=68GB，INT4量化权重+KV用FP8 → 32GB单卡可塞!