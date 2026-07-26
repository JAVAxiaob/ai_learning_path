# vLLM大模型高吞吐推理面试题汇总（上篇）- PagedAttention核心与调度（15题 附详细标准答案）

---

## 一、PagedAttention核心原理（Q1-Q8）

---

### Q1. 操作系统虚拟内存页机制 × KV Cache结合 = PagedAttention！页表/Block/FreeList三个关键结构设计

**⭐ 标准定义**

PagedAttention（2023年vLLM团队提出，出自论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》）= **直接照搬操作系统虚拟内存管理的设计思想**来管理LLM推理的KV Cache内存。

**🔥 核心思想类比表（面试用OS概念类比，秒懂）：**

| 操作系统概念 | PagedAttention对应概念 | 作用 |
|---|---|---|
| **虚拟页（Virtual Page）** | **逻辑Block（Logical Block）** | 每个请求按Token序列划分的逻辑块，比如每16个Token=1个逻辑块 |
| **物理页（Physical Page）** | **物理Block（Physical Block）** | GPU显存中预先切好的固定大小内存块，比如16 Token×KV头数×头维度 = 一块固定显存 |
| **页表（Page Table）** | **Block Table（块表）** | 每个请求1张表：`逻辑Block号 → 物理Block号` 的映射 |
| **缺页中断（Page Fault）** | **Block分配请求** | 逻辑Block没映射物理块时，从FreeList申请新的物理Block |
| **空闲链表（Free List）** | **KV Cache Free List** | 系统所有空闲物理Block的链表，O(1)时间分配/回收 |
| **内存共享（Shared Pages）** | **前缀共享（Prefix Sharing）** | 相同系统Prompt/Same前缀的请求 → 映射同一物理Block，零拷贝共享KV |
| **写时复制（COW）** | **Copy-on-Write** | 共享Block某请求要写新Token → 才分配新Block拷贝再写 |

**📐 核心数据结构代码思路（面试手写伪代码）：**

```cpp
// ============ 1. 物理Block：GPU显存上一块固定大小KV内存 ============
struct PhysicalBlock {
    int block_id;           // 全局唯一物理块号
    int ref_count;          // 引用计数（共享前缀时>1，COW判断）
    void* kv_ptr_gpu;       // 指向实际GPU显存（key+value各16token内存）
};

// ============ 2. Block Table 块表（每个请求1张，类比页表） ============
struct RequestState {
    int req_id;
    vector<int> block_table; // ⭐ 核心！size=逻辑块数
                             // block_table[逻辑块号i] = 物理块号（PhysicalBlock ID）
    int prompt_len;          // prompt token数
    int generated_len;       // 已生成token数
    int total_len() const { return prompt_len + generated_len; }
};

// ============ 3. KVCacheManager 全局管理器 ============
class KVCacheManager {
    vector<PhysicalBlock> all_blocks;     // 所有物理块（启动时一次性预分配）
    list<int> free_list;                  // ⭐ 空闲块链表（存block_id）
    mutex mtx;

public:
    KVCacheManager(int num_total_blocks, int block_size, int num_kv_heads, int head_dim) {
        // 启动时一次性cudaMalloc所有显存，避免推理时动态malloc（有锁慢）
        all_blocks.resize(num_total_blocks);
        size_t bytes_per_block = 2 * block_size * num_kv_heads * head_dim * sizeof(half);
        // 2表示 key + value
        for (int i = 0; i < num_total_blocks; ++i) {
            cudaMalloc(&all_blocks[i].kv_ptr_gpu, bytes_per_block);
            all_blocks[i].block_id = i;
            all_blocks[i].ref_count = 0;
            free_list.push_back(i);  // 初始所有块在空闲链表里
        }
    }

    // 给请求分配1个新的逻辑块 → 映射到1个空闲物理块
    int allocate_block(RequestState& req) {
        lock_guard<mutex> lock(mtx);
        assert(!free_list.empty());  // OOM前会触发换入换出/拒绝新请求
        int phys_id = free_list.front(); free_list.pop_front();
        all_blocks[phys_id].ref_count = 1;
        req.block_table.push_back(phys_id);  // ⭐页表写一条映射
        return phys_id;
    }

    // 释放请求某逻辑块 → 物理块引用计数减1，归0就回FreeList
    void free_block(RequestState& req, int logical_idx) {
        lock_guard<mutex> lock(mtx);
        int phys_id = req.block_table[logical_idx];
        if (--all_blocks[phys_id].ref_count == 0) {
            free_list.push_back(phys_id);  // ⭐ 块可回收复用，无需cudaFree
        }
    }

    // ========== 前缀共享（加分点）：新请求prefix和已完成请求相同 → 直接映射同一块 ==========
    void share_prefix(RequestState& new_req, RequestState& existed_req, int num_shared_logical_blocks) {
        lock_guard<mutex> lock(mtx);
        for (int i = 0; i < num_shared_logical_blocks; ++i) {
            int phys_id = existed_req.block_table[i];
            new_req.block_table.push_back(phys_id);
            all_blocks[phys_id].ref_count++;  // ⭐ 引用计数+1，不拷贝显存！
        }
    }
};
```

---

### Q2. KV Cache内部碎片问题：传统Transformers推理为什么显存浪费高达75%？PagedAttention怎么解决

**⭐ 标准定义**

传统HuggingFace TGI/原生推理：每个请求分配一个**最大长度的连续KV Tensor**（比如预分配max_seq_len=8192的连续显存），但用户请求实际长度差异极大 → 产生巨大的**内部碎片（Internal Fragmentation）**。

**📐 传统方式 vs vLLM碎片率对比（数学计算，面试可以算）：**

```
场景假设：max_seq_len=8192，线上真实请求分布（按业界线上统计）：
  - 70% 请求是短对话：prompt+生成长度≈512 token（6.25% max）
  - 20% 请求是中等长度：prompt+生成≈2048 token（25% max）
  - 10% 请求是长文档问答：prompt+生成≈8192 token（100% max）

传统连续内存分配（每个请求固定分配8192 token KV）：
  70%请求：分配8192，只用512 → 碎片率 = (8192-512)/8192 = 93.75%！💥
  20%请求：分配8192，用2048 → 碎片率 = 75%
  10%请求：分配8192，用8192 → 碎片率=0
  加权平均碎片率 = 70%×93.75% + 20%×75% = 80.625% ≈ 80%！
  → 结论：80%显存被浪费，只能跑1/5的并发量 ❌

vLLM PagedAttention按Block分配：
  Block大小=16 Token，用完多少块就分配多少块
  短请求512 Token = 32块 → 只分配32块，1块碎片（最后一块可能没用满）
  碎片率 = 1/33 ≈ 3% ！✅
  中等请求2048 Token = 128块 → 碎片率≈1/128<1%
  加权平均碎片率 ≈ 3% × 70% + <1% × 30% ≈ 2~4%
  → 显存利用率从传统20% → 95%+ ⭐ 并发量×5！
```

**💡 除了内部碎片，传统方式还有两大浪费：**

| 浪费类型 | 传统HuggingFace | vLLM PagedAttention |
|---|---|---|
| **内部碎片**（分配了没用的部分） | ⭐ **80%+**（上面计算） | **<4%**（按块按需分配） |
| **外部碎片**（释放后空穴太小无法复用） | 严重：短请求结束后留下8192空洞，新长请求插不进去 → cudaMalloc连续失败 | ❌ 无外部碎片！物理块离散，页表映射，任何空洞都是1个完整Block |
| **前缀冗余拷贝**（多用户相同系统prompt） | 每个请求拷贝一份系统prompt的KV，100用户=100份重复KV | ✅ **前缀共享**：COW+引用计数，100用户共享1份，零额外开销 |

**📊 论文实测结果（面试直接说数据加分）：**
> 相同A100 GPU，Llama-7B模型，相同延迟SLA下：
> - HuggingFace baseline：并发 max-num-seqs = 64，吞吐 785 token/s
> - **vLLM PagedAttention：并发 max-num-seqs = 512，吞吐 5120 token/s**
> → **6.5倍吞吐量提升，完全来自显存利用率提升！**

---

### Q3. Block Size 8/16/32 调参依据？小模型/长上下文/大模型各选多少

**⭐ 标准定义**

Block Size（vLLM启动参数`--block-size`，默认值=16）= 每个物理Block对应多少个Token位置，直接影响：①碎片率（越大碎片越多）②页表大小（越小页表越大）③CUDA Kernel访存效率（越大连续访问越多带宽越高）。

**📊 Block Size三因素权衡表：**

| 权衡维度 | Block Size 太小（如4） | Block Size 太大（如128） |
|---|---|---|
| 碎片率 | ✅ 极低 <1%（1个Token浪费=4/1浪费25%） | ❌ 很高 60%+（最后一块没用完浪费127个） |
| 页表/元数据开销 | ❌ 巨大：8K序列要2000条目，Block Table占显存+CPU内存 | ✅ 很小：8K只要64条目 |
| CUDA Kernel带宽 | ❌ 低：小块太多，指针跳跃访存次数多，DRAM带宽跑不满 | ✅ 高：128连续Token访存合并，DRAM带宽95%+ |
| 前缀共享粒度 | ✅ 极细：4个Token以上相同就能共享 | ❌ 很粗：要128个Token对齐相同才共享，实际很少 |

**✅ 黄金调参表（不同模型/场景推荐值，面试背这表）：**

| 场景/模型 | 硬件 | 推荐Block Size | 理由 |
|---|---|---|---|
| **通用默认值** | 任意 | **16 ⭐⭐⭐** | 三因素平衡点，官方默认值经过大量实测，90%场景最优 |
| 小模型（<3B）/短对话为主（<1K） | A10G/T4 | 8 | 碎片更敏感，短请求多，8块碎片率<6%，长请求少Kernel影响小 |
| **大模型（7B~70B）/通用场景** | A100/H100 | **16 ⭐** | 论文Ablation实验16最优，实际生产用16几乎不用调 |
| 超长上下文为主（32K~128K RAG） | A100 80G/H100 | **32** | 长序列Kernel带宽优先，大Block带宽高10-20%；碎片率也低（最后1块浪费32/2048≈1.5%） |
| 多轮对话/大量相同系统Prompt | 任意 | 8或16 | 前缀共享优先，小粒度共享更充分 |
| FP8/INT4 KV Cache量化 | H100 | 16 | 量化后每Block显存更小，16带宽与碎片均衡 |

**💡 面试加分点：** Block Size不是参数乱调，vLLM源码里有Ablation实验（Llama-13B A100）：
- bs=4：12.2ms/it
- bs=8：11.8ms/it
- bs=16：**11.2ms/it ⭐ 最快**
- bs=32：11.6ms/it
→ U型曲线，16是甜点。

---

### Q4. PagedAttention计算时KV在离散物理Block不连续，CUDA Kernel怎么正确算？指针跳跃访问的Kernel写法思路

**⭐ 标准定义**

普通Attention Kernel：`K_cache[batch, seq, head, head_dim]` 是连续4维Tensor → pointer arithmetic直接寻址，SIMT线程读连续显存块。

PagedAttention Kernel特殊的地方：**KV Cache不是连续Tensor，而是由Block Table查出来的离散物理Block指针**。

**📐 普通Attention vs PagedAttention 寻址方式对比：**

```cpp
// ========== 普通连续KV Attention（HuggingFace朴素版） ==========
__global__ void vanilla_attention_kernel(
    half* K, half* V, half* Q, half* O,
    int seq_len, int num_heads, int head_dim)
{
    // 计算当前线程负责：query token = q_idx, head = h, d = 头内维度
    int q_idx = blockIdx.x;
    int h     = blockIdx.y;
    int d     = threadIdx.x;

    float sum = 0.0f;
    for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
        // 连续KV → 直接指针偏移：
        int offset = (k_idx * num_heads + h) * head_dim + d;
        half K_val = K[offset];  // ✅ 连续，直接数组访问
        sum += (float)Q[(q_idx * num_heads + h) * head_dim + d] * (float)K_val;
    }
    // ... softmax + 乘V
}

// ========== PagedAttention CUDA Kernel 核心思路 ==========
__global__ void paged_attention_kernel(
    // ⭐ 第一个区别：传进来的不是K/V大Tensor！是物理Block指针数组 + Block Table
    void**        block_ptrs,      // [num_phys_blocks] 每个物理Block的GPU显存首地址
    const int*    block_tables,    // [max_num_seqs, max_logical_blocks_per_seq] 页表
    const int*    seq_lens,        // [max_num_seqs] 每个请求的当前长度
    const half*   Q,               // [num_seqs, num_heads, head_dim] 当前step只算1个Q（Decode阶段）
    half*         O,
    int block_size, int num_heads, int head_dim, int num_seqs_in_batch)
{
    int seq_id  = blockIdx.x;       // 哪个请求
    int h       = blockIdx.y;       // 哪个Attention头
    int kv_head = h / (num_heads/num_kv_heads); // GQA情况：多Q头共享一组KV头
    int d       = threadIdx.x;

    int cur_seq_len = seq_lens[seq_id];

    // ========== 核心2步：先查Block Table再取KV ==========
    float sum = 0.0f;
    for (int k_token = 0; k_token < cur_seq_len; k_token++) {
        // Step 1：Token在哪个逻辑块？逻辑块号 + 块内偏移
        int logical_blk_id = k_token / block_size;   // 比如123号token → 逻辑块7(block16: 7*16=112)
        int blk_offset    = k_token % block_size;    // 块内123-112=11号位置

        // Step 2：⭐⭐⭐ 查 Block Table（页表）→ 得到物理Block号
        int phys_blk_id = block_tables[seq_id * MAX_LOGICAL_BLOCKS + logical_blk_id];

        // Step 3：从物理Block指针数组拿此块的显存首地址
        half* phys_block_kv = (half*)block_ptrs[phys_blk_id];
        // ⭐ Block内部是连续的！格式= [block_size, 2, kv_head, head_dim]
        int k_within_block = (blk_offset * 2 * num_kv_heads + kv_head) * head_dim + d;
        half K_val = phys_block_kv[k_within_block];  // ✅ 一次查表+偏移，拿到KV值！

        sum += (float)Q[(seq_id * num_heads + h) * head_dim + d] * (float)K_val;
    }
    // ... 和普通Attention一样 softmax(qk) × V，取V时也是同样查表逻辑
}
```

**💡 面试高频追问：这么多指针跳跃查表不会慢吗？**
→ 有两个优化让它几乎和连续KV一样快：
1. **Block内部是连续的**：16个Token连续排列 → 1个block内部=16次连续读，warp内32线程可以Memory Coalescing合并访存，1次DRAM burst读16×16=256字节
2. **Block Table SRAM缓存**：block_tables 小表先读入Shared Memory / L1 Cache，查表就是SRAM访问≈1ns，和算术指令同级别延迟

实测：PagedAttention Kernel 比 朴素HuggingFace Attention 只慢<3%，但靠5倍并发量总吞吐×5+。

---

### Q5. Block Table页表每请求维护：实际物理Block号 + 请求逻辑Token位置怎么映射

**⭐ 图文示意（请求生成过程中Block Table变化过程）：**

```
配置：block_size=16, 请求prompt=53个Token

Step0: 请求刚进来，没分配Block
  block_table = []

Step1: 预分配prompt=53Token需要多少逻辑块？
  ceil(53 / 16) = 4个逻辑块（4×16=64 slots）
  从FreeList申请4个空闲物理块 → 假设拿到 phys 12, 45, 7, 31
  block_table = [12, 45, 7, 31]  ← 下标=逻辑块号，值=物理块号
  Token 0-15   → phys12的offset 0~15
  Token 16-31  → phys45的offset 0~15
  Token 32-47  → phys7的offset 0~15
  Token 48-63  → phys31的offset 0~15（53只用了offset0~4，5~15空闲）

Step2: Prefill阶段 → 计算K/V → 按映射写入各自物理Block ✅

Step3: Decode阶段开始生成新Token，每生成1个Token新KV写哪儿？
  generated token #54 → ceil(54/16) = ceil(3.375) = 4号逻辑块（已分配）
    → 写 phys31, blk_offset = 54%16 = 6 ✅ 还在已有的第3块（块0-indexed）
  generated token #63 → 刚好写满phys31的最后一个slot(blk_offset=15)
  generated token #64 → ceil(64/16)=5号逻辑块 现在没分配！
    → FreeList再取1个空闲块 phys=22
    → block_table 变成 [12, 45, 7, 31, 22] ✅ 动态追加
    → Token 64~79 现在映射到phys22啦
```

**✅ 数学公式（面试要会写）：**

```python
# 已知：第 i 个Token（0-indexed），block_size=16, block_table是请求的页表
# 求：物理块号 + 块内偏移 → 就能定位这个Token的KV在GPU哪块显存

logical_block_idx = i // block_size   # 整除 → 第几号逻辑块
blk_offset        = i %  block_size   # 取余 → 块内第几个slot

# 查Block Table：逻辑块号 → 物理块号
phys_block_id    = block_table[logical_block_idx]  

# 通过phys_block_id → 从全局block_ptrs[phys_block_id]拿到此物理块GPU显存首地址
# 再加上 blk_offset × [2 × num_kv_heads × head_dim × sizeof(half)]
# = 此 i 号 Token的K/V具体地址，KV就可以读写了。
```

**💡 共享前缀场景映射特例（加分点）：**
请求A和B都是"你是XX公司的AI助手，请..."相同128字系统prompt →
请求A prompt=512 tokens, 请求B进来，先做prefix匹配，发现前128 tokens相同 = 8个逻辑块(16×8) →
请求B的block_table[0..7] = 请求A的block_table[0..7]，对应8个物理块的ref_count各+1 →
**零拷贝共享**，B不需要重新Prefill这128 tokens的KV，Prefill时间省掉25%！✅

---

### Q6. GPU显存利用率：从传统HuggingFace 20% → vLLM 95%的核心3设计 预分配池 + 按用分配 + 无碎片回收

**📊 三大设计分别解决显存浪费的哪个维度：**

| 设计 | 对应传统浪费 | 机制 | 单独带来利用率提升 |
|---|---|---|---|
| **1. 启动时显存预分配池（Block Pool）** | 传统推理时动态cudaMalloc/cudaFree有锁+有最小粒度，很多显存Driver层就没释放导致OOM | 启动时根据公式一次性cudaMalloc出所有KV Block，放进FreeList，推理全程零malloc/free！ | +20%（减少CUDA驱动层内存碎片） |
| **2. 按需按Block分配（PagedAttention）** | 传统按max_seq_len预分配→内部碎片80% | 请求有多少Token就分配多少块，不预支 | +40%（内部碎片从80%降到<4%）|
| **3. 引用计数+FreeList无碎片回收** | 传统请求结束释放连续大块后，空穴太小无法给新请求用（外部碎片） | Block固定大小（16T），任何Block回FreeList都可以给任何请求的任何逻辑块用，完全没外部碎片 | +15%（消除外部碎片） |
| **合计** | | | **20% → 95% ⭐⭐⭐** |

**📐 显存预分配公式（vLLM源码默认计算，面试要会说）：**

```
vLLM启动时：
  Step1: 先测1次空显存：torch.cuda.mem_get_info() → 比如A100 80GB剩75GB可分配
  Step2: 模型权重（含优化器？推理无！）：
         Llama-7B FP16 = 13GB
         Llama-13B FP16 = 26GB
         Llama-70B FP16 TP4 = 每卡32.5GB
  Step3: 激活/工作区保留 = 固定比例 5-10%（~4GB/80GB卡）
  Step4: 剩下的全给KV Cache池 → 75-13-4 = 58GB
  Step5: 每个Block(16token, Llama-7B, 32kv_heads, head_dim=128, half)字节数：
         2(K+V) × 16 × 32 × 128 × 2B = 262,144 B = 256KB / block
  Step6: 总物理Block数 = 58GB / 256KB ≈ 226,560 Blocks
         总可服务Token数 = 226,560 × 16Token = 3,624,960 Tokens ⭐
         → 并发512请求 × 平均7K tokens = 3.5M 刚好合适
```

---

### Q7. PagedAttention v1 vs v2 改进点？v2支持前缀共享/分页表/多维度Block

**📊 PagedAttention v1（2023初版） vs v2（2023年末vLLM 0.2+）对比表：**

| 维度 | v1（早期vLLM 0.1.x） | v2（vLLM 0.2.x+ 默认） |
|---|---|---|
| **Block Table存储** | CPU内存存储，每次Kernel启动前整块拷贝到GPU | ✅ GPU显存常驻Block Table，无需每次H2D拷贝 |
| 每次调度拷贝开销 | 大：`num_seqs × avg_blocks × 4B` 每次都要H2D | ❌ 无开销：GPU直读 |
| 大批量并发延迟 | 高：CPU→GPU拷贝占总Decode时间10-20% | ✅ 延迟-15%，吞吐+20% |
| **前缀共享(Prefix Sharing)** | ❌ 不支持，Block是每请求独有的 | ✅ 支持，PhysicalBlock加ref_count做COW共享 |
| 多轮对话长Prefix重复计算率 | 100%（每会话重算所有KV） | **最多-80%**（历史system prompt/历史对话KV直接共享引用）|
| **Block Table分页** | 大请求8K序列要512条目=2KB连续表 | ✅ 页表也分页，2级页表，大请求页表不占连续GPU显存 |
| **滑动窗口Attention** | ❌ 不支持，Token超长只能丢 | ✅ KV超过窗口大小时，头Block自动detach ref_count→0回FreeList，尾部分配新块，滑动O(1) |
| **FP8/INT4 KV量化** | ❌ Block格式只支持FP16/BF16 | ✅ Block格式支持FP8 per-tensor/INT4 AWQ量化 |

**💡 面试加分点：** v2性能数据（vLLM官方Benchmark Llama-7B A100 80G）：
- v1：并发512，吞吐4950 tok/s
- v2：同设置，吞吐**5980 tok/s** → **+21%**，主要来自Block Table零拷贝和前缀共享命中节省Prefill。

---

### Q8. RadixAttention (SGLang) 相比PagedAttention的优势：前缀树缓存/多轮对话共享Prefix部分KV

**⭐ 标准定义**

RadixAttention是SGLang（2024年伯克利新LLM推理框架）提出的KV Cache管理，是PagedAttention+**Radix Tree基数树**的结合，核心改进：vLLM前缀共享需要请求ID手工匹配前缀，RadixAttention把所有历史KV Cache用基数树（Trie前缀树）组织起来，任何新请求进来自动最长前缀匹配，能共享的KV自动命中引用，不用显式传prompt hash。

**📐 Radix Tree（基数树）KV索引结构示意：**

```
配置：树节点 = 1个逻辑Block(16Token KV)

根 [Root, phys_block=null]
├─ "你是一个" [Block 17, ref=3]  ← 100个会话都有"你是一个XX助手"
│    ├─ "智能客" [Block 22, ref=1] (客服)
│    │    └─ 服助手，请" [Block 40, ref=1] → 后面是客服FAQ知识，共享命中2个Block=32Token
│    └─ "编程助" [Block 28, ref=1] (程序员助手)
│         └─ 手，帮我写" [Block 55, ref=1]
└─ "请把以下" [Block 19, ref=2] (翻译请求)
     └─ 英文翻译" [Block 61, ref=2]
```

新请求进来：prompt = "你是一个智能客服助手，请帮我查订单..." →
从根走Trie自动匹配：
→ 命中 Root → "你是一个" (Block 17) → "智能客" (Block 22) → 下一个"服助手"没匹配上（树里是"服助手，请"）
→ **自动共享2个Block = 32 Token KV Prefill不用算！** 比vLLM的手工前缀匹配更自动、更细粒度。

**📊 三大推理引擎KV共享能力对比表（面试直接说结论）：**

| 能力 | 原生HF TGI | vLLM PagedAttn v2 | SGLang RadixAttention |
|---|---|---|---|
| 内部碎片率 | 80% | <4% | <4%（同PagedAttn）|
| 相同请求显式前缀共享 | ❌ 不支持 | ✅ 需传prefix_hash | ✅ 自动（基数树匹配）|
| 系统Prompt跨请求全量共享 | ❌ 每请求重算 | ✅ 传enable_prefix_caching=true | ✅ 默认自动 |
| 多轮对话历史KV跨轮复用 | ❌ 每轮重算 | ⚠️ 多轮在同一会话内保留，但不同会话相同历史不共享 | ✅ 相同历史对话段自动跨会话共享 |
| 相同长文档RAG多问 | ❌ 每问题重算整个文档KV | ⚠️ 传doc_id手工共享 | ✅ 相同doc前缀自动共享，QA同一篇文档100问Prefill只做1次 |
| Prefill节省（RAG多问长文档） | 0% | 手工90%+ | 自动90%+ |

---

## 二、调度与批处理（Q9-Q15）

---

### Q9. Continuous Batching动态批处理 vs 静态Batch等待一批全完才下一批：为什么vLLM平均等待低10倍

**⭐ 标准定义**

静态批处理（Static / Orca式批处理）：N个请求一起进入Batch，**必须等N个请求全部Decode生成结束（最长那个请求最后一个token）**，才能放新请求进GPU。结果：短请求生成完了，在GPU里空等长请求几十轮Decode → GPU算力空转+短请求P95极高。

连续批处理（Continuous Batching / Iterative Batching，vLLM默认）：**每1个Decode步结束后（~10-20ms粒度）就重新调度一次**：结束的请求立刻踢出去释放Block，排队中的新请求立刻插进来 → GPU每一步都是满载的，短请求不用等长请求。

**📐 对比示意（2请求，请求A生成32token，请求B生成256token）：**

```
静态Batch (A+B一起进Batch):
  t0 ---- Decode Step 1~32 ---- A结束了但B还没生成完，A占着GPU VRAM啥也不干等！
  t32 --- Decode Step 33~256 -- B终于结束
  总耗时 = 256 steps
  A的等待= 256步（明明32步就生成完却在Batch里占坑等224步！💥）
  B的等待= 256步
  GPU利用率前32步100%，后224步只有B在跑=50%利用率（Batch只剩1个请求浪费SM）

vLLM Continuous Batching（每步重调度）：
  t0~32： A+B Decode 1~32 → Step 32结束后，Scheduler看到A EOS
          → ✅ 立即释放A的Block回FreeList（显存腾出来了）
          → ✅ 同一时刻，把排队的请求C/D/E 3个直接插入现在的Batch！
  t33~256：B + C + D + E 4个请求同时Decode
  A等待：32步（生成完立刻返回给用户，不等待B！✅ 延迟×8下降）
  B等待：256步（没变，但吞吐涨了）
  C/D/E等待：提前224步就开始生成！总吞吐 ×3-4
  GPU利用率：全程 95%+，每一步都是满满当当的Batch
```

**📊 实际Benchmark数据（Llama-7B A100，泊松到达流量，平均到达率=15req/s）：**

| 指标 | 静态Batch (HF TGI bs=64) | 连续Batch (vLLM max-seqs=512) | 提升倍数 |
|---|---|---|---|
| P50 请求端到端延迟 | 3.8s | **0.35s** | **10.8x ⭐** |
| P99 请求端到端延迟 | 21.4s | **1.9s** | 11.2x |
| 吞吐 token/s | 1,240 | **6,780** | 5.5x |
| GPU SM平均利用率 | 58% | **94%** | 1.6x |

**✅ Continuous Batching调度算法伪代码（面试要能说步骤）：**

```python
def scheduler_tick(every_decode_step_done_after):
    # 1. 先处理Running Batch中的每个请求
    for req in running_batch[:]:  # 遍历快照
        req.generated_len += 1
        if req.last_token_is_EOS or req.generated_len >= req.max_tokens:
            # ⭐ 本步生成完了 → 立刻释放，不等其他请求
            running_batch.remove(req)
            kv_cache_manager.free_all_blocks(req)
            send_response_to_user(req)

    # 2. 再从Waiting队列 按策略（FCFS/Priority）加新请求
    while len(running_batch) < args.max_num_seqs and waiting_queue:
        next_req = waiting_queue.popleft()  # FCFS
        if kv_cache_manager.can_allocate(next_req.prompt_len_blocks):
            # ⭐ 插入当前Running Batch！下一个Decode步和老请求一起算
            kv_cache_manager.allocate_prefill_blocks(next_req)
            running_batch.append(next_req)
        else:
            break  # 显存不够了，后面请求肯定也不够，跳出

    # 3. 构造下一个Decode Step的Batch参数（新的seq_lens/block_tables）
    next_step_batch = build_batch(running_batch)
    run_decode_step_async(next_step_batch)  # 启动GPU下一轮Decode
```

**💡 面试加分点：** Continuous Batching是Orca 2022年论文（来自CMU/Stanford）首先提出，叫**Iterative Scheduling**，vLLM把PagedAttention+Orca Continuous Batching两者结合才实现了×10延迟/×5吞吐的效果，单独一个PagedAttention没用。

---

### Q10. Chunked Prefill分块预填充：8K长Prompt Prefill 200ms阻塞200个Decode，怎么切成4块插空跑把P99从212ms降到62ms

**⭐ 标准定义**

问题背景：vLLM早期Prefill是原子操作，一个8192长Prompt Prefill要独占GPU算200ms，这200ms里200个正在Decode的短请求完全停摆不能Decode → 它们的P99延迟突增150ms+，出现毛刺（Tail Latency Spike）。

Chunked Prefill（vLLM 0.2.3+ 引入，参数`--enable-chunked-prefill`，默认True）：把长Prompt Prefill切成**多个固定大小Chunk（默认512 tokens）**，Prefill算完1个Chunk就让Decode请求插空跑几步，再算下一个Chunk → 避免长Prefill阻塞。

**📐 时序对比（8K Prompt vs 200 Decode请求）：**

```
❌ 无Chunked Prefill（老版本）：
  |←           Prefill 8K tokens = 200ms GPU独占            →|
  Decode req1~200: |← 全部空等200ms，每步延迟+200ms →|
  Decode req P99延迟 = 原本12ms + 200ms 阻塞 = 212ms 💥 毛刺

✅ Chunked Prefill = 8K 切成 16个Chunk × 512 tokens 每个Chunk=12.5ms
  |←P512→|D1|D2|D3|←P512→|D1|D2|D3|←P512→|D1|D2|D3| ... （共16轮P+3D交替）
  每个Chunk Prefill=12.5ms后让Decode跑3步(~0.3ms/step×3=1ms)
  Decode req最多只等12.5ms（不是200ms！）
  Decode P99延迟 = 12ms + 12.5ms 最坏等待 = 24.5ms，实测62ms（考虑其他开销）✅
```

**✅ Chunked Prefill关键参数调优（面试说参数）：**

```bash
$ python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --enable-chunked-prefill \           # ⭐ 开Chunked Prefill（默认开）
    --max-num-batched-tokens 8192 \      # ⭐ 每Step批处理最大Token总数
                                            # （Prefill Chunk Tokens + Decode Requests数）
    --prefill-chunk-size 512             # ⭐ 每个Chunk大小=512 tokens 默认
                                            # ↓小 = 延迟更低但调度开销略大
                                            # ↑大 = 吞吐高但P99毛刺大
```

**📊 实测性能（论文/官方Benchmark，长Prompt RAG流量混合场景）：**

| 指标 | 无Chunked Prefill | Chunked Prefill Chunk=512 | 改善 |
|---|---|---|---|
| 长Prefill请求P50延迟 | 204ms | 228ms | +11%（略慢，正常因为插空） |
| **Decode请求P99延迟** ⭐用户体感 | **212ms** | **62ms** | **-71% 毛刺消失！✅** |
| 总吞吐 token/s | 5,780 | 5,510 | -4.6%（吞吐轻微降，延迟大幅改善必tradeoff） |

---

### Q11. Scheduler调度三策略：FCFS公平/优先级VIP插队/LengthAware短任务优先 适用业务场景

**📊 三策略对比表（7维度）：**

| 策略 | 全称 | 入队顺序 | 优点 | 缺点 | 最适合业务场景 |
|---|---|---|---|---|---|
| **FCFS (默认)** | First Come First Serve 先来先服务 | 请求按到达时间排序 | ✅ 绝对公平，无饥饿问题 | ❌ VIP付费用户和免费用户一样等 | C端普通产品、无付费分级 |
| **Priority 优先级** | 优先级队列 | priority字段高的先排，如VIP=0插队到最前 | ✅ 付费用户低延迟，差异化SLA | ❌ 低优先级可能一直被插队饿死（要加aging：每等1s优先级+1） | SaaS企业版/VIP用户分层产品 |
| **Length-Aware SJF** | 最短作业优先 / LengthAware | 预估生成token数少的先排 | ✅ 总平均等待时间最短（排队论结论）| ❌ 长请求可能饿死（同样要aging），而且要能预估长度 | 短对话占比高的闲聊客服 |

**✅ vLLM自定义Scheduler接入伪代码（加分点面试说思路）：**

```python
# vLLM支持自定义Scheduler，通过 --scheduler-class 参数指定
from vllm.core.scheduler import Scheduler

class PriorityScheduler(Scheduler):
    def __init__(self, config, cache_config):
        super().__init__(config, cache_config)
    
    def schedule(self):  # 重写schedule方法
        # 重排waiting队列：按priority + 到达时间 aging
        self.waiting.sort(key=lambda req: (
            req.priority,                    # VIP=0先排
            -(time.time() - req.arrival_time),  # ⭐ aging：等了越久优先级越高，防饿死
            req.req_id
        ))
        return super().schedule()  # 剩下的逻辑复用父类
```

---

### Q12. --max-num-seqs参数：并发请求数设置多少最合适？A100=512 H100=2048 显存+计算权衡公式

**⭐ 标准定义**

`--max-num-seqs N` = vLLM同一时刻GPU上最多同时跑多少个并发请求（Continuous Batch的最大Batch Size）。
- 设太小：GPU SM跑不满，吞吐上不去
- 设太大：每个请求分到的计算资源太少，Decode步长延迟飙升+KV Cache不够用频繁OOM拒绝请求

**📐 经验值表（不同硬件+模型大小，面试背下表）：**

| 模型大小 | GPU型号 | 推荐max-num-seqs | 备注 |
|---|---|---|---|
| 7B (FP16/BF16) | A10G 24G / L4 24G | **128** | 显存小，KV Cache总Token约1M |
| 7B | A100 40G | **256** | |
| 7B | A100 80G | **512 ⭐** | 论文实验值 |
| 7B | H100 80G | **1024** | 算力强+KV大 |
| 13B | A100 80G | **256~384** | 每token KV更大 |
| 13B | H100 80G | **512** | |
| 70B TP8 (8×A100 80G) | 8xA100 | **64~128** | 每卡分到的KV少 |
| 70B TP8 | 8xH100 | **128~256** | |

**📐 理论估算公式（自己也能算）：**

```
Step1: 计算总可用KV Token数
  KV可用显存 = GPU总显存 - 模型权重FP16 - 5GB工作区 (A/B百分比如Q6)
  每个Token KV字节数(FP16) = 2 × num_kv_heads × head_dim × 2B
    Llama-7B: 2×32×128×2 = 16,384 B = 16KB/token
    Llama-13B: 2×40×128×2 = 20,480 B = 20KB/token
    Llama-70B: 2×8×128×2 = 4,096 B = 4KB/token (GQA: 8KV头)
  Total_KV_Tokens = KV可用显存 / 每Token字节数
    A100 80G 跑 7B → 58GB / 16KB ≈ 3.6M tokens

Step2: 平均每请求预计占用多少Token？（根据业务数据拿）
  线上业务平均值：avg_tokens = 1K prompt + 1K生成 = 2K tokens

Step3: max-num-seqs理论上限 = Total_KV_Tokens / avg_tokens
  → 3.6M / 2K = 1800
  但Decode阶段Batch太大（1800并发）SM分不过来，每步Decode延迟太高
  所以经验值：除以3~4 → 1800/3.5 ≈ 514 ≈ 512 ✅ 就是官方默认推荐！
```

---

### Q13. 投机采样Speculative Decoding：小模型草稿猜5Token, 大模型并行验证，速度×2-3的零精度损失加速原理

**⭐ 标准定义（2023 Google DeepMind论文《Accelerating Large Language Model Decoding with Speculative Sampling》）：**

核心思想：**用小模型（60M~7B，快）来猜未来N个Token的"草稿"，然后大模型一次性并行验证这些猜的Token对不对，猜对的直接接受（整步不用Decode算），猜错的第一个位置按大模型真实分布采样，后面丢弃重算。** 因为小模型猜的准的概率非常高（~70-80%位置猜中），所以平均能省2-3x大模型Decode次数。

**🔥 关键前提：输出分布P_big(x)和P_small(x)的交叉熵低时才有效，且必须用相同tokenizer，**且不需要重新训练**，数学上证明了这个算法输出和纯大模型的分布**完全一致（零精度损失）**！**

**📐 工作流程示意（草稿N=5 tokens，小模型=DistilGPT-2 124M，大模型=Llama-2-70B）：**

```
时刻t0: 当前已生成 = [我,爱,深度,学,习]，小模型 + 大模型KV都算到这一步
  Step1: 小模型Draft Model 单独连续Decode 5步 → 出草稿：
           draft_tokens = [我,们,一,起,加] （5个token，小模型算5步很快，总耗时~大模型0.15步）
  Step2: ⭐ 把5个草稿Token拼回原文，做1次并行的大模型Prefill Forward
           大模型1次前向 → 并行算出 6个位置（原来第1个+5个草稿）的Logits分布
           （1次Prefill 6 tokens ≈ 1.5个Decode步 耗时，比5次Decode快多了）
  Step3: 逐个验证每个草稿位置：
           大模型真实argmax位置 = "们" == 草稿[0]"们"？✅ 接受
           大模型真实argmax位置 = "一" == 草稿[1]"一"？✅ 接受
           大模型真实argmax位置 = "起" == 草稿[2]"一"？✅ 接受
           大模型真实argmax位置 = "深" vs 草稿[3]"起"？❌ 不匹配！
               → 第0,1,2个Token直接保留（✓一次拿3个Token！）
               → 第3个Token按大模型真实分布采样"深"
               → 第4,5个草稿丢弃不要
  Step4: 更新KV Cache（3个accept + 1个reject采样 = 4个新Token！不是1个是4个）✅
  回到Step1循环
```

**📊 实测加速效果：**

| 模型组合（Draft+Target） | GPU | Draft N | 平均Accepted Tokens/步 | 纯大模型 tok/s | SpecDec tok/s | 加速比 |
|---|---|---|---|---|---|---|
| GPT-124M + Llama-2-70B | 2xA100 80G | 5 | **3.4** | 128 tok/s | **372 tok/s** | **2.9x ⭐** |
| GPT-6B + Llama-2-70B | 2xA100 80G | 7 | **4.8** | 128 tok/s | **520 tok/s** | **4.1x** |
| TinyLlama-1.1B + Mixtral-8x7B | H100 | 8 | 5.2 | 560 tok/s | 1,458 tok/s | **2.6x** |

**✅ vLLM 开启SpecDec一行参数：**
```bash
$ python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b \
    --speculative-model TheBloke/Llama-2-70B-GPTQ \  # 草稿模型，大模型的GPTQ/AWQ小版或同家族小模型
    --num-speculative-tokens 5   # 每次猜几个，一般4-7最优
```

**💡 面试加分点（数学严谨性）：**
Speculative Sampling不是启发式，而是有数学证明：如果Proposal模型q(x)和Target p(x)满足 acceptance prob = min(1, p(x)/αq(x))，则最终采样分布严格等于p(x)，**零精度损失**。不会出现"小模型拉低大模型智商"的问题。

---

### Q14. Prefill算力密集型 vs Decode访存密集型：为什么两个阶段瓶颈完全不同（计算 vs 显存带宽）

**📊 Prefill vs Decode 阶段特性对比表（面试全背下来）：**

| 维度 | **Prefill 预填充阶段**（Prompt输入进去算KV+第一个输出Token） | **Decode 生成阶段**（之后每生成1个Token一步） |
|---|---|---|
| **输入Token数** | 大：Prompt 512~64K tokens | ⭐ 极小：就 **1个新Token**（上一步生成的那个 + 旧KV）|
| 计算量（FLOPs） | 巨大：Q=全Prompt，K=全Prompt → Self-Attention矩阵O(L²d)；FFN O(Ld²)；L大计算巨多 | 小：Q=1个Token，K=L历史KV → Attention矩阵O(1×L×d)；FFN O(d²)；L大但Q=1 |
| **瓶颈** | ⭐ **算力密集 （Compute Bound）** → 吃FP16/BF16 TensorCore FLOPS | ⭐ **显存带宽密集（Memory Bound）** → 吃HBM 3带宽，TensorCore利用率低 |
| GPU SM利用率 | 高 ~90%+ | 低 <40%（单个请求，warp占不满SM） |
| 每步时间（7B A100 bs=1） | 512 tokens ≈ 40ms | **1 token ≈ 14ms** |
| 每ms处理Tokens | 12.8 tok/ms（算力强） | 0.07 tok/ms（带宽限制） |
| 优化方向 | 算子融合FlashAttention、增加并行度、TensorRT调算子 | ⭐ Continuous Batching把几百个请求拼成大Batch → 每次读KV一次Warp处理N个Q，HBM带宽复用 |
| Batch影响 | 大Batch Prefill：计算/通信比例好，效率更高 | ⭐ Continuous Batch越大越好（显存允许时）：512并发读KV = 512次请求合成1次HBM读 |

**📐 算力/带宽 Roofline模型图解（面试解释）：**

```
Roofline模型：性能= min(峰值算力×计算强度, 峰值带宽×计算强度)
                ↑ GFlops/s
    算力屋顶线 (1000 TFlops/s, H100 FP8)
          ╱
        ╱  Prefill计算强度= 200 Flops/Byte → 撞算力屋顶 ✅ Compute Bound
      ╱───────────────────────────────────
    ╱    ⚠️ Decode计算强度= 0.8 Flops/Byte → 撞带宽屋顶 ❌ Memory Bound
  ╱───────────────────────────────────────────
╱  ↗ 带宽屋顶线（3.35 TB/s H100）
↗----------------------------------------------→ 计算强度 Flops/Byte
```

**💡 面试结论话术：** 这就是为什么vLLM的Continuous Batching是神来之笔——Decode阶段每个请求单独跑是Memory Bound（慢），但把512个请求**拼Batch一起算Decode**，就可以把HBM带宽一次读取复用给512个Q计算 → 让Roofline上的实际点向计算强度高处移动20x，接近算力屋顶 → 吞吐×5+。

---

### Q15. CUDA Graph优化：相同形状批量请求为什么开Graph省30%Kernel Launch CPU开销

**⭐ 标准定义**

CUDA Kernel Launch（CUDA Driver在CPU上发Kernel命令到GPU队列）有固定开销：每个小Kernel ~5-10μs，Decode阶段1步有几十个小Kernel（每个Attention头、FFN每层、LayerNorm）→ 每步CPU Launch开销 ~0.5-1ms，占Decode总延迟10-30%，在大并发下CPU Launcher线程成为瓶颈。

CUDA Graph：把**一组固定输入形状/固定参数的Kernel序列**，提前录制成1个Graph（类似静态图），然后直接replay这个Graph（1次Launch调用顶几十个小Kernel Launch）→ 省掉所有小Kernel Launch开销。

**✅ 工作流程：**

```
1. 【录制阶段 Capture】（启动时/第一次遇到该形状时做1次）
   cudaStreamBeginCapture → 跑1次完整Decode Step：
        LayerNorm → QKVLinear → Attention → FFN1 → GELU → FFN2 → ... 正常Launch每个Kernel
   → cudaStreamEndCapture → 得到一个 cudaGraph_t 对象（保存了所有Kernel和参数引用）

2. 【实例化 Instantiate】（每个不同形状1次）
   cudaGraphInstantiate → 生成 cudaGraphExec_t（编译好的可重放执行计划）

3. 【重放 Replay】（每步Decode调用，省30%时间）
   → cudaGraphLaunch(exec_graph, stream)
   🔥 1个函数调用顶几十个 cudaKernelLaunch，CPU侧零开销！
```

**📊 vLLM 实测开启 `--enforce-eager` vs CUDA Graph：**

| 模型 | 硬件 | max-num-seqs | 不开CUDA Graph (eager) | 开CUDA Graph | 每步Decode加速 |
|---|---|---|---|---|---|
| Llama-2-7B | A100 | 64 | 7.8ms/step | 5.5ms/step | **-29.5%** ⭐ |
| Llama-2-7B | A100 | 512 | 14.2ms/step | 10.1ms/step | **-28.9%** |
| Llama-2-70B TP8 | 8xA100 | 128 | 18.8ms/step | 13.6ms/step | **-27.7%** |

**⚠️ CUDA Graph 限制（面试说缺点加分）：**
- ✅ 静态形状友好：同样batch size、同样seq len的请求可以复用Graph Exec
- ❌ 形状变了要重新Capture（所以vLLM默认开了**CUDAGraphPool**：常见形状比如bs=1,2,4,8,16,32,64,128,256,512都提前Capture好放池子里，进来直接复用）
- ❌ 如果某请求形状不在池子（比如bs=97），就Fallback到eager模式，略慢，但保证功能正确
- vLLM控制参数：`--gpu-memory-utilization 0.9`（影响Graph预分配），默认开CUDA Graph，除非 `--enforce-eager` 强制关。