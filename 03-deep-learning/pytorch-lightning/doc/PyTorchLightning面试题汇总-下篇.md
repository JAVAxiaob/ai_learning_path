# PyTorch Lightning面试题汇总（下篇：分布式训练与工程实践）

> 适用对象：AI算法工程师/深度学习工程师岗位面试 | 覆盖Q19-Q35共17题 | 配合上篇使用

---

## 三、分布式训练深入（Q19-Q26）

---

### Q19. DDP AllReduce算法原理？为什么Ring AllReduce比Parameter Server带宽扩展性好？计算公式

**✅ AllReduce定义：** 分布式训练每个GPU有独立梯度，需要聚合所有GPU的梯度平均值再下发给每个GPU做参数更新。AllReduce = Reduce（求和）+ Broadcast（广播）的融合操作。

**📊 三种AllReduce算法对比表：**

| 算法 | 通信步骤 | 每个GPU通信数据量 | 带宽利用率 | 扩展性(N卡) | 适用场景 |
|---|---|---|---|---|---|
| **Parameter Server** | 2步：Worker→PS收梯度→PS发新参数 | 每个Worker：2M（发M收M），PS瓶颈：2N·M | ❌ 差，PS带宽瓶颈 | N>32时PS打满，线性扩展崩溃 | 小集群<8卡，异步训练 |
| Tree AllReduce（二叉树） | 2·log₂N步 | M·log₂N | ⭐⭐ 50%左右 | 中等，N=8时8M/卡 | PyTorch默认NCCL内部fallback |
| **✅ Ring AllReduce（Baidu 2017，NCCL默认）** | **2(N-1)步（N-1次ScatterReduce + N-1次AllGather）** | **2M·(N-1)/N ≈ 2M 几乎与N无关！** | ⭐⭐⭐⭐⭐ 95%+，每卡带宽均衡 | **线性扩展！N=64卡仍≈2M/卡** | 任何>2卡训练必选 |

**🔍 Ring AllReduce两步拆解（N=4卡）：**
```
假设总梯度大小=M，每卡切成N=4块，每块大小=M/4

【Phase 1: ScatterReduce — 顺时针传N-1=3次】
初始：卡0=[a0,b0,c0,d0], 卡1=[a1,b1,c1,d1], 卡2=[a2,b2,c2,d2], 卡3=[a3,b3,c3,d3]
第1步传：卡0→1发d0, 卡1→2发a1, 卡2→3发b2, 卡3→0发c3
       卡1累加d块: d0+d1,  卡2累加a块: a1+a2, 卡3累加b块: b2+b3, 卡0累加c块: c3+c0
第2步传：卡0→1发(c0+c3), 卡1→2发(d0+d1), ...
       卡2累加d块: d0+d1+d2, 卡3累加a块: a1+a2+a3,...
第3步传：【最终】每卡各自拥有1块Complete Sum！
       卡0的c块: c0+c1+c2+c3=Σc, 卡1的d块: Σd, 卡2的a块: Σa, 卡3的b块: Σb

【Phase 2: AllGather — 再顺时针传N-1=3次】
每轮把自己的Complete块传给邻居，最终每卡都拥有Σa Σb Σc Σd = 完整Σ梯度！

✅ 总数据量：N-1次ScatterReduce每卡传M/N + N-1次AllGather每卡传M/N 
           = 2(N-1)·M/N ≈ 2M 与卡数N无关！线性扩展！
```

**💡 PL DDP使用要点：** PL Trainer(strategy="ddp") 底层自动调用NCCL Ring AllReduce，**不要手动在training_step里all_reduce**，optimizer.step()之前DDP hook会自动完成梯度同步。但`self.log("loss", x, sync_dist=True)`才会跨卡聚合指标。

> ⚠️ 常见坑：计算节点之间万兆网卡<10Gbps时Ring效率暴跌，必须用InfiniBand/RoCE RDMA网卡 + nvlink（同机GPU互联）。A100 8卡同机NVLink AllReduce吞吐比PCIe快8~12倍！

---

### Q20. FSDP FULL_SHARD vs SHARD_GRAD_OP vs HYBRID_SHARD切分策略对比？70B模型选型表

**✅ FSDP定义（Fully Sharded Data Parallel，Meta 2022 + PyTorch原生 + PL Strategy）：** DDP每卡存完整参数M+梯度M+优化器状态2M=4M显存；FSDP按维度切分到N卡，每卡仅存4M/N → 70B模型单卡A100 80G也能塞下！

**📊 四种Sharding策略对比选型表：**

| 策略名 | 切分内容 | 每卡显存占用（对比DDP=100%） | 通信开销（每step） | 适用模型规模 | 典型场景 |
|---|---|---|---|---|---|
| **NO_SHARD = DDP** | 不切，每卡全量 | 100% (模型4M) | 仅梯度AllReduce 2M | ≤7B模型 / 显存充足 | 小模型追求最高速度 |
| **SHARD_GRAD_OP (ZeRO-2)** | ✅切梯度+优化器状态，❌参数不切（forward前全Gather参数） | ~25-30% (模型M + 梯度/优化器切分) | AllReduce参数一次+梯度AllReduce | 7B~30B模型 / A100 80G | 推荐大多数7B/13B训练首选，速度快 |
| **✅ FULL_SHARD (ZeRO-3，FSDP最常用)** | ✅切 参数+梯度+优化器状态 全部三切 | **~10-15% 最低！** (每卡仅4M/N) | 每层forward前AllGather参数→计算完丢弃 | 30B~70B+ / 必须开 | 70B Llama2必选，唯一能装下 |
| **HYBRID_SHARD (FSDP v2新)** | 同机8卡内NO_SHARD（用NVLink高速），跨机FULL_SHARD切 | ~20% 折中 | 跨机通信量减为1/N_node | 多机8xA100集群 / 64卡+ | 企业集群16机+首选，跨机带宽瓶颈首选 |

**💡 PL启用FSDP代码：**
```python
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ✅ 关键：按Transformer Layer层粒度wrap（默认按整个模型切，通信爆炸）
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer,}
)

fsdp = FSDPStrategy(
    sharding_strategy="FULL_SHARD",        # ZeRO-3 70B必选
    auto_wrap_policy=auto_wrap_policy,     # 每层独立切分，逐层通信
    cpu_offload=True,                      # 优化器状态/卸载到CPU内存（极端情况）
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ),
    activation_checkpointing_policy={LlamaDecoderLayer,} # ✅ 每层重计算省显存
)

trainer = pl.Trainer(accelerator="cuda", devices=8, num_nodes=4, strategy=fsdp, precision="bf16-mixed")
```

> ⚠️ 三大常见坑：① **没设auto_wrap_policy** — 整个模型一张大卡切，每层都全Gather整个模型，比DDP还慢10倍；② **FULL_SHARD + gradient_accumulation_steps > 1** — 切记`fsdp.set_state_dict_type(STATE_DICT_TYPE.SHARDED_STATE_DICT)`，否则checkpoint加载显存OOM；③ **fsdp + torch.compile** 2.1以下版本有bug，需升级PyTorch 2.2+。

---

### Q21. FSDP vs DeepSpeed ZeRO-3对比表？企业生产选型建议

**📊 FSDP vs DeepSpeed ZeRO-3 全方位对比表：**

| 维度 | PyTorch原生FSDP | Microsoft DeepSpeed ZeRO-3 |
|---|---|---|
| **维护方** | PyTorch官方，与PT版本同步发布 | 微软独立仓库，独立发版（滞后PT 1-3个月） |
| **接入代码改动** | ⭐⭐⭐⭐⭐ 几乎0改动，Lightning仅`strategy="fsdp"` | ⭐⭐ 需要DeepSpeedConfig JSON + monkey patch模型 |
| **ZeRO Offload CPU/NVMe** | 支持CPU Offload（但实验性） | ✅✅ 成熟支持CPU + NVMe Offload（ZeRO-Offload/ZeRO-Infinity） |
| **混合精度** | bf16/fp16原生，但fp16部分算子不稳定 | ✅ fp16非常成熟，大量生产验证 |
| **通信优化** | NCCL原生 + HYBRID_SHARD多机优化 | NCCL + 自研通信，8卡以下相当，64卡以上DS略优 |
| **Flash Attention 2兼容** | ✅ 直接兼容（因为不patch模型） | ❌ 经常冲突（DeepSpeed重写了很多Attention算子） |
| **torch.compile兼容** | ✅ PT2.2+完美兼容 | ❌ 大部分情况Graph Break，不支持compile |
| **Checkpoint格式** | 原生PT state_dict / Sharded格式 | ✅ 自定义DS格式，转换工具多 |
| **Debuggability** | ⭐⭐⭐⭐⭐ 原生NCCL error，断点正常 | ⭐⭐ 堆栈深，很多C++层报错看不懂 |
| **3D并行 (TP+PP+DP)** | 需手动组合（FSDP+TensorParallel） | ✅ DeepSpeed-Ulysses/3D-Parallel开箱即用 |
| **生产稳定性（70B模型）** | ⭐⭐⭐⭐ 2.2+版本稳定 | ⭐⭐⭐⭐⭐ 最多大厂线上验证 |

**🎯 选型建议（PL环境下）：**
| 场景 | 推荐 |
|---|---|
| 新启动项目 / 追求简洁 / 用torch.compile / 用FlashAttn2 | ✅ FSDP |
| 单卡显存不够必须NVMe Offload / 64卡以上大集群 / 老项目沿用 | ✅ DeepSpeed ZeRO-3 |
| 70B模型 8xA100单机 | FSDP FULL_SHARD + CPU Offload（稳） |
| 70B模型 4机32卡 | FSDP HYBRID_SHARD（最优） |

> 💡 PL同时支持两者：`strategy="deepspeed_stage_3"`直接用ZeRO-3，JSON配置可以传`strategy=DeepSpeedStrategy(config="ds_config.json")`。

---

### Q22. DDP梯度同步时机？no_sync()上下文管理器有什么用？梯度累积时为什么要开

**✅ DDP默认同步机制（每step）：**
```
每一个training_step iteration的生命周期：
1. forward：✅ DDP hook自动AllGather参数（其实DP才需要，DDP参数初始已广播好，一直全量）
2. loss.backward()：✅【关键】计算梯度过程中，每个梯度反向算完后自动触发AllReduce异步通信！
                     （不是等所有梯度算完再同步，是边算边同步，通信与计算重叠）
3. optimizer.step()：梯度已聚合完毕，直接本地更新参数
```

**📌 no_sync()定义：** 临时禁用DDP反向传播的梯度AllReduce通信，梯度**只算不通信**，等退出上下文后再手动触发一次批量同步。

**💡 两大核心使用场景：**

**场景1：梯度累积（Gradient Accumulation）必开！省N-1次通信**
```python
# ❌ 错误写法（纯DDP代码，PL默认自动帮你优化了！）
accum_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch) / accum_steps
    loss.backward()                    # 每轮都AllReduce！4次通信=浪费3次
    if (i+1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ 正确写法（3次跳过同步，只1次通信，速度+30%）
for i, batch in enumerate(loader):
    # 前accum_steps-1次，no_sync不通信！
    with model.no_sync() if ((i+1) % accum_steps != 0) else nullcontext():
        loss = model(batch) / accum_steps
        loss.backward()                # 只算本地梯度，不同步
    if (i+1) % accum_steps == 0:
        # 最后一次：退出no_sync → 自动触发AllReduce同步一次汇总梯度
        optimizer.step()
        optimizer.zero_grad()
```

**场景2：多个loss分别反向（GAN / 多任务），统一同步一次**
```python
# Generator + Discriminator 两个forward/backward，只想同步一次
with ddp_model.no_sync():
    g_loss = ddp_model(gen_input)
    g_loss.backward()   # G的梯度只算不同步
d_loss = ddp_model(dis_input)
d_loss.backward()       # 退出no_sync，这里一次性 AllReduce G+D全部梯度！
opt_G.step(); opt_D.step()
```

> 💡 PL自动优化：`Trainer(accumulate_grad_batches=4)`时，**PL底层自动在LightningModule上包no_sync()**，你写的代码完全不用改！这就是框架的价值。面试如果被问"梯度累积怎么优化通信"，要答出来no_sync机制。

---

### Q23. 多机多卡训练 NCCL/SHM/MPI backend怎么选？常见NCCL错误排查套路

**📊 三种分布式通信Backend对比选型：**

| Backend | 维护方 | CPU通信 | GPU通信 | 速度 | 支持平台 | 推荐场景 |
|---|---|---|---|---|---|---|
| **✅ NCCL (NVIDIA Collective)** | NVIDIA | ❌ 不支持CPU Tensor | **✅⭐⭐⭐⭐⭐ 最快** | GPU通信最快（NVLink/PCIe/RDMA全部加速） | Linux + NVIDIA GPU ONLY | **所有GPU训练默认唯一选择** |
| Gloo | Meta/Facebook | ✅ 支持 | ⭐⭐ 能用，但比NCCL慢2~5倍 | 中等 | Linux/Windows CPU + GPU | **纯CPU训练 / Windows调试** |
| MPI (OpenMPI) | 社区 | ✅ 支持 | ⭐⭐⭐ 中等 | 靠MPI实现 | Linux需装OpenMPI | 老项目/多机InfiniBand RDMA老集群 |

**🏭 多机多卡启动命令（4机×8卡=32卡，PL标准写法）：**
```bash
# 机器1（master，IP=192.168.1.100，PORT=29500）— 4机都要执行！
torchrun \
  --nnodes=4 \                 # 4台机器
  --node_rank=0 \              # 机器号0/1/2/3，每台改这个
  --nproc_per_node=8 \         # 每台机器8卡
  --master_addr=192.168.1.100 \  # 机器0的IP
  --master_port=29500 \
  train.py --accelerator cuda --strategy ddp
# PL Trainer会自动读取环境变量LOCAL_RANK/RANK/WORLD_SIZE
```

**🔍 NCCL常见错误排查三板斧（90%问题都在这里）：**

| 错误类型 | 症状 | 排查步骤 | 修复方法 |
|---|---|---|---|
| **NCCL WARN Connection refused** | 初始化卡住10min超时，master连不上slave节点 | ① ping 对端IP通不通；② telnet master_ip 29500端口 | 防火墙放行端口29500~29600；用内网IP不用公网；ECS开安全组 |
| **NCCL WARN P2P access between device 0 and 1 is NOT supported** | 同机GPU互联失败，走PCIe很慢 | `nvidia-smi topo -m`看NVLink | PCIe卡正常，但设`NCCL_P2P_DISABLE=1`跳过检查；GPU之间通过同一个NUMA节点插卡 |
| **NCCL WARN misc/socket.cc:54 retrying connect... timeout** | 多机互联RDMA卡不通 | `ibv_devinfo`看RDMA网卡；`nccl-tests`测带宽 | 设`NCCL_IB_DISABLE=1`临时走TCP（降速）；或加载MLNX_OFED驱动 |
| **NCCL error unhandled system error / Cuda Error invalid device ordinal** | backward()突然崩 | 每台机器`echo $LOCAL_RANK`是否0-7 | 检查torchrun参数；不要手动`CUDA_VISIBLE_DEVICES`冲突 |
| **DDP各卡loss不一致** | 同step 8个GPU loss差别很大>5% | ①打印各卡输入；②打印grad范数 | DistributedSampler **必须每个epoch set_epoch(epoch)**！否则所有卡同一份数据 |

> 🔑 Debug必杀技：设置环境变量 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` 打印NCCL详细日志，看哪一步卡了。

---

### Q24. PL的 Strategy列表？ddp vs ddp_spawn vs ddp_fork vs deepspeed vs fsdp区别

**📊 PL常用分布式Strategy完整对比表：**

| Strategy名 | 启动方式 | 进程模型 | 支持num_nodes>1多机 | Windows支持 | 单机启动速度 | 与DataLoader num_workers兼容 |
|---|---|---|---|---|---|---|
| `dp` (DataParallel) | 单进程多线程 | 1进程，多卡多线程forward | ❌ 仅单机 | ✅ | 快（无fork） | ❌ GIL锁严重，CPU预处理超慢 |
| **`ddp` ✅ 生产首选** | torchrun外部启动 | 每卡1独立进程 | ✅ 多机 | ❌ 仅Linux | 快（外部已fork） | ✅ 完美兼容 |
| `ddp_spawn` | Trainer内部spawn() | Trainer里起多进程 | ❌ 仅单机 | ✅ Win可用 | 慢（pickle序列化模型大文件） | ⚠️ ddp_spawn + num_workers>0 经常死锁！ |
| `ddp_fork` | Trainer内部fork() | 同上，用unix fork | ❌ 仅单机Linux | ❌ | 比spawn快（不pickle） | ⚠️ OpenCV/多线程不兼容fork，core dump |
| `ddp_notebook` | Jupyter Notebook内 | 同上spawn | ❌ 仅单机 | ❌ | 慢 | ❌ notebook调试玩具 |
| `deepspeed_stage_1/2/3` | torchrun / Trainer内部 | 同ddp进程模型 | ✅ 多机 | ❌ | 同ddp | ✅ 兼容 |
| **`fsdp` ✅ 大模型首选** | torchrun | 同ddp进程模型 | ✅ 多机 | ❌ | 同ddp | ✅ 兼容，HYBRID_SHARD跨机 |
| `bagua` | 第三方 | 同ddp | ✅ | ❌ | 比ddp快10%（自研算法） | ✅ |

**🎯 选型建议：**

| 场景 | 推荐Strategy |
|---|---|
| **生产训练脚本** | **永远选 ddp**，用torchrun启动，别用内部起进程 |
| 本机调试快速看loss跑通 | ddp_spawn（Windows可用）但别开num_workers>2 |
| 7B~70B大模型 | fsdp 或 deepspeed_stage_3 |
| 多机多卡 | ddp + torchrun （fsdp/deepspeed底层也是ddp进程启动） |
| 700M小模型速度极致优化 | bagua（对小模型有特殊梯度压缩算法快10%+） |

> ⚠️ 千万别在生产用`dp`：① DP master卡瓶颈，8卡只能到5倍速（DDP 7.6倍）；② DP不支持多机；③ DP模型的batch norm同步不对。

---

### Q25. PL SyncBatchNorm原理？什么时候必须转？多卡小batch精度掉了怎么办

**✅ SyncBN定义（跨卡同步BatchNorm）：** 普通BN每卡**独立**计算自己的均值μ/方差σ²（64卡×8bs=每卡8样本，μ统计严重不准，BN层毁精度）。SyncBN = AllReduce所有卡的μ和σ²，用全局统计量做归一化 → **精度恢复和大batch单卡一致**。

**🔍 SyncBN计算流程：**
```
普通BN（每卡独立）：
卡0 bs=8 → μ_0, σ²_0 → 归一化自己的8张图  ❌ 不准
卡1 bs=8 → μ_1, σ²_1 → 归一化自己的8张图  ❌ 不准

SyncBN（全局统计）：
1️⃣ 每卡算局部sum=Σx, sum_sq=Σx², count=N（纯标量，通信量极小！）
2️⃣ AllReduce sum / sum_sq / count 三个标量（总共24字节，通信可以忽略）
3️⃣ 全局μ = total_sum/total_count，全局σ² = total_sum_sq/N - μ²
4️⃣ 每卡各自用全局μ/σ²归一化自己的特征图 → 统计一致！
```

**💡 PL一键开启：**
```python
# ✅ 自动把模型中所有BatchNorm2d/3d/1d替换成SyncBatchNorm
trainer = pl.Trainer(
    accelerator="cuda", devices=8, strategy="ddp",
    sync_batchnorm=True,   # 🔑 一行搞定！
)
```

**🎯 什么时候必须开SyncBN：**

| 情况 | 是否开SyncBN | 原因 |
|---|---|---|
| 单机8卡，batch_size=128（每卡16） | ⭐ 可选（提升0.5~1%） | 每卡16样本勉强够BN统计 |
| 单机8卡，batch_size=64（每卡8） | ✅ **必须开** | 每卡8样本BN统计偏差大，掉点2%+ |
| 检测分割（batch_size本来就小 bs=2/卡） | ✅ **必须开！** | 否则mAP可以掉5~8个点！ |
| 多机32卡，batch_size=32（每卡1） | ✅✅ 不开完全没法训练！ | 单样本BN=灾难，退化成LayerNorm |
| 用GroupNorm / InstanceNorm / LayerNorm | ❌ 不需要 | 这些Norm不依赖batch维度统计 |

> ⚠️ 常见坑：① SyncBN虽然通信量小，但层数多（ResNet152 200+BN层）累加也有10%左右开销；② 不要和`torch.nn.SyncBatchNorm.convert_sync_batchnorm()`手动一起用，会双重包装报错；③ ViT/Transformer用LayerNorm→完全不用关心SyncBN。

---

### Q26. Lightning DeepSpeed集成要点？ZeRO-3 + CPU Offload 70B配置示例

**✅ PL DeepSpeed 零代码集成：** 只要把strategy改成deepspeed，不需要改LightningModule代码！

**📊 四种Stage选型（和FSDP对应关系）：**

| DeepSpeed Stage | 等价FSDP策略 | 显存占用 | 推荐场景 |
|---|---|---|---|
| stage 0 | DDP（只混合精度） | 100% | 小模型baseline对比 |
| stage 1 | 优化器状态切分 | 75% | 不常用 |
| stage 2 ✅ 推荐7B/13B | SHARD_GRAD_OP ZeRO-2 | 40% | 大多数场景首选，速度快 |
| **stage 3 ✅ 70B必用** | FULL_SHARD ZeRO-3 | **10%~20%** | 70B/130B大模型 |

**💡 70B模型 ZeRO-3 + CPU Offload 完整配置（ds_config.json + PL）：**
```jsonc
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "steps_per_print": 10,
  "optimizer": { "type": "AdamW", "params": { "lr": 2e-5, "weight_decay": 0.1, "adam_w_mode": true } },
  "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": 0, "warmup_max_lr": 2e-5, "warmup_num_steps": 100 } },
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "cpu", "pin_memory": true, "nvme_path": "/mnt/nvme" },
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "allgather_partitions": true, "allgather_bucket_size": 5e8,
    "overlap_comm": true, "reduce_scatter": true, "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9
  },
  "gradient_clipping": 1.0,
  "activation_checkpointing": {
    "partition_activations": true, "cpu_checkpointing": true,
    "contiguous_memory_optimization": true
  }
}
```

**PL代码集成：**
```python
from pytorch_lightning.strategies import DeepSpeedStrategy

strategy = DeepSpeedStrategy(config="ds_zero3_cpuoffload_70b.json")

trainer = pl.Trainer(
    accelerator="cuda", devices=8, num_nodes=4, strategy=strategy,
    max_epochs=1, precision="bf16-mixed",
    gradient_clip_val=1.0, accumulate_grad_batches=8
)
trainer.fit(model)
```

> 🚨 典型ZeRO-3坑：checkpoint加载默认是分片的，单卡加载会OOM → DeepSpeed必须`deepspeed --launcher torchrun --num_gpus=1 zero_to_fp32.py . /tmp/fp32_ckpt.pt`把分片转成FP32单文件再推理用。

---

## 四、工程实践与高级话题（Q27-Q35）

---

### Q27. PL Callback vs Hook区别？写一个自定义ModelCheckpoint只保存Top3 val_acc的ckpt

**📊 Callback系统 vs LightningModule Hooks 对比：**

| 维度 | Callback（插件式） | LightningModule Hook（继承override） |
|---|---|---|
| **代码位置** | 独立class，可复用到任意项目 | 写在你的LightningModule里 |
| **可组合性** | ✅ 无限多个堆叠，不互相影响 | ❌ 一个Hook只能override一次 |
| **关注点** | 通用横切逻辑：保存ckpt、打印日志 | 业务相关逻辑：模型本身的train_step |
| **举例** | ModelCheckpoint, EarlyStopping, LRMonitor | training_step, validation_step |
| **常用生命周期** | on_train_start / on_validation_end | 同上，但方法是Module上的 |

**💡 实战：自定义ModelCheckpoint Callback**
```python
import os
from pytorch_lightning.callbacks import ModelCheckpoint

class TopKModelCheckpoint(ModelCheckpoint):
    def __init__(self, top_k=3, monitor="val/acc", mode="max", **kwargs):
        super().__init__(monitor=monitor, mode=mode, save_top_k=top_k, **kwargs)
        self.custom_best_path = None
    
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        current = self.current_score
        if self.best_model_score is None or self._is_better(current, self.best_model_score):
            target = os.path.join(self.dirpath, "best-model.pt")
            trainer.save_checkpoint(target, weights_only=False)
            self.custom_best_path = target
            print(f"✅ 新best模型保存: {target}, score={current:.4f}")

checkpoint_cb = TopKModelCheckpoint(
    top_k=3, monitor="val/acc", mode="max",
    dirpath="checkpoints/",
    filename="model-epoch{epoch:02d}-valacc{val/acc:.4f}",
    auto_insert_metric_name=False,
    save_last=True, every_n_epochs=1, save_on_train_epoch_end=False
)
trainer = pl.Trainer(callbacks=[checkpoint_cb])
```

> ⚠️ 常见坑：**在on_train_epoch_end存ckpt**（val还没跑！存的是上个epoch指标！）→ 必须在on_validation_end后存。

---

### Q28. EarlyStopping Callback三大参数patience/min_delta/stopping_threshold？自定义F1早停

**📊 EarlyStopping核心参数详解表：**

| 参数 | 作用 | 典型值 | 说明 |
|---|---|---|---|
| `monitor` | 监控哪个指标 | `"val/loss"` / `"val/acc"` | 必须是self.log过的key！ |
| `mode` | min/max | `"min"`（loss）/`"max"`（acc） | ❌ 写错mode反着判断=模型永远不收敛 |
| `patience` ⭐ 最常用 | 连续多少epoch没变好才停 | 5 / 10 / 20 | patience=5 = 至少等5个涨平才停 |
| `min_delta` | 必须至少提升多少才算变好 | `0.0001` / `1e-4` | 防止第N位小数抖动无限训练 |
| `stopping_threshold` | 达到阈值直接停 | `0.95` acc / `0.001` loss | 指标到天花板直接停，省电费 |
| `divergence_threshold` | 高于阈值直接发散停 | `loss > 1e+6` | loss爆NaN立刻停 |
| `check_finite` | 检测NaN/Inf自动停 | 默认True | 别关 |
| `verbose` | 打印每次不提升日志 | True | 调试用 |

**💡 实战示例（推荐配置）：**
```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop_loss = EarlyStopping(
    monitor="val/loss", mode="min", patience=8,
    min_delta=1e-4, stopping_threshold=0.02,
    divergence_threshold=5.0, verbose=True, log_rank_zero_only=True
)

early_stop_acc = EarlyStopping(
    monitor="val/acc", mode="max", patience=5,
    min_delta=0.001, stopping_threshold=0.96, verbose=True
)

trainer = pl.Trainer(callbacks=[checkpoint_cb, early_stop_loss, early_stop_acc])
```

**💡 自定义：按batch_level的F1早停**
```python
from pytorch_lightning.callbacks import Callback
from torchmetrics import F1Score

class BatchLevelF1EarlyStop(Callback):
    def __init__(self, patience_batches=1000, mode="max"):
        self.patience = patience_batches
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.no_improve = 0
    
    def _better(self, a, b): return a > b if self.mode=="max" else a < b
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 != 0: return
        val_batch = next(iter(trainer.datamodule.val_dataloader()))
        x, y = val_batch[0].to(pl_module.device), val_batch[1].to(pl_module.device)
        pl_module.eval()
        with torch.no_grad():
            logits = pl_module(x)
            f1 = F1Score(task="multiclass", num_classes=10).to(pl_module.device)(logits.argmax(1), y)
        pl_module.train()
        
        if self._better(f1.item(), self.best):
            self.best = f1.item(); self.no_improve = 0
        else:
            self.no_improve += 100
        
        if self.no_improve >= self.patience:
            print(f"🛑 Batch级早停！best F1={self.best:.4f}")
            trainer.should_stop = True
```

---

### Q29. LearningRateMonitor + CSVLogger + WandbLogger使用要点？自定义日志到TensorBoard

**📊 三种主流Logger对比选型：**

| Logger | 存储 | 可视化 | 团队协作 | 线上部署 | 推荐场景 |
|---|---|---|---|---|---|
| **CSVLogger**（PL内置） | 本地metrics.csv | ❌ Excel画 | ❌ | ✅ 无需网络 | 离线环境 |
| **TensorBoardLogger** ✅ 官方首选 | 本地events文件 | TensorBoard WebUI | ❌ 本地共享 | ✅ 内网可看 | **单机调试必备** |
| **WandbLogger** (Weights&Biases) | 云端SaaS | 超豪华WebUI+GPU监控 | ✅ 团队链接分享 | ❌ 涉密不行 | **公司团队协作首选**，90%大厂用 |

**💡 三大Logger + LRMonitor组合拳：**
```python
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

loggers = [
    TensorBoardLogger(save_dir="logs/tb_logs/", name="resnet50_exp", version="v1"),
    CSVLogger(save_dir="logs/csv_logs/", name="resnet50_exp", version="v1"),
    WandbLogger(project="image-classification", name="resnet50-bs128-adamw",
                save_dir="logs/wandb/", log_model="all", tags=["baseline"])
]

lr_monitor = LearningRateMonitor(
    logging_interval="step", log_momentum=True, log_weight_decay=True
)

trainer = Trainer(logger=loggers, callbacks=[lr_monitor], log_every_n_steps=20)
```

**💡 自定义图像日志：**
```python
class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        
        if batch_idx == 0 and self.current_epoch == 0:
            preds = logits.argmax(dim=1)
            grid = torchvision.utils.make_grid(x[:8], nrow=4, normalize=True)
            self.logger[0].experiment.add_image(
                tag="train/predictions", img_tensor=grid, global_step=self.global_step
            )
        return loss
```

---

### Q30. PL超参搜索Optuna集成？LightningCLI一键调度用法

**📊 超参搜索框架对比：**

| 框架 | 算法 | 并行 | 早停剪枝 | 易用性 | 推荐 |
|---|---|---|---|---|---|
| **Optuna** ✅ 最流行 | TPE贝叶斯采样 | ✅ 多进程多机 | ✅ Median/SuccessiveHalving | ⭐⭐⭐⭐⭐ | **首推**，生态最成熟 |
| Ax (Facebook) | 贝叶斯BO+多目标 | ✅ 多机 | ✅ | ⭐⭐⭐ | FB生态团队 |
| Ray Tune (HpBandSter) | BOHB=贝叶斯+HyperBand | ✅ Ray集群 | ✅ HyperBand最专业 | ⭐⭐⭐ | 百卡集群百组实验 |

**💡 Optuna + PL 标准集成：**
```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.Trial):
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }
    model = ImageClassifier(**hparams)
    dm = CIFARDataModule(batch_size=hparams["batch_size"])
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val/acc")
    
    trainer = Trainer(max_epochs=30, accelerator="cuda", devices=1,
                      callbacks=[pruning_cb], enable_checkpointing=False,
                      enable_progress_bar=False)
    trainer.fit(model, dm)
    return trainer.callback_metrics["val/acc"].item()

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(direction="maximize", pruner=pruner, storage="sqlite:///optuna.db")
study.optimize(objective, n_trials=30, n_jobs=1)
print("🏆 Best:", study.best_params, "val_acc=", study.best_value)
```

**💡 LightningCLI 零代码调参：**
```python
# main.py
from pytorch_lightning.cli import LightningCLI
from my_model import ImageClassifier
from my_datamodule import CIFARDataModule

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=ImageClassifier, datamodule_class=CIFARDataModule,
        save_config_kwargs={"overwrite": True}
    )
```
```bash
# 命令行零改代码玩遍所有参数：
python main.py fit --trainer.max_epochs 1 --model.hidden_dim 512 --data.batch_size 256
# 4机32卡DDP
torchrun --nnodes=4 --node_rank=$NODE_RANK --nproc_per_node=8 main.py fit --config config.yaml
# 测试最优ckpt
python main.py test --ckpt_path logs/version_0/checkpoints/best.ckpt
```

---

### Q31. 断点续训resume_from_checkpoint注意事项？自动恢复哪些状态？哪些要手动

**✅ PL ckpt保存内容（完整状态）：**
```
checkpoint.ckpt = {
  "state_dict":               → model参数 ✅ 自动恢复
  "optimizer_states":         → Adam动量/二阶矩 ✅ 自动恢复（重要！）
  "lr_schedulers":            → scheduler步数 ✅ 自动恢复
  "loops":                    → global_step, current_epoch ✅ 自动恢复
  "callbacks":                → EarlyStopping/Checkpoint的best值 ✅ 自动恢复
  "hyper_parameters":         → save_hyperparameters()的参数 ✅ 自动恢复
}
```

**💡 断点续训正确姿势：**
```python
trainer = pl.Trainer(max_epochs=100, accelerator="cuda", devices=8, strategy="ddp")
# 上次跑到epoch=32被中断了，直接接着跑！
trainer.fit(model, datamodule=dm, ckpt_path="logs/version_0/checkpoints/last.ckpt")
# → epoch从33开始，global_step接上次，完美！
```

**📊 自动恢复 vs 手动确认清单：**

| 状态 | 是否自动恢复 | 注意事项 |
|---|---|---|
| model参数weights | ✅ 是 | |
| optimizer state（Adam动量） | ✅ 是 | 同一个optim类；改lr手动覆盖 |
| scheduler state（Cosine步数） | ✅ 是 | 改max_epochs会错 |
| current_epoch / global_step | ✅ 是 | |
| EarlyStopping patience计数 | ✅ 是 | 换数据集会误判 |
| DataLoader shuffle种子 | ❌ 否 | 数据增强轻微差0.1%正常 |
| RNG随机种子 | ❌ 否 | 手动seed_everything(seed, workers=True) |
| 混合精度GradScaler | ✅ 是 | |
| 你self.xxx自定义变量 | ❌ 否 | 需override on_save_checkpoint/on_load_checkpoint |

**🔍 迁移学习场景：只加载backbone权重**
```python
# ❌ 错误：ckpt_path把optimizer/epoch也加载，接着之前的epoch跑
# trainer.fit(model, ckpt_path="pretrain.ckpt")

# ✅ 正确：只加载模型参数（从头训练）
ckpt = torch.load("pretrain_imagenet.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)  # head层不匹配忽略
trainer.fit(model, dm)  # epoch从0开始
```

> ⚠️ 三大死坑：
> 1. FSDP/DeepSpeed分片ckpt **必须同样N卡数恢复**！8卡训的不能4卡加载；
> 2. resume后`max_epochs`要设100（原总epoch），不是68（剩下的），PL自动从33跑到100；
> 3. `ModelCheckpoint(save_last=True)` 一定要开！last存最新，best存最优。

---

### Q32. PL自定义Training Loop？manual optimization实现GAN

| 场景 | 是否自定义Loop |
|---|---|
| 普通分类/检测/分割 | ❌ 不用，自动模式完美 |
| GAN（D/G交替，两个optim） | ⭐ manual optimization即可 |
| RL PPO（rollout+多步） | ✅ 自定义Loop |
| 蒸馏（Teacher/Student） | ⭐ manual optimization |
| MAML元学习 | ✅ 自定义Loop |

**💡 实战1：GAN用Manual Optimization（优先用这个，比自定义Loop简单！）**
```python
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=64, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # ✅ 关闭自动优化
        self.generator = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
        self.discriminator = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()
        
        # 1. Train Discriminator
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        d_real = self.discriminator(real_imgs.flatten(1)).view(-1)
        d_fake = self.discriminator(fake_imgs.detach().flatten(1)).view(-1)
        d_loss = -torch.mean(0.5*torch.log(d_real.sigmoid()+1e-8) + 0.5*torch.log(1-d_fake.sigmoid()+1e-8))
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        
        # 2. Train Generator
        d_fake2 = self.discriminator(fake_imgs.flatten(1)).view(-1)
        g_loss = -torch.mean(torch.log(d_fake2.sigmoid()+1e-8))
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        
        self.log_dict({"loss/d": d_loss, "loss/g": g_loss}, prog_bar=True)
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_d, opt_g], []
```

---

### Q33. PL常见错误排查SOP？CUDA OOM / 梯度NaN / DDP挂起

**📋 面试高频故障排查SOP表：**

| 故障 | 典型报错 | 优先级解决方法 |
|---|---|---|
| **CUDA OOM 显存爆** | `CUDA out of memory` | 1️⃣ batch/2 → 2️⃣ precision bf16 → 3️⃣ activation checkpointing → 4️⃣ FSDP ZeRO-3 |
| **Loss NaN / Inf** | `loss=nan` | 1️⃣ detect_anomaly定位层数 → 2️⃣ 降lr + 梯度裁剪 → 3️⃣ 数据检查label → 4️混合精度改fp32 debug |
| **DDP永远挂起** | `initializing process group`卡住 | 1️⃣ 防火墙/端口29500 → 2️各节点torch/nccl版本一致 → 3️NCCL_DEBUG=INFO |
| **DataLoader死锁** | enumerate()卡死 | 1️Windows+ddp_spawn别开num_workers>0 → 2️Dataset别开多线程 → 3️先num_workers=0跑通 |
| **val_acc比train_acc高** | train 80% val 85% | 正常现象！train模式dropout/BN噪声大；val少采样偏差；数据增强太狠 |
| **self.log key找不到** | `KeyError: 'val/acc'` | 打印trainer.callback_metrics.keys()；key大小写完全一致；val_step里确实self.log过 |
| **DDP各卡loss不同** | rank0/1差2倍 | 1DistributedSampler.set_epoch(epoch)漏了！ → 2️SyncBatchNorm没开 → 3️数据划分重叠 |

**💡 CUDA OOM定位工具：**
```python
from pytorch_lightning.callbacks import DeviceStatsMonitor
trainer = pl.Trainer(callbacks=[DeviceStatsMonitor()])  # 每step打印显存
```

---

### Q34. 迁移学习冻结backbone在PL里优雅处理？解冻+微调两步策略代码

**✅ 经典两阶段微调：**
```
Stage 1 (0-5 epoch)：冻结Backbone，只训Head（lr=1e-3，快）
Stage 2 (5-30 epoch)：解冻Backbone全量微调（backbone lr=1e-5，head lr=1e-4）
```

**💡 Callback优雅控制冻结/解冻：**
```python
from pytorch_lightning.callbacks import Callback

class BackboneFinetuningFreeze(Callback):
    def __init__(self, unfreeze_at_epoch=5, backbone_attr="backbone"):
        self.unfreeze_epoch = unfreeze_at_epoch
        self.backbone_attr = backbone_attr
    
    def setup(self, trainer, pl_module, stage):
        backbone = getattr(pl_module, self.backbone_attr)
        for p in backbone.parameters(): p.requires_grad = False
        print(f"❄️  Stage1: 冻结{self.backbone_attr}，仅训head")
    
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            backbone = getattr(pl_module, self.backbone_attr)
            for p in backbone.parameters(): p.requires_grad = True  # 解冻
            optimizer = trainer.optimizers[0]
            for pg in optimizer.param_groups:
                if "backbone" in pg.get("name", ""):
                    pg["lr"] = 1e-5
                else:
                    pg["lr"] = 1e-4
            print(f"🌱 Stage2: epoch{self.unfreeze_epoch}解冻，差分学习率启用")

# LightningModule参数分组（差分学习率必备）
class ResNetTransfer(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        import torchvision.models as models
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(in_feat, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    
    def configure_optimizers(self):
        param_groups = [
            {"params": self.backbone.parameters(), "lr": self.hparams.lr*0.1, "name": "backbone", "weight_decay": 1e-4},
            {"params": self.head.parameters(),     "lr": self.hparams.lr,     "name": "head",     "weight_decay": 1e-4}
        ]
        return torch.optim.AdamW(param_groups, lr=self.hparams.lr)

trainer = pl.Trainer(max_epochs=30, callbacks=[BackboneFinetuningFreeze(unfreeze_at_epoch=5)])
```

**📊 微调学习率经验表：**

| 层 | 推荐lr（相对head） | 原因 |
|---|---|---|
| 分类Head fc | 1x = 1e-3 | 随机初始化，快收敛 |
| Layer3/4最后两层 | 0.1x = 1e-4 | 特征通用，微调 |
| Layer1/2前两层 | 0.01x = 1e-5 | 边缘纹理，几乎别改 |
| Conv1/bn1最底层 | 冻结 or 0.001x | 改了反而掉点 |

---

### Q35. PL+生产推理模型导出？TorchScript/ONNX/state_dict三种对比

**📊 三种导出方式对比表：**

| 格式 | 依赖 | 跨语言 | 推理速度 | 体积 | 量化 | 推荐场景 |
|---|---|---|---|---|---|---|
| `.pt` state_dict | PyTorch+模型代码 | ❌ Python only | 慢 | 大 | ⚠️ 手动 | 训练内断点推理 |
| **TorchScript .ts** ✅ C++ | libtorch | ✅ C++/Java/Go | ⭐⭐⭐ 1.3x | 中 | ✅ 原生量化 | **移动端/边缘C++ Serving** |
| **ONNX .onnx** ✅ 跨框架标准 | ORT/TensorRT | ✅ 所有语言 | ⭐⭐⭐⭐⭐ TRT 4~10x | 小 | ✅ 工具链最成熟 | **生产Serving首选** |

**💡 PL三种导出完整代码：**
```python
# 加载最优ckpt（PL一行搞定，hparams自动恢复）
model = MyLightningModel.load_from_checkpoint("best.ckpt", map_location="cpu")
model.eval()
model.freeze()  # 等效eval + no_grad优化
dummy = torch.randn(1, 3, 224, 224)

# ====== 方法1: 纯state_dict（Python内用） ======
torch.save(model.state_dict(), "production/weights.pt")

# ====== 方法2: TorchScript（C++/移动端部署） ======
try:
    scripted = torch.jit.script(model)  # Scripting（静态类型支持好）
except:
    scripted = torch.jit.trace(model, dummy)  # Tracing回退（有控制流慎用）
scripted.save("production/model_scripted.ts")
# C++加载: torch::jit::load("model_scripted.ts")

# ====== 方法3: ONNX（生产Serving+TensorRT加速）======
torch.onnx.export(
    model=model, args=dummy, f="production/model.onnx", opset_version=17,
    input_names=["images"], output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},  # 动态batch
    do_constant_folding=True
)
# 验证：onnxruntime推理
import onnxruntime as ort
ort_sess = ort.InferenceSession("production/model.onnx", providers=["CUDAExecutionProvider"])
ort_out = ort_sess.run(["logits"], {"images": dummy.numpy()})[0]
with torch.no_grad():
    pt_out = model(dummy).numpy()
print(f"✅ 精度验证 最大误差: {np.abs(ort_out-pt_out).max():.2e}")  # <1e-5通过
```

---

> 📌 配合**上篇（Q1-Q18基础架构+性能优化）**，PL 35题完整覆盖面试全部考点，从单机单卡到大模型多机多卡分布式，再到工程化落地全链路。
