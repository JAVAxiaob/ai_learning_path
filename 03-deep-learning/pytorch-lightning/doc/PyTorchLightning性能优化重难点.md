# PyTorch Lightning 性能优化重难点解析

> 位置: 03-deep-learning/pytorch-lightning/doc/
> 配套文档: PyTorchLightning模块化训练框架.md | PyTorchLightning流程图详解.md | PyTorchLightning面试题汇总.md

---

## 一、训练速度优化 6大维度全景

### 1.1 加速手段效果对比表

| 优化手段 | 单机单卡加速比 | 显存占用变化 | 实现复杂度 |
|---------|--------------|-------------|-----------|
| ⭐混合精度 precision=16-mixed/bf16-mixed | **1.5-2.5x** | ↓×2 省一半显存 | 0 配置一行 |
| ⭐TF32 (A100/3090+) Ampere架构 | **1.2-1.5x** | 无变化 | 0 自动 |
| ⭐Flash Attention 2.0 (Transformer) | **2-4x** | ↓×5 显存爆减 | 1 一行代码 |
| 梯度累积 accumulate_grad_batches=N | ≈1x (等效大Batch) | ↓×N 可跑超大Batch | 0 Trainer参数 |
| 梯度检查点 activation_checkpointing | **0.7-0.9x**速度略慢 | ↓×10 大模型才能跑 | 1 一行装饰器 |
| torch.compile(mode='max-autotune') | **1.3-2x** PyTorch2.0+ | 编译时+显存 | 0 一行 |
| DDP 4卡数据并行 | **3.5-3.8x** 近线性加速 | 每张卡独立 | 0 Trainer(devices=4) |
| FSDP 大模型 70B+ | 多卡显存共享×N | ↓分×N卡 | 2 strategy参数 |
| DeepSpeed ZeRO Stage3 | 同上 + CPU offload | ↓×10-50 超巨大模型 | 2 strategy='deepspeed_stage_3' |

> 🏆 **黄金组合起步**: Trainer(precision="bf16-mixed", devices=8, strategy="ddp", accumulate_grad_batches=4) + FlashAttn + torch.compile = **单机8卡 40× 起步加速**

---

## 二、DataLoader 数据加载优化 (30%训练瓶颈在IO!)

### 2.1 6大Dataloader参数黄金配置

```python
def train_dataloader(self):
    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        # 🔴 #1 num_workers: CPU核数的一半~2/3 别设0 也别太大
        num_workers=8,  # 推荐值: min(CPU_CORES//2, 8) 16核设8足够
        # 🔴 #2 pin_memory=True: CUDA主机端锁页内存 → 减少CPU→GPU拷贝耗时×2
        pin_memory=True,
        # 🔴 #3 prefetch_factor: 预取2-4批数据GPU空闲时就加载
        prefetch_factor=4,
        # 🔴 #4 persistent_workers=True: 避免每epoch重建worker进程
        persistent_workers=True,
        # 🔴 #5 drop_last=True: 丢掉最后不完整batch 不会因为BatchNorm炸
        drop_last=True,
        # 🔴 #6 collate_fn: 自定义批处理(如NLP padding动态长度省padding)
        collate_fn=dynamic_padding_collate_fn,
    )
```

### 2.2 更高级的格式：提前把数据转成二进制/内存映射格式

| 数据集格式 | 加载速度 | CPU占用 | 适用场景 |
|----------|---------|---------|---------|
| 原始小图片文件夹(JPG) | 1x基线 | 高 解压慢 | 数据集<1万张 |
| WebDataset (tar包流式) | **3-5x** | 低 顺序读快 | 百万级图片ILSVRC |
| LMDB / HDF5 / Parquet | **10-20x** | 极低 | 超大数据集推荐⭐ |
| NVIDIA DALI GPU解码增强 | **20-50x** | 全GPU加速极快 | 极致CV训练 |

---

## 三、混合精度与数值稳定性

### 3.1 FP16 vs BF16 选型 (高频面试题)

| 对比项 | FP16 (半精度) | BF16 (bfloat16) | FP32 (全精度) |
|--------|-------------|----------------|-------------|
| 硬件支持 | 所有主流GPU | A100/A10/4090/M系列 ⭐新架构 | 全部 |
| 动态范围 | ±65504 较窄 ⚠️易溢出 | ±3e38 和FP32一样宽 ✅稳 | ±3e38 |
| 精度尾数 | 10位 ⚠️训练后期不稳 | 7位 精度略差 | 23位 |
| Grad Underflow梯度下溢 | 常见 必须GradScaler放大 | 极少出现 ✅几乎不要Scaler | 无 |
| 训练稳定性 | 中等需调参 | **最稳**推荐优先用 | 最稳 |
| 适用场景 | 老显卡 | 新显卡(3090/4090/A100)首选⭐ | 基线对比 |

### 3.2 常见FP16训练不稳定 + 修复

| 症状 | 根因 | 解决 |
|-----|------|-----|
| loss直接NaN成not a number | 梯度溢出Inf | 换bf16 / 降lr×0.5 / 加大梯度裁剪 max_norm=1.0 |
| 训练一段时间后loss突然爆涨 | 梯度爆炸/scale_factor太小 | Trainer(gradient_clip_val=1.0) 全局梯度裁剪 |
| 验证集acc比FP32低2%+ | 小梯度被GradScaler当成0跳过了 | 换bf16 / precision=32 baseline对比 |
| LayerNorm/Softmax层数值异常 | 数学范围超FP16表示极限 | 关键层手动保持FP32: LayerNorm().to(torch.float32) |

---

## 四、分布式策略选型 DDP vs FSDP vs DeepSpeed

```
模型大小 × 可用GPU数 选型地图:
───────────────────────────────────────────────────
模型大小      单卡能装？            最佳策略
───────────────────────────────────────────────────
< 7B       → ✅ 单卡装的下       → DDP 数据并行
               Trainer(strategy='ddp', devices=8)
               → 8卡同步梯度 线性加速 ~7.6x

7B ~ 70B   → ❌ 单卡装不下       → FSDP ZeRO-3
               Trainer(strategy='fsdp', devices=8)
               → 每层切片分到8卡 显存/8
               → 再加CPU Offload: 参数卸载到内存

> 70B     → ❌ 8卡也装不下      → DeepSpeed Stage3 + NVMeOffload
               Trainer(strategy="deepspeed_stage_3_offload")
               → 权重卸载到CPU内存还不够 + SSD磁盘虚拟内存
               → 牺牲速度 但能跑
```

### 4.1 FSDP 分片策略3级对比

| FSDP Sharding策略 | 显存节省 | 通信开销 | 适用 |
|-----------------|---------|---------|------|
| SHARD_GRAD_OP (ZeRO-2) | ~×2 省 | 中等 | 10B模型推荐 |
| FULL_SHARD (ZeRO-3) ⭐ | ~×N卡 8卡省8倍 | 高 | 70B大模型标配 |
| HYBRID_SHARD 节点内分片 | ×节点数 | 低 | 多机多卡大集群 |

FSDP 配置代码:
```python
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ViT/Transformer 最优: 每层独立分片, 通信重叠计算
my_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock, VisionTransformerBlock},
)

strategy = FSDPStrategy(
    auto_wrap_policy=my_auto_wrap_policy,  # ⭐必须按层分片 不然没用
    cpu_offload=True,   # 显存不够就卸到CPU 慢×2 但能跑
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ),
)
Trainer(devices=8, strategy=strategy, accelerator="cuda")
```

---

## 五、Flash Attention 2 优化 (Transformer加速神器)

### 5.1 为什么能省×10显存 + ×3速度？

```
传统Attention显存瓶颈:
Q[B,H,N,D] × K.T = S[B,H,N,N] 超大中间矩阵 N=4096 → B=16 → 16×16GB爆显存!
                              ↓ Softmax(S) × V = O
Flash Attention 2 分块计算 Tiling:
把N切成128小块, 每块分别算 → 全程不用存大N×N矩阵!
→ 显存复杂度 从 O(N²) → O(N) 线性!
→ 多的显存 → 能加2倍BatchSize → 再×2速度
```

代码接入 (1行):
```python
# PyTorch2.0+ 原生就支持: torch.nn.functional.scaled_dot_product_attention
# 只要不用自定义Attention写法 就自动走FlashAttn!
# Trainer(precision="bf16-mixed") 开bf16效果最好

# 或者手动启用 (最可靠写法)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)  # 关慢的数学实现
```

---

## 六、超参数 & 学习率调参黄金公式

### 6.1 BatchSize × LearningRate 线性缩放法则

```
✅ 经验黄金法则 (Linear Scaling Rule, 由Facebook何恺明论文验证):
BatchSize × k倍 → LearningRate 也×k倍
────────────────────────────────────
例:
单卡 Batch=32,  LR=1e-4   → 基线
4卡 DDP Batch=128 (×4倍) → LR = 4e-4  (×4倍)
4卡×累积×2 → 等效Batch=256 → LR = 8e-4

⚠️ 配合warmup步数线性热身:
WarmupSteps = max(1000, 总步数×5%) 前5%步数LR从0线性升到目标LR
→ 防止大Batch初始梯度炸
```

CosineAnnealing 配置 (95%场景最优):
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(),
                                  lr=4e-4,         # 根据Batch缩放
                                  weight_decay=0.1, # Transformer默认0.1
                                  betas=(0.9, 0.999)) # AdamW默认
    total_steps = self.trainer.estimated_stepping_batches  # PL自动算总数
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=self.lr * 0.01
    )
    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
```

---

## 七、梯度消失/爆炸诊断 & Callback修复

自定义梯度监控Callback 面试逐行代码:
```python
class GradientMonitor(Callback):
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % 100 != 0:
            return
        total_norm, layer_norms = 0.0, {}
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                layer_norms[name] = param_norm.item()
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log("grad/total_norm", total_norm, prog_bar=True)
        # ✅ 诊断规则:
        # grad_norm → 0.00001以下: 梯度消失! 降lr/换初始化/加LayerNorm
        # grad_norm → 1000+:      梯度爆炸! 加大gradient_clip_val=5.0
        # grad_norm在 0.01 ~ 10:  ✅ 健康范围
```