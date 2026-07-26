# PyTorch 面试题汇总（下篇）- 高级主题与生产部署（20题 附详细标准答案）

---

## 目录

- [七、推理加速与模型部署（第21-28题）](#七推理加速与模型部署第21-28题)
- [八、TorchScript 与模型序列化（第29-32题）](#八torchscript-与模型序列化第29-32题)
- [九、性能调优与内存优化（第33-38题）](#九性能调优与内存优化第33-38题)
- [十、自定义算子与扩展开发（第39-40题）](#十自定义算子与扩展开发第39-40题)

---

## 七、推理加速与模型部署（第21-28题）

---

### 21. model.eval() 和 torch.no_grad() 有什么区别？推理时两个都要写吗？

**⭐ 标准定义**

- `model.eval()`：**切换模型为推理模式**，影响特定Layer的行为（Dropout关闭、BatchNorm用running_mean/var而非batch统计）
- `torch.no_grad()`：**关闭梯度追踪上下文管理器**，让所有操作不构建计算图，节省显存和算力

**✅ 正确用法：推理时两个都要写，缺一不可！**

```python
# ❌ 错误1：只写eval不写no_grad → 仍构建计算图，显存浪费
model.eval()
output = model(x)

# ❌ 错误2：只写no_grad不写eval → Dropout仍工作，BN用batch统计 → 结果错误！
with torch.no_grad():
    output = model(x)

# ✅ 正确写法：两个都写
model.eval()
with torch.inference_mode():  # PyTorch 1.9+ 推荐，比no_grad更快更彻底
    output = model(x)
```

| 对比项 | model.eval() | torch.no_grad() | torch.inference_mode() |
|---|---|---|---|
| 关闭Dropout | ✅ 是 | ❌ 否 | ❌ 否 |
| BN用running统计 | ✅ 是 | ❌ 否 | ❌ 否 |
| 不记录梯度 | ❌ 否 | ✅ 是 | ✅ 是 |
| 节省显存 | ❌ 几乎不 | ✅ 省~50% | ✅ 省更多 |
| 能否取tensor.grad | ✅ 能（没必要） | ❌ 不能 | ❌ 不能 |
| 版本要求 | 所有版本 | 所有版本 | PyTorch≥1.9 |

**💡 常见面试坑：** 推理忘写model.eval() → 线上准确率掉5-10%，面试必问经典坑！

---

### 22. PyTorch模型导出ONNX的完整流程是什么？有哪些常见踩坑点？

**⭐ 标准定义**

ONNX（Open Neural Network Exchange）是跨框架模型格式，PyTorch→ONNX用`torch.onnx.export`基于**追踪（tracing）**或**脚本化（scripting）**将计算图序列化。

**✅ 完整导出流程（生产级模板）**

```python
import torch
import torchvision.models as models
import numpy as np

model = models.resnet50(weights="IMAGENET1K_V1")
model.eval()  # ⭐ 必须先切eval模式！

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,               # ⭐ 越高支持的算子越多，推荐14/17
    dynamic_axes={                  # ⭐ 动态轴声明
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    do_constant_folding=True,
    export_params=True,
    verbose=False
)

# ========== 验证：PyTorch vs ONNX输出一致性 ==========
import onnxruntime as ort
with torch.inference_mode():
    torch_out = model(dummy_input).numpy()
sess = ort.InferenceSession("resnet50.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"input": dummy_input.numpy()})[0]
np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-5)
print("✅ ONNX导出验证通过！")
```

**🚨 常见导出踩坑（面试必问）**

| 坑点 | 原因 | 解决方案 |
|---|---|---|
| 动态控制流导出错误（if/for依赖数据） | tracing只记录一次执行路径 | 用torch.jit.script或确保dummy覆盖所有分支 |
| Tensor.shape[i]做循环范围 | Python int被固化成常量 | 用torch.jit.script + 声明dynamic_axes |
| 输出全零/NAN | 忘了model.eval()，BN用batch=1统计 | 必须先eval再导出 |
| 自定义算子不支持 | ONNX无对应opset | register_custom_op_symbolic注册映射 |
| opset过低 | GELU/LayerNorm低版本不支持 | 升级opset到14+ |

---

### 23. 什么是混合精度训练？AMP自动混合精度怎么用？能省多少显存？

**⭐ 标准定义**

混合精度训练（Mixed Precision）：模型**权重用FP32保存**（保证精度），**前向/反向计算用FP16/BF16**（加速+省显存），配合Loss Scaling防止FP16梯度下溢。

**✅ PyTorch AMP 标准用法（PyTorch 1.6+ 原生支持）**

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ========== 核心1：GradScaler 梯度缩放器 ==========
scaler = GradScaler(
    init_scale=2**16,      # 初始缩放因子，防FP16梯度下溢
    growth_factor=2.0,     # 连续N步没溢出就×2
    backoff_factor=0.5,    # 溢出一次就÷2
    growth_interval=2000   # 连续2000步无溢出才放大
)

for epoch in range(NUM_EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        # ========== 核心2：autocast 上下文 ==========
        with autocast(dtype=torch.float16):  # 或bfloat16（Ampere+推荐）
            logits = model(x)
            loss = criterion(logits, y)
        # ========== 核心3：scaler.scale(loss).backward() ==========
        scaler.scale(loss).backward()
        # ========== 核心4：scaler.step(optimizer) ==========
        scaler.step(optimizer)
        # ========== 核心5：scaler.update() ==========
        scaler.update()
```

**📊 显存/速度收益（典型ResNet-50，单A100）**

| 精度模式 | 显存占用 | 训练速度 | 精度影响 |
|---|---|---|---|
| FP32纯单精 | ~14GB（bs=128） | 1x基准 | ✅ 最高 |
| AMP FP16混合 | ~7.5GB（省~46%） | 1.5x~1.8x | ✅ 几乎无损（<0.1%） |
| AMP BF16混合 | ~7.5GB（省~46%） | 1.6x~1.9x | ✅ 完全无损（Ampere+） |

---

### 24. FP16、BF16、FP32 三种浮点格式有什么区别？怎么选？

**📊 三种格式参数对比表（面试必须背下来）**

| 属性 | FP32 (单精) | FP16 (半精) | BF16 (bfloat16) |
|---|---|---|---|
| 总位数 | 32 bit | 16 bit | 16 bit |
| 符号位 | 1 | 1 | 1 |
| 指数位 | 8 | **5** | **8**（和FP32一样） |
| 尾数位 | 23 | 10 | 7 |
| 数值范围 | ±3.4e38 | **±6.5e4**（小！） | **±3.4e38**（和FP32一样） |
| 精度（最小增量） | ~1e-7 | ~1e-3 | ~1e-2 |
| 需要Loss Scaling? | 不需要 | ✅ **必须** | ❌ **不需要** |
| 支持硬件 | 所有GPU | Volta+ (V100+) | **Ampere+ (A100/3090/4090)** |

**💡 FP16 为什么必须 Loss Scaling？**

```
FP16最小正数值 = 2^(-24) ≈ 5.96e-8

当梯度 = 1e-8 时：
  FP16直接存储 → 太小超出表示范围 → round成0 → 梯度消失！❌

Loss Scaling 机制：
  step1: loss × S (S=65536) → 梯度也自动×S → 1e-8×65536≈6.55e-4 ✅ FP16存得下
  step2: backward 用缩放后的loss算梯度
  step3: scaler.unscale_() → 梯度÷S → 还原真实值
  step4: 检查溢出(INF/NAN) → 溢出则跳过本次step

BF16指数位=8bit（和FP32一样），最小正数=2^(-134)，根本不会下溢，所以不用Scaler！
```

---

### 25. 推理时批量处理（Batch Inference）怎么优化？动态Batch怎么做？

**⭐ 标准定义**

推理吞吐量优化核心：**把多个单请求打包成一个Batch**，利用GPU并行算力，QPS可提升3-10倍。

**✅ 方案1：静态Batch（服务端攒批）**

```python
from collections import deque
import time

class StaticBatcher:
    def __init__(self, model, max_batch=32, max_wait_ms=10):
        self.model = model
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = deque()

    def infer(self, x_single):
        start = time.time()
        self.queue.append(x_single)
        # 攒批条件：队列满 OR 等待超时
        while len(self.queue) < self.max_batch and (time.time()-start) < self.max_wait:
            time.sleep(0.001)
        batch = [self.queue.popleft() for _ in range(min(self.max_batch, len(self.queue)))]
        batch_tensor = torch.cat(batch, dim=0)
        with torch.inference_mode():
            out = self.model(batch_tensor)
        return torch.split(out, 1, dim=0)
```

**✅ 方案2：动态Batch + 填充（NLP变长场景）**

```python
from torch.nn.utils.rnn import pad_sequence

def dynamic_batch_collate(batch_list):
    input_ids = [item[0] for item in batch_list]
    labels    = [item[1] for item in batch_list]
    # pad到本batch最长长度，而不是全局最大
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded_ids != 0).long()
    return padded_ids, attention_mask, torch.stack(labels)

dl = DataLoader(dataset, batch_size=32, collate_fn=dynamic_batch_collate)
```

---

### 26. 如何把PyTorch模型从CPU迁移到GPU？.cuda()/.to()/pin_memory 分别有什么用？

**✅ 推荐写法（可移植性好）**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # model.to()是in-place
x = x.to(device)          # tensor.to()不是in-place！必须赋值！
```

**🚨 关键区别（面试必考）**

| 操作 | 是否 in-place | 作用对象 | 常见坑 |
|---|---|---|---|
| model.to(device) | ✅ 是 | 整个Module（递归移动parameters/buffers） | - |
| tensor.to(device) | ❌ 否（返回新Tensor） | 单个Tensor | ❌ 不赋值：x.to(device) 不写 x=x.to(...) |

**✅ DataLoader 的 pin_memory + non_blocking 加速数据搬运**

```python
# pin_memory=True：锁页内存，GPU可直接DMA搬运，速度快2-3倍
# 普通可分页内存 → GPU要先拷到临时锁页buffer，多一次拷贝
# 锁页内存 → 永远不swap，地址固定，GPU直接DMA
train_loader = DataLoader(
    dataset, batch_size=64,
    num_workers=4, pin_memory=True,
)

for x, y in train_loader:
    # non_blocking=True：异步拷贝，数据搬运和GPU计算重叠
    # 生效条件：1) pin_memory=True 2) CPU→CUDA
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
```

---

### 27. PyTorch 2.0 的 torch.compile() 怎么用？能提升多少性能？

**✅ 基础用法（一行代码加速）**

```python
model = models.resnet50().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ⭐ 一行编译，后面用法完全不变
model = torch.compile(model, mode="max-autotune")
# mode="default"           平衡型：编译快 + 运行中等加速
# mode="reduce-overhead"   推理/小模型推荐：极致降Python开销
# mode="max-autotune"      训练/大模型推荐：花久编译，运行最快

for x, y in train_loader:
    # 第一次调用会触发编译（有点慢），第二次起开始加速
    logits = model(x.cuda())
```

**📊 实际性能收益（A100 80GB实测）**

| 模型 | 模式 | 原生PyTorch | torch.compile后 | 加速比 |
|---|---|---|---|---|
| ResNet-50 训练 | max-autotune | 1500 img/s | 2250 img/s | **1.5x** |
| GPT-2 1.5B 训练 | max-autotune | 38 tok/s/GPU | 57 tok/s/GPU | **1.5x** |
| BERT-Large 推理 | reduce-overhead | 62 QPS | 93 QPS | **1.5x** |
| ViT-B 小模型推理 | reduce-overhead | 1800 QPS | 3060 QPS | **1.7x** |

**🔥 torch.compile 核心优化：**
1. 算子融合（Conv+BN+ReLU三变一，减少显存读写）
2. 内核自动调优（枚举多种GEMM选最快）
3. Python开销消除（静态图绕开解释器）
4. 内存规划优化（自动复用临时Tensor）

---

### 28. 量化（Quantization）有哪些类型？PyTorch怎么实现INT8量化？

**📊 三种量化方式对比表（面试必考）**

| 量化方式 | 原理 | 精度损失 | 速度提升 | 适用场景 |
|---|---|---|---|---|
| **动态量化 (Dynamic PTQ)** | 权重量INT8，**激活运行时临时量化** | 小 (~0.5%) | 1.5-2x（CPU明显） | NLP模型（BERT/GPT）、CPU推理 |
| **静态量化 (Static PTQ)** | 权重+激活**都提前量化**，校准集统计范围 | 中 (~1-2%) | 2-4x ⭐最快 | CNN模型（ResNet）、移动端/端侧 |
| **QAT 量化感知训练** | 训练时模拟量化噪声，微调补偿 | 极小 (<0.2%) | 2-4x | 高精度要求场景 |

**✅ PyTorch 三种量化代码示例**

```python
from torch.ao.quantization import (
    get_default_dynamic_quant_config,
    get_default_qconfig,
    quantize_dynamic, quantize, prepare_qat, convert
)

model_fp32 = models.resnet18(weights="IMAGENET1K_V1").eval()

# ============== 方案1：动态量化（一行搞定）==============
model_dq = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)  # 模型大小 46MB → 12MB

# ============== 方案2：静态量化（需要校准集）==============
model_sq = models.resnet18(weights="IMAGENET1K_V1").eval()
model_sq.qconfig = get_default_qconfig("x86")  # x86服务器/qnnpack移动端
torch.ao.quantization.prepare(model_sq, inplace=True)
# 校准：几百张图统计激活min/max
for x, _ in calibration_loader:
    with torch.no_grad():
        _ = model_sq(x)
torch.ao.quantization.convert(model_sq, inplace=True)

# ============== 方案3：QAT 量化感知训练 ==============
model_qat = models.resnet18(weights="IMAGENET1K_V1")  # train模式
model_qat.train()
model_qat.qconfig = get_default_qconfig("x86")
prepare_qat(model_qat, inplace=True)  # 插入伪量化节点
# 微调1-3个epoch
for epoch in range(3):
    train_one_epoch(model_qat, train_loader)
model_qat.eval()
convert(model_qat, inplace=True)
```

---

## 八、TorchScript 与模型序列化（第29-32题）

---

### 29. TorchScript 是什么？tracing 和 scripting 两种模式有什么区别？

**⭐ 标准定义**

TorchScript是PyTorch的**中间表示（IR）**，把Python模型转成**独立于Python运行时**的序列化格式，可以在C++/Java/移动端部署。

**📊 Tracing vs Scripting 对比（面试必考）**

| 对比项 | torch.jit.trace | torch.jit.script |
|---|---|---|
| 转换方式 | 给dummy跑一遍前向，记录执行路径 | 编译Python AST源码 |
| 支持if/for控制流 | ❌ **只记录执行到的分支** | ✅ 完整支持 |
| 依赖数据的循环 | ❌ 循环次数被固化为常量 | ✅ 运行时动态 |
| 需要dummy输入 | ✅ 必须给 | ❌ 不需要 |
| 报错时机 | 追踪运行时 | 编译语法时 |
| 代码兼容性 | ✅ 几乎都能trace | ❌ 部分语法不支持 |

**❌ Tracing失败典型案例**

```python
class ControlFlowModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:   # 依赖输入的if分支
            return x * 2
        else:
            return x + 100

traced = torch.jit.trace(ControlFlowModel(), torch.tensor([1.0, 2.0]))
# sum=3>0 → 只记录了x*2路径！
print(traced(torch.tensor([-1.0, -2.0])))
# 预期：[-1+100, -2+100]=[99,98]
# 实际：[-1*2,  -2*2 ]=[-2,-4] ❌ BUG！

scripted = torch.jit.script(ControlFlowModel())
print(scripted(torch.tensor([-1.0, -2.0])))  # ✅ 正确返回[99,98]
```

---

### 30. state_dict 包含哪些内容？保存/加载完整模型有哪几种方式？

**📊 三种模型保存/加载方式对比**

| 方式 | 保存命令 | 加载命令 | 推荐度 | 说明 |
|---|---|---|---|---|
| **state_dict（官方推荐）** | `torch.save(model.state_dict(),"w.pt")` | `m=Model(); m.load_state_dict(...)` | ⭐⭐⭐⭐⭐ | 体积小，兼容好 |
| **整个模型** | `torch.save(model,"full.pt")` | `m=torch.load("full.pt")` | ⭐ | 耦合代码路径，常报错 |
| **TorchScript部署** | `scripted.save("model.ts")` | `m=torch.jit.load("model.ts")` | ⭐⭐⭐⭐ | 跨语言，不需Python |

**✅ 生产级Checkpoint保存模板（支持中断续训）**

```python
# ============== 保存（不止是权重！）==============
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),  # Adam动量等
    "scheduler_state_dict": scheduler.state_dict(),  # LR调度状态
    "scaler_state_dict": scaler.state_dict(),        # AMP缩放器状态
    "best_acc": best_acc,
    "loss_history": loss_list,
}
torch.save(checkpoint, f"checkpoint_ep{epoch}.pt")

# ============== 加载（完整恢复训练现场）==============
start_epoch = 0
if RESUME:
    ckpt = torch.load("checkpoint_ep9.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]
```

**🚨 常见加载坑**

| 报错 | 原因 | 解决方案 |
|---|---|---|
| Missing key in state_dict | 权重少了key | strict=False 忽略（确认模型没错） |
| Unexpected key "module.layer1..." | DDP训练保存多了module.前缀 | `{k.replace('module.',''):v for k,v in sd.items()}` |
| size mismatch | 模型结构改了（类别数变了） | 删对应key不加载，strict=False |
| Attempting to deserialize on CUDA | 无GPU机器加载GPU模型 | map_location="cpu" 必加！ |

---

### 31. torch.load 的 map_location 参数有什么用？怎么正确加载跨设备模型？

**✅ 四种map_location写法**

```python
# 1. 全加载到CPU（最通用，一定不报错）
sd = torch.load("gpu.pt", map_location="cpu", weights_only=True)

# 2. 加载到指定GPU（原本存GPU0→直接加载到GPU1）
sd = torch.load("gpu0.pt", map_location="cuda:1")

# 3. 函数式动态映射
def map_fn(storage, loc):
    return storage.cuda(0)  # 强制都放到GPU0
sd = torch.load("multi_gpu.pt", map_location=map_fn)

# 4. PyTorch 2.0+ 安全加载（防pickle漏洞）
sd = torch.load("w.pt", map_location="cpu", weights_only=True)
# weights_only=True：只解析Tensor，不反序列化Python对象 → 防恶意pt执行代码
```

---

### 32. parameters()、buffers()、named_parameters()、state_dict() 四者区别？

**⭐ 标准定义**

Module持久化数据分两类：
1. **Parameters（参数）**：需梯度更新的可学习权重（nn.Parameter包装），requires_grad=True
2. **Buffers（缓存）**：不需梯度但要跟着模型保存（BN的running_mean/var等），requires_grad=False

**📊 四者对比表**

| API | Parameters | Buffers | 带name | 返回类型 |
|---|---|---|---|---|
| model.parameters() | ✅ 全部 | ❌ 无 | ❌ | Generator[Tensor] |
| model.buffers() | ❌ 无 | ✅ 全部 | ❌ | Generator[Tensor] |
| model.named_parameters() | ✅ 全部 | ❌ 无 | ✅ | Generator[(name, Tensor)] |
| model.named_buffers() | ❌ 无 | ✅ 全部 | ✅ | Generator[(name, Tensor)] |
| model.state_dict() | ✅ 全部 | ✅ 全部 | ✅ 完整层级名 | OrderedDict |

**💡 面试超级坑（90%人答错）：**

问：BN的weight和bias是parameter还是buffer？
→ ✅ **是Parameter！** BN公式 y = γ*(x-μ)/σ + β，γ(bn.weight)和β(bn.bias)是可学习参数
→ 只有running_mean/running_var（滑动平均统计值）是Buffer

普通属性tensor不会被state_dict记录，必须用register_buffer注册。

---

## 九、性能调优与内存优化（第33-38题）

---

### 33. 怎么查看PyTorch显存占用？memory_allocated() 和 memory_reserved() 区别？

**⭐ 标准定义**

- **Allocated（已分配）**：Tensor真实占用的显存（实际用了多少）
- **Reserved/Cached（已缓存）**：PyTorch向CUDA Driver申请的显存池总量（分配的+空闲没还的）

**✅ 显存监控函数**

```python
def print_gpu_mem(prefix=""):
    alloc = torch.cuda.memory_allocated() / 1024**3
    res   = torch.cuda.memory_reserved()  / 1024**3
    max_a = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{prefix}] Alloc={alloc:.2f}GB, Reserved={res:.2f}GB, Peak={max_a:.2f}GB")
```

**📊 典型训练阶段显存构成（BERT-Large bs=16，单GPU）**

| 阶段 | 显存占用 | 主要构成 |
|---|---|---|
| 刚加载模型 | ~1.2GB | FP32权重（4B×340M参数≈1.36GB） |
| 前向传播中 | ~10GB ⭐最大 | 权重 + **中间激活值（占大头！）** |
| 反向传播后 | ~2.4GB | 权重 + 梯度（和权重等大） |
| +Adam优化器 | +3.6GB | Adam每个参数存m和v（=3×权重大小） |
| **总需求** | ~12GB | |

核心结论：中间激活值是显存消耗大头（batch越大越多），优化激活能省最多显存。

---

### 34. 显存不够怎么办？（OOM）请列举至少8种显存优化手段？

按实施难度从易到难：

| 方案 | 原理 | 显存节省 | 精度影响 |
|---|---|---|---|
| 1. **减小batch_size** | 激活量与batch成正比 | 线性减少 | ✅ 无（用梯度累积补） |
| 2. **混合精度AMP** | 激活/计算用FP16（2B） | 40-50% ⭐ | ✅ 几乎无损 |
| 3. **梯度累积N步** | N步才更新一次，等效大batch | 省(1-1/N)激活 | ✅ 等效大bs |
| 4. **Adam→SGD** | SGD无m/v状态 | ~33%总显存 | ⚠️ 需调LR |
| 5. **梯度检查点** | 前向不存激活，反向重算 | 激活省50-80% | ✅ 无损（慢20-30%） |
| 6. **冻结部分层** | 浅层不回传，不存激活 | 按层数比例省 | ⚠️ 微调用 |
| 7. **优化器Offload** | 优化器状态放CPU（ZeRO） | 50%+ | ✅ 无损 |
| 8. **FSDP/ZeRO-3** | 参数/梯度分片到多卡 | 多卡时80%+ | ✅ 无损 |

**✅ 梯度累积代码（面试手写题）**

```python
accum_steps = 4   # 等效 batch_size × 4
optimizer.zero_grad()

for step, (x, y) in enumerate(train_loader):
    x, y = x.cuda(), y.cuda()
    with autocast():
        loss = criterion(model(x), y) / accum_steps  # ⭐ 必须除以accum_steps！
    scaler.scale(loss).backward()  # 梯度累加到.grad

    if (step + 1) % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**✅ 梯度检查点代码（省激活显存神器）**

```python
from torch.utils.checkpoint import checkpoint

# 原理：时间换空间，前向不存中间激活，反向重跑前向算激活
# 省显存：50-80%的激活显存
# 代价：训练速度慢20-30%（重算激活）

# 写法1：函数式API
y = checkpoint(my_block, x, use_reentrant=False)  # 包耗显存的子模块

# 写法2：Sequential分块checkpoint
from torch.utils.checkpoint import checkpoint_sequential
segments = 5  # 50层分成5段，每段尾存checkpoint点
out = checkpoint_sequential(model.layers, segments, input, use_reentrant=False)
```

---

### 35. DataLoader 的 num_workers 怎么设置最优？pin_memory 原理？

**📊 num_workers 选型经验值**

| 硬件环境 | 推荐num_workers |
|---|---|
| CPU < 8核 | 2~4 |
| CPU 16核 + HDD机械盘 | 4~8 |
| CPU 32核 + NVMe SSD | 8~16 ⭐ 最常见 |
| CPU 64核 + 内存盘 | 16~32 |

**✅ 生产级DataLoader配置模板**

```python
def worker_init_fn(worker_id):
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True,
    num_workers=12,
    pin_memory=True,           # 锁页内存，DMA加速
    drop_last=True,            # 丢弃不完整batch（BN避免bs=1）
    prefetch_factor=4,         # 每worker预取4个batch（1.7+）
    persistent_workers=True,   # epoch间不销毁worker（1.8+）
    worker_init_fn=worker_init_fn,  # 保证可复现
    collate_fn=my_collate,
)
```

---

### 36. 如何判断训练瓶颈是CPU还是GPU？怎么定位？

**✅ 定位工具（从简到全）**

```bash
# 1. nvidia-smi 看GPU SM利用率（最快定位）
$ nvidia-smi dmon -s u
# gpu   sm  mem  enc  dec
#   0   40%  25%   0%   0%   ← ⚠️ sm<70% → GPU闲，瓶颈CPU/数据！
#   0   95%  60%   0%   0%   ← ✅ GPU跑满，瓶颈在GPU

# 2. PyTorch Profiler + TensorBoard火焰图（最专业）
```

**✅ PyTorch Profiler 代码**

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True, profile_memory=True,
) as prof:
    for i, (x, y) in enumerate(train_loader):
        with record_function("forward"):
            logits = model(x.cuda())
        with record_function("backward"):
            criterion(logits, y.cuda()).backward()
        optimizer.step(); optimizer.zero_grad()
        prof.step()

# Top15耗时算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# tensorboard --logdir ./log → Trace Viewer看火焰图
```

---

### 37. tensor.item()/.cpu()/print() 为什么会让训练变慢？

**⭐ 核心原因：破坏CUDA异步模型**

PyTorch CUDA操作是**异步**的：CPU发起Kernel就返回，GPU后台排队执行。
但`.item()`/`.cpu()`/`.numpy()`/`print()`/`if tensor>0:`是**GPU-CPU同步点**，CPU必须阻塞等GPU跑完才能继续 → GPU流水线断了。

**✅ 反例 vs 正例**

```python
# ❌ ❌ ❌ 慢：每步3次同步！
for x, y in loader:
    logits = model(x.cuda())
    loss = criterion(logits, y.cuda())
    print(f"loss={loss.item():.4f}")    # ⚠️ 同步点1
    acc = (logits.argmax(1)==y.cuda()).float().mean()
    print(f"acc={acc.item():.4f}")      # ⚠️ 同步点2
    all_loss.append(loss.item())        # ⚠️ 同步点3
    loss.backward()

# ✅ ✅ ✅ 快：每50步才同步1次，摊薄成本
LOG_INTERVAL = 50
step_loss = 0.0; step_acc = 0.0
for step, (x, y) in enumerate(loader):
    logits = model(x.cuda())
    loss = criterion(logits, y.cuda())
    step_loss += loss.detach()  # GPU侧累加，异步！
    step_acc  += (logits.argmax(1)==y.cuda()).float().mean().detach()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    if (step+1) % LOG_INTERVAL == 0:
        print(f"Step {step+1} loss={step_loss.item()/LOG_INTERVAL:.4f}")
        step_loss = 0.0; step_acc = 0.0
```

测算子时间的正确姿势（也是面试题）：
```python
torch.cuda.synchronize(); t0 = time.time()  # 先同步等前面跑完
y = model(x)
torch.cuda.synchronize(); dt = time.time()-t0  # 再同步等model()真跑完
```

---

### 38. in-place 操作（x+=1, add_()）有什么好处？为什么用了可能报错？

**⭐ 优点**：直接修改原Tensor内存，不分配新内存 → 省显存。

**🚨 三大雷区（和Autograd冲突时）**

```python
# 雷区1：叶子节点requires_grad=True → 直接报错
w = torch.randn(3, requires_grad=True)
w.add_(1)  # ❌ RuntimeError: leaf Variable in-place op
# Autograd需要保存叶子原值算梯度，改了就没法算了

# 雷区2：前向修改中间节点，反向重放时找不到旧值
x = torch.randn(3, requires_grad=True)
y = x * 2
y.add_(10)  # ❌ 覆盖y的内存，反向回放时y已经变了
z = y.sum(); z.backward()

# 雷区3：索引赋值in-place
x = torch.randn(3, 5, requires_grad=True)
x[:, 0] = 1.0  # ⚠️ 也是in-place修改
```

**✅ 安全使用场景**

1. 推理模式（torch.no_grad()/inference_mode()）下随便用
2. 不需要梯度的普通Tensor修改
3. nn.ReLU(inplace=True)/BatchNorm(inplace=True) 等标准Layer自带参数（已特殊处理）

最佳实践：省不了多少显存还容易Debug半天，**自己写逻辑尽量不用in-place**，标准Layer放心开inplace。

---

## 十、自定义算子与扩展开发（第39-40题）

---

### 39. 如何写自定义Autograd Function？需要实现哪两个方法？

**⭐ 标准定义**

继承`torch.autograd.Function`，手动实现：
- `forward(ctx, *args)`：前向计算逻辑，ctx.save_for_backward存反向要用的Tensor
- `backward(ctx, grad_output)`：手写梯度公式，返回对应每个输入的梯度

**✅ 代码示例：手动实现带ReLU的Linear**

```python
class MyLinearReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # input:[N, in_dim], weight:[out_dim, in_dim], bias:[out_dim]
        z = input @ weight.T + bias        # [N, out_dim]
        a = z.clamp_min(0.0)               # ReLU
        ctx.save_for_backward(input, weight, z)  # ⭐ 存反向需要的量
        return a

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = dL/dy [N, out_dim]
        input, weight, z = ctx.saved_tensors
        mask = (z > 0).float()             # ReLU梯度
        grad_z = grad_output * mask        # dL/dz
        grad_input  = grad_z @ weight      # dL/dinput [N, in_dim]
        grad_weight = grad_z.T @ input     # dL/dweight [out, in]
        grad_bias   = grad_z.sum(dim=0)    # dL/dbias [out_dim]
        # ⚠️ 返回梯度个数 = forward输入个数（input/weight/bias三个）
        return grad_input, grad_weight, grad_bias

# 调用方式：apply
y = MyLinearReLU.apply(x, W, b)

# 或封装成nn.Module
class MyLinearReLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        return MyLinearReLU.apply(x, self.weight, self.bias)
```

**📌 要点：**
- ctx.save_for_backward只能存Tensor，Python其他类型直接赋ctx.attr
- 梯度个数必须等于forward输入个数（某输入不需要梯度返回None即可）
- 用torch.autograd.gradcheck做数值梯度校验

---

### 40. PyTorch 分布式训练 DP 和 DDP 区别？为什么DDP更快？

**📊 DP vs DDP 对比（面试必考）**

| 对比项 | DataParallel (DP) | DistributedDataParallel (DDP) |
|---|---|---|
| 进程模型 | 单进程多线程（GIL限制） | 多进程（每卡1进程，无GIL） |
| 并行方式 | Parameter Server（主卡0收集） | Ring All-Reduce（所有卡平等） |
| 负载均衡 | ❌ 主卡显存+计算都更重 | ✅ 每卡完全相同工作量 |
| 通信量 | 2N次点对点（N卡数） | 2(N-1)/N（卡越多优势越大） |
| 支持多机 | ❌ 仅单机 | ✅ 单机+多机 |
| 推荐程度 | ❌ 官方不推荐 | ⭐⭐⭐⭐⭐ 生产标准 |

**✅ DDP 启动方式（torchrun）**

```bash
# 单机4卡
$ torchrun --nproc_per_node=4 train_ddp.py

# 2机4卡=8卡：Node 0 (master)
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.10 --master_port=29500 train_ddp.py

# Node 1
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.10 --master_port=29500 train_ddp.py
```

**💡 DDP为什么比DP快？三大核心原因：**

1. **多进程绕开Python GIL**：DP单进程多线程，CPU侧DataLoader/预处理串行；DDP每卡独立进程，CPU也并行
2. **Ring All-Reduce 通信更高效**：DP主卡0收所有卡梯度→算平均→分发，O(N)通信；DDP Ring AllReduce固定2(N-1)/N通信量，16卡时通信量只有DP的~12%
3. **计算与通信重叠**：DDP反向传播时，每算完一层的梯度就立刻开始bucket打包通信，和后面层的反向计算重叠，总耗时≈max(计算,通信)而不是相加