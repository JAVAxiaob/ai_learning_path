# PyTorch 性能优化重难点分析

> 位置: 03-deep-learning/pytorch-examples/

---

## 一、训练性能优化

### 1. 数据加载优化

```python
# 优化配置
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,           # 子进程数
    pin_memory=True,         # 锁页内存加速GPU拷贝
    persistent_workers=True  # 保持worker进程不销毁
)
```

### 2. AMP混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. 梯度检查点

```python
# 节省显存，适合大模型
torch.utils.checkpoint.checkpoint(model, input)
```

### 4. 分布式训练

```python
# DDP配置
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[rank])
```

### 5. FSDP（大模型专用）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

## 二、推理性能优化

### 1. 模型转换

```python
# TorchScript导出
script_model = torch.jit.script(model)
script_model.save('model.pt')

# ONNX导出
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### 2. 推理模式

```python
model.eval()

# 禁用梯度
with torch.no_grad():
    output = model(input)

# 更激进的推理模式
with torch.inference_mode():
    output = model(input)
```

### 3. 量化

```python
# 动态量化
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.ao.quantization.prepare(model)
model_prepared(input)
model_quantized = torch.ao.quantization.convert(model_prepared)
```

## 三、常见坑点

### 坑1：显存爆炸

**原因**：batch太大/中间变量太多/梯度累积

**解决方案**：
```python
# 减小batch
batch_size = 32  # 从64降到32

# 梯度累积
accum_steps = 4
for i, (data, target) in enumerate(train_loader):
    loss = criterion(model(data), target)
    loss.backward()
    
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 坑2：Loss=NaN

**原因**：梯度爆炸/除0/log负数

**解决方案**：
```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 加epsilon防止除0
loss = -torch.mean(y * torch.log(pred + 1e-8))
```

### 坑3：数据Loader卡死（Windows）

**原因**：Windows不支持多进程

**解决方案**：
```python
# Windows设置num_workers=0
train_loader = DataLoader(
    dataset,
    num_workers=0,  # Windows必须设为0
    pin_memory=True
)
```

### 坑4：DDP多卡不收敛

**原因**：SyncBatchNorm没加/LR没缩放

**解决方案**：
```python
# 加SyncBatchNorm
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model)

# LR线性缩放
lr = 1e-3 * num_gpus
```

## 四、性能监控

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"Time: {self.elapsed:.2f}s")

# 使用
with Timer():
    output = model(input)
```