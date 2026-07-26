# PyTorch 面试题汇总（35题 附详细标准答案+可运行代码）

> 位置: 03-deep-learning/pytorch-examples/
> 配套文档: PyTorch训练流程图详解.md | PyTorch性能优化重难点.md

---

## 一、基础概念与核心API（8题）

---

### Q1. PyTorch核心五大组件 + Tensor基础操作全景（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. 五大核心组件定义

| 组件 | 作用一句话 | 核心类/方法 | 类比 |
|-----|-----------|------------|------|
| **Tensor** | 多维数组，所有计算的基础载体 | `torch.Tensor` / `torch.tensor()` | NumPy ndarray + GPU |
| **Autograd** | 自动求导引擎，构建反向计算图 | `torch.autograd` / `.backward()` | 链式法则自动计算器 |
| **nn.Module** | 神经网络层/模型基类，封装参数+前向 | `nn.Module` / `nn.Linear` `nn.Conv2d` | Keras Layer |
| **Optimizer** | 根据梯度更新模型参数 | `torch.optim.SGD` / `optim.Adam` | 梯度下降策略实现 |
| **DataLoader** | 高效加载批量数据，多进程+洗牌 | `DataLoader` / `Dataset` / `Sampler` | TF Dataset + Keras Sequence |

#### 2. ⭐ Tensor 20个高频操作面试必背（代码可运行）

```python
import torch
import numpy as np

# ========== ① 创建类 ==========
a = torch.tensor([1,2,3], dtype=torch.float32)   # 从list创建 指定dtype
b = torch.zeros((2,3), device='cuda:0')          # 全0 GPU上
c = torch.ones_like(a)                           # 和a同shape全1
d = torch.randn(3,4)                             # 标准正态 N(0,1)
e = torch.arange(0, 10, 2)                       # [0,2,4,6,8] 同range
f = torch.linspace(0, 1, steps=5)                # [0.,0.25,0.5,0.75,1.] 线性等分
g = torch.from_numpy(np.array([1,2]))             # NumPy→Tensor 共享内存！

# ========== ② 形状操作 必考 ==========
x = torch.randn(2, 3, 4)              # [B=2, C=3, H=4]
x.shape                               # torch.Size([2,3,4])
x.view(2, 12)                         # 改形状 内存连续 注意reshape更通用
x.reshape(6, 4)                       # ✅ 推荐reshape：自动决定是否拷贝
x.permute(0, 2, 1)                    # 维度换位 [2,4,3] 常用BCHW→BHWC
x.unsqueeze(0)                        # 升维 [1,2,3,4] 加batch维
x.squeeze()                           # 降维 所有size=1的都去掉
x.expand(4, 3, 4)                     # 广播size=1的维度到4，不分配新内存

# ========== ③ 设备/类型转换 ==========
x_cpu = x.cpu()                       # GPU→CPU
x_gpu = x.cuda()                      # CPU→GPU（有CUDA才行）
x32   = x.float()                     # 转FP32
x16   = x.half()                      # 转FP16 显存减半
xnp   = x.detach().cpu().numpy()      # ✅ Tensor→NumPy标准写法 先detach脱离图

# ========== ④ 数学+规约 ==========
torch.mean(x, dim=1, keepdim=True)    # 沿dim1求均值 keepdim保持维度不丢
torch.sum(x, dim=-1)                  # 最后一维求和
torch.max(x, dim=1)                   # 返回 (values, indices) 两个Tensor！
torch.argmax(x, dim=-1)               # 只返回下标
torch.softmax(x, dim=1)               # 分类输出概率
torch.clamp(x, min=-1, max=1)         # 截断到[-1,1] 梯度裁剪用
x.dot(torch.tensor([...]))            # 向量点积
x @ y                                 # 矩阵乘法 等价 torch.matmul
```

#### 3. 常见坑点面试追问
- **坑1**：`torch.Tensor([1,2])` 大写Tensor = 旧API默认float32；推荐小写`tensor()`自动推断dtype。
- **坑2**：`view()` 要求内存连续（`x.is_contiguous()`），否则报错；**`reshape()`永远可以用**（不连续时自动拷贝一份）。面试写reshape更稳妥。
- **坑3**：NumPy→Tensor `from_numpy()` 是共享内存的！修改np数组Tensor也变；想独立就`.clone()`。
- **坑4**：转numpy前必须`.cpu().detach()`：`.detach()`脱离计算图避免autograd追踪，`.cpu()`保证在CPU上（GPU的Tensor不能直接转numpy）。

---

### Q2. Variable与Tensor区别 + Autograd计算图原理解密（⭐⭐⭐⭐）

**【标准答案】**

#### 1. Variable vs Tensor 历史演变

| 版本 | 状况 | 说明 |
|-----|------|------|
| PyTorch < 0.4.0 | 两个独立类 | `Variable(Tensor, requires_grad=True)` 包装Tensor才有梯度；`data`拿裸Tensor，`.grad`拿梯度 |
| PyTorch ≥ 0.4.0 | **Variable废弃** | Tensor直接支持所有属性：`x.requires_grad=True` / `x.data` / `x.grad` / `x.grad_fn` |

> 面试答：**「现在Variable已经合并进Tensor了，就像Python3把unicode和str合并了一样。面试官问这个是在考察你有没有接触过老版本，或者有没有看过PyTorch历史演进。」**

#### 2. 🔥 Autograd计算图工作原理（底层必背）

```python
import torch

# 例：y = a * b + c；求y对a,b,c的梯度
a = torch.tensor([2.0], requires_grad=True)   # 叶子节点 用户创建
b = torch.tensor([3.0], requires_grad=True)   # 叶子节点
c = torch.tensor([4.0], requires_grad=True)   # 叶子节点

y = a * b + c           # 中间结果 非叶子；grad_fn = <AddBackward>
y.backward()            # 反向传播 链式法则

print(a.grad)           # dy/da = b = 3 ✅
print(b.grad)           # dy/db = a = 2 ✅
print(c.grad)           # dy/dc = 1 ✅
print(y.grad)           # None！非叶子节点默认不存grad（省显存）
```

**核心四概念**：
| 概念 | 定义 | 上例中 |
|-----|------|--------|
| **叶子节点 (is_leaf)** | `requires_grad=True`且是用户直接创建（不是任何操作的输出） | a, b, c |
| **grad_fn** | 这个节点是由什么操作产生的？反向时从这里开始走 | y: AddBackward；a*b那个中间节点: MulBackward |
| **retain_graph** | 默认`backward()`后计算图释放；要多次backward就得传`retain_graph=True`否则第二次报错 | 高阶导数/多任务loss叠加用 |
| **grad 累加** | 每次`backward()`是**累加**梯度！不是覆盖！所以训练必须`optimizer.zero_grad()`清梯度 | 忘记清零 = 梯度越堆越大 直接爆炸 |

#### 3. 面试追问：为什么非叶子节点.grad默认是None？
> 省显存！99%的场景用户只关心模型参数（都是叶子）的梯度。真想存就调`y.retain_grad()`或者用`y.register_hook(lambda g: print(g))`钩子函数看。

---

### Q3. 完整标准训练循环 + 为什么每步顺序不能乱？（⭐⭐⭐⭐⭐ 100%必考）

**【标准答案】**

#### 1. 🔥 工业级训练循环标准7步法（顺序背死！）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ========== Step 0: 初始化 ==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))

model     = ToyNet().to(device)
criterion = nn.CrossEntropyLoss()          # 分类损失：含Softmax 不用自己加
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scaler    = torch.cuda.amp.GradScaler()    # 混合精度FP16用

train_loader = DataLoader(ToyDataset(), batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)  # 多进程加载

# ========== Step 1: 训练大循环 ==========
NUM_EPOCHS = 30
for epoch in range(NUM_EPOCHS):
    model.train()                  # ⭐0. 先切训练模式！Dropout/BN训练态
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        # ======= ⭐ 核心7步法 背顺序！ =======
        # ① 数据搬GPU
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # ② 梯度清零！🔥（放在每batch最开头；set_to_none=True更快）
        optimizer.zero_grad(set_to_none=True)

        # ③ 前向传播（mixed precision上下文FP16加速）
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        # ④ 反向传播 = 算梯度
        scaler.scale(loss).backward()          # scaler包装 防FP16下溢

        # ⑤ ⭐【可选但常考】梯度裁剪！防止梯度爆炸 RNN/LSTM必加
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ⑥ 优化器更新参数 = 用梯度改W
        scaler.step(optimizer); scaler.update()  # 混合精度版
        # optimizer.step()  # 纯FP32版

        # ========== 以下统计用 不影响训练 ==========
        total_loss += loss.item() * imgs.size(0)  # .item()拿Python标量 别存Tensor省显存
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    # ⑦ 每个epoch结束：学习率调度 + 打印日志 + 验证
    scheduler.step()                        # ⚠️ 大多数调度器写在epoch后
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Loss={total_loss/total:.4f} Acc={correct/total:.4f} "
          f"LR={optimizer.param_groups[0]['lr']:.6f}")
```

#### 2. 🔥 为什么顺序一定是「零梯度→前向→反向→更新」？不能乱？

| 如果颠倒顺序 | 后果 | 出现频率 |
|-----------|------|---------|
| ❌ `backward()`之后才`zero_grad()` | 第一次batch梯度就包含了**之前累积的** → loss巨抖不收敛 | 新手50%会犯 |
| ❌ `step()`之后才`backward()` | 先更新了参数，但那时候梯度还没算（是None/旧的）→ 模型什么都没学到 |  |
| ❌ 忘了`model.train()` | Dropout不工作/BN用了训练统计 → 训练准确率异常低 | 20%新手犯 |
| ❌ scheduler.step()放在batch循环里 | 学习率每个batch就调一次，本来想10 epoch降一次 结果几十batch就降没了 → 完全训不动 | 常见坑！ |

#### 3. 面试追问：`set_to_none=True vs =False` 区别？
> False = 梯度设成全0的Tensor（还占内存，下回还要分配）；True = 直接把grad变量设成None → 速度快约15%+省内存。PyTorch官方推荐True。

---

### Q4. model.train()/eval() + torch.no_grad()/inference_mode() 四件套深度对比（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. 四大函数作用对象全景对比表

| API | 作用对象 | 影响哪些层 | 显存/速度收益 | 什么时候调用 |
|-----|---------|-----------|-------------|------------|
| `model.train()` | 所有子Module内部flag | Dropout(启用随机失活) / BatchNorm(用当前batch均值方差 + 更新running统计) / LayerNorm(无影响) / DropPath/StochDepth | 0 | 训练epoch开始前 |
| `model.eval()` | 同上flag | Dropout(完全禁用=恒等映射) / **BatchNorm(固定running_mean/var不再更新，推理就用训练集统计的)** | 0 | 验证/测试/推理前 **必须先调！** |
| `torch.no_grad()` | Autograd引擎全局上下文 | **禁用梯度追踪**：所有操作不建计算图，不分配grad内存 | +30%速度 +50%显存 | eval模式下推理套这个 |
| 🔥`torch.inference_mode()` | Autograd更激进禁用 | 除了no_grad的所有，还禁用**视图追踪/版本计数器**等Autograd元数据 | 比no_grad再快5-10%，显存再省点 | PyTorch≥1.9推荐，**新代码一律用inference_mode代替no_grad** |

#### 2. ⚠️ 经典坑代码 vs 正确代码（面试官最爱的找茬题）

```python
# ❌❌❌ 错误写法：80%初级工程师踩过的三大错
@torch.no_grad()                       # 好，用了no_grad
def validate_wrong(model, loader):
    # model.eval()                      # 🐛 坑1：忘了切eval！Dropout还在随机丢！
    acc = 0
    for x, y in loader:
        pred = model(x.cuda())
        # loss = F.cross_entropy(pred, y); loss.backward()  # 🐛 坑2：验证也反向没必要
        acc += (pred.argmax(1)==y.cuda()).sum()
    print(acc/len(loader.dataset))
    # model.train()                     # 🐛 坑3：验证完不切回train！下一轮训练时Dropout永远关着！
```

```python
# ✅✅✅ 标准验证模板 面试直接默写
@torch.inference_mode()               # inference_mode > no_grad
def validate_correct(model, loader, device):
    was_training = model.training     # ⭐先记原来状态
    model.eval()                      # 100%先切eval！
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)             # 不用with autocast验证一般FP32稳
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.numel()
    if was_training: model.train()    # ⭐切回训练态 不破坏外部状态
    return correct / total
```

#### 3. BatchNorm面试必追问：为什么eval()很关键？
> 训练时BN每个batch算自己的μ/σ²，同时用动量EMA更新全局`running_mean = 0.9*old + 0.1*batch_mean`。
> **eval()时BN直接用训练累积的running统计量**，而不是推理那一张图的统计（单张图的均值方差毫无意义）。
> 忘了eval → BN用推理batch的均值方差 → 同一模型同一张图每次推理结果可能都不一样，准确率暴跌10-30%！

---

### Q5. view vs reshape vs permute vs transpose 四大形状操作区别（⭐⭐⭐⭐）

**【标准答案】**

#### 核心区别表（面试画这张表直接满分）

| API | 是否改变数据内存顺序？ | 是否要求连续contiguous？ | 会拷贝数据吗？ | 典型用途 |
|-----|---------------------|----------------------|-------------|---------|
| `x.view(new_shape)` | ❌ 不改内存，只改stride元数据 | ✅ **必须连续** 不满足直接报错RuntimeError | ❌ 永远view共享内存 | 把特征展平 [B,C,H,W]→[B, C*H*W] |
| 🔥`x.reshape(new_shape)` | ❌ 也不改 | ❌ 不要求；不连续时**自动**先`.contiguous()`（这一步会拷贝）再view | ⚠️ 可能拷贝可能不 | **万能推荐** 什么时候都能写 不用想连续 |
| `x.permute(dims)` | ✅ 换维度顺序 = 改内存语义（通道维换位置） | ❌ 输出一定非连续！之后想.view必须先.contiguous() | ❌ 只改stride | BCHW ↔ BHWC / NLP [L,B,D]↔[B,L,D] |
| `x.transpose(dim0, dim1)` | ✅ 交换两个维度 permute的简化版 | 同上permute非连续 | 同上 | 转置矩阵 [3,4]→[4,3] |

```python
import torch
x = torch.randn(2, 3, 4)

# ========== reshape vs view ==========
y = x.reshape(2, 12)                     # ✅ OK 万能
try:
    z = x.permute(0,2,1).view(2, 12)     # ❌ Error！permute完不连续
except RuntimeError as e:
    print(e)  # view size is not compatible

z_fix = x.permute(0,2,1).contiguous().view(2, 12)  # ✅ 先contiguous再view
z_or  = x.permute(0,2,1).reshape(2, 12)            # ✅ reshape自动做，更简洁

# ========== 共享内存 vs 拷贝 ==========
a = torch.arange(6).view(2,3)
b = a.reshape(3,2)
a[0,0] = 999
print(b[0,0])  # 999！reshape在连续时确实共享内存
```

#### 面试追问：什么是contiguous？底层stride怎么理解？
> Tensor实际是一维内存+shape/stride元数据。`stride[i]`=第i维每加1，内存指针需要跳多少个float。
> `view`只改shape/stride，要求stride符合「降维紧凑排列」；permute把stride打乱了就不符合 → 非连续。
> 别深记，一句话：**写reshape永远不会错**。

---

### Q6. torch.gather / scatter / index_select 高级索引三剑客（⭐⭐⭐⭐）

**【标准答案】**

#### 1. 三大高级索引使用场景+代码

```python
import torch

# ======================================
# ① gather: 按index在指定dim上「收集」元素
#    经典用：分类任务取 正确类别的logit值 （NLP beam search也常用）
# ======================================
B, C = 3, 5
logits = torch.randn(B, C)         # 3样本5分类 [ [2.1, 0.3, -1, 0.5, -0.2], ... ]
labels = torch.tensor([2, 0, 4])   # 正确类别下标

# 想取：第0样本的第2类值 / 第1样本的第0类值 / 第2样本的第4类值
# gather要求：index的shape必须 和 输出shape相同；除了dim其他维度要对齐broadcast
correct_logits = torch.gather(logits, dim=1, index=labels.unsqueeze(1)).squeeze(1)
print(correct_logits)  # 就是CrossEntropyLoss内部用来算的那个标量序列

# ======================================
# ② scatter: gather的反向操作 = 按index「散布」填入Tensor
#    经典用：one-hot编码 手动写 / 混淆矩阵统计
# ======================================
one_hot = torch.zeros(B, C, dtype=logits.dtype)
one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1.0)  # ⚠️ 带下划线_ = 原地操作in-place
print(one_hot)  # labels=[2,0,4] → 一行一个1在对应列

# ======================================
# ③ index_select: 按index列表 取整行/整列/整块（比gather简单，dim维度挑若干条）
# ======================================
all_features = torch.randn(1000, 768)   # 1000个样本特征
idx = torch.tensor([0, 15, 333, 999])   # 想挑这4个样本的整行
picked = torch.index_select(all_features, dim=0, index=idx)  # shape [4,768]
# 等价 fancy索引：all_features[idx]  但index_select更快且不会触发高级索引拷贝警告
```

#### 2. gather最容易写错的地方
- 坑：`index`的dtype必须是`torch.int64`(long)！int32直接报错。CV/NLP里常从numpy拿的index可能是int32 → `.long()`转一下。
- 坑：index的shape！要在`dim`维取k个值，index的dim维大小就得是k，其他维度得和原tensor一样或能广播。
- 面试加分：带下划线`scatter_() / add_() / mul_()` = PyTorch约定俗成的**原地in-place操作**，直接改调用者不返回新Tensor → 省显存但有Autograd视图风险（叶子节点别原地改值会报错）。

---

### Q7. nn.Module的parameters/buffers子模块管理 + state_dict/checkpoint（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. Module内的三类"东西" 必考区分

| 类别 | 是什么？ | 例子 | 是否参与梯度更新？ | 是否被`state_dict()`收录？ |
|-----|---------|------|------------------|-------------------------|
| **Parameters** | 可训练参数 用`nn.Parameter()`包 | Linear.weight / Conv.bias | ✅ 自动算grad 被optimizer.step()更新 | ✅ |
| **Buffers** | 不变/不由梯度更新的张量 | BatchNorm.running_mean / running_var / PositionalEncoding | ❌ 不参与梯度 | ✅（保存/加载要一起！） |
| 普通属性（Python list/dict/int） | Python变量 | `self.drop_p = 0.5`超参 / 用普通list存的子模块 | ❌ | ❌ **用普通list存子Module = 不被model.parameters()收录！训不动！** |

```python
import torch
import torch.nn as nn

# ========== 🔥 Buffer 正确注册 vs 错误写法 ==========
class MyBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # ✅ 正确：用register_buffer 会被state_dict收录、自动to(device)
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var',  torch.ones(dim))
        # ❌ 错误：self.running_mean = torch.zeros(dim)
        #   坑：model.cuda()时这个tensor不会自动搬去GPU！存checkpoint也不会被存！

        # ⭐ Parameter = 默认可训练
        self.gamma = nn.Parameter(torch.ones(dim))   # 可训练缩放
        self.beta  = nn.Parameter(torch.zeros(dim))  # 可训练偏移

# ========== 🔥 子Module 别用Python list！用nn.ModuleList / nn.Sequential ==========
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10,10), nn.ReLU()]  # ❌ Python list！
        # 坑：len(list(model.parameters())) == 0！训不了！

class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10,10), nn.ReLU()])  # ✅ ModuleList
        # 或者 nn.Sequential(nn.Linear(10,10), nn.ReLU()) ✅ Sequential自带forward
    def forward(self, x):
        for l in self.layers: x = l(x)  # ModuleList要自己写forward
        return x

# ========== state_dict = 整个模型的参数字典 ==========
model = GoodNet()
sd = model.state_dict()   # 键是"层级路径.name"如 "layers.0.weight" / "layers.0.bias"
for k, v in sd.items():
    print(f"{k:25s} shape={list(v.shape)} dtype={v.dtype}")
```

#### 2. 🔥 Checkpoint 保存/加载的三大标准写法

```python
PATH = "checkpoint.pt"

# ========== ✅ 标准1：只存参数（推荐 体积小 通用） ==========
torch.save(model.state_dict(), PATH)

# 加载：先实例化模型结构 再load_state_dict
model2 = GoodNet()
state = torch.load(PATH, map_location='cpu')  # ⭐ map_location='cpu' 跨GPU/CPU加载不报错
missing, unexpected = model2.load_state_dict(state, strict=False)
# strict=False：多的少的key不报错 打印出来手动排查；strict=True默认完全匹配否则报错
print(f"加载情况：缺失={missing} 多余={unexpected}")

# ========== ✅ 标准2：训练断点恢复全量（存epoch/optimizer/scheduler/loss） ==========
checkpoint = {
    'epoch': epoch,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss':        best_loss,
}
torch.save(checkpoint, PATH)

# 恢复
ckpt = torch.load(PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])  # ⭐连Adam动量状态也恢复！
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch'] + 1

# ========== ❌ 反模式：torch.save(model, PATH) 存整个对象 ==========
# 坑：加载时要求Python路径、类名、导入方式必须完全一样；换了代码结构直接无法加载！
```

#### 面试追问：`strict=False`什么时候用？
> 迁移学习/微调：加载预训练 backbone 但分类头换了（类别数不一样）→ strict=False让它跳过不匹配的head层。打印missing看是不是只剩分类头没加载 → 正常。

---

### Q8. 注册钩子Hook：forward/backward钩子原理 + 特征可视化/梯度裁剪实战（⭐⭐⭐⭐）

**【标准答案】**

#### 1. 三大钩子类型及使用场景

| Hook类型 | 注册API | 触发时机 | 能拿到什么 | 经典用途 |
|---------|--------|---------|-----------|---------|
| **Forward钩子** | `module.register_forward_hook(fn)` | module前向计算完后立即触发 | (module, input, output) 输入输出Tensor | 🔥取中间层特征图 / 特征可视化 / 打印shape调试 / 统计激活分布 |
| **Backward钩子** | `module.register_full_backward_hook(fn)` | module反向梯度算完后触发 | (module, grad_input, grad_output) 对输入的梯度/对输出的梯度 | 梯度范数统计 / 梯度裁剪 / 检测梯度消失爆炸 |
| Tensor钩子 | `tensor.register_hook(fn)` | 某个Tensor梯度算出来后 | 该Tensor的grad值 | 修改特定参数的梯度 / 打印某个叶子的grad |

```python
import torch
import torch.nn as nn

# ========== 实战1：Forward钩子 拿ResNet中间层feat做可视化 ==========
class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(32, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 想抓这个输出
        x = torch.relu(self.conv2(x))  # 想抓这个输出
        return self.fc(self.pool(x).flatten(1))

model = DemoNet()
# 字典存所有层输出
activations = {}
def get_act_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()  # 记得detach不然显存爆
    return hook

# 给两个conv层都挂hook
h1 = model.conv1.register_forward_hook(get_act_hook("conv1"))
h2 = model.conv2.register_forward_hook(get_act_hook("conv2"))

# 跑一次，自动触发hook填充activations
with torch.inference_mode():
    out = model(torch.randn(2, 3, 32, 32))
print(activations["conv1"].shape)  # [2,16,32,32] ✅ 拿到中间特征了

h1.remove(); h2.remove()  # 用完卸钩子 不然后面每次都跑 越来越慢

# ========== 实战2：Backward钩子 监控梯度范数 ==========
grad_norms = {}
def grad_norm_hook(name):
    def hook(module, grad_input, grad_output):
        # grad_output是 对module输出的梯度 一般看这个判断流进来多少梯度
        if grad_output[0] is not None:
            grad_norms[name] = grad_output[0].norm().item()
    return hook

for n, m in model.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.register_full_backward_hook(grad_norm_hook(n))

# 跑一次训练后grad_norms就有所有层的梯度范数了
# 可视化：最底层conv梯度范数≈0 = 梯度消失！
```

#### 2. 面试追问：Hook用了忘记remove会怎么样？
> 钩子是全局注册的，不remove每跑一次forward/backward都会执行一次，越积越多越跑越慢、最后内存泄漏。
> 最佳实践：**用try-finally / 上下文管理器** 保证必remove。或者用`with torch.no_grad():`这种天然临时的机制代替。

---

## 二、损失函数与优化器（6题）

---

### Q9. CrossEntropyLoss/BCEWithLogitsLoss内部原理 + 为什么不用自己加Softmax？（⭐⭐⭐⭐⭐ 必考）

**【标准答案】**

#### 1. 三大分类损失对比表 + 计算拆解

| 损失 | 适用任务 | 输入要求（最后一层） | 内部做了什么 | 公式一句话 |
|-----|---------|-------------------|-----------|-----------|
| 🔥 `nn.CrossEntropyLoss` | **多分类**（互斥，10选1） | 输出logits，shape=[B,C]，**⚠️不要加Softmax！** | `LogSoftmax` + `NLLLoss` 两步合并 | `-log( exp(logits[y_i]) / Σ_j exp(logits[j]) )` |
| 🔥 `nn.BCEWithLogitsLoss` | **多标签分类**（不互斥，每个标签独立是/否）/二分类 | 输出logits，shape=[B,num_labels]，**⚠️不要加Sigmoid！** | `Sigmoid` + `BCELoss` 合并 | 每个label独立做 `-y·logσ(x) - (1-y)·log(1-σ(x))` |
| `nn.NLLLoss` | 多分类（你自己先过了LogSoftmax） | 输入是LogSoftmax后的结果 | 只取正确类那列的负值求和 | `-input[class_idx]` |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== CrossEntropyLoss 拆解验证 ==========
logits = torch.tensor([[2.0, 0.5, -1.0]])   # [B=1, C=3] 三分类
labels = torch.tensor([0])                   # 正确类是第0类

# 方式1：官方一站式（生产必用，数值稳定）
ce_loss = nn.CrossEntropyLoss()(logits, labels)

# 方式2：手动拆开算 验证等价
log_softmax = F.log_softmax(logits, dim=1)   # 先LogSoftmax（比先Softmax再log数值稳）
nll_loss    = nn.NLLLoss()(log_softmax, labels)  # 再NLL = 取正确类那列负值
torch.allclose(ce_loss, nll_loss)  # True ✅ 等价

# ❌❌❌ 致命错误：自己又加了一次Softmax！
# logits → Softmax → CrossEntropy内部又LogSoftmax → 相当于Softmax两次！梯度极小完全训不动
loss_bug = nn.CrossEntropyLoss()(F.softmax(logits, dim=1), labels)
# 这种情况验证集准确率卡在1/C（随机猜的概率）不动 直接看loss代码！

# ========== BCEWithLogitsLoss：多标签分类 ==========
logits_multi = torch.tensor([[1.5, -0.3, 2.0]])   # 3个独立标签 正负不相关
labels_multi = torch.tensor([[1.0, 0.0, 1.0]])    # 第0和第2个标签是正
bce_loss = nn.BCEWithLogitsLoss()(logits_multi, labels_multi)
# 同样不要自己加Sigmoid再BCEWithLogits 错误同上！
```

#### 2. 🔥为什么内部要Softmax+NLL合并写？两大原因（面试答到就加分）
1. **数值稳定性**：`log(sum(exp(x)))`直接算大值会溢出。内部用`max`技巧：`x_max + log(sum(exp(x - x_max)))`，保证任何情况都不NaN。
2. **梯度公式化简**：合并后梯度是`Softmax(logits) - one_hot(y)`，一步直接算；分开写Softmax反向有除法，容易数值下溢。

#### 面试追问：MSE能不能用于分类？
> 能但不推荐。MSE假设高斯噪声（回归适用），分类是伯努利/多项式分布 → 用交叉熵（MLE等价）是统计上正确的损失。且MSE对分类是**非凸**的，优化困难；交叉熵梯度在错的时候大、对的时候小，更符合直觉（错得多就更新得狠）。

---

### Q10. 七大优化器SGD/Momentum/NAG/Adagrad/RMSprop/Adam/AdamW对比表（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. 七大优化器核心对比（面试画这张表封神）

| 优化器 | 全称一句话 | 超参 | 凸/非凸收敛性 | 适合场景 | 主要问题 | 一句话记忆点 |
|-------|-----------|------|-------------|---------|---------|-------------|
| **SGD** (plain) | vanilla 随机梯度：W -= lr·g | lr | 凸O(1/√T) | 所有场景baseline | 鞍点/峡谷震荡、选lr累 | 纯裸奔 |
| + **Momentum** SGD | 加动量Vt=βV + (1-β)g；W-=lr·V | lr, β≈0.9 | 同上+ 加速峡谷 | 通用；CV常用 配合Nesterov | 冲过极小点 | 山坡下滚 带惯性 |
| + **NAG** | Nesterov：先算"预估走一步后"的梯度 | 同上 | 凸问题理论加速 | 少用，被Adam淘汰 |  | 向前看一步再滚 |
| **Adagrad** | 每个参数自适应lr：累加历史梯度平方倒数开方 | lr | 凸O(1/√T) | 稀疏特征NLP/CTR | lr非增，训到后期太小完全停 | 稀疏好 后期萎 |
| **RMSProp** | Adagrad改进：梯度平方用EMA指数移动平均（不再单调增） | lr, α≈0.9 | 非凸实践好 | RNN/LSTM基准 | 无偏修正缺失 | Adagrad加了滑动窗 |
| 🔥 **Adam** | RMSProp + Momentum 双EMA + 偏置修正 (1-β^t) | lr, β1=0.9, β2=0.999, ε=1e-8 | 非凸最主流 | **通用默认首选** 99%任务开箱即用 | 可能不收敛 / 泛化比SGD+M差1% | 双动量+修偏 全能选手 |
| 🔥 **AdamW** | Adam + 解耦权重衰减（原Adam的weight_decay写法错了！） | 同上 + weight_decay=1e-4/1e-2 | 同上但泛化明显更好 | **🔥现在推荐用AdamW代替Adam** |  | Adam但weight decay方式正确 |

```python
import torch

# ========== 工业界推荐三种配置 背下来 ==========
# ① 首选 AdamW（Transformer/LSTM/MLP 99%情况）
opt1 = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,          # MLPerf标准
    betas=(0.9, 0.999),
    weight_decay=1e-2  # ⭐关键：AdamW必须配weight_decay！别用0也别太大
)

# ② CV 图像任务经典 SGD+Momentum（泛化性略好于Adam 代价是调参）
opt2 = torch.optim.SGD(
    model.parameters(),
    lr=0.1,           # SGD lr要比Adam大10-100倍 这是新手常错！
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True     # 加Nesterov提0.5% 成本0
)

# ③ 🔥 进阶：分组参数设置（分类头lr×10，bias不weight decay）
def build_group_optimizer(model, base_lr=1e-3, wd=1e-2):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.endswith(".bias") or "bn" in name or "LayerNorm" in name:
            no_decay.append(p)      # bias/BN/LN参数不加权重衰减 业界共识
        else:
            decay.append(p)
    groups = [
        {"params": decay,    "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr)
```

#### 2. AdamW vs Adam面试必背公式差别
- ❌ 旧Adam的`weight_decay`实现：`g_t = g_t + weight_decay · w_t` → 先把wd加到梯度里，然后除以Adam自适应lr（除以v的开方）→ 自适应后wd效果被削弱了，达不到正则化目的！
- ✅ AdamW解耦：**`w_t = w_t - lr · (m_t_hat / (sqrt(v_t_hat)+ε) + weight_decay · w_t)`** → wd直接乘权重独立减，不经过自适应lr归一化 → 正则作用稳定，泛化能力大大提升。
- 面试一句话：**「AdamW是现在的标配，比旧Adam泛化好1%左右，必须配合weight_decay=1e-2左右使用。」**

---

### Q11. 学习率调度器七大种 + Warmup必要性 + 写在batch还是epoch后？（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. 主流调度器对比表

| 调度器 | 公式一句话 | 曲线形状 | 适用场景 | 标准参数 |
|-------|-----------|---------|---------|---------|
| `StepLR` | 每N步 ×gamma | 阶梯状递减 | 通用baseline，分类最常见 | step_size=30 epoch, gamma=0.1 |
| `MultiStepLR` | 在指定milestone节点乘gamma | 自定义阶梯 | 已知收敛节点 | milestones=[60,80], gamma=0.1 (ResNet论文) |
| `CosineAnnealingLR` | 余弦函数从lr_max降到0（或min_lr） | 平滑U型 | 🔥CV/NLP全能首选 泛化略好 | T_max=总epochs，eta_min=1e-6 |
| `ReduceLROnPlateau` | 监控val_loss/acc N个epoch不下降才降 | 自适应形状 | ⭐小数据集/调参复杂任务 | mode='min', patience=5, factor=0.5 |
| `OneCycleLR` | 先线性升到max再cos降下来（Super-Convergence） | 驼峰形 | 🔥小epoch 快速收敛 4倍加速 | max_lr=lr×10, total_steps=总batch数 |
| `LinearWarmupCosineAnnealingLR` | 先Warmup k个epoch升lr → 再余弦退到0 | 🔥NLP/Transformer标配 | BERT/ViT/GPT训练 | warmup_epochs=5~10%总epoch |
| `ExponentialLR` | 每个epoch × gamma | 指数衰减 | 少见 | gamma=0.95 |

```python
import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR)

# ========== ① 最流行 CosineAnnealing + Warmup 手动版 ==========
EPOCHS = 30; WARMUP = 3
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

def get_lr(epoch):  # 返回当前的lr系数（乘初始lr）
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP     # 线性warmup：第0 epoch lr=lr/3 第2=2lr/3 第3=lr
    return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP) / (EPOCHS - WARMUP)))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=get_lr)

# 训练中：
for epoch in range(EPOCHS):
    train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step()   # ⭐ Cos/Step/MultiStep 这类按epoch调度的写在epoch后

# ========== ② ReduceLROnPlateau 监控val_loss ==========
scheduler2 = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
scheduler2.step(val_loss)   # ⚠️ 必须把要监控的指标传进去！

# ========== ③ OneCycleLR 按batch调度 ==========
total_steps = len(train_loader) * EPOCHS
scheduler3 = OneCycleLR(opt, max_lr=1e-2, total_steps=total_steps)
for batch_idx, _ in enumerate(train_loader):
    train_step(...)
    scheduler3.step()  # ⚠️ OneCycle是按batch调的！写在batch循环里！
```

#### 2. 🔥Warmup为什么必要？面试灵魂三问
1. **为什么前几个epoch要小lr慢慢升？** → 初始参数是随机的，一开始梯度范数很大（因为算出来的loss巨大），直接大步学把预训练参数/初始状态搞坏了；小lr"热身"让参数进入合理范围再正常学习率训练。
2. **Transformer为什么必须Warmup？** → 原论文公式`lr = d_model^-0.5 · min(step^-0.5, step·warmup_steps^-1.5)`，不Warmup的话Attention模块输出巨大数值直接NaN/Inf训崩。
3. **写在batch还是epoch？** → 记住口诀：**「按总steps数变化的（OneCycle/LinearWarmupCosine）写在batch后；按时间周期变化的（Step/Cosine/ReduceLROnPlateau）写在epoch后。」**

---

### Q12. 梯度裁剪clip_grad_norm_ / clip_grad_value_ 原理 + RNN必加（⭐⭐⭐⭐）

**【标准答案】**

#### 1. 两种梯度裁剪对比

| 方式 | API | 做什么 | 对梯度方向影响 | 适用场景 | 推荐参数 |
|-----|-----|-------|---------------|---------|---------|
| 🔥 **范数裁剪（推荐）** | `nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type=2)` | 算所有参数梯度拼成的大向量的L2范数；若>max_norm → 整体等比例缩小 `g = g * max_norm / ||g||` | ✅ 不改变梯度方向，只缩放大小 | **通用首选**；RNN/LSTM/Transformer训练标配 | 1.0 / 5.0 保守值 |
| **值裁剪** | `nn.utils.clip_grad_value_(params, clip_value=0.5)` | 每个梯度分量单独截断到 [-clip_value, +clip_value] | ❌ 破坏梯度方向（某个分量被截其他没截=方向变了） | 少用；梯度极端尖刺时 | 0.5~1 |

```python
# ========== 标准写法位置：loss.backward()之后 optimizer.step()之前 ==========
for imgs, labels in loader:
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        loss = criterion(model(imgs), labels)
    scaler.scale(loss).backward()

    # ⭐⭐⭐ 梯度裁剪必须在backward之后、step之前！
    # scaler.unscale_ 先反放缩回真实梯度再剪（混合精度必须！否则剪的是被scale×1024的梯度）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

#### 2. 面试追问：为什么RNN/Transformer比CNN更需要梯度裁剪？
> RNN时间步共享权重，BPTT沿时间反向求导 = 同一个W多次相乘 → `W^T` T=100的话：
> - W的特征值>1 → 梯度爆炸到NaN
> - W的特征值<1 → 梯度消失到0
> LSTM门控+梯度裁剪 = 工业界解决RNN梯度爆炸的标准组合，没加clip_norm的LSTM 80%概率训崩过NaN。
> Transformer虽然用残差+LayerNorm缓解了，但仍可能因为初始化/特殊token出现梯度尖刺，加clip_norm几乎是零成本。

---

### Q13. 权重衰减Weight Decay vs L2正则化 数学等价但Adam中不一样（⭐⭐⭐⭐）

**【标准答案】**

#### 1. 标准SGD+Momentum下的数学等价性
| 理论L2正则（加loss项） | 工程实现 Weight Decay（直接减参数） |
|---|---|
| `L_total = L_data + (λ/2)·‖w‖²` | `w_t+1 = w_t - lr·(g_t + λ·w_t)` |
| 对w求导 ∂L/∂w = g_t + λ·w | 代入SGD：w ← w - lr(g_t+λw) = (1-lrλ)w - lrg_t |
| ✅ SGD下 两种方式完全等价 λ = weight_decay |  |

#### 2. 🔥 在Adam/RMSProp自适应优化器中 不等价！这就是AdamW提出的原因
- 旧PyTorch的Adam的`weight_decay`是写进梯度的：`g_t = g_t + wd·w_t` → 下一步会被Adam的分母√v_t归一化除一下 → 权重衰减的实际效果被自适应lr抵消了一部分；lr自适应越小的参数（稳定少变的）衰减越弱 → 正则化效果大打折扣。
- ✅ AdamW解耦：`w ← w - lr·(adam_update(w, g_t) + wd·w_t)` → wd直接对w本体生效，不走m/v归一化 → 正则化效果稳定如预期。

**面试一句话总结**：**「SGD加weight_decay=L2正则等价；Adam加weight_decay不是！要用AdamW这个专门解耦的实现，否则白加正则了。」**

---

### Q14. Focal Loss解决什么问题？公式 + α平衡/γ难易样例 面试手写（⭐⭐⭐⭐）

**【标准答案】**

#### 1. Focal Loss背景与公式
**解决的问题**：目标检测/分类中正负样本极度不平衡（1个正 vs 10000个负）+ 大部分负样本是「简单易分」的 → 它们损失虽然小但数量巨大，总和淹没正样本贡献 → 模型学不到。

**标准CE二分类**：`CE = -[y·logσ(x) + (1-y)·log(1-σ(x))] = -log(p_t)，p_t = 模型对正确类的概率`

**🔥 Focal Loss**：`FL = -α_t · (1-p_t)^γ · log(p_t)`

两项超参分工：
| 超参 | 作用 | 推荐值 |
|-----|------|-------|
| α (alpha) | 平衡正负样本数量：正样本少就α大 | α=0.25（正少）/ 看正负比调 |
| γ (gamma) > 0 | 降低「简单样本(p_t≈1)」的权重，只让「难例(p_t小)」主导梯度；γ越大对难例越专注 | γ=2.0 原论文最佳 |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Focal Loss 标准实现 面试手写 ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha       # 正负平衡因子
        self.gamma = gamma       # 难例聚焦因子
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C] 多分类；[B,1]或[B]二分类
        # targets: [B] 类别下标 0..C-1
        B, C = logits.shape

        # ① pt = 模型对正确类的概率
        log_pt = F.log_softmax(logits, dim=1)                # [B,C] LogSoftmax
        pt     = torch.exp(log_pt)                            # [B,C] 概率
        log_pt_target = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B] 正确类的logp
        pt_target     = pt.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B] 正确类的p

        # ② α_t：正类用α 负类用1-α；多分类可扩展成α[C]数组每类自己的权重
        alpha_t = torch.where(targets > 0, self.alpha, 1.0 - self.alpha)

        # ③ Focal 主体 = α_t × (1-p_t)^γ × (-log_p_t)
        focal_weight = alpha_t * (1 - pt_target) ** self.gamma
        loss = focal_weight * (-log_pt_target)

        return loss.mean() if self.reduction == 'mean' else loss.sum()

# 快速验证：pt大的（简单样例）loss会被压得很小
fl = FocalLoss(alpha=0.25, gamma=2.0)
logits_test = torch.tensor([[3.0, -1.0, 0.5], [0.1, 0.2, 0.0]])  # 第1样本很确定 第2不确定
labels_test = torch.tensor([0, 1])
print(fl(logits_test, labels_test))
```

#### 2. 面试追问：γ=0 FocalLoss退化成什么？
> γ=0 → `(1-p)^0=1` → 只剩`-α·logp` = 加权CrossEntropyLoss（α平衡正负的CE）。所以Focal Loss就是CE加了**可学习的样本难度权重**。

---

## 三、数据加载与增强（4题）

---

### Q15. Dataset + DataLoader + Sampler + CollateFn 四件套工作机制（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. 四件套定义与职责链路

```mermaid
graph LR
A[Dataset.__getitem__] --> B[Sampler决定取哪些idx的顺序 shuffle/sorted]
B --> C[多Worker进程 并行取数据 num_workers>1]
C --> D[collate_fn 把list[sample]拼成batch Tensor + padding]
D --> E[训练循环for batch in DataLoader 每次出一组]
```

| 组件 | 负责什么 | 默认行为 | 自定义场景 |
|-----|---------|---------|-----------|
| `Dataset` | 定义「第idx个样本长啥样」，`__len__`/`__getitem__` | - | 从磁盘读图+label / 文本tokenize |
| `Sampler` | 定义「每个epoch 样本idx的取出顺序」 | `SequentialSampler` / shuffle=True时`RandomSampler` | 🔥按序列长度分桶采样（BucketSampler NLP少padding） / 类别不平衡加权WeightedRandomSampler |
| **num_workers** | 开多少个独立子进程 并行跑Dataset.__getitem__填队列 | 0（主进程自己干 慢） | Linux CPU核数一半 或 batch_size相关；**Windows必须0否则卡死** |
| **collate_fn** | 把worker返回的list[dict/Tensor] **拼成一个batch的大Tensor** | 默认：stack同shape 自动加batch维 | 🔥NLP不等长句子padding到最长 + 生成attention_mask / 目标检测多size图片padding / MixUp/CutMix在这一步做 |

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

# ========== 例：文本分类 自定义Dataset + CollateFn（NLP必考）==========
class TextCLSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        # 单个样本：不等长！所以不能直接stack，要靠collate_fn pad
        encoded = self.tok(self.texts[idx], truncation=True, max_length=self.max_len)
        return {"input_ids":  torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

def collate_fn_padding(batch_samples: list[dict]):
    # 手动pad 不等长 input_ids 到当前batch的最长
    ids   = [s["input_ids"]  for s in batch_samples]
    masks = [s["attention_mask"] for s in batch_samples]
    labels = torch.stack([s["label"] for s in batch_samples])

    padded_ids   = torch.nn.utils.rnn.pad_sequence(ids,   batch_first=True, padding_value=0)
    padded_masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return {"input_ids": padded_ids, "attention_mask": padded_masks, "labels": labels}

# ========== WeightedRandomSampler：类别不平衡（1:99）给少的类加采样权重 ==========
labels_np = np.array([0]*9900 + [1]*100)  # 1%正例 极度不平衡
class_counts = np.bincount(labels_np)
class_weights = 1.0 / class_counts         # 少的类权重高 9900:1
sample_weights = class_weights[labels_np]  # 每个样本的权重
sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                num_samples=len(labels_np),  # 抽多少条
                                replacement=True)            # 有放回抽才能加权重

# 组装DataLoader
loader = DataLoader(
    TextCLSDataset(...),
    batch_size=32,
    sampler=sampler,               # ⚠️ sampler和shuffle互斥！指定了sampler就不能shuffle=True
    num_workers=4,                 # Linux=4；Windows一定写0！
    pin_memory=True,               # ⚠️ 配合to(device, non_blocking=True)异步搬GPU更快
    collate_fn=collate_fn_padding, # 自定义padding
    drop_last=False,               # 最后不够batch_size的要不要扔掉？训练可True 验证False
    prefetch_factor=2,             # 每个worker预取2个batch（num_workers>0才生效）
)
```

#### 2. 常见坑点
- 坑1 Windows：**num_workers>0 必卡死**（Windows多进程spawn方式会重新import整个模块触发嵌套）→ 必须0。
- 坑2 sampler与shuffle互斥：传了自定义sampler（如上WeightedRandomSampler）就不能写shuffle=True，否则报错。
- 坑3 pin_memory + non_blocking配套：`pin_memory=True`把数据锁页内存（不被swap到硬盘）→ `imgs.to(device, non_blocking=True)` CPU→GPU拷贝和GPU计算异步重叠 → 提速约15%，零成本。

---

### Q16. torchvision八大增强 + RandAugment/AutoAugment + MixUp/CutMix（⭐⭐⭐⭐）

**【标准答案】**

#### 1. 标准训练增强流水线（ImageNet分类）

```python
from torchvision import transforms as T
import torch

# ========== 训练集：强组合增强 ==========
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),  # ⭐随机裁剪+缩放
    T.RandomHorizontalFlip(p=0.5),                                     # 水平翻转 无信息损
    T.TrivialAugmentWide(),                                            # ⭐🔥PyTorch 1.13+ 官方现成的轻量AutoAugment
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet统计量
    T.RandomErasing(p=0.5, scale=(0.02, 0.20)),                       # 随机擦一块 提升遮挡鲁棒性
])

# ========== 验证/测试集：只有CenterCrop + Normalize（确定性，保证两次推理结果相同）==========
val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),                       # ⚠️验证/测试一定用CenterCrop 不能Random！
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 和训练完全相同的Normalize！
])
```

#### 2. MixUp / CutMix 正则化技术 面试手写（在collate_fn里做）

| 技术 | 操作一句话 | 标签处理 | 图像形状变化 |
|-----|-----------|---------|-------------|
| **MixUp** (ICLR18) | 两张图按 λ 线性加权混合像素 | 标签也按 λ 加权（软标签） | 不变，逐像素加和 |
| **CutMix** (ICCV19) | 从一张图切一块矩形区域 贴到另一张图对应位置 | 标签按面积比例加权 | 不变，像素替换 |
| CutOut / RandomErasing | 单张图随机挖一块 填0/均值 | 标签不变 | 不变 |

```python
# ========== MixUp 标准实现（lambda~Beta(α,α)分布 α=0.2常见） ==========
def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)   # λ∈[0,1] α=0.2 大多数时候接近0或1
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)  # 打乱配对 找另一张图
    mixed_x = lam * x + (1 - lam) * x[index, :]    # 图像加权混合
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 训练循环中：
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    if np.random.rand() < 0.5:     # 一半概率做MixUp
        x, ya, yb, lam = mixup_data(x, y, alpha=0.2)
        loss = mixup_criterion(criterion, model(x), ya, yb, lam)
    else:
        loss = criterion(model(x), y)
    ...
```

#### 面试追问：Normalize的mean/std是ImageNet的，但我训自己的数据集怎么办？
> **正确做法**：在自己的训练集上算一遍所有图片RGB三通道的均值和方差 → 用这组值Normalize。只有迁移学习加载ImageNet预训练权重时才必须用ImageNet的mean/std（和预训练时预处理一致，预训练模型才能认出输入）。

---

## 四、分布式训练与混合精度（6题）

---

### Q17. DP vs DDP对比 + DDP启动三方式（⭐⭐⭐⭐⭐）

**【标准答案】**

#### 1. DP(DataParallel) vs DDP(DistributedDataParallel) 灵魂对比表

| 维度 | DP (nn.DataParallel) | 🔥 DDP (nn.parallel.DistributedDataParallel) |
|-----|----------------------|-------------------------------------------|
| **进程模型** | ❌ **单进程 多线程**（Python GIL锁的锅！并行度差） | ✅ **多进程**，每张GPU一个独立进程，无GIL |
| **梯度同步** | rank0主GPU收集所有GPU梯度 计算平均 → 主GPU更新 → 广播权重给其他GPU（主从瓶颈） | ✅ 所有GPU同时 **Ring AllReduce 梯度同步**，每卡都算所有卡平均，无中心瓶颈 |
| **速度** | 2卡≈1.4x 4卡≈2.2x（递减严重 DP的50%效率天花板） | ✅ 2卡≈1.9x 4卡≈3.7x 8卡≈7.4x（95%近线性加速！） |
| **多机** | ❌ 只能单机器 | ✅ 支持多机多卡 TCP/NCCL通信 |
| **BN同步** | ❌ 每张卡各自BN统计不准（batch被拆） | ✅ `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` 全卡统计BN |
| **官方推荐** | ❌ 官方文档已经标注"legacy遗留不更新" | ⭐✅ 官方唯一推荐分布式训练方式 |
| **代码改动** | 极少：`model = nn.DataParallel(model)` 一行 | 中等：要加启动/初始化/分散数据/Sampler |

#### 2. DDP最简启动 + 代码模板

```bash
# 启动方式B 最常用：torchrun一键启动（PyTorch ≥1.9 推荐）
# 单机8卡：
torchrun --nproc_per_node=8 --master_port=29500 train_ddp.py

# 多机 2机各8卡（在master机器主IP=10.0.0.1上分别执行）：
# Node 0 (master):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 train_ddp.py
# Node 1 (worker): 只改node_rank=1
```

```python
# train_ddp.py 极简DDP模板
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    # ========== 1. 后端初始化 ==========
    local_rank = int(os.environ["LOCAL_RANK"])        # 当前进程 在本机的GPU号 0..7
    rank       = int(os.environ["RANK"])