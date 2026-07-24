# 🔥 03 - 深度学习与PyTorch 章节导览

> 现代AI的基础：从神经网络到Transformer，所有CV/NLP/LLM的底层技术。面试必考！
> 预计学习周期：3周 (21天) | 目标掌握度：⭐⭐⭐⭐ L4熟练级
> 配套项目路径：`../../03-deep-learning/annotated-transformer/` / `pytorch-examples/` / `vit-pytorch/` / `onnx-tutorials/` / `pytorch-lightning/`

---

## 📚 本章节文件索引

| 文件名 | 内容重点 | 优先级 |
|-------|---------|--------|
| **README.md** (本文) | 路线图 + 选型对比 + 高频模型 | ⭐⭐⭐ 先读 |
| **神经网络基础与反向传播.md** | MLP/激活函数/损失函数/优化器/BN/Dropout+反向传播推导 | ⭐⭐⭐⭐⭐ 必学 |
| **CNN与视觉模型.md** | LeNet→ResNet→YOLO→U-Net→ViT/Swin 视觉模型演进 | ⭐⭐⭐⭐ 视觉岗必背 |
| **Transformer架构详解.md** ⭐⭐⭐⭐⭐ | Attention公式/MultiHead/位置编码/Encoder-Decoder/BERT vs GPT | ⭐⭐⭐⭐⭐ 每面必考 |
| **PyTorch训练工程化.md** | DataLoader/Trainer/混合精度AMP/GradClip/Lightning模板 | ⭐⭐⭐⭐⭐ 项目必备 |
| **ONNX模型转换部署.md** | 转ONNX+动态轴+simplify+TensorRT+性能基准测试 | ⭐⭐⭐⭐ 部署必备 |
| **代码实战.md** | MNIST/ResNet50 CIFAR-10/小Transformer训练代码 | ⭐⭐⭐⭐ 必做 |
| **面试题库.md** | 100道深度学习面试题+答案 | ⭐⭐⭐⭐⭐ 面试前背 |
| **GitHub项目推荐.md** | 5个配套项目+代码阅读路线 | ⭐⭐⭐ 参考 |

---

## 🧠 深度学习演进路线图 (按年代)

```mermaid
graph LR
    2012[AlexNet 8层 赢了ImageNet<br>深度学习元年] --> 2014[VGG 19层<br>3×3小卷积堆叠]
    2014 --> 2015[GoogLeNet Inception 22层<br>多尺度卷积]
    2015 --> 2016[**ResNet 152层**⭐⭐⭐⭐⭐<br>残差连接=深层网络能训]
    2016 --> 2017[*DenseNet 密集连接* + SENet 通道注意力]
    2017 --> 2018[⭐Transformer Attention Is All You Need<br>NLP领域革命]
    2018 --> 2020[EfficientNet 复合缩放策略]
    2018 --> 2020[**ViT** Vision Transformer<br>Transformer打进CV]
    2020 --> 2021[Swin Transformer 分层+窗口注意力<br>CV通用骨干]
    2021 --> 2023[Llama/GPT-4 大模型时代]
```

---

## 🌟 三大必考架构（面试100%考一个）

### 🥇 1. ResNet 残差网络 (必考！任何视觉岗第一问)

> 为什么ResNet能训152层而之前VGG只能训19层？→ 残差连接！

```
残差块公式 ⭐ (背下来):
  H(x) = F(x) + x

  x ──────────→ (+) ─→ H(x)    ← 这条叫 "shortcut/skip connection 捷径分支"
  └─→ Conv-BN-ReLU-Conv-BN ─→ F(x)
```

面试标准答案：
> `"残差连接：让网络学的是残差F(x)=H(x)-x，而不是直接学H(x)。如果某个层已经最优了，残差F(x)学成0即可，恒等映射x直接通过，不会退化。没有残差的话深层网络在反向传播时梯度连乘→消失，训不动。"`

衍生知识：**Batch Normalization 批归一化** (ResNet块里必有BN)
> `"BN在每个Channel维度做均值0方差1标准化：减均值除标准差 + γ放缩+β偏置。作用：① 允许更大学习率 收敛快2~5倍 ② 减少初始化敏感性 ③ 轻微正则化防过拟合"`

### 🥈 2. Transformer ⭐⭐⭐⭐⭐ (每面必考！LLM全靠它)

**必背 Scaled Dot-Product Attention 公式：**
```
Attention(Q, K, V) = Softmax( Q·K^T / √d_k ) · V
```
每一项的直觉理解 (面试挨个问)：
1. **Q Query 查询**：我现在这个词想找什么 （我想去注意谁）
2. **K Key 键**：每个词提供什么信息 （我有什么值得被注意）
3. **V Value 值**：每个词的实际内容是什么 （注意到我之后拿什么信息）
4. **Q·Kᵀ 矩阵乘法**：每个Q和所有K做点积=相似度分数 (n×n矩阵)
5. **÷ √d_k 为什么除以？⭐必问**：防止K维度d_k大时点积结果太大，Softmax之后梯度过小 → 除以√d_k把方差拉回1，梯度稳定
6. **Softmax**：相似度分数归一化，变成权重（和为1）
7. **· V 乘Value**：权重加权求和所有V，得到融合了全局注意力的输出向量

**Multi-Head Attention 多头注意力：**
> `"8个不同的Q/K/V线性投影矩阵 (Wq,Wk,Wv 各8个)，并行算出8个Attention结果，最后Concat起来再过一个Wo线性投影。好处：8个头学不同的注意力模式，一个学语法关系，一个学指代消解，一个学实体..."`

**位置编码 Positional Encoding：**
> `"Transformer本身没有序列顺序概念(注意力是置换不变的)，要显式加位置信息。原论文用 sin/cos 奇偶位置编码：PE(pos,2i)=sin(pos/10000^(2i/d_model)), PE(pos,2i+1)=cos(...)`
> `"实际工作中直接用 **可学习Position Embedding** (BERT/GPT都用这个，效果更好)"`

Transformer Encoder vs Decoder 区别（面试题）：
| Encoder (BERT用) | Decoder (GPT用) |
|-----------------|----------------|
| Self-Attention 双向看全句 | Masked Self-Attention 只能看前面的词(Causal因果掩码) |
| 输入：句子全部 | 每步输入：之前生成的所有Token |
| 输出：每个位置的理解向量 | 输出：下一个Token的概率分布 |
| 适合：文本分类/命名实体识别/理解 | 适合：文本生成/对话/续写 |
| BERT / RoBERTa / ALBERT | GPT / Llama / Qwen / Claude |

### 🥉 3. ViT + Swin Transformer 视觉版Transformer

**ViT 流程一句话：**
> `"把224×224图片切成16×16=196个Patch，每个Patch拉平到768维+一个<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>分类Token，再加可学习Position Embedding→197个Token送进标准Transformer Encoder×12层→最后取<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>输出向量→MLP分类头→1000类概率"`

**Swin Transformer改进ViT的两个问题：**
| ViT 痛点 | Swin 解决方法 |
|---------|--------------|
| 自注意力196×196 = O(n²)，高分辨率图算不动 | **W-MSA窗口自注意力**：只在7×7窗口内部算注意力，复杂度降O(HW) |
| ViT只有1/16下采样，多尺度检测/分割不好用 | **Patch Merging降采样** + 分层特征 (类似ResNet C2/C3/C4/C5) + SW-MSA移位窗口跨窗口通信 |

---

## ⚡ PyTorch 训练工程模板 (100行 = 生产可用)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # 混合精度AMP

class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防过自信
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)
        self.scaler = GradScaler()  # AMP FP16混合精度 速度×1.5~2 显存÷2
        self.device = device

    def train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)  # 比zero_grad()快一点

            with autocast():  # AMP上下文：自动选算子FP16/FP32
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()          # 梯度放大后反传，防FP16下溢
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪 防爆炸
            self.scaler.step(self.optimizer)            # 先unscale再更新，超NaN跳过
            self.scaler.update()

        self.scheduler.step()  # 每个epoch调学习率(CosineAnnealing是epoch级)
        print(f"Epoch {epoch:2d} | LR: {self.scheduler.get_last_lr()[0]:.6f} | Loss: {loss.item():.4f}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()  # ⭐ 别忘了! 不然BN/Dropout在推理模式行为不对
        correct, total = 0, 0
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            pred = self.model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
        acc = correct / total * 100
        return acc

# 主流程：训练30轮 + 最佳模型保存
best_acc, best_ckpt = 0.0, None
for epoch in range(1, 31):
    trainer.train_one_epoch(epoch)
    val_acc = trainer.validate()
    print(f"Val Acc: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc, best_ckpt = val_acc, model.state_dict().copy()

torch.save(best_ckpt, "best_model.pth")
print(f"Training done! Best Val Acc: {best_acc:.2f}%")
```

> ⚠️ 面试必问的坑：**为什么 train() 和 eval() 必须切换？** → BatchNorm在train用batch均值方差+更新running统计量，eval用固定running统计；Dropout在train开随机失活，eval全通。忘记切eval()准确率掉5~15%！

---

## 📦 配套推荐项目（优先级按序）

| 项目 | 学习时间 | 必看源码文件 | 面试价值 |
|-----|---------|-------------|---------|
| **annotated-transformer** ⭐⭐⭐⭐⭐ | 15h | `the_annotated_transformer.py` 完整Attention/MultiHead/PositionalEncoding | Transformer面试考啥你就写过啥 |
| **pytorch-examples** | 10h | `mnist/main.py` Trainer+AMP+检查点 | 标准训练代码模板直接抄到简历 |
| **vit-pytorch** | 8h | `vit_pytorch/vit.py` ViT实现不到200行 | ViT/Swin面试200行写完超加分 |
| **pytorch-lightning** | 10h | `src/lightning/pytorch/core/module.py` LightningModule接口 | 大项目工程化必备 |
| **onnx-tutorials** | 5h | 导出ONNX+动态轴+TensorRT推理 | 部署岗重点 |

---

## 🎯 章节结业标准

- [ ] 能背诵Attention公式，解释÷√d_k的作用
- [ ] 能说ResNet残差连接的作用，为什么深层网络不退化
- [ ] 能解释BN/Head的区别，train()/eval()必须切换的原因
- [ ] 能独立写出PyTorch训练循环：DataLoader/AMP/梯度裁剪/保存最佳权重
- [ ] 能解释Multi-Head/Self-Attention/Cross-Attention的区别
- [ ] 能导出PyTorch模型到ONNX，写一句验证输出误差<1e-5的代码
- [ ] 能解释GPT(Decoder-only) vs BERT(Encoder-only) vs Encoder-Decoder(T5)三类区别
- [ ] 面试题库正确率 ≥ 70%