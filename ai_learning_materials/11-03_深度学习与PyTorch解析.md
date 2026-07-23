# 11-03 深度学习与Transformer解析

> 📂 项目: `annotated-transformer/` `pytorch-examples/` `pytorch-lightning/` `vit-pytorch/` `onnx-tutorials/`
> ⭐ 简历推荐: ⭐⭐⭐⭐⭐ | 🎯 岗位: 深度学习/NLP/CV/大模型工程师

---

## 一、Annotated Transformer 源码拆解

### 1.1 架构对照源码

```
Transformer (论文Attention is All You Need)
├── Encoder (6层)
│   ├── PositionalEncoding      正弦/余弦位置编码
│   ├── [EncoderLayer ×6]
│   │   ├── MultiHeadAttention    自注意力
│   │   ├── 残差 + LayerNorm      Add & Norm
│   │   ├── PositionwiseFFN       FFN(512→2048→512)
│   │   └── 残差 + LayerNorm
│   └── LayerNorm
└── Decoder (6层)
    ├── PositionalEncoding
    ├── [DecoderLayer ×6]
    │   ├── Masked MultiHeadAttn  自注意力(下三角mask,不看未来)
    │   ├── 残差 + LayerNorm
    │   ├── EncoderDecoderAttn    Q来自Decoder, KV=Encoder输出
    │   ├── 残差 + LayerNorm
    │   ├── PositionwiseFFN
    │   └── 残差 + LayerNorm
    ├── Linear + Softmax         词表分类
    └── Generator
```

### 1.2 缩放点积注意力 (必考)

```python
def attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    # Step1: Q·K^T / sqrt(d_k)  除以sqrt(d_k)防方差太大导致softmax饱和
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Step2: Mask (设为-∞ → softmax后≈0)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Step3: Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    # Step4: 加权和Value
    return torch.matmul(p_attn, V), p_attn
```

> 🎯 **连环4问**:
> 1. **除√d_k的原因?** → QK^T方差是d_k,除以√d_k让方差=1,softmax不饱和
> 2. **3种Attention区别?** → Encoder自Attn(无mask) / Decoder Masked自Attn(下三角) / Encoder-Decoder Attn(Q=Dec, KV=Enc)
> 3. **多头为什么好?** → 不同子空间学不同依赖(语法/语义/位置),拼接后表达丰富
> 4. **复杂度?** → O(n²·d),n是序列长,平方级→催生Linformer/Swin/Longformer

### 1.3 位置编码PE

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数维sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数维cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不训练但保存
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
```

### 1.4 Noam学习率调度

```python
# lr = d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)
# 前warmup线性上升 → 后按step^-0.5衰减
class NoamOpt:
    def rate(self, step=None):
        step = step or self._step
        return self.factor*(self.model_size**(-0.5) * min(step**(-0.5), step*self.warmup**(-1.5)))
```

---

## 二、PyTorch标准训练框架

### 2.1 Production级训练循环

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def train_epoch(self, epoch):
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad(set_to_none=True)  # 清空梯度
            loss = self.criterion(self.model(data), target)
            loss.backward()          # 反向
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 防爆炸
            self.optimizer.step()     # 更新
        self.scheduler.step()

    @torch.no_grad()  # 推理禁用梯度,省显存
    def validate(self):
        self.model.eval()
        val_loss, correct = 0, 0
        for data, target in self.val_loader:
            data, target = data.cuda(), target.cuda()
            out = self.model(data)
            val_loss += self.criterion(out, target).item()
            correct += (out.argmax(1) == target).sum().item()
        return 100.*correct/len(self.val_loader.dataset)
```

### 2.2 PyTorch常见坑速查

| 症状 | 根因 | 修复 |
|-----|------|------|
| Loss不下降 | 没zero_grad / LR太大 / 数据问题 | 每次backward前zero_grad;LR调小10倍 |
| Loss=NaN | 梯度爆炸 / /0 / log(负) | clip_grad_norm_; 检查输入; 标签平滑 |
| Train好Val差 | 没切换model.eval() / 过拟合 | 手动切换模式; 加数据增强/正则 |
| CrossEntropy Loss大 | 最后层重复加Softmax | PyTorch CE内置Softmax, 输出Linear直接喂 |

---

## 三、ViT视觉Transformer

```
图像224×224×3
  ↓ (切成16×16块=14×14=196个Patch)
每个Patch展平 → 768维
  ↓ Linear投影 768→768
  ↓ 前面加 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token (197个)
  ↓ 加可学习位置编码 197×768
  ↓ Transformer Encoder ×12层
  ↓ 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 输出 768维
  ↓ MLP Head
  → 分类
```

```python
# ViT核心 (简化版)
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 196
        self.patch_embed = nn.Linear(patch_size**2 * 3, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = TransformerEncoder(dim, depth, heads)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        B, C, H, W = img.shape
        # [B,3,224,224] → [B,3,14×16,14×16] → [B,196,768]
        patches = img.unfold(2,16,16).unfold(3,16,16).reshape(B,196,-1)
        x = self.patch_embed(patches)
        x = torch.cat([self.cls_token.repeat(B,1,1), x], dim=1)  # [B,197,768]
        x += self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])  # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 分类
```

---

## 四、PyTorch Lightning 工程化

### 4.1 样板代码分离

```python
# Lightining帮你做: DDP/AMP/Checkpoint/EarlyStop/Logging/梯度累积
class LitModel(pl.LightningModule):
    def __init__(self): super().__init__(); self.net=MyModel()
    def forward(self, x): return self.net(x)
    def training_step(self, batch, _):
        x,y=batch; loss=F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss    # Lightning自动 backward+step+zero_grad
    def validation_step(self, batch, _):
        x,y=batch; loss=F.cross_entropy(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)

# 一键工程化
trainer = pl.Trainer(
    max_epochs=50, accelerator="gpu", devices=2, strategy="ddp",  # 双卡DDP
    precision="16-mixed", gradient_clip_val=1.0, accumulate_grad_batches=4,
    callbacks=[ModelCheckpoint(monitor="val_loss"), EarlyStopping(monitor="val_loss", patience=5)],
)
trainer.fit(LitModel(), train_loader, val_loader)
```

---

## 五、ONNX模型部署

### 5.1 PyTorch → ONNX → ORT推理

```python
# 1. 导出ONNX
model = resnet50(pretrained=True).eval()
torch.onnx.export(
    model, torch.randn(1,3,224,224), "resnet.onnx",
    opset_version=17, do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":{0:"b"}, "output":{0:"b"}}
)
# 2. 验证 + ORT推理
import onnxruntime as ort
ort_sess = ort.InferenceSession("resnet.onnx", providers=["CUDAExecutionProvider"])
out = ort_sess.run(None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})
# CPU推理2~5x加速; +INT8量化 4~8x; +TensorRT EP 10x
```

---

## 六、简历亮点 + 面试题

### ✍️ 简历写法 (直接套用)

| 方向 | 简历句式 (含量化) |
|-----|-----------------|
| Transformer | 「基于annotated-transformer复现德译英系统，Multi30k数据集BLEU=37.6；自定义RoPE旋转位置编码替换正弦编码，长句(>30词)BLEU额外+2.1」 |
| ViT/CV | 「复现ViT-B/16，ImageNet-1K Top-1 Acc=77.9%；小样本迁移(1%标签)比ResNet50高6.3%，验证大模型泛化优势」 |
| 训练框架 | 「PyTorch Lightning搭建训练平台：双卡DDP+AMP混合精度+梯度累积4倍，单epoch从2h→35min (3.4x加速)」 |
| 模型部署 | 「ResNet50部署：PyTorch→ONNX+INT8量化，CPU QPS从35→142 (4.1x)，延迟28ms→8ms，Acc仅-0.6%」 |

### 🎯 高频面试题

**基础Q**: 为什么LayerNorm不用BatchNorm？→ NLP batch小+padding，BN在batch维统计噪声大；LN每个样本独立统计。

**进阶Q**: KV Cache加速生成原理？→ 第t步生成时，缓存前t-1步所有层的K/V，新token只用自己的Q和历史KV内积 → 计算从O(t²d)降到O(td)，速度提升几十倍。

**高级Q**: 长文本Transformer优化方案？→ 稀疏注意力(Longformer滑窗+全局token) / 低秩(Linformer KV投影k<<n) / 核近似(Performer φ(Q)·(φ(K)^T V) )。

---

**下一篇**: 👉 [11-04 Android端侧AI解析](11-04_Android端侧AI项目解析.md)