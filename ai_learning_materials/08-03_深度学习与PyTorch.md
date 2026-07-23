# 🔷 技术方向3：深度学习与PyTorch

## 3.1 知识点展开详解

### 3.1.1 神经网络基础

**MLP（多层感知机）核心流程**：
- 线性层：`z = W·x + b`
- 激活函数（引入非线性）：
  - ReLU: `max(0, x)` —— 简单，缓解梯度消失，但可能"神经元死亡"
  - sigmoid: `1/(1+e^(-x))` —— 输出[0,1]，但深层网络梯度消失
  - tanh: `2·sigmoid(2x)-1` —— 输出[-1,1]，零中心化
  - **GELU**: `x·Φ(x)`（高斯误差线性单元）—— **Transformer默认**

**BatchNorm / LayerNorm**：
- BatchNorm: `BN(x) = γ·(x - μ_b)/√(σ_b²+ε) + β` —— 沿batch维度归一化，CNN常用
- LayerNorm: `LN(x) = γ·(x - μ_l)/√(σ_l²+ε) + β` —— 沿特征维度归一化，**Transformer/RNN常用**，对batch size不敏感

**残差连接（Residual Connection）**：
- `H(x) = F(x) + x` —— 让深层网络可以训练，梯度可直接流过上一层
- 核心创新（ResNet 2015何恺明）：152层网络可训练，之前20层就退化

### 3.1.2 CNN（卷积神经网络）

**核心公式**：
- 卷积输出尺寸：`H_out = (H_in - F + 2P) / S + 1`
  - F=卷积核大小, P=填充, S=步长
- 池化：Max/Average Pooling，降维+增加感受野

**典型架构演进**：
- LeNet (1998) → AlexNet (2012, ReLU+Dropout) → VGG (2014, 3×3小核叠加) → Inception (2014, 多尺寸并行) → **ResNet (2015, 残差)** → EfficientNet (2019, 复合缩放) → **ViT (2020, Vision Transformer, 图像=序列)**

### 3.1.3 RNN/LSTM（序列建模）

**LSTM门机制**：
- 输入门 `i_t = σ(W_i·[h_{t-1}, x_t])` —— 新信息写入多少
- 遗忘门 `f_t = σ(W_f·[h_{t-1}, x_t])` —— 旧记忆丢弃多少
- 输出门 `o_t = σ(W_o·[h_{t-1}, x_t])` —— 输出多少到隐藏态
- 细胞状态 `c_t = f_t·c_{t-1} + i_t·tanh(W_c·[h_{t-1}, x_t])` —— 长期记忆载体
- 隐藏态 `h_t = o_t·tanh(c_t)`

**问题**：难以并行计算（t依赖t-1），长序列梯度消失 → **Transformer用自注意力替代**

### 3.1.4 Transformer（核心重点）

**架构**：Encoder-Decoder结构，多头自注意力 + 前馈网络(FFN) + 残差+LayerNorm

**自注意力（Self-Attention）核心公式**：
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```
- Q/K/V = 输入序列经过3个不同线性层投影得到
- `Q·Kᵀ` = 计算每个位置与其他所有位置的"相似度分数"
- `/√d_k` = 缩放因子，防止点积过大导致softmax梯度消失（**重要工程细节**）
- softmax归一化 → 得到注意力权重 → 加权求和V → 输出

**多头自注意力（Multi-Head Attention）**：
- 将Q/K/V切分为h个子空间（通常h=8或12），分别计算注意力 → 拼接 → 线性层
- 直觉：不同头关注不同模式（语法/语义/全局/局部）

**位置编码（Positional Encoding）**：
- Transformer本身是位置无关的（排列不变）→ 需要显式注入位置信息
- 正弦位置编码（原始论文）：
  ```
  PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
  ```
- 现代模型多用**可学习的位置嵌入**（直接学习位置向量）

**与Android结合**：
- PyTorch训练的模型 → 导出 ONNX (`torch.onnx.export`) → 转为 TFLite → Android部署
- 或者直接用 **PyTorch Mobile** (`org.pytorch:pytorch_android:1.13.1`)

### 3.1.5 PyTorch工程实践

| 概念 | 代码模式 | 说明 |
|------|---------|------|
| Module | `class Net(nn.Module): def __init__(...)` | 所有模型的基类，参数自动注册 |
| DataLoader | `DataLoader(dataset, batch_size=32, shuffle=True)` | 多线程数据加载 |
| 训练循环 | `optimizer.zero_grad(); loss.backward(); optimizer.step()` | 清零梯度→反向传播→更新 |
| 推理模式 | `with torch.no_grad(): model.eval()` | 禁用梯度计算+切换BatchNorm/LayerNorm模式 |
| 保存/加载 | `torch.save(model.state_dict(), 'model.pt')` | 只保存参数（推荐）vs 保存整个模型 |
| ONNX导出 | `torch.onnx.export(model, dummy_input, 'model.onnx')` | 跨平台部署格式 |

---

## 3.2 GitHub项目推荐

### 📦 项目1：annotated-transformer（Transformer逐行注释实现）

- **GitHub链接**：https://github.com/harvardnlp/annotated-transformer
- **下载命令**：`git clone --depth 1 https://github.com/harvardnlp/annotated-transformer.git`
- **解压到目录**：`D:\ai_learning\deep_learning\annotated-transformer\`
- **Star数**：约12k · 最后更新：2024
- **技术栈**：Python + PyTorch
- **项目简介**：哈佛大学NLP组对"Attention Is All You Need"原始论文的逐行注释实现。每一行代码都有详细注释，是理解Transformer架构的绝佳项目。
- **学习价值**：
  - 理解Transformer的每个组件（自注意力/多头/位置编码/FFN）的精确实现
  - 学习PyTorch的nn.Module组织方式
  - 为后续学习BERT/GPT等衍生模型打基础
- **代码阅读路线**：
  - `AnnotatedTransformer.ipynb`（完整实现，逐行注释 ⭐⭐⭐）
  - `MultiHeadAttention` 类 → `ScaledDotProductAttention` 函数
  - `PositionalEncoding` 类 → 正弦位置编码公式实现
- **与传统开发结合点**：
  - Java：理解Transformer后，可用DJL加载ONNX格式的Transformer模型推理
  - Android：PyTorch训练 → ONNX导出 → TFLite → 端侧部署

### 📦 项目2：pytorch-examples（PyTorch官方示例）

- **GitHub链接**：https://github.com/pytorch/examples
- **下载命令**：`git clone --depth 1 https://github.com/pytorch/examples.git`
- **解压到目录**：`D:\ai_learning\deep_learning\pytorch-examples\`
- **Star数**：约25k · 最后更新：2025
- **技术栈**：Python + PyTorch
- **项目简介**：PyTorch官方示例仓库，覆盖MNIST、ImageNet、Word Language Model、DCGAN等经典任务。代码质量高，是学习PyTorch工程实践的最佳起点。
- **代码阅读路线**：
  - `mnist/main.py`（MNIST手写数字识别 ⭐）
  - `imagenet/main.py`（ImageNet图像分类，ResNet训练流程 ⭐⭐）
  - `word_language_model/main.py`（RNN/LSTM语言模型）
  - `dcgan/main.py`（生成对抗网络）

### 📦 项目3：vit-pytorch（Vision Transformer简洁实现）

- **GitHub链接**：https://github.com/lucidrains/vit-pytorch
- **下载命令**：`git clone --depth 1 https://github.com/lucidrains/vit-pytorch.git`
- **解压到目录**：`D:\ai_learning\deep_learning\vit-pytorch\`
- **Star数**：约10k · 最后更新：2025
- **技术栈**：Python + PyTorch
- **项目简介**：Vision Transformer（ViT）的简洁PyTorch实现。ViT将图像视为序列（Patch Embedding），用纯Transformer架构替代CNN，在图像分类任务上超越ResNet。
- **学习价值**：
  - 理解图像如何变成序列（Patch Embedding）
  - Transformer在视觉任务中的应用
  - lucidrains的代码风格简洁优雅，适合学习

### 📦 项目4：onnx/tutorials（ONNX导出与部署教程）

- **GitHub链接**：https://github.com/onnx/tutorials
- **下载命令**：`git clone --depth 1 https://github.com/onnx/tutorials.git`
- **解压到目录**：`D:\ai_learning\deep_learning\onnx-tutorials\`
- **Star数**：约4k · 最后更新：2025
- **技术栈**：Python + ONNX + PyTorch/TensorFlow
- **项目简介**：ONNX官方教程仓库，覆盖PyTorch/TensorFlow/scikit-learn到ONNX的导出流程，以及ONNX Runtime的部署示例。
- **学习价值**：
  - 掌握 `torch.onnx.export()` 的完整用法
  - 理解ONNX格式如何跨平台部署（Java/Android/浏览器）
  - 与方向4（Android端侧）和方向5（Java后端）无缝衔接

### 📦 项目5：pytorch-lightning（PyTorch Lightning）

- **GitHub链接**：https://github.com/Lightning-AI/pytorch-lightning
- **下载命令**：`git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git`
- **解压到目录**：`D:\ai_learning\deep_learning\pytorch-lightning\`
- **Star数**：约28k · 最后更新：2025
- **技术栈**：Python + PyTorch
- **项目简介**：PyTorch Lightning是PyTorch的高层封装，标准化训练循环。将训练逻辑抽象为LightningModule，将工程逻辑（分布式/混合精度/日志）交给Trainer。
- **学习价值**：
  - 大型PyTorch项目的代码组织（Module/Callback/Logger分离）
  - 理解深度学习工程化的最佳实践
  - 为方向8（MLOps）做铺垫

---

### 项目清单汇总表

| 项目名 | 技术栈 | 难度 | 预计学习时长 | 核心学习点 | git clone 命令 |
|--------|--------|------|-------------|-----------|---------------|
| annotated-transformer | Python+PyTorch | ⭐⭐⭐⭐ | 15h | Transformer逐行理解 | `git clone --depth 1 https://github.com/harvardnlp/annotated-transformer.git` |
| pytorch-examples | Python+PyTorch | ⭐⭐ | 15h | PyTorch官方示例 | `git clone --depth 1 https://github.com/pytorch/examples.git` |
| vit-pytorch | Python+PyTorch | ⭐⭐⭐ | 10h | ViT图像=序列 | `git clone --depth 1 https://github.com/lucidrains/vit-pytorch.git` |
| onnx/tutorials | Python+ONNX | ⭐⭐ | 10h | 模型导出与跨平台部署 | `git clone --depth 1 https://github.com/onnx/tutorials.git` |
| pytorch-lightning | Python+PyTorch | ⭐⭐⭐ | 10h | 深度学习工程化 | `git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git` |

---

## 3.3 完整可运行代码示例

### 代码1：PyTorch从零实现简化版Transformer（可直接运行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        # Q/K/V形状: (batch, n_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # (batch, heads, seq, d_k)
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须被n_heads整除"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        # 线性投影 + 拆分为多头: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # 合并头: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(attn_output), attn_weights


class PositionWiseFeedForward(nn.Module):
    """前馈网络 FFN = Linear → GELU → Linear"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # buffer不参与梯度更新
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class EncoderLayer(nn.Module):
    """单个Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LN结构（现代Transformer常用）: LN → Attention → Dropout → 残差
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class SimplifiedTransformer(nn.Module):
    """简化版Transformer分类器（适用于文本/序列分类）"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256, n_classes=2, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        # x形状: (batch, seq_len) = token indices
        x = self.embedding(x) + self.pos_encoding(x)  # 嵌入+位置编码
        for layer in self.encoder_layers:
            x = layer(x)
        # 取全局平均作为分类特征
        pooled = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(pooled)  # (batch, n_classes)


if __name__ == "__main__":
    print("="*60)
    print("简化版Transformer - 从零实现测试")
    print("="*60)
    
    # 模拟数据: batch=2, seq_len=10, 词汇表=1000, 二分类
    torch.manual_seed(42)
    batch_size, seq_len, vocab_size = 2, 10, 1000
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.tensor([0, 1])
    
    model = SimplifiedTransformer(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256)
    
    # 前向传播测试
    output = model(input_tokens)
    print(f"输入形状: {input_tokens.shape}  (batch={batch_size}, seq_len={seq_len})")
    print(f"输出形状: {output.shape}  (batch={output.shape[0]}, n_classes={output.shape[1]})")
    print(f"输出logits:\n{output}")
    print(f"预测类别: {output.argmax(dim=-1).tolist()}")
    
    # 训练步骤测试
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_tokens)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: loss = {loss.item():.4f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Transformer前向/反向传播成功！参数数量: {total_params:,}")
    print("="*60)
```

**运行**：`pip install torch` → `python transformer_from_scratch.py`

---

### 代码2：PyTorch CNN图像分类 + ONNX导出

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class SimpleCNN(nn.Module):
    """简化版CNN - 适用于MNIST手写数字识别"""
    def __init__(self, n_classes=10):
        super().__init__()
        # 输入: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # →(batch,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # →(batch,64,28,28)
        self.pool = nn.MaxPool2d(2, 2)                                      # →减半
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 两次pooling后: 28→14→7
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (batch,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))   # (batch,64,7,7)
        x = x.flatten(1)                        # (batch, 64*7*7=3136)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    acc = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch}: 训练loss={total_loss/len(train_loader):.4f}, 准确率={acc:.1f}%")


def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print(f"测试集准确率: {100.*correct/len(test_loader.dataset):.1f}%")


def export_to_onnx(model, device, onnx_path='mnist_cnn.onnx'):
    """导出ONNX - 供Java/Android推理使用"""
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)  # 样例输入
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17
    )
    print(f"✅ 模型已导出: {onnx_path}")
    print("   Java/Android端可用 DJL(OnnxRuntime) 或 TFLiteConverter 加载推理")


if __name__ == "__main__":
    print("="*60)
    print("PyTorch CNN图像分类 + ONNX导出")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    print(f"数据集: 训练{len(train_dataset)}, 测试{len(test_dataset)}")
    
    # 模型训练
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()
    for epoch in range(1, 4):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    print(f"训练耗时: {time.time()-start:.1f}s")
    
    # 导出ONNX（供Java/Android部署）
    export_to_onnx(model, device)
    print("="*60)
```

**运行**：`pip install torch torchvision` → `python cnn_mnist_onnx.py`

---

### 代码3：Java - DJL加载ONNX模型推理（与方向5衔接）

```java
package com.ai.dl.inference;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/**
 * Java端加载PyTorch导出的ONNX模型推理
 * 
 * Python端导出代码:
 *   torch.onnx.export(model, dummy_input, 'mnist_cnn.onnx', opset_version=17)
 */
public class PytorchOnnxInference {

    public static class MnistTranslator implements Translator<float[], Integer> {

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            NDManager manager = ctx.getNDManager();
            // 输入: 784维向量 (28*28) → 重塑为 (1, 1, 28, 28)
            NDArray tensor = manager.create(input).reshape(1, 1, 28, 28);
            // 归一化（与Python端transforms.Normalize((0.1307,), (0.3081,))保持一致）
            tensor = tensor.sub(0.1307f).div(0.3081f);
            return new NDList(tensor);
        }

        @Override
        public Integer processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.get(0);
            // 输出形状 (1, 10) → 取argmax
            return (int) output.argMax().getLong();
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(60));
        System.out.println("Java DJL加载PyTorch ONNX模型推理示例");
        System.out.println("=".repeat(60));

        Criteria<float[], Integer> criteria = Criteria.builder()
                .setTypes(float[].class, Integer.class)
                .optModelPath(java.nio.file.Paths.get("models"))
                .optModelName("mnist_cnn")  // 加载 models/mnist_cnn.onnx
                .optTranslator(new MnistTranslator())
                .optEngine("OnnxRuntime")
                .build();

        try (ZooModel<float[], Integer> model = criteria.loadModel();
             Predictor<float[], Integer> predictor = model.newPredictor()) {

            System.out.println("\n模型加载成功！");

            // 模拟一条MNIST测试数据（28*28=784维）
            float[] testImage = new float[784];
            // 简单模拟：中间画一个"数字"（实际应加载真实MNIST图像）
            for (int i = 0; i < 784; i++) {
                testImage[i] = (float) Math.random() * 0.5f;  // 随机噪声
            }

            Integer prediction = predictor.predict(testImage);
            System.out.printf("预测结果: 数字 %d\n", prediction);
        }

        System.out.println("\n✅ Java ONNX模型推理成功！");
        System.out.println("=".repeat(60));
    }
}
```

**pom.xml依赖**：`ai.djl:api:0.27.0`, `ai.djl.onnxruntime:onnxruntime-engine:0.27.0`

---

## 3.4 面试题库（深度学习与PyTorch）

### 📝 理论题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 3.1 | ReLU/sigmoid/tanh/GELU激活函数的区别？为什么ReLU训练更快？ | 中 | ⭐⭐⭐ |
| 3.2 | 什么是梯度消失/爆炸？BatchNorm/LayerNorm/残差连接分别如何缓解？ | 中 | ⭐⭐⭐⭐ |
| 3.3 | 写出自注意力公式。为什么要除以√d_k？ | 中 | ⭐⭐⭐⭐⭐ |
| 3.4 | 卷积输出尺寸公式 `(H-F+2P)/S+1`，举例说明3×3核+stride=2+padding=1的作用 | 简 | ⭐⭐⭐ |
| 3.5 | Transformer为什么需要位置编码？正弦位置编码和可学习位置编码的区别？ | 中 | ⭐⭐⭐⭐ |
| 3.6 | BatchNorm和LayerNorm的区别？为什么Transformer用LayerNorm不用BatchNorm？ | 中 | ⭐⭐⭐⭐ |
| 3.7 | LSTM的4个门机制及直觉理解。为什么LSTM比普通RNN更能处理长序列？ | 中 | ⭐⭐⭐ |
| 3.8 | 多头注意力的直觉：为什么"多头"比"单头"效果好？ | 中 | ⭐⭐⭐ |
| 3.9 | ViT（Vision Transformer）如何把图像变成序列？Patch Embedding的过程？ | 中 | ⭐⭐⭐ |
| 3.10 | Pre-LN和Post-LN的区别（残差前后的LayerNorm位置）？现代模型用哪种？ | 中 | ⭐⭐⭐ |

### 🐍 PyTorch代码题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 3.11 | 用PyTorch实现一个2层MLP：`nn.Linear(784→256→10) + ReLU + CrossEntropy`，前向/反向传播完整流程 | 简 | ⭐⭐⭐ |
| 3.12 | 用PyTorch实现简化版自注意力层，输入`(batch, seq, d_model)`，输出同形状 | 中 | ⭐⭐⭐⭐ |
| 3.13 | `model.eval()` + `torch.no_grad()` 和 `model.train()` 有什么区别？写一个推理函数 | 简 | ⭐⭐⭐⭐ |
| 3.14 | 写出用 `torch.onnx.export()` 导出模型的完整代码，包括dummy input的创建 | 简 | ⭐⭐⭐ |
| 3.15 | 用PyTorch写一个图像分类的完整训练循环（DataLoader+train+test+保存模型） | 中 | ⭐⭐⭐⭐ |

### 🔗 与传统开发结合题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 3.16 | PyTorch训练的CNN模型如何部署到Android？描述从训练到手机推理的完整路径（至少2种方案） | 中 | ⭐⭐⭐⭐ |
| 3.17 | Java用DJL加载ONNX格式的Transformer模型推理，描述Translator的processInput/processOutput | 中 | ⭐⭐⭐ |
| 3.18 | PyTorch Mobile (`org.pytorch:pytorch_android`) 与 TFLite 在Android上的对比？ | 中 | ⭐⭐⭐ |

---

> ✅ **方向3（深度学习与PyTorch）学习完成自检清单**：
> - [ ] 能手写自注意力公式并解释缩放因子√d_k的作用
> - [ ] 能独立用PyTorch实现Transformer的一个编码器层
> - [ ] 能导出ONNX格式并理解跨平台部署流程
> - [ ] 能解释BatchNorm vs LayerNorm的区别和适用场景