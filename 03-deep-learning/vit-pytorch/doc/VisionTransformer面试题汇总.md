# VisionTransformer 面试题汇总 (50题)

> 位置: 03-deep-learning/vit-pytorch/doc/
> 配套文档: VisionTransformer架构与实现.md | VisionTransformer流程图详解.md | VisionTransformer性能优化重难点.md

---

## 📊 题目分布

| 类别 | 题数 | 出现频率 |
|-----|-----|---------|
| ViT基础架构 | 15题 | ⭐⭐⭐⭐⭐ |
| 注意力机制 | 12题 | ⭐⭐⭐⭐⭐ |
| 训练优化 | 10题 | ⭐⭐⭐⭐ |
| 变体与对比 | 8题 | ⭐⭐⭐⭐ |
| 工程部署 | 5题 | ⭐⭐⭐ |

---

## 一、ViT基础架构题（15题）

### Q1. ViT和CNN的核心区别是什么？ViT为什么能在视觉任务上超越CNN？
> **A**:
> 核心区别三大点：
> 1. **表征方式**：CNN用局部卷积核，具有平移不变性和局部性归纳偏置；ViT用全局自注意力，一上来每个Patch都看全图，无先验假设
> 2. **长距离依赖**：CNN要逐层叠加才能看全局(50层感受野才全图)，ViT第1层第1个头就看到全部196个Patch
> 3. **数据尺度效应**：CNN小数据集表现好(归纳偏置帮大忙)，ViT在>1亿张超大数据集上，能学到比手工归纳偏置更优的规律 → 表示上限更高，JFT-3B预训练ViT能到90.45% Top1
>
> 超越原因：**规模+预训练范式**。不是ViT结构天生比CNN强，而是Transformer架构更能吃算力和数据，加上自监督预训练(MAE等)释放了潜力。同等参数量(28M)的Swin和ResNet-50比，Swin赢3-5个点。

### Q2. 画一下ViT-B/16的完整架构图，标注各部分维度变化
> **A**: (面试官盯着白板看你画的每一步)
> ```
> Input: [B, 3, 224, 224]
>   ↓ Patch Embedding: 16×16 Conv / 或 Rearrange+Linear
> [B, 196, 768]  (196=14×14=224²/16², 768=16×16×3 Linear投影)
>   ↓ 拼CLS Token
> [B, 197, 768]  (第0位: <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 全局分类Token)
>   ↓ +Position Embedding 197×768可学习
> [B, 197, 768]
>   ↓ Transformer Encoder ×12层 (每层: MHA+FFN+残差Pre-LN)
> [B, 197, 768]  ← 形状从头到尾不变! (这是Transformer好处)
>   ↓ 切片取[:,0,:] CLS位置的输出
> [B, 768]
>   ↓ LayerNorm + Linear
> [B, 1000] → Softmax → 分类概率
> ```
> ⚠️ 常见坑：**忘记提Pre-LN还是Post-LN** → ViT是Pre-LN，先归一化再进MHA/FFN

### Q3. 为什么要加<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token？不用的话有什么替代方案？
> **A**:
> 作用：**整张图的全局抽象表示**，放在序列最前面，最后分类直接用这个位置的输出做MLP分类，不用额外池化操作
>
> 为什么这么做？直接来源于BERT的<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token设计，Transformer输出的每个位置都只"聚合"了和它相关的信息。CLS位置在每一层自注意力都和所有Patch做交互，最后自然包含全局信息
>
> 替代方案（3种）：
> 1. **Global Average Pooling (GAP)**: 最后一层197个Token在序列维做均值池化得到768维向量 → Swin Transformer就是这么做的！
> 2. **Global Max Pooling**: 序列维取最大值（较少用）
> 3. **所有Token加权求和**: 额外学一个注意力权重α对所有位置加权，相当于学一个动态的CLS
>
> 面试追问：「CLS和GAP哪个好？」→ 论文实验差距<0.5%。CLS更稳定，GAP少一个参数，Swin选GAP主要为了输出特征图方便检测分割。

### Q4. ViT的Position Embedding为什么用可学习的？和正弦/RoPE/ALiBi对比？
> **A**:
> ViT作者原始消融实验：可学习PE vs 正弦PE 效果几乎相同（<0.1%差异），选可学习纯粹是**实现简单，调参少**
>
> 四大PE对比表（面试高频！）：
>
> | PE类型 | 公式/原理 | 外推能力(超训练长度) | 相对位置建模 | 代表模型 |
> |-------|----------|------------------|------------|---------|
> | 可学习 | nn.Parameter 直接梯度下降 | ❌ 极差，超长度随机初始化 | ❌ 隐式学，不保证 | ViT/CLIP/BERT |
> | 正弦固定 | sin/cos(pos/10000^(2i/d)) | ✅ 天生可外推 | ❌ 绝对位置 | 原始Transformer |
> | **RoPE旋转** | Q/K逐位置乘复数旋转矩阵 e^(imθ) | ✅ 外推非常好(NTK缩放) | ✅ 完美相对位置 | **LLaMA/GPT-NeoX 主流LLM** |
> | **ALiBi** | 不加PE，Attention分数减\|i-j\|·m线性惩罚 | ✅ 外推最佳 简单高效 | ✅ 自然相对位置 | MPT/BLOOM/百川 |
>
> 选型建议：视觉新工作 → 可学习PE就够；做LLM长文本 → **RoPE+NTK**或ALiBi

### Q5. Patch Size 16 vs 32 怎么选？影响什么？
> **A**:
>
> | Patch Size | 序列长N | 计算量N²d | 单Token语义粒度 | 准确率 |
> |-----------|---------|----------|--------------|-------|
> | 14×14=196 (PS=16) | 196个 | ~30M MACs | 细(16×16约占原图0.5%) | 高 +2~3% |
> | 7×7=49 (PS=32) | 49个 | ~2M (少16倍!) | 粗(32×33占2%) | 低一些 |
>
> 经验规则：
> - **追求精度**：PS=16，大模型甚至PS=8（N=784，但显存爆）
> - **追求速度端侧**：PS=32优先，后续补检测分割的话PS=16更兼容
> - 下游检测一般用更小的PS=4或8，保证细粒度定位
>
> 追问：「PS=16为什么序列是196？」→ (224/16)² = 14² = 196，会算就行

### Q6. ViT里MHSA的计算量是多少？和MLP比哪个大？
> **A**:
> ViT-B/16 N=197, d=768, h=12, d_h=64, 序列长度N=197：
>
> 1. **MHSA计算量**：4个Linear投影(Q/K/V/Out) + 注意力分数计算
>    - QKV三个Linear: 3 × (N·d·d) = 3·197·768² ≈ **349M**
>    - Output Linear: N·d·d ≈ **116M**
>    - Q·K^T注意力分数: h·N²·d_h = 12·197²·64 ≈ **30M**
>    - Softmax·V: h·N²·d_h ≈ **30M**
>    - **MHA合计**: ~**525M FLOPs**
>
> 2. **FFN计算量**：Linear(768→3072) + GELU + Linear(3072→768)
>    - 第一层: N·d·4d = 197·768·3072 ≈ **464M**
>    - 第二层: N·4d·d = 197·3072·768 ≈ **464M**
>    - **FFN合计**: ~**928M FLOPs**
>
> 结论：**FFN计算量 > MHSA**，比例约2:1。所以优化ViT很多是在优化FFN（SwiGLU、MoE等）
> ⚠️ 但注意：上面是**计算量**，实际**运行速度瓶颈**是MHSA的N²操作，因为是Memory-Bound访存受限，计算量少但跑起来慢。

### Q7. 归纳偏置Inductive Bias是什么？对比CNN/RNN/Transformer的归纳偏置
> **A**:
> 定义：模型在**没见过数据前**，由于结构设计带来的**对数据规律的先验假设**，假设越对，小数据收敛越快；但假设错了，会限制上限
>
> 三大架构归纳偏置对比：
>
> | 架构 | 归纳偏置 | 优点（小数据） | 缺点（大数据） |
> |-----|---------|-------------|-------------|
> | **CNN** | ① 局部性(Locality)：附近像素相关 ② 平移不变性(Translation Invariance) ③ 权重共享 | ✅ 小图片数据集快速收敛泛化好 | ❌ 归纳偏置太强，大数据下不如无偏置的学更好 |
> | **RNN** | ① 时序性(Sequential order) ② 马尔可夫假设(当前依赖前几步) | ✅ 文本顺序天生适配 | ❌ 长距离梯度消失 串行难并行 |
> | **Transformer/ViT** | ① 几乎**零归纳偏置** ② "万物皆可能相关"全局自注意力 | ❌ 小数据会过拟合/不收敛，需要超大数据或CNN初始化 | ✅ 大数据(>1M张)释放表示上限，学到更好的规律 |
>
> 黄金结论：**归纳偏置是双刃剑**。小数据需要偏置助收敛→CNN胜；大数据不需要限制模型，让模型自己学规律→ViT胜。

### Q8. Transformer Encoder每层具体有什么？前向传播的完整流程？
> **A**:
> 单层Encoder（ViT用Pre-LN结构，Post-LN是老版本）：
> ```python
> def encoder_layer_forward(x):
>     # --- 第1个残差分支: Multi-Head Self-Attention ---
>     x_norm1 = LayerNorm(x)                        # Pre-LN先归一化
>     attn_out = MultiHeadSelfAttention(x_norm1)   # QKV→Softmax→加权V→Linear
>     attn_out = Dropout(attn_out)
>     x = x + attn_out                              # 残差连接1 (主路保留梯度!)
>
>     # --- 第2个残差分支: FFN前馈网络 ---
>     x_norm2 = LayerNorm(x)                        # Pre-LN先归一化
>     ffn_out = Linear1(x_norm2)                    # 768 → 3072
>     ffn_out = GELU(ffn_out)
>     ffn_out = Dropout(ffn_out)
>     ffn_out = Linear2(ffn_out)                    # 3072 → 768
>     ffn_out = Dropout(ffn_out)
>     x = x + ffn_out                               # 残差连接2
>     return x
> ```
> 记忆口诀：**"先Norm再做事，做完残差回"** (Pre-LN)，每层2个残差块，12层就是24个残差分支

### Q9. ViT-Base/Large/Huge/giant 的配置差异？参数量怎么算？
> **A**:
>
> | 模型 | 层数L | 维度d | 头数h | MLP膨胀 | Patch | 参数量 | ImageNet Top1 |
> |-----|------|------|------|---------|-------|--------|--------------|
> | ViT-Ti/16 | 12 | 192 | 3 | 4× | 16 | 5.7M | 72.2% |
> | ViT-S/16 | 12 | 384 | 6 | 4× | 16 | 22M | 75.9% |
> | **ViT-B/16** | 12 | 768 | 12 | 4× | 16 | **86M** | **77.9%** |
> | ViT-L/16 | 24 | 1024 | 16 | 4× | 16 | 307M | 76.5% (IN-1K)/85.2% (JFT) |
> | ViT-H/14 | 32 | 1280 | 16 | 4× | 14 | 632M | 88.5% (JFT-300M) |
>
> 参数量估算（忽略LayerNorm/PE等小头≈<2%）：
> ```
> Embedding参数: Patch+CLS+PE ≈ 768*768+768+197*768 ≈ 0.7M
> 每层Encoder:
>   MHA: 4 * d² (Q/K/V/Out 四个Linear) = 4*768² = 2.36M
>   FFN: 2 * d * 4d = 8d² = 8*768² = 4.72M
>   单层合计: ~7.08M
> 12层Encoder: 12 * 7.08M = 84.96M
> Head: LN(768*2) + Linear(768*1000) = 0.77M
> 总参数量≈84.96 + 0.7 + 0.77 ≈ 86M ✓ 对上了
> ```

### Q10. 为什么ViT在ImageNet-1K(128万张)上训练不如ResNet，但是在JFT-300M(3亿)上大幅超越？
> **A**:
> 这就是**"数据规模决定归纳偏置的胜负"**的经典例子：
>
> 1. **IN-1K数据少(1.28M)**：ViT归纳偏置为零，需要从数据中自己学"图像是局部相关的、平移不变"这些规律 → 128万不够，拟合得差。而ResNet自带这些先验，直接跳过学习阶段，上来就比ViT强5-10个点
>
> 2. **JFT-300M数据足(3亿)**：足够ViT从数据中完整学到比CNN手工先验**更优、更灵活、更泛化**的视觉规律，此时CNN的强归纳偏置反而成了**瓶颈**，限制了模型表示上限 → ViT反而反超ResNet 3-8个点
>
> 证据图要点：画两条Acc-DataSize曲线，CNN先平后缓，ViT线性涨超，交叉点大约在千万级样本量附近
>
> 追问：「有办法让ViT小数据集也好吗？」→ 有！① DeiT用CNN蒸馏 ② 加少量卷积层前几层(ConViT/ViT-C) ③ 数据增强拉满 ④ 多任务预训练

### Q11. MLP Head的具体结构？为什么用LayerNorm + Linear？不用先Dropout？
> **A**:
> ```python
> mlp_head = nn.Sequential(
>     nn.LayerNorm(dim),       # 先归一化，因为CLS出来的值范围不稳定
>     nn.Linear(dim, num_classes)  # 直接分类，不要隐藏层(参数量小避免过拟合)
> )
> ```
> 为什么先LN？因为最后一层Transformer输出是残差和，数值波动可能大；LN稳定分布到N(0,1)附近，Linear分类更容易优化
>
> 为什么Dropout一般不加在Head？因为CLS只有一个Token，Dropout概率p会把768维里p*768个维度直接置0，信号损失太大。正则化靠前面的DropPath和Weight Decay就够了

### Q12. Pre-LN vs Post-LN 结构区别？为什么ViT选Pre-LN？
> **A**:
> ```
> Post-LN (老Transformer/ResNet):
>   x → Attn → +x → LN → FFN → +x → LN   ❌ 主路有LN!
>
> Pre-LN (ViT/GPT-2/LLaMA 新标配):
>   x → LN → Attn → +x → LN → FFN → +x   ✅ 主路直通!
> ```
> 核心差异：**梯度回传主路径有没有LN**
> - Post-LN：回传路径必须过LN，梯度被归一化缩小，10层衰减1000倍 → 深层训不动，必须warmup很长 + LR很小
> - Pre-LN：主路就是一个恒等分支f(x)=x，梯度直接回传一模一样的值 → 100层也能稳定训，不用warmup都行
>
> 面试加分：「Xiong et al. 2020 ON LAYER NORMALIZATION IN THE TRANSFORMER 这篇论文从理论上证明了Pre-LN梯度 Lipschitz 常数是1，所以稳定」

### Q13. ViT里用的GELU和ReLU有什么区别？为什么Transformer家族都用GELU？
> **A**:
> GELU = Gaussian Error Linear Unit，公式是：`GELU(x) = x * Φ(x)`，其中Φ(x)是标准正态累计分布函数
>
> 近似公式（工程实现）：`0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`
>
> 对比表：
>
> | 激活 | 梯度饱和 | 平滑性 | 负值 | ViT/BERT实际 | 速度 |
> |-----|---------|--------|-----|-------------|-----|
> | ReLU | ❌ 正半区不饱和 负半区直接0(Dead ReLU) | ❌ 在x=0处不可导 不平滑 | ❌ 0截断 | - | 快 |
> | GELU | ✅ 两端软饱和 | ✅ 处处可导 平滑 | ⚠️ 有很小的负值区(约x<-0.1时才负) | ✅ 标准ViT用GELU | 略慢 能接受 |
> | SwiGLU | ✅ | ✅ | ✅ | ✅ LLaMA/PaLM用 SOTA | 中 |
>
> Transformer选GELU原因：
> 1. 平滑可导 → 优化更稳定，不担心Dead ReLU
> 2. 自注意力出来的数值分布是N(0,1)附近，GELU在0附近形状≈linear，梯度回传好
> 3. 2018年GPT验证过在NLP任务上GELU > ReLU 1-2个点

### Q14. ViT的DropPath (Stochastic Depth)是什么？和Dropout区别？
> **A**:
> **DropPath / 随机深度**：在残差分支上**随机整条分支置零**，整条残差短路
> ```python
> # 伪代码
> def drop_path(x, drop_prob=0.1):
>     if train and random.random() < drop_prob:
>         return x  # 直接返回输入! 残差分支整个删掉
>     return f(x) + x  # 正常走
> ```
>
> Dropout是随机置0**部分神经元**，但矩阵还是参与计算；DropPath是**整个残差块跳过**，相当于这次前向传播少了1层。深度越深，后面的层drop_prob可以设越大（线性递增0→0.5）
>
> | 方法 | 置零对象 | 效果 | ViT常用值 |
> |-----|---------|------|----------|
> | Dropout | 元素级 神经元 | 防止特征共适应 | Attention后0.0 / FFN内0.1 |
> | **DropPath** | 层级 整个残差分支 | 隐式模型集成 + 缓解过深 | **0.1~0.5 (B/L/H不同)** |
>
> 为什么Transformer用DropPath多？因为深度大(24/32层)，直接训深层容易后面的层学不到东西（退化问题），DropPath相当于每次随机训不同深度的子网络，最后泛化更好

### Q15. Patch Embedding为什么等价于一个特殊的Conv2d？用哪个好？
> **A**:
> **数学上严格等价！** 两种写法完全一样：
> ```python
> # 写法1: Rearrange+Linear (更直观)
> x1 = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)  # 展平Patch
> x1 = Linear(patch_dim, dim)(x1)  # 线性投影
>
> # 写法2: Conv2d (更快 工业界常用!)
> conv = Conv2d(in_c=3, out_c=dim, kernel_size=16, stride=16)
> x2 = conv(img).flatten(2).transpose(1,2)  # [B,D,H,W]->[B,H*W,D] = 同上
>
> # x1 == x2 ✓  (只要Linear.weight形状和Conv.weight reshape后一样)
> ```
> 为什么Conv快？CuDNN/Conv内核有硬件加速，Rearrange+Linear是通用矩阵乘，实际跑CPU/GPU Conv版会快5-15%。
>
> 选型：**做部署/生产环境选Conv2d版；做论文/教学用Rearrange更直观**

---

## 二、注意力机制深度题（12题）

### Q16. 逐行手写Scaled Dot-Product Attention，解释为什么除√d_k？
> **A**:
> 面试官要求白板写！逐行解释：
> ```python
> import math
> import torch.nn.functional as F
>
> def attention(Q, K, V, mask=None, dropout_p=0.1):
>     d_k = Q.size(-1)  # 64 for ViT-B
>
>     # 第1步: Q·K^T 计算两两相关性分数
>     # Q: [B,h,N,d_k]  K^T: [B,h,d_k,N]  → scores: [B,h,N,N]
>     scores = torch.matmul(Q, K.transpose(-2, -1))
>
>     # 第2步: ✅ 除以 sqrt(d_k) ← 面试重点!
>     scores = scores / math.sqrt(d_k)
>     """
>     原理: 假设Q,K每个元素独立 N(0,1), 则Q·K^T的方差是 d_k * Var(q_i*k_i) = d_k
>          除√d_k后方差回到1!
>     不除的后果: d_k=64方差=64，scores值域[-40,+40] → Softmax之后要么≈1要么≈0
>              → one-hot分布 梯度全0! 没法训练 (梯度消失!!)
>     除了之后方差=1，Softmax输出平滑分布，梯度正常回传
>     """
>
>     # 第3步: Mask (Decoder用下三角/Encoder不用)
>     if mask is not None:
>         scores = scores.masked_fill(mask == 0, float('-inf'))
>         # 置-∞ softmax后=0 不参与注意力
>
>     # 第4步: Softmax得到注意力权重
>     attn_weights = F.softmax(scores, dim=-1)  # 每行和=1
>
>     # 第5步: Dropout (训练时!)
>     attn_weights = F.dropout(attn_weights, p=dropout_p)
>
>     # 第6步: 加权求和Value得到最终输出
>     output = torch.matmul(attn_weights, V)  # [B,h,N,N]·[B,h,N,d_k]→[B,h,N,d_k]
>
>     return output, attn_weights
> ```
> 追问：「如果d_k=8方差是多少？softmax输入方差大会怎样？」→ 答方差8+画softmax饱和梯度消失图

### Q17. Multi-Head Attention比单头好在哪？多头计算量和单头一样吗？
> **A**:
> 多头好处（4点）：
> 1. **子空间多样性**：每个头在不同d/h维子空间学不同类型的注意力 - 有的头学"颜色相似性"，有的头学"物体相邻关系"，有的头学"全局中心位置"，拼起来信息更丰富
> 2. **多粒度表示**：不同头可以关注不同距离范围 - 近邻头(局部模式) + 远距离头(全局依赖)
> 3. **优化稳定性**：多个独立头相当于集成，梯度噪声更小训练更稳
> 4. **工程并行友好**：多个头可以在GPU上并行算，实际不慢
>
> 计算量：**几乎一模一样！没有增加FLOPs**。推导：
> ```
> 单头: Q/K/V/Out 都是 d×d矩阵 → 4d² FLOPs
> 多头(h=12):
>   Q投影拆成12个 d×(d/12) → 12*(d*d/12) = d² FLOPs
>   K, V, Out 同理各 d² → 合计也是 4d²!
> ```
> ⚠️ 参数量也是一模一样！不是多头就更多参数，只是把大矩阵拆成小的，最后再拼回去。这是Vaswani论文的精妙设计。
>
> 面试官追击：「那为什么现在都用GQA/MQA(少头K/V)节省推理显存？」→ 答K/V Cache在长序列推理时是显存大头，2头/1头K/V比Q少很多参数，效果差不多(GQA)

### Q18. 自注意力的复杂度是O(N²d)，怎么优化到O(N)或O(NlogN)？
> **A**:
> 5大主流优化方案（面试至少说3个）：
>
> | 方法 | 复杂度 | 核心思想 | 代表模型 | 精度损失 |
> |-----|--------|---------|---------|---------|
> | **Flash Attention (硬件)** | 同O(N²)但快3-5倍显存少 | Tiling分块+SRAM在线softmax 不构造N×N矩阵 | 工业标准 LLaMA/GPT | 无 (bf16/fp16等效) |
> | **滑动窗口/局部** | O(N·W²) W=窗口大小 | 每个Token只看左右W个邻居 不看全局 | Longformer/Swin | W≥128几乎无损 |
> | **线性Attention (数学)** | **O(N·d²)** | 用kernel近似: (ϕ(Q)·ϕ(K)ᵀ)·V → ϕ(Q)·(ϕ(K)ᵀ·V) 结合律换顺序 | Linformer/Performer | 2-5%下降 看任务 |
> | **降采样** | O((N/r)²) r=压缩率 | Key/Value先降采样r倍再算注意力 | Linformer/Synthesizer | r=4几乎无损 |
> | **稀疏模式** | O(N·k) k=固定连接数 | 每个Token只看固定k个特殊位置(轴/跨步/地标) | BigBird/ETC | k~64时几乎无损 |
>
> 面试加分：「Flash Attention不是算法优化是**硬件优化**，纯靠GPU SRAM-Tiling技巧，理论复杂度没变但访存量从O(N²)降到O(N)，是实际生产中最有效的手段，所有LLM推理必用」

### Q19. Flash Attention的原理？分块Tiling算法和两次pass是啥？
> **A**:
> **核心问题**: 标准Attention要构造 [B,h,N,N] 的中间分数矩阵，N=4096时就是4096²=16M浮点数/头，H800 HBM显存带宽不够用（Memory Bound）
>
> **Flash Attention解**: 用GPU SRAM(10-20MB超级快但极小)做**分块Tiling计算**，不把完整N×N矩阵存HBM：
>
> ```
> 第一次Forward Pass:
>   把 Q 分成 T_r 块 q_1..q_T_r, 每块大小 Br×d
>   把 K,V 分成 T_c 块 k_1..k_c, 每块大小 Bc×d
>   (Br,Bc选能刚好塞进SRAM的最大尺寸: 例如~256)
>
>   外层循环 遍历每个Q块 i:
>      从HBM加载q_i到SRAM  (一次读)
>      初始化 m_i=-∞ (行max), l_i=0 (行和), o_i=0 (输出块)
>
>      内层循环 遍历每个KV块 j:
>          从HBM加载k_j,v_j到SRAM
>          s_ij = q_i·k_j^T / √d   [Br×Bc]  存在SRAM!
>          修正数值稳定性:
>              m_new = max(m_i, rowmax(s_ij))
>              P_ij = exp(s_ij - m_new)        ← 减去max防溢出
>              l_new = exp(m_i - m_new)*l_i + rowsum(P_ij)
>              o_new = exp(m_i - m_new)*o_i + P_ij @ v_j
>          更新 m_i=m_new, l_i=l_new, o_i=o_new
>      循环结束 o_i = o_i / l_i → 写回HBM
>   最终o就是完整注意力输出! (没存过完整N×N矩阵)
>
> 第二次Backward Pass:
>   因为Forward没存Softmax矩阵 → 反向必须重算一遍注意力数值
>   但HUGE优势: 重算的数值量 < 从HBM读整个Softmax矩阵的带宽量
>   → 反向依然比标准Attention快!
> ```
> 记忆：**Flash Attention = 分块Tiling + 数值稳定Online Softmax + 反向重计算省HBM带宽**
> 面试绝杀：「Dao et al. 2022 论文里给了数值证明，因为用了Online-Softmax+Kahan补偿累加，实际精度比FP16标准Attention还略好」

### Q20. 注意力图可视化后很多头是均匀分布/无意义的，这说明什么？
> **A**:
> 正常！ViT-B有12层×12头=144个头，**大多数头(约60-70%)学到的是"非语义通用功能"**，不代表没用：
>
> 典型头功能分类：
> 1. **语义头 (20-30%)**：才是你想象中的"看猫耳朵/狗眼睛"，和物体部分强对应
> 2. **空间平滑/去噪头 (20%)**：均匀看周围Patch，相当于做了一次空间Blurring，特征去噪平滑
> 3. **低频纹理头 (15%)**：关注颜色相似/纹理连续区域，保留视觉低频信息
> 4. **SOS/EOS特殊头 (10%)**：所有Patch都强烈关注CLS Token或角落Patch，负责"全局特征汇聚到CLS"的功能
> 5. **冗余/未激活头 (15-25%)**：Prune裁剪掉完全不影响准确率 → 这就是结构化剪枝ViT能压2/3参数量的依据
>
> 面试追问：「怎么处理冗余头？」→ 回答结构化剪枝（训练中逐步mask掉低重要度的头）+ 蒸馏（用大模型训好的logits教小模型）

### Q21. Q,K,V三个矩阵能不能共享权重？或者干脆砍掉V变成Q,K两个？
> **A**:
>
> 1. **Q=K=V共享**：**自注意力可以**（因为Q,K,V来源相同都是x），但性能会掉1-3%。工程上可以省2/3的Embedding投影参数，小模型场景确实有人用。
>    ```python
>    shared = Linear(d, d)  # 只一个Linear
>    Q = shared(x); K = shared(x); V = shared(x)
>    ```
>
> 2. **砍掉V，QK做完直接乘某个X**：不行！V是"内容载体"，QK是"决定看谁的路由"。没有V的话输出只能是输入的线性组合加权，表达能力严重下降。
>
> 本质理解Attention三件套：
> | 角色 | 功能类比 | 去掉后果 |
> |-----|---------|---------|
> | Query Q | 搜索引擎的查询词"我要找什么" | 没有搜索意图，全attention没意义 |
> | Key K | 文档索引的关键词标签 | 没索引没法匹配 |
> | Value V | 文档实际内容 | 匹配上了也拿不到信息，输出空壳 |
>
> 面试官会追问：「那Cross Attention的Q和KV不同来源，能不能共享权重？」→ 答不能，跨模态的话Query是文本/视觉，KV是另一模态，分布完全不同共享没意义。

### Q22. 为什么用Softmax归一化注意力？能不能改成Sigmoid每个位置独立打分？
> **A**:
>
> **标准Softmax Attention**：所有Token注意力和=1，是**"有限注意力预算分配"**思想 → 一个位置多看10%，另一个就少看10%，竞争性分配
>
> **改成Sigmoid独立打分**：每个Token 0-1独立，和可能=0.1也可能=10，没有预算限制 → 实际效果上：
> - 优点：稀疏性(很多自然0)，可以并行计算，长序列友好
> - 缺点：缺乏归一化，训练不稳定，数值尺度乱飘，最终准确率差2-5%
>
> 但是！**Sigmoid Attention在超长序列场景(N>4K)有实际用途**：
>   - 论文「Sigmoid Loss for Language Modeling」和GPT-4技术报告暗示用了变体，因为softmax随序列变长，分母求和越来越大导致小权重直接下溢到0
>   - 加温度系数τ调尺度：`σ((q·k/√d)/τ)` 实际可以接近softmax效果
>
> 面试回答结构：先答不能直接换(准确率掉) → 再说为什么Softmax好(竞争性预算分配+梯度稳定) → 最后加Sigmoid的特例场景（加分）

### Q23. 什么是Attention温度系数Temperature？调大调小的影响？
> **A**:
> 公式：`Softmax( (Q·K^T / √d_k) / T )`，T是Temperature，默认T=1.0
>
> | T值 | 注意力分布 | 效果 | 适用场景 |
> |----|----------|------|---------|
> | T→0+ （如0.1） | 尖峰One-hot，最大位置≈1其他≈0 | 硬选择，只看最相关1个Token | 推理时想要确定答案/检索要精确匹配 |
> | T=1.0 (默认) | 平滑分布，头几名权重递减 | 兼顾多数相关Token | 训练默认值 |
> | T→∞ （如10） | 均匀分布，每个位置≈1/N | 全平均看所有Token，特征模糊 | 训练初期防止过拟合/增加多样性 |
>
> 延伸：Tuning技巧
> - 训练前期用大T(2.0)让注意力探索，后期退火到小T(0.7)让决策更锐利
> - 蒸馏模型用T>1(通常T=3~5)把Teacher的"暗知识"logits软分布传给学生
>
> 面试官追击：「为什么知识蒸馏要T大？」→ 答Softmax(T大)把类别间细小差异放大，例如Teacher对猫照片输出猫=99%狗=0.5%兔=0.3%，T=1时Softmax近似one-hot，学生学到的和硬标签没差；T=10后输出变成猫≈50%狗≈30%兔≈20%，学生能学到"猫比狗更像但都像哺乳动物"这类**类间相似性暗知识**

### Q24. Cross Attention和Self Attention区别？ViT用了Cross吗？
> **A**:
>
> | 类型 | Q来源 | K,V来源 | 作用 | 用在哪 |
> |-----|------|--------|------|--------|
> | **Self Attention** | 同一个序列 x | 同一个序列 x | 模态内交互 建模序列内关系 | ViT Encoder / BERT Encoder |
> | **Cross Attention** | 序列A (例: 文本Query) | **另一个序列B** (例: 图像KV) | 跨模态/跨序列对齐 用A去"查"B | Transformer Decoder第二层 / CLIP / BLIP2 |
>
> 标准ViT（纯分类）**只有Self Attention**，没有Cross。
>
> 延伸：ViT家族用Cross的场景：
> - **CLIP**：文本Transformer最后输出的特征当Query，去Cross Attend图像ViT的Patch特征 → 图文对比学习对齐
> - **BLIP-2**：Q-Former学的Query向量Cross Attend ViT特征 → 用少量可学习Query抽图像信息给LLM
> - **Detection DETR**：Object Query (可学习向量) Cross Attend Encoder输出的图像特征 → 直接出检测框

### Q25. 注意力里KV Cache是啥？为什么推理能加速几十倍？
> **A**:
> 典型LLM自回归推理场景：第t步生成第t个词，下一个词只依赖前t个历史
>
> **无KV Cache的朴素推理**：
> ```
> Step 1: 输入 [A]          → 算 K_1,V_1 → 预测 B  (计算N=1²=1)
> Step 2: 输入 [A,B]        → 重算 K_1,V_1,K_2,V_2 → 预测 C  (N=2²=4 多了!)
> Step 3: 输入 [A,B,C]      → 重算 K_1..K_3 → 预测 D (N=9)
> Step 512: ...                                             (N=262144 巨慢!)
> 累计: 1+4+9+...+T² = O(T³)  生成512个词要算4400万次矩阵乘法
> ```
>
> **有KV Cache的推理**：
> ```
> Step 1: 输入 [A] → 算 K_1,V_1 → 存下来Cache → 预测 B
> Step 2: 只用新Token [B] → 算 K_2,V_2 → 和Cache拼接 → [K_1,K_2], [V_1,V_2] → 预测 C
> Step 3: 只用新Token [C] → 算 K_3,V_3 → 拼接 → [K_1..K_3] → 预测 D
> Step t: 只算1个新Token的K,V, 拼接历史Cache → Attention是 (1 × t) 小矩阵
> 累计: O(T²) → 实际因为第t步只算新token的线性投影，矩阵乘都是向量×矩阵量级
> ```
>
> 速度提升：生成长序列(1024词)时 30-80x加速。显存代价：每层要存2个KV矩阵，大小=2·T·d_model，所以才有MQA(多查询共享KV)/GQA(分组查询KV)来省显存（LLaMA 70B用GQA 32组）

### Q26. 可视化Attention有哪些方法？能看出模型好坏吗？
> **A**:
> 3大主流可视化方法：
>
> 1. **CLS→Patch热力图 (最常用)**：
>    - 取最后一层所有头平均的 attention_weights[0, 0, CLS_idx, 1:] → [196]
>    - Reshape成14×14 → 双线性插值上采样到224×224
>    - matplotlib imshow叠加到原图上 → 热区就是模型"看哪里"做决策
>
> 2. **Attention Rollout (跨多层聚合)**：
>    - 朴素热力图只看最后1层1个头，Rollout把所有L层注意力矩阵"递归乘起来"，加上残差恒等权重：
>      `Ã = 0.5*A + 0.5*I` （残差算一半直连）
>      `Rollout = Ã_L @ Ã_{L-1} @ ... @ Ã_1`
>    - Rollout比单层的完整，显示从底层边缘到高层语义的全链路注意力
>
> 3. **t-SNE/UMAP特征降维**：
>    - 所有Patch的输出特征196×768 → t-SNE降到2D → 看相似语义的Patch有没有聚在一起
>    - 猫耳朵聚成一团、天空背景又一团 → 模型学到合理聚类
>
> 看模型好坏：
> ✅ 好模型热力图：热区集中在物体主体(猫脸/车轮)，不飘在背景或割裂
> ❌ 坏模型：热区全图散或盯着底部/角落水印（数据集泄露），或直接盯着CLS死看

### Q27. 注意力里残差连接为什么那么重要？去掉会怎样？
> **A**:
> 数学视角：残差连接让前向变成`y = x + F(x)`，反向梯度变成`∂L/∂x = ∂L/∂y * (1 + ∂F/∂x) = 主项1*梯度 + 小项残差分支梯度`
> → **无论残差分支的∂F/∂x多小甚至0，主路1这一项保证梯度不会消失！原封不动传下去**
>
> 去掉残差的后果：
> - ViT-B/16 12层训出来Top1准确率下降10-20个点
> - ViT-L 24层直接Train Loss爆炸不收敛，和普通深度网络退化问题一模一样
> - 梯度范数逐层指数衰减：0.5^12 ≈ 0.000244，第1层拿到的梯度是输出层的万分之二
>
> 记忆口诀：**残差连接 == 梯度高速公路**，信息走残差主路直通，梯度不衰减，这是能训1000层网络的根基（Pre-LN+Residual两兄弟合璧才好使）

---

## 三、训练优化与调参（10题）

### Q28. ViT训练超参数的黄金配置是什么？和CNN有什么不一样？
> **A**:
> ViT-B/16 ImageNet-1K 标准训练（DeiT-III 训练配方）：
>
> | 超参 | ViT推荐值 | CNN(ResNet-50)推荐值 | 差异说明 |
> |-----|----------|---------------------|---------|
> | **总Epoch** | 300 (甚至1000!) | 90-120 | ViT吃训练步数，训越久越好 |
> | **Batch Size** | 4096 (大Batch!) | 256-1024 | 大Batch稳定ViT，用LARS优化器 |
> | **基础LR (Cosine)** | 3e-3 (3e-4 for AdamW) | 1e-1 (SGD) | ViT LR必须比CNN小10-100倍! |
> | **LR调度** | Cosine + Warmup | Step/MultiStep | 余弦更平滑，warmup前期稳 |
> | **Warmup** | 10k步 (5-10 epoch) | 无或1 epoch | ⚠️ ViT必须warmup 否则初期震荡发散 |
> | **优化器** | **AdamW** weight_decay=0.05 | SGD+Momentum wd=1e-4 | AdamW适合Transformer，权重衰减大3个数量级 |
> | **梯度裁剪** | ✅ norm=1.0 | ❌ 不需要 | 防梯度爆炸 |
> | **Dropout** | 0.0 / Attention后0.1 | 0.2-0.5 | ViT数据增强拉满代替Dropout |
> | **DropPath** | 0.1~0.5 随深度线性增 | 无 | 深度Transformer正则神器 |
> | **MixUp α** | 0.8 ✅必开 | 0.2 可选 | ViT归纳偏置差，MixUp提供额外"类间平滑"先验 |
> | **CutMix α** | 1.0 ✅必开 | 可选 | 同上，数据强正则 |
> | **RandAugment** | 9层 ✅必开 | 可选 | 小数据增广ViT |
>
> 追问回答「为什么ViT要超大Batch？」→ 答小Batch的梯度噪声对归纳偏置弱的ViT影响极大，Batch<1024训练曲线像股票K线上下震荡，大Batch(4096+)的梯度更平滑，更容易收敛

### Q29. Adam和AdamW区别？为什么Transformer都用AdamW？
> **A**:
> 核心区别：**权重衰减Weight Decay加在哪里**
>
> ```
> ❌ Adam (错误做法 L2正则写在梯度里):
>   g_t = ∇_θ L(θ_{t-1}) + λ * θ_{t-1}   ← λ就是权重衰减 加到梯度里
>   m_t = β1*m_{t-1} + (1-β1)*g_t
>   v_t = ...
>   θ_t = θ_{t-1} - lr * m_t/(√v_t+ε)
>   问题: Adam的m_t和v_t会因为β1/β2的平均效应，把λθ衰减系数也给"平滑了"
>         → 实际衰减量被Adam自己的动量机制稀释了!  WD打折扣
>
> ✅ AdamW (Decoupled Weight Decay ICLR 2019 正确做法):
>   g_t = ∇_θ L(θ_{t-1})                  ← 梯度里不加WD!
>   m_t = β1*m_{t-1} + (1-β1)*g_t
>   v_t = β2*v_{t-1} + (1-β2)*g_t²
>   θ_t = θ_{t-1} - lr * (m_t/(√v_t+ε) + λ * θ_{t-1})  ← 最后独立步骤直接减θ
>         两部分独立: 标准Adam更新  + 解耦的权重衰减
> ```
>
> 影响：相同λ=0.01下，Adam实际衰减只有理论的1/5-1/10，模型容易过拟合；AdamW衰减到位，泛化更好。
>
> Transformer特征维度大(768+)，参数多(86M)，权重衰减正则化至关重要 → **全行业默认AdamW**

### Q30. ViT微调的3种常用策略？各适合多少数据量？
> **A**:
>
> | 策略 | 可训参数占比 | 数据量需求 | 耗时 | 准确率 |
> |-----|------------|----------|------|-------|
> | **Linear Probe (线性探测)** | <1% (只有分类头) | 少 (<1k张) | 最快 | 基线最差 |
> | **Partial FT (部分微调)** | 10-50% (后N层+Head) | 中 (1k~1万) | 中 | 性价比高 |
> | **Full FT (全量微调)** | 100% | 多 (>1万) | 最慢 | 最高 |
>
> 进阶4：**LoRA / Adapter 轻量微调** → 可训参数0.1-2%，但能达到Full FT 95%+的效果（见Q12详解）
>
> **Partial FT经验法则**: 从解冻最后2层Head开始 → 不行再解冻倒数第3-4层 → 再不行解冻全部 + 分层LR
> 追问「解冻全部了还是准确率上不去？」→ 答：**降低整体LR一个数量级**，加分层学习率越靠底层越小，加warmup不要一上来LR大

### Q31. 高分辨率微调(384×384) ViT遇到PE形状不匹配怎么办？
> **A**:
> 预训练PE是[1, 197, D]，对应224×224=196+CLS；高分辨率576个Patch需要[1, 577, D]
> → **Position Embedding 2D双线性插值**
>
> 代码步骤（高频代码题白板写）：
> ```python
> def interpolate_pos_embed(pos_embed, orig_h=14, orig_w=14, new_h=24, new_w=24):
>     # 1. 把CLS单独拿出来
>     cls_pe = pos_embed[:, :1, :]        # [1, 1, D]
>     patch_pe = pos_embed[:, 1:, :]      # [1, 196, D]
>
>     # 2. Reshape成图像网格形状 (先拆2D再插值)
>     patch_pe_2d = patch_pe.reshape(1, orig_h, orig_w, -1)  # [1,14,14,D]
>     patch_pe_2d = patch_pe_2d.permute(0, 3, 1, 2)         # [1,D,14,14]  通道放第2维
>
>     # 3. PyTorch双三次插值 (NEAREST会掉点!)
>     patch_pe_2d_new = F.interpolate(
>         patch_pe_2d,
>         size=(new_h, new_w),
>         mode='bicubic',        # 双三次 > 双线性 > 最近邻
>         align_corners=False    # ⚠️ False 对齐像素中心
>     )
>
>     # 4. 还原形状 + 拼回CLS
>     patch_pe_2d_new = patch_pe_2d_new.permute(0, 2, 3, 1).flatten(1, 2)  # [1, 576, D]
>     new_pos_embed = torch.cat([cls_pe, patch_pe_2d_new], dim=1)
>     return new_pos_embed  # [1, 577, D]
> ```
>
> 面试官会追击：「为什么PE插值不会严重掉点？」→ ViT位置编码学到的是相对空间关系(左上角/右上角/中间)，插值相当于平滑缩放，物理意义不变；实验显示IN-1K从224迁移到384插值，精度只掉0.1%以内

### Q32. DeiT是什么？用什么方法让ViT在ImageNet-1K上也能赢ResNet？
> **A**:
> DeiT = Data-efficient Image Transformer (Facebook 2021)，**不需要3亿张JFT，只用IN-1K 128万张就能让ViT超越ResNet**，是工业界ViT训练基线
>
> 核心创新3个：
> 1. **最强训练配方** (贡献60%)：
>    - 300+Epoch、AdamW、Cosine+Warmup、RandomErasing
>    - MixUp(0.8) + CutMix(1.0) + RandAugment(9层) ← 对ViT最关键的数据增强三件套
> 2. **蒸馏Token distill_token (贡献20%)**：
>    - 序列除了<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>再加一个蒸馏Token [1, 1, D]
>    - 训练时两个Loss：CLS用真实交叉熵，蒸馏Token用Teacher(RegNetY-16GF)的Soft蒸馏KL散度
>    - 推理：cls和蒸馏logits平均当输出 → 额外+1个点
> 3. **模型家族规范** (贡献20%)：Ti/S/B三个尺度，Patch=16主流
>
> DeiT-B/16 IN-1K Top1：83.1% (原始ViT-B是77.9%，**+5.2%**，超过ResNet-50 76.1%!)
>
> 面试绝杀：「DeiT-III 2022更新：3个Deeper(更深)/Shorter(更短Patch)/Wider(更宽)改动 + 1000Epoch + CutMix+MixUp+RepeatedAug，DeiT-B/16到**85.8%**」

### Q33. MAE自监督预训练ViT的原理？为什么掩码比例75%？
> **A**:
> MAE = Masked Autoencoders (HeKaiming大神CVPR 2022 Oral)
>
> 核心思想类比NLP的BERT MLM：
> ```
> 【编码器】输入: 随机75%的Patch (遮挡25%!)
>           输出: 这些可见Patch的特征 走标准ViT Encoder
> 【轻量解码器】输入: Encoder输出 + [被Mask的位置用可学习MASK Token填充]
>             输出: 所有196个Patch重建RGB像素值
> 【Loss】只在被遮挡的Patch上算MSE (占75%的位置)，可见Patch不算!
> ```
>
> 为什么75%遮挡比例？
> - BERT只遮15%是因为语言信息密度高，遮太多语义直接没了
> - 图像Patch冗余度极高（相邻Patch像）：遮25%的话旁边Patch一猜就猜到，训练没有挑战性 → 模型学不到深层语义
> - 遮75%后：必须理解"桌子上有个咖啡杯"这种高层语义才能猜出被挡的咖啡杯区域，逼模型学抽象知识
>
> 结果：**只用ImageNet-1K自监督预训练 + 1%标签(12.8万张)微调** → Top1 73.5% (和监督训练的ResNet-50 76%只差2.5%)
>
> 下游检测任务：MAE预训练ViT-L比监督训练高5 COCO mAP，这是自监督的威力
>
> 追击回答「MAE和SimCLR/MoCo对比？」→ MAE是生成式重建任务，学到的特征更通用；对比学习需要大Batch/Negative Sample，硬件更贵；现在检测分割主流预训练是MAE类

### Q34. 训练ViT梯度爆炸/Loss NaN怎么排查？
> **A**:
> 按概率从高到低的Checklist（面试实际场景排序）：
>
> 1. **(概率80%) 学习率太大！**：ViT的LR阈值非常敏感，超过3e-3就炸，降10倍试；没开warmup直接上大LR也炸
> 2. **(10%) 初始化错误**：Weight/PE/CLS初始化std>0.1（建议固定0.02），或者Linear bias没初始化zero（nn.init.zeros_）
> 3. **(5%) 没做Pre-LN**：用了Post-LN在深ViT梯度爆炸（10+层），改Pre-LN立刻好
> 4. **(3%) AMP混合精度下溢出**：GradScaler的init_scale设得太大(默认65536)，试试设成1024，或关AMP用FP32试
> 5. **(2%) 数据没归一化**：图像是0-255 uint8输入，没/255也没Normalize，值范围太大Attention QK²计算直接10000+ → Softmax后饱和NaN
>
> Debug代码：在每次Forward之前加：
> ```python
> for name, param in model.named_parameters():
>     if torch.isnan(param).any() or torch.isinf(param).any():
>         print(f"NaN/Inf in parameter {name}, shape {param.shape}")
>     grad_norm = param.grad.norm(2).item() if param.grad is not None else 0
>     if grad_norm > 1000:
>         print(f"⚠️ Large grad in {name}: {grad_norm:.1f}")
> ```
> 面试官延伸：「为什么AMP会NaN？」→ FP16最大数值只有65504，中间乘积Q·K^T很容易超，GradScaler就是通过先×scale把loss放大算，step再÷回来，overflow就跳步更新

### Q35. ViT模型压缩有哪些方法？能压到多少？
> **A**:
> 4类主流方法，实际组合能压ViT-B 86M→5M（17x），精度掉<3%：
>
> | 方法 | 原理 | ViT-B→小 典型压缩率 | 精度损失 |
> |-----|------|-------------------|---------|
> | **知识蒸馏KD** | 大模型(Teacher)的logits/特征/注意力图教小模型(Student) | 3-5x (DeiT蒸馏版) | <1% |
> | **量化 INT8/INT4** | FP32权重→8位/4位定点，scale+zp | 4x-8x | INT8 <0.5%，INT4 <2% |
> | **结构化剪枝** | 移除整头/整层/整通道(结构化硬件友好) | 2-4x (可剪60%头无损失) | <1% |
> | **NAS/Arch搜索** | 搜每层宽度/深度/头数的最优组合(AutoML) | 10x+ (Auto-ViT) | <2% |
>
> 工业组合拳案例：ViT-B → Structured Prune(50%通道×头) → 8-bit Quantization → Tiny Student KD → **Final: 3-5M参数，原始97%准确率，CPU单张推理<10ms**

### Q36. CLIP和ViT关系？CLIP的对比学习loss怎么写？
> **A**:
> CLIP (Contrastive Language-Image Pre-training OpenAI 2021) = 用了两个Transformer做**图文对比学习**，视觉端就是标准ViT
>
> 架构：
> - 图像塔：ViT-B/32 or ViT-L/14 → 输出512维图像特征
> - 文本塔：Transformer Text Encoder (和GPT类似) → 输出512维文本特征
> - Projection Head：两塔特征各自投影归一化到512维球面 (L2 Norm=1)
>
> Loss：对比学习双塔对称InfoNCE Loss
> ```python
> def clip_loss(img_feat, text_feat, temperature=0.07):
>     # 两个特征都已经L2归一化到单位球面了
>     B = img_feat.shape[0]
>     logits = img_feat @ text_feat.T / temperature   # [B,B] 两两cos相似度 / τ
>     # 对称: i2t 和 t2i 两边都算交叉熵
>     labels = torch.arange(B, device=img_feat.device)  # 对角线是正样本对！
>     loss_i2t = F.cross_entropy(logits, labels)
>     loss_t2i = F.cross_entropy(logits.T, labels)
>     return (loss_i2t + loss_t2i) / 2
> ```
> 核心：对角线是正样本(同索引的图文对)，非对角线都是负样本，一个Batch里(B²-B)个负例！Batch越大效果越好(CLIP用32768超大Batch)
>
> 面试绝杀：CLIP是Zero-Shot Learning里程碑，训练好后ImageNet分类**不需要任何训练**，直接用"一张{类别}的照片"文本模板生成1000个text embedding，挑最相似的 → Top1 76.2%，比肩ResNet-50监督训练

---

## 四、ViT变体与对比（8题）

### Q37. Swin Transformer和ViT核心区别？三大创新点是什么？
> **A**:
> 解决ViT 3大痛点：① 单尺度特征 ② N²全局自注意力计算爆炸 ③ 小数据集泛化差
>
> 三大创新：
> 1. **分层金字塔Hierarchical Feature Maps**（像ResNet C2-C5）
>    - Stage1输出H/4×W/4 通道=C
>    - Stage2 PatchMerging 2×2合并 → H/8×W/8 通道=2C
>    - Stage3 → H/16×W/16 4C
>    - Stage4 → H/32×W/32 8C
>    - ✅ 天生适合接FPN做检测/分割/实例分割，ViT做不到
>
> 2. **Window based Multi-Head Self-Attention (W-MSA)**
>    - 把特征图分成不重叠M×M窗口(默认7×7)，只在窗口内算自注意力
>    - 计算量从全局O((HW)²·C) → HW×M²·C 线性！高分辨率N×优势巨大
>    - 缺点：窗口间无通信 → 第3创新点解决
>
> 3. **Shifted Window based MSA (SW-MSA)** 相邻层窗口偏移
>    - 偶数层标准7×7划分，奇数层窗口向右下偏移(⌊M/2⌋, ⌊M/2⌋) = (3,3)像素
>    - 这样新窗口会跨上一层的窗口边界 → 信息自然流动！
>    - 实现用Cyclic Shift循环移位 + Mask屏蔽无效组合，额外开销极小
>
> Swin-T vs ViT-B对比：Swin-T参数28M(少3x)，IN-1K 81.3%比ViT-B 77.9%高3.4%，COCO检测用Swin-T backbone比ResNet-50高7.1mAP

### Q38. 为什么Shifted Window要用Cyclic Shift+Mask？直接分不规则窗口不行？
> **A**:
> 直接分9块不规则窗口：
> - 图8×8，M=4，偏移后3×3=9个窗口：大小分别是(3×3, 3×1, 3×4, 1×3, 1×1, 1×4, 4×3, 4×1, 4×4)
> - 每个窗口形状不一样，无法Batch矩阵运算 → 循环9次每个单独算，速度骤降10x
>
> Cyclic Shift技巧（论文精髓！速度关键）：
> 1. 把特征图的行和列按偏移量(3,3)做**torch.roll循环移动**（行最后3行移最前，列最后3列移最前）
> 2. 移动后又是**规整的2×2=4个标准7×7窗口**，和上一层一样可以Batch一次算完
> 3. 算完Attention，输出再**反向roll回去**恢复原位置
>
> 但是！循环移动把A区(原图左上)和B区(原图右下)凑到了一个窗口里，它们实际上不相邻不能算注意力 → 加**Attention Mask**把不该看的位置分数设为-∞，Softmax后权重=0
>
> 总结：Cyclic Shift解决形状不规整无法Batch计算，Mask解决错误配对，两者配合实现0额外计算开销的跨窗口通信
>
> 面试官追击：「反向传播时roll和mask怎么处理？」→ roll是线性变换矩阵转置就是反向roll，mask直接在前向算完分数mask掉，反向不影响梯度流通(softmax 0位置梯度0)

### Q39. ConvNeXt是什么？和ViT/Swin比优劣？
> **A**:
> ConvNeXt (FAIR 2022) = **把ViT/Swin所有训练/结构trick迁移回纯CNN架构**，证明了很多ViT涨点来自"配方"不是"注意力结构"
>
> 向ViT偷师的7个改动：
> 1. 训练策略：和Swin完全一样(300Epoch AdamW Cosine MixUp CutMix) → CNN老配方90Epoch直接换，+3个点
> 2. 宏观设计：Stage比例1:1:9:1 (Swin比例)，不是ResNet的1:1:3:4 → +0.8
> 3. Patchify Stem：4×4 Conv Stride=4（ViT Patch Embedding等价），不是7×7 Conv+MaxPool → +0.5
> 4. Depthwise Conv：3×3 DW Conv 逐通道(和MHSA每个头独立同理)，宽度系数从64→96 → +1.0
> 5. 激活：ReLU→GELU (ViT同款) → +0.3
> 6. 归一化：BN→LayerNorm (ViT同款)，每个Block只在Conv后用1次LN不是每次 → +0.4
> 7. 下采样：单独一层LayerNorm + 2×2 Conv Stride=2 (Swin PatchMerging) → +0.2
>
> 结果：ConvNeXt-B (88M) IN-1K **87.8%**，Swin-B是87.3%，几乎打平
>