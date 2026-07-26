# Transformer架构 性能优化重难点解析

> 位置: 03-deep-learning/annotated-transformer/doc/
> 配套文档: Transformer架构逐行解析.md | Transformer架构流程图详解.md | Transformer面试题汇总.md

---

## 一、训练稳定性核心难点

### 1.1 为什么原始Transformer难训？6大痛点

| 痛点 | 根因 | 后果 |
|-----|------|------|
| 梯度消失 | Post-LN + 深层叠加 → 梯度逐层乘0.1衰减 | N>6层基本训不动，Loss平的 |
| 学习率敏感 | 无warmup直接上目标LR | 初始几步梯度爆炸NaN |
| 位置编码外推差 | 正弦/可学习长度外推乱掉 | 测试长句翻译BLEU暴跌 |
| Softmax数值不稳 | Q·K^T方差=d_k，太大导致one-hot饱和 | 梯度消失不更新 |
| 标签过拟合 | One-Hot硬标签，模型过度自信 | 泛化差，OOT翻译错 |
| Batch分布漂移 | NLP Padding多长度不一，BN均值方差飘 | LayerNorm代替BN |

### 1.2 解决方案：原始论文7大关键参数

```python
# 论文超参配置 (严格复制)
class TransformerConfig:
    N = 6                     # Encoder/Decoder各6层
    d_model = 512             # 维度
    d_ff = 2048               # FFN膨胀4倍
    h = 8                     # 8头注意力
    d_k = d_v = 64            # 每头512/8=64
    dropout = 0.1             # 各处统一0.1
    warmup_steps = 4000       # warmup 4000步
    lr_factor = 2             # Noam系数
    label_smoothing = 0.1     # ✅必须开！正则化神器
    max_grad_norm = 1.0       # 梯度裁剪

# 4000步warmup图示:
# LR ∝ min(step^-0.5, step·warmup^-1.5)
#      先线性冲到峰值(约4000)，再按1/√step下降
```

> 📌 训练前自检：`warmup=True` + `Noam调度` + `LabelSmoothing` + `GradClip` = 不翻车4件套

---

## 二、Attention计算优化 (O(N²d) → 实用加速)

### 2.1 内存瓶颈：标准Attention 显存公式

```
标准Attention 显存占用 (训练时) per sample per layer:
├── Q,K,V: 3 × N × d (FP16=×2字节) = 3Nd×2 = 6Nd
├── scores: N × N (最大!) = N²×2
├── P_attn: N × N (Softmax后)   = N²×2
├── O: N × d                    = 2Nd
└── dO/dS 反向中间: N × N       = N²×2 (Softmax梯度)

合计 ≈ 4N² + 8Nd  (单位: FP16字节约等于2)

实际数值 N=512, d=512:
  QKV: 6×512×512×2 = 3.1MB
  scores: 4×(512²)×2 = 2.1MB  ← 还好
  N=4096 长文本时:
  scores: 4×16M×2 = 128MB 每层每头 ×12层×8头 = 爆12GB！
```

### 2.2 工程优化方案对照

| 手段 | 速度提升 | 显存下降 | 代码改动 | 影响准确率 |
|-----|---------|---------|---------|----------|
| Flash Attention v2 | **2-4x** | **3-10x** | 替换attention函数 | <0.1% 无感知 |
| XFormers Memory Efficient | 2x | 4x | 一行替换 | <0.5% |
| KV Cache (推理) | **10-100x** | 少量 | 加缓存逻辑 | 0% |
| Grouped Query Attention | ~20% | **K/V省4-8x** | 改维度 | <1% |
| Sliding Window (长文本) | 线性~N | 线性~N | 窗口Mask | N>2K时长文本更准 |

### 2.3 Flash Attention Tiling 图示要点

```
原理图示 (N=4096分Br=Bc=128块)：
┌─────────────────────────────────┐ HBM: 慢但大 (80GB A100)
│ Q[4096,d] K[4096,d] V[4096,d]   │
└──┬──────────────┬───────────────┘
   │Load 切块      │Load
   ▼               ▼
┌──────────┐  ┌──────────┐ SRAM: 快2TB/s 极小(20MB)
│ Q_i[Br,d]│  │K_j,Vj[Bc]│
└────┬─────┘  └────┬─────┘
     │ s = Q_i·K_j^T  │  算[128×128]小块
     │ 在线Softmax(不减Max稳定)│
     │ o_i += P·V_j   │
     └────────┬───────┘
            ▼
       写回 HBM: O_i[Br,d]

结果：完全不构造[4096×4096]中间矩阵！HBM读写字节数从O(N²)降O(N)
```

---

## 三、LayerNorm vs BatchNorm 深度辨析

### 3.1 为什么Transformer必须用LayerNorm？

```
对比归一化维度：
BatchNorm2d (CNN图像): [B, C, H, W] → 在B,H,W维度上归一化 ← 依赖大Batch
  μ_c = 1/(B·H·W) Σ样本 Σ高 Σ宽  X[:,c,:,:]  ← 每个通道独立

LayerNorm1d (NLP文本): [B, N, d] → 在d维度上归一化 ← 和Batch完全无关!
  μ_bn = 1/d Σ维度 X[b,n,:]  ← 每个样本每个位置独立

为什么BN在NLP惨败：
 ① NLP batch size通常小(B=32)，Padding多，BN统计量噪声大(±50%波动)
 ② 序列长度不一，短样本被Pad统计污染 (Pad全0拉低μ)
 ③ Batch维度跨样本独立 → 预测时要running_mean 训练/推理差异大

LayerNorm优点:
  ✅ Batch=1都能算 (和B无关)
  ✅ Train/Eval一致 (无running统计)
  ✅ 每个样本独立归一化，分布稳定
```

### 3.2 Pre-LN vs Post-LN (训练稳定性分水岭)

```
Post-LN (老Transformer Vaswani 2017):
  X → MHA → Add → **LN** → FFN → Add → **LN**
  问题: 梯度回传时要过两次LN，梯度被缩放~0.1^L倍，L=6时=百万分之一

Pre-LN (GPT-2/现在所有LLM标准):
  X → **LN** → MHA → Add → **LN** → FFN → Add
  优势: 残差主路直通! 梯度直接传 1×梯度 深层也稳
  数学证明 (Xiong 2020): Pre-LN 每一层的梯度 Lipschitz ≤1 保证稳定

迁移经验:
  NLP翻译任务老Post-LN: 必须严格warmup 4000步，否则NaN
  CV/LLM新Pre-LN: 直接上LR也不炸，warmup可以短到100步
```

---

## 四、位置编码深度优化

### 4.1 位置编码选型对比表

| 编码 | 外推长文本 | 相对位置 | 计算量 | 代表 | 推荐场景 |
|-----|----------|---------|-------|-----|---------|
| 可学习Learned | ❌极差，>训练长=乱 | ❌隐式 | O(1)参数 | ViT/BERT | 长度严格固定 <512 |
| 正弦Sinusoid | ✅还好 | ❌绝对 | O(1)无参数 | 原Transformer | 经典基线，老论文复现 |
| **RoPE旋转** | ✅极佳 + NTK外推 | ✅完美 | O(d)计算 | **LLaMA/GPT-J** | ⭐ 现在主流LLM必选 |
| **ALiBi** | ✅外推最佳 | ✅自然线性 | O(1) | MPT/BLOOM | 超长上下文8K+首选 |

### 4.2 RoPE核心思想（面试画图）

```
RoPE = 旋转位置编码 Rotary Position Embedding
核心: 不是给Embedding加位置，是给Q/K在Attention之前乘旋转矩阵

复数形式 (每个头的偶数/奇数维组成复数对):
  q'_m = f(q_m, m) = q_m · e^{i m θ}  复数: 模不变，角度旋转mθ
  k'_n = f(k_n, n) = k_n · e^{i n θ}

点积性质: <q'_m, k'_n> = Re(q'_m · conj(k'_n)) = Re(q_m·conj(k_n) e^{i(m-n)θ})
  ✅ 只依赖相对位置(m-n)！和绝对位置无关！
  ✅ 序列长度任意长，旋转θ和长度无关，自然外推！

实数旋转矩阵实现 (每对维度):
  [q_i  ]   =  [cos(mθ)  -sin(mθ)] [q_i]
  [q_i+1]      [sin(mθ)   cos(mθ)] [q_i+1]

推理外推 (训练长度4K，推理8K):
  原始 RoPE: θ_i = 10000^(-2i/d)  固定频率
  NTK-aware: θ'_i = 10000^(-2i/d · α)  α=缩放因子>1 降低高频
  效果: 无需微调，直接长度×4 外推，困惑度几乎不涨
```

---

## 五、Label Smoothing + 优化调度

### 5.1 Label Smoothing正则化机制

```
硬标签 One-Hot:  [0, 0, 1, 0, 0, ...]  (第3类是真)
软标签 Smooth:   [ε/V, ε/V, 1-ε+ε/V, ε/V, ...]   ε=0.1
  所有错误类给一点点概率 ε/V，真类= 1-ε + ε/V

为什么有效 (对应3大好处):
  ① 防过拟合: 模型不能把真类学到1.0，否则其他类-KL散度会罚
  ② 防数值不稳: 预测真类0.99999和0.999的Loss差很小，梯度不会爆
  ③ 知识蒸馏友好: 软标签本身就有类间相似度信息，Teacher输出一样

Loss公式:
  Loss = - Σ_c y_smooth[c] · log(p[c])
       = (1-ε)·CE_硬 + ε · 均匀分布的交叉熵

代码 (annotated-transformer自带):
  class LabelSmoothing(nn.Module):
      def __init__(self, size, padding_idx, smoothing=0.1):
          self.criterion = nn.KLDivLoss(size_average=False)  # KL散度距离
      def forward(self, x, target):
          true_dist = x.data.clone().fill_(ε/(V-1))
          true_dist.scatter_(1, target.data.unsqueeze(1), 1-ε)
          return self.criterion(F.log_softmax(x), true_dist)
```

### 5.2 NoamOpt调度器数值例

```
公式: LR = factor · d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)

例: factor=2, d_model=512, warmup=4000
┌──────┬───────────────┬─────────────────┐
│ step │ 公式分支       │ 学习率数值       │
├──────┼───────────────┼─────────────────┤
│ 1    │ step·warmup^-1.5│ 2·0.044·~0    ≈ 2e-7 │
│ 1000 │ 线性分支       │ 2·0.044·1e-11·1000 ≈ 8.8e-5 │
│ 4000 │ 交叉点=峰值     │ 2·0.044·4000^-0.5 ≈ 1.4e-3 │
│ 16000│ 衰减分支       │ 2·0.044·16000^-0.5 = 7e-4 │
│ 64000│ 衰减分支       │ 2·0.044·(64000)^-0.5 = 3.5e-4 │
└──────┴───────────────┴─────────────────┘
图示: 先快速冲到1.4e-3 (warmup保护初期参数随机)，再缓慢1/√x下降
```

---

## 六、推理性能优化 (Beam Search / KV Cache)

### 6.1 朴素推理 vs KV Cache加速比

```
任务: 翻译T=100个词，Batch=1，N=100源句

❌ 无Cache (愚蠢版):
  词1: 输入<sos>            算Encoder 1次 + Decoder QKV[1,100]×1次
  词2: 输入<sos, w1>       算Encoder 1次 ❌(重算!) + Decoder QKV[2,100]×1次 ❌(前面K,V重算)
  ...
  词100: 输入长度100        Decoder算: N²+dN = 100²+100·d
  总计算量: Σ_{t=1..T} t² ≈ T³/3 ≈ 33万次矩阵乘

✅ 有KV Cache (工业标准):
  词1: 算Encoder 1次 ✅ 结果存！Decoder第1层K1,V1存Cache
  词2: 只算w2的K2,V2 → 和Cache [K1..K1] 拼 → Attention不用重算历史
       Encoder复用！Decoder: 只算新Token的QKV，O(d·N)不是O(t²)
  总计算量: T·N·d ≈ T×线性  约5万，实际速度**30-80x**
  显存代价: 每层2×T×d FP16 = 2·100·512·2B=204KB/层，6层=12MB 划算！
```

### 6.2 Beam Search参数优化

| 参数 | 推荐值 | 影响 |
|-----|-------|------|
| Beam Size | 4~6 (翻译)，1 (摘要) | Beam越大质量越好但速度×Beam，>8边际递减 |
| Length Penalty α | 0.6 (翻译短) / 1.0 / 1.2 (长文档) | 惩罚短句，防止翻译偷懒给"I am fine"这种废话 |
| Coverage Penalty | β=0.1~0.5 | 惩罚源端没覆盖到的词，翻译漏词减少 |
| Early Stop | True | 所有Beam出EOS就停，不用跑到max_len |

---

## 七、常见Bug & Debug checklist

| 症状 | 排查项 (按概率) |
|-----|---------------|
| Loss初始就NaN | ① Noam warmup关了？② Label Smoothing ε设太大(>0.2)？③ LR×10了？ |
| 训练20Epoch Loss不降 | ① 有没有加subsequent_mask？Decoder偷看未来直接背下来！② Padding_idx设对没？ |
| 训练好测试乱码 | ① 推理Batch有没有加Padding Mask？② 有没有Shift Right输入<SOS>？ |
| BLEU低(翻译<30) | ① Beam=1？开成4试试+2~5BLEU ② Checkpoint取Best还是Last？(取Best) |
| 速度比基线慢3x | ① 有没有用KV Cache？② 推理Batch设1？改成大Batch/Beam用矩阵 |
| 长文本翻译崩掉 | ① 位置编码外推差？换RoPE+ALiBi ② Context窗口截断滑动窗口解 |