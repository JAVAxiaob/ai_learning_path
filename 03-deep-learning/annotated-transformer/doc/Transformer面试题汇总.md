# Transformer架构 面试题汇总 (45题)

> 位置: 03-deep-learning/annotated-transformer/doc/
> 配套文档: Transformer架构逐行解析.md | Transformer架构流程图详解.md | Transformer性能优化重难点.md

---

## 📊 题目分布

| 类别 | 题数 | 出现频率 |
|-----|-----|---------|
| 基础架构 (Encoder/Decoder) | 15题 | ⭐⭐⭐⭐⭐ |
| 注意力机制深度 | 10题 | ⭐⭐⭐⭐⭐ |
| 训练优化与正则 | 10题 | ⭐⭐⭐⭐ |
| 位置编码 & 变体 | 5题 | ⭐⭐⭐⭐ |
| 工程部署 | 5题 | ⭐⭐⭐ |

---

## 一、基础架构题（15题）

### Q1. 画原始Transformer完整架构图，标出各模块名和数据流
> **A**:
> ```
> (源词嵌入+PE)→Dropout
>     │
>     ▼
> ┌───────────────┐  ×N=6
> │ MultiHead Attn│ ← 自注意力 + Padding Mask
> │  Add & Norm   │
> │ Feed Forward  │ ← FFN 512→2048→512
> │  Add & Norm   │
> └───────┬───────┘
>         │ Memory (Encoder输出给所有Decoder)
>         ▼
> ┌──────────────────────┐ ×N=6
> │ Masked MultiHead Attn│ ← 自注意力 + 下三角Mask(防偷看未来)
> │  Add & Norm          │
> │ MultiHead Attn       │ ← Cross注意力! Q=Decoder, KV=Encoder Memory
> │  Add & Norm          │ ← 源序列Padding Mask
> │ Feed Forward         │
> │  Add & Norm          │
> └──────────┬───────────┘
>            ▼
>    Linear → LogSoftmax → 预测下一个词
> ```
> ⚠️ 面试官常卡3点：有没有标**下三角Mask**、**Cross注意力KV来源**、**2个Add&Norm每子层**

### Q2. Transformer相比RNN/LSTM的3大优势？为什么能取代RNN？
> **A**:
> 1. **并行计算**：RNN第t步输出要等第t-1步算完(串行)；Transformer Self-Attention所有位置一次性矩阵乘并行算，训练速度×10-100倍，GPU利用率拉满
>
> 2. **长距离依赖建模**：RNN要传1000步梯度，中间每步乘Wh，梯度×(Wh)^1000直接指数消失；Transformer任意两个位置一跳直达，梯度O(1)路径，不管多长文本依赖都能建模
>
> 3. **可解释性 + 多模态对齐**：注意力权重天然可视化，知道模型"看了哪些词"，Encoder-Decoder Cross注意力直接看哪个源词对应当前翻译目标词；RNN是黑盒隐状态，难解释
>
> 追问「那Transformer没有缺点吗？」→ 答：① O(N²)计算量，超长序列爆 ② 位置信息不如RNN天生自带，要额外加PE ③ 小数据比RNN/LSTM泛化差(归纳偏置弱)

### Q3. Encoder和Decoder各有哪些Attention？Q/K/V来源分别是？
> **A**:
> 三种Attention，Q/K/V来源是面试必考点，答错直接挂：
>
> | 位置 | Attention类型 | Query Q来源 | Key K来源 | Value V来源 | Mask |
> |-----|--------------|------------|----------|------------|------|
> | Encoder每层 | 自注意力 | 上层Encoder输出 | 同Q | 同Q | 源端 Padding Mask |
> | Decoder第1个 | **Masked自注意力** | 上层Decoder输出 | 同Q | 同Q | 下三角+目标Padding |
> | Decoder第2个 | **Cross交叉注意力** | 上层Decoder输出 | **Encoder最顶层Memory** | **Encoder最顶层Memory** | 源端Padding Mask |
>
> 记忆口诀：**自注意力QKV一家亲，跨注意力Decoder是Q去KV家做客(Encoder)**

### Q4. 为什么Decoder第一个注意力要Mask？subsequent_mask怎么生成？
> **A**:
> 训练时Decoder输入是完整目标句子(例如 "<SOS> I am a student")，如果不加Mask，预测第3个词"am"的时候Attention能看到后面"student"答案 → 直接作弊背下来，推理时没有未来词就会崩
>
> 代码（白板写）：
> ```python
> def subsequent_mask(size):
>     "下三角全1矩阵，size×size，对角线以上=0(看不到)"
>     attn_shape = (1, size, size)
>     mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
>     # 上三角(包括对角线以上)=1，取反=下三角True=能看到
>     return torch.from_numpy(mask) == 0
>
> # 应用: scores.masked_fill(mask == 0, -1e9) 把看不到的位置Softmax前=-∞→0权重
> ```
> 示例 size=4: Mask矩阵是 [[T,F,F,F],[T,T,F,F],[T,T,T,F],[T,T,T,T]] → 第i行看前i个

### Q5. FFN前馈网络结构是什么？为什么先膨胀4倍再收缩？
> **A**:
> ```python
> class PositionwiseFeedForward(nn.Module):
>     def __init__(self, d_model, d_ff, dropout=0.1):
>         super().__init__()
>         self.w_1 = nn.Linear(d_model, d_ff)   # 512→2048  4x膨胀
>         self.w_2 = nn.Linear(d_ff, d_model)   # 2048→512  还原
>         self.dropout = nn.Dropout(dropout)
>     def forward(self, x):
>         return self.w_2(self.dropout(F.relu(self.w_1(x))))
> ```
> 为什么要"胖-瘦"结构？
> - 增加非线性容量：d_ff=4d是论文经验，越大容量越高(d_ff=8d也可但过拟合风险大)
> - 类比CNN 1×1 Conv：先升维(更多通道组合特征) + ReLU非线性 + 降维回压缩
> - 参数量：2×d×4d = 8d²，和MHA的4d²互补，约占每层67%计算量
>
> 面试加分：SwiGLU激活(LLaMA/PaLM)替代ReLU + w1/w2/w3三个Linear，FFN从2个Linear变3个(SwiGLU)，同参数量SOTA

### Q6. 残差连接+LayerNorm的顺序？Pre-LN和Post-LN哪个好？
> **A**:
> ```
> ❌ Post-LN (原始Transformer Vaswani 2017老版):
>   X → Attn → Add → LN → FFN → Add → LN
>   主路梯度经过LN，被缩放，6层后梯度范数×0.001 → 深层难训，必须严格warmup
>
> ✅ Pre-LN (GPT2/3、BERT、现在所有LLM标准):
>   X → LN → Attn → Add → LN → FFN → Add
>   残差主路 = X + Attn(LN(X)) 梯度直接恒等映射传！
>   100层也能训不炸，warmup可短到100步
> ```
> 结论：**所有新模型都用Pre-LN**，Post-LN只在复现老论文才用。面试必须答出Pre-LN主路梯度直通的关键

### Q7. 为什么要除以√d_k (Scaled Dot-Product)？不除会怎样？
> **A**:
> 假设Q,K每个元素独立N(0,1)，内积q·k = Σ_{i=1..d_k} q_i k_i
> 方差：Var(q·k) = d_k × Var(q_i)×Var(k_i) = d_k
>
> 不除√d_k的后果：
> - d_k=64 → 方差=64 → 分数分布±40范围 → Softmax前极大极小数 → 输出one-hot (最大位置≈1其他≈0)
> - Softmax饱和区梯度0！**梯度消失** → 模型训不动
>
> 除√d_k后：方差=1 → 分数~N(0,1) ±3范围 → Softmax平滑分布 → 梯度正常回传
>
> 追问：「T5模型不除√d_k怎么处理的？」→ 答T5把Q在投影后直接加了1/√d_k缩放，属于变种最终效果一样

### Q8. Transformer参数量怎么估算？(Encoder 6层 d=512 h=8)
> **A**:
> 逐层估算（忽略LN/bias小项 <3%）：
> ```
> Embedding: V×d (词表×维度) = 37000×512 ≈ 19M
>
> Encoder单层:
>   MHA: Q/K/V/Out 4个Linear → 4×(d×d) = 4·512² ≈ 1.05M
>   FFN: Linear1+Linear2 → d·4d + 4d·d = 8·512² ≈ 2.1M
>   单层≈3.15M
> Encoder 6层: 6 × 3.15M ≈ 18.9M
>
> Decoder单层 (多1个Cross Attention!):
>   Masked-MHA: 4d² = 1.05M
>   Cross-MHA : 4d² = 1.05M  ← 这里多一套QKV/KV
>   FFN       : 8d² = 2.1M
>   单层≈4.2M
> Decoder 6层: 6 × 4.2M ≈ 25.2M
>
> Generator Head: d×V = 512×37000 ≈ 19M
>
> 合计: 19M + 18.9M + 25.2M + 19M ≈ **82M**
> ```
> 快速公式：每层Encoder≈12×d²（6×2×d²？修正：MHA4d²+FFN8d²=12d²）；Decoder每层16d²。总≈12Nd²+16Nd²+2Vd

### Q9. 为什么FFN叫Position-wise？和普通MLP区别？
> **A**:
> Position-wise = **对每个位置独立、相同的MLP**，所有位置参数共享
> 输入[B,N,d] → Linear1作用在最后d维，等价于对N个位置分别做相同变换：
> ```python
> # for循环等价写法(但慢)
> for b in range(B):
>     for n in range(N):
>         out[b,n] = w2(relu(w1(x[b,n])))  # 每个位置独立! 用同一套w1/w2
> ```
> 和普通MLP区别：普通MLP会把N×d展平成N*d，位置之间会乘权重交叉；这里Position-wise**不跨位置交互**，跨位置交互只在Self-Attention里做 → 解耦"跨位置信息路由"和"单位置特征变换"两件事

### Q10. Transformer为什么需要Multi-Head？单头不行吗？
> **A**:
> 4大好处：
> 1. **子空间分离**：8个头各自在8个64维子空间学习不同模式 → 头1学语法主谓一致，头2学指代消解，头3学短语搭配…单头只能混在一起学
> 2. **集成效应**：多个独立头相当于Bagging集成，预测更稳，梯度噪声低
> 3. **计算量无增加**：4×(d·(d/8)) = 4d²，和单头大矩阵的4×d·d FLOPs一模一样！白嫖多样性
> 4. **微调灵活性**：下游任务可以剪枝掉不重要的头(通常可以剪1/3头无损)
>
> 单头效果：INMT任务单头BLEU比8头低3-5个点，但在超小模型/超小数据集上可以单头防过拟合

### Q11. Encoder输出的Memory是怎么给Decoder用的？6层都给还是只给最后一层？
> **A**:
> 只给**Encoder最顶层(第6层)的输出**作为Memory，传给**每一层Decoder的Cross Attention** (不是只给第6层Decoder)
>
> 流程：`Memory = Encoder6(x_src)`，然后`Decoder1→CrossAttn(Q=Dec1_hidden, KV=Memory)`，...，`Decoder6→CrossAttn(Q=Dec6_hidden, KV=同一个Memory)`
>
> 设计好处：6层Decoder每一层可以在不同语义层级去对齐源端 → 低层Decoder对齐原词/短语，高层Decoder对齐整句语义

### Q12. Dropout在Transformer里用在哪几处？(面试容易漏)
> **A**:
> 原始论文共5个Dropout，p=0.1统一：
> ```
> ① Embedding+PE之后  dropout(embed(x)+pe)
> ② Self-Attention权重 Softmax之后，加权V之前 dropout(p_attn)
> ③ Self-Attention残差相加之前  dropout(attn_out) 再 +x
> ④ FFN激活之后Linear2之前 dropout(relu(w1(x)))
> ⑤ FFN残差相加之前  dropout(ffn_out) 再 +x
> ```
> 现代LLM趋势：Dropout逐渐降低→0.0/0.05，大模型参数量够大本身正则化足够，Dropout反而拖训练速度

### Q13. Beam Search vs Greedy Search优劣？Beam Size怎么选？
> **A**:
> | 方法 | 原理 | 翻译质量 | 速度 |
> |-----|------|---------|-----|
> | Greedy贪心 | 每步取概率最大1个词 | 差 容易局部最优死路 | 快 1x |
> | Beam Search束搜索 | 每步保留TopK(Beam)个候选 | 好 BLEU高4-8点 | K倍慢 |
>
> Beam Size选型经验：
> - 机器翻译BS=4-6最优，>8边际收益递减，速度×8不划算
> - 生成摘要/创意写作BS=1 (多样！贪心太确定Beam重复模板)
> - 代码生成 BS=10 (正确性优先)
>
> Beam Search坑：
> - 不出<EOS>越长越好 → 加Length Penalty / 概率归一化除长度
> - 重复"NLP NLP NLP" → 加N-gram重复惩罚 (禁止3-gram重复)

### Q14. Label Smoothing作用？ε=0.1怎么理解？
> **A**:
> 把硬标签One-Hot软化，防止模型过度自信(学到P(正)=1.0，梯度0死区)
> 公式：真类=1-ε，其他V-1个类各分ε/(V-1)
> 例：三分类 ε=0.1，真类是2 → 软标签=[0.05, **0.90**, 0.05]
>
> 效果三方面：
> 1. **正则化**：KL散度惩罚类别过度自信，泛化提1-2% BLEU
> 2. **数值稳定**：Softmax+Log 真类预测0.99999和0.99的Loss差很小
> 3. **知识蒸馏桥梁**：软标签本身和Teacher模型输出的分布是同一形式，天然兼容
>
> 注意坑：训练Accuracy会**看起来更低**(0.9 vs 0.995)是正常的！是标签软化不是模型变差，测试集指标反而更高

### Q15. Multi30k/WMT14翻译任务标准评估指标？怎么算BLEU？
> **A**:
> BLEU (Bilingual Evaluation Understudy) = N-gram精确度的几何平均 × 长度 brevity penalty
> ```
> 例:
> 参考翻译: "the cat is on the mat"
> 模型翻译: "the the the the the the the"
> Unigram 精确度: 7个"the"里参考最多2个 → min(7,2)/7 = 2/7
> Bigram: 所有"the the"参考里0 → BP(过短)
>
> BLEU = BP × exp( Σ 1/4 log(Precision_n) ), n=1..4
> BP = 1 if len(hyp)≥len(ref), else exp(1 - len(ref)/len(hyp))
> ```
> 常见参考分：Multi30k德译英 基线≈35，Annotated-Transformer正确训练≈37-38，SOTA≈42

---

## 二、注意力机制深度题（10题）

### Q16. 逐行推导QKV Multi-Head Attention + Concat过程
> **A**:
> (面试官会盯着白板写矩阵维度！)
> ```python
> # 输入: query [B, N_q, d], key [B, N_k, d], value [B, N_k, d], h=8
> d_k = d_v = d // h = 64
> B = query.size(0)
>
> # 1. 线性投影
> Q = linear_q(query)  # [B, N_q, 512]
> K = linear_k(key)    # [B, N_k, 512]
> V = linear_v(value)  # [B, N_k, 512]
>
> # 2. 拆多头: 把最后d维拆成 h × d_k, 再把头挪到第2维方便并行
> Q = Q.view(B, -1, h, d_k).transpose(1, 2)  # [B, 8, N_q, 64]
> K = K.view(B, -1, h, d_k).transpose(1, 2)  # [B, 8, N_k, 64]
> V = V.view(B, -1, h, d_v).transpose(1, 2)  # [B, 8, N_k, 64]
>
> # 3. Scaled Dot-Product Attention 所有头并行算!
> scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # [B,8,Nq,Nk]
> if mask is not None:
>     scores = scores.masked_fill(mask == 0, -1e9)
> attn = F.softmax(scores, dim=-1)  # [B,8,Nq,Nk] 每行和=1
> attn = dropout(attn)
> x = attn @ V  # [B,8,Nq,64]  加权求和V
>
> # 4. Concat多头还原回去: 头的维度挪回来再合并最后两维
> x = x.transpose(1, 2).contiguous()  # [B, N_q, 8, 64]
> x = x.view(B, -1, h * d_v)          # [B, N_q, 512]  Concat!
>
> # 5. 输出Linear投影
> return linear_out(x), attn  # [B, N_q, 512] 和输入形状一模一样!
> ```
> 面试官最爱卡：`.contiguous()` 不加view会报错。为什么？transpose后张量非物理连续，view要求连续内存。

### Q17. Attention计算里的两种Mask (Padding/Src/Tgt)区别和应用
> **A**:
> 3种Mask，场景分清楚：
>
> | Mask名 | 形状 | True=保留的位置 | 用在哪 | 目的 |
> |-------|------|---------------|-------|------|
> | **Key Padding Mask** | [B,1,1,N_k] | 非<PAD> | Encoder+Decoder Cross | 不把<PAD>零向量当有效Key |
> | **Subsequent/因果Mask** | [1,1,N_t,N_t] | 下三角 对角线及以下 | Decoder第一层自注意力 | 训练时防偷看未来词 |
> | **联合Target Mask** | [B,1,N_t,N_t] | 两者交集 | Decoder第一层 | SubseqMask AND PaddingMask |
>
> 面试加分：现代实现中都是在forward前把所有Mask统一转成加性的bias（True→0，False→-inf），加到scores上不用if判断，GPU效率高

### Q18. Flash Attention原理？为什么显存少速度快？(硬通货)
> **A**:
> 核心 = **GPU内存层次利用 + 在线Softmax避免HBM读写中间矩阵**
>
> 标准Attention的显存杀手：显式构造[N,N] scores矩阵存在HBM(80GB大慢)，读写巨慢
>
> Flash算法三步骤：
> 1. **分块Tiling**：把Q分成Br块、K/V分成Bc块 → Br×Bc刚好塞进SRAM(20MB小快)
> 2. **在线Softmax**：传统Softmax要知道全局max做数值稳定；Flash用增量式`m_new=max(old,local_max)`和修正的`exp(oldmax-newmax)`系数，分块算也能得到**数学精确等价的完整Softmax结果**
> 3. **反向重计算**：Forward阶段不存Softmax矩阵(省显存)，Backward在SRAM重算局部注意力+梯度 → 虽然多算了点FLOPs但省了HBM带宽(速度瓶颈是带宽不是算力!)
>
> 量化效果：A100 FP16 N=4096：标准=1.0x 90ms；FlashAttn2=3.2x 28ms，显存占用从**2.2GB→320MB (7x)**

### Q19. Cross Attention为什么是Decoder用Q、Encoder提供KV？反过来行吗？
> **A**:
> 直觉理解："我(Decoder当前目标词)现在要找答案，我有个Query(想知道什么)，去所有源端KV(资料库里的键值)检索最相关的" → 非常自然
>
> 反过来（目标KV给源端Q查）**数学上完全可以**，但任务语义不匹配：
> - 机器翻译：是生成目标词的时候要去"查"源端，所以Q在目标端
> - 图文captioning：生成文字时查图像区域 → Q在文本Decoder，KV在视觉Encoder
>
> 对称场景（如CLIP对比学习）两边各有CrossAttn（可选），是双向对齐，反过来也行，主要看谁是"主动查询方"

### Q20. GQA/MQA (分组/多查询注意力) 是什么？和标准MHA区别？
> **A**:
> LLM推理时K/V Cache是最大的显存瓶颈（生成4K词，70B模型要存2×层数×d×B×4B ≈ 单样本几十GB！）
>
> 3种Attention结构：
> ```
> 标准 MHA: 8头Q + 8头K + 8头V   ← KV Cache是8份，显存大
>         Q Q Q Q Q Q Q Q
>         K K K K K K K K
>         V V V V V V V V
>
> MQA: 8头Q + **1头共享K** + **1头共享V**  ← Google PaLM 2022
>         Q Q Q Q Q Q Q Q
>         ↘ ↓ ↙ ↘ ↓ ↙ ↘ ↓
>             K    K    (共享，只1套!)
>             V    V
>
> GQA: 8头Q + 2组K + 2组V (4头Q共享1套KV)  ← Meta LLaMA 2 70B
>         QQQQ QQQQ  (分2组)
>          K      K
>          V      V
> ```
>
> 效果：MQA推理显存×8(同吞吐)但质量略降(1%)；GQA 70B是**质量和显存最优点**，LLaMA 2 70B→70B-Chat推理标配

### Q21. 为什么自注意力可以捕获长距离依赖？和CNN感受野对比
> **A**:
> 数学上：第L层CNN感受野大小 = 1 + L×(k-1)，k=3，要覆盖512长度 → L≥256层
>
> 第1层Transformer自注意力：**任意两个位置一跳直达！感受野=全序列长度N**
> 路径长度：O(1)，梯度只过1个Attention节点就能传，不会指数消失
>
> 形象理解：CNN是"传话游戏"1→2→3→...→500，传到第500位消息变味；Transformer是"500人群聊"，1号和500号直接聊天，没有中间商
>
> 代价：CNN感受野局部→计算量线性O(N)；Transformer全局→O(N²)，所以有局部窗口/滑动窗口注意力(N>4K时妥协)

### Q22. 时间复杂度和空间复杂度对比 Transformer vs RNN vs CNN (k=3)
> **A**:
> N=序列长, d=维度, k=卷积核
>
> | 模型 | 每层时间复杂度 | 每层空间(参数量) | 顺序最大路径长(依赖路径) |
> |-----|-------------|----------------|----------------------|
> | RNN | O(N·d²) | O(d²) | O(N) 串行最长 |
> | CNN (k=3) | O(N·k·d²) | O(k·d²) | O(log_k N) 跳表式 |
> | **Transformer** | **O(N²·d)** | O(d²) | **O(1) 最短！** |
>
> 选型拐点：当N < ~4096时，N²≈16M × d=512 ≈ 80亿次运算，GPU刚好撑住；N>8192必须用线性/稀疏注意力变体

### Q23. 可视化Attention Map有哪些模式？(解释性)
> **A**:
> 机器翻译Encoder-Decoder Cross注意力图最有信息量：
> 1. **对齐对角线**（正常）：Deutschland→Germany 基本1:1对应在对角线上附近
> 2. **多对一**："New York"两个源Token对应翻译"NuevaYork"一个目标Token → 模型学到短语合并
> 3. **长距离跳连**：(定冠词/指代) "The cat ... it"中目标"it"对齐到源端"cat"，距离20词以上 → 证明学到长依赖
> 4. **乱序散点**（坏训练）：无规律，可能超参错，注意力没收敛
>
> 头功能分工：低层头抓相邻词(局部n-gram)，高层头抓指代对齐/语义长距离依赖

### Q24. 为什么要Concat多头再乘一个Out Linear？去掉Out Linear会怎样？
> **A**:
> 去掉可以跑，但会有两个问题：
> 1. **子空间叠加没加权**：8个头各输出64维拼512维，直接输出的话每个头的64维是"硬拼接"，不能学"给头1×1.5头2×0.2"的线性组合调权重
> 2. **表达能力损失证明**：设h个头输出Concat是v=[v1;v2;...vh]，加个Out=W_out×v可以学任意线性映射，参数量d²；没有的话相当于W_out=I，少了d²参数(相当于整个MHA的25%参数量)，BLEU掉2-4点

### Q25. KV Cache为什么不缓存Q？只缓存K和V？
> **A**:
> 自回归第t步，只生成第t个新Token xt，所以新Q只来自xt（1个位置），Q矩阵大小是[B, h, 1, d]，没有历史Q要算 → **不需要缓存Q**
>
> 但K和V：计算Attention要 Q_t · [K_1..K_t]^T，需要前面t-1步所有的历史K/V！不存的话就要重算前t-1步的K/V = T²/2计算量 → 所以缓存K/V是O(T)存储换O(T²)→O(T)计算，是自回归推理的核心优化

---

## 三、训练优化与正则（10题）

### Q26. Noam学习率调度器公式？为什么warmup必须有？
> **A**:
> 公式: `lr = factor · d_model^(-0.5) · min( step^(-0.5), step·warmup_steps^(-1.5) )`
> 两条线取min：
> ```
>  0~4000步: step·W^-1.5 = 线性增长 (约等于warmup)
>  4000步+: step^-0.5    = 1/√step 余弦近似衰减
>  交叉点 step=warmup=4000 达到峰值
> ```
> 为什么warmup？初期参数随机，Q/K乱投影，Attention分数方差大，LR大了会直接"过拟合初始噪声"+梯度爆。warmup用小LR逐渐进入稳定区，是Transformer训练稳定性的命脉

### Q27. Adam/AdamW/SGD在Transformer上怎么选？对比表
> **A**:
> 选型结论：**所有Transformer默认AdamW**
>
> | 优化器 | 适合Transformer | 典型LR | 收敛速度 | 泛化 | 备注 |
> |-------|---------------|-------|---------|-----|------|
> | SGD+Momentum | ❌ 几乎不用 (warmup要特别久) | 1e-1 | 慢 10x Epoch | 略好 | CNN还在用 |
> | Adam | ⚠️ 可用但不推荐 | 1e-3 | 快 | 一般 | Weight Decay和动量耦合 → 实际衰减被稀释 |
> | **AdamW** | ✅ 标配首选 | 3e-4(大模型5e-5) | 最快 | 最好 | Decoupled WD衰减到位，LLM 100%选 |
> | LAMB | ⚠️ 超大Batch>4096 | 1e-2 | 快 | 一般 | 分层归一化LR，专门解决大Batch训练不稳定 |

### Q28. 梯度裁剪为什么Transformer必加？max_norm选多少？
> **A**:
> 原因：Attention分数矩阵偶尔会出极端值(1e-99/1e+99)，导致某一步梯度范数1e+6直接参数飞出去→权重NaN。Clip后即使梯度爆，也最多更新norm的幅度，不炸

> 公式：`g_clip = g * min(1, max_norm / ||g||₂)` 梯度模超max_norm时按比例缩

> max_norm经验值：1.0 通用；RNN可以5.0，**Transformer统一1.0**(论文默认+实操稳定)

### Q29. 训练Transformer Loss不下降的Top 5排查原因 (80%概率命中)
> **A**:
> 1. **(40%) 学习率错！** ：AdamW LR设成1e-2了？(应该1e-3~3e-4)；Noam没加warmup？
> 2. **(20%) Mask写错**：Decoder subsequent_mask把上三角保留了(看未来直接作弊) → 训练Loss奇低，测试时傻的
> 3. **(10%) Embedding没乘√d**：`x = self.dropout(self.embed(x) * math.sqrt(self.d_model))` → 漏乘×√d数值范围不对，PE被淹
> 4. **(8%) 输入没Shift Right**：训练时Decoder tgt输入应该是`<SOS> I am`，tgt_y目标是`I am <EOS>`，tgt=tgt_y对齐错位
> 5. **(5%) 词表Embedding和Generator共享权重没做**：现代标配省参数量，不做也行但慢+过拟合
>
> Debug技巧：先跑**1个极小Batch=2样本**过拟合，应该100Epoch内Loss→0 → 如果连这都做不到，一定是代码Bug不是数据问题

### Q30. 梯度消失/爆炸怎么判断？怎么解决？
> **A**:
> 判断：Tensorboard/WandB画每层参数梯度L2范数随Step曲线
> - 梯度消失：第1层梯度范数<输出层1% → 1e-7左右 → Post-LN深网络
> - 梯度爆炸：梯度>1e+4 → 参数更新跳步Loss暴涨NaN
>
> 解决全套：
> ```
> 消失 → ① 换Pre-LN不用Post-LN  ② 残差连接确保没写错  ③ 降低LR
> 爆炸 → ① 必加GradClip norm=1.0  ② AMP 调小GradScaler  ③ 小学习率+更长warmup
> 通用 → 100%用AdamW，加warmup，LabelSmoothing减少数值极端值
> ```

### Q31. 为什么要做标签平滑(Label Smoothing)？会不会影响训练准确率？
> **A**:
> 前面讲过机制，但**面试追问训练准确率**是区分点：
> - 训练Accuracy会**看起来低**：真类标签是0.9，模型预测0.9就是最优，不可能像one-hot那样做到99.9%Acc → 是**正常现象**不要改回去！
> - 但是**验证/测试集的BLEU/Acc都会更高**：正则化有效，模型不会去学噪声样本的死记硬背
> - 知识蒸馏坑：Student用Label Smoothing训练过 → Teacher Soft Target + Smooth标签要一致，不然信号冲突
>
> 典型ε值：机器翻译0.1，图像分类0.1，超大模型0.0~0.05(不需要太强正则)

### Q32. Weight Decay在AdamW里设置多少？哪些参数不做WD？
> **A**:
> ViT/Transformer标准：weight_decay=**0.05** (比CNN 1e-4大500倍！)
>
> 不做Weight Decay的参数3类（面试细节点）：
> ```python
> no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
> # PyTorch分组优化:
> optimizer_grouped_parameters = [
>     {"params": [p for n,p in model.named_parameters()
>                 if not any(nd in n for nd in no_decay)],
>      "weight_decay": 0.05},  # Linear/Conv权重正常衰减
>     {"params": [p for n,p in model.named_parameters()
>                 if any(nd in n for nd in no_decay)],
>      "weight_decay": 0.0},   # bias / LN参数不罚!
> ]
> optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4)
> ```
> 为什么不罚bias/LN？LN的beta/gamma是平移缩放，数值小罚了破坏归一化；bias正则化作用极小又增加不必要约束

### Q33. Transformer数据增强有哪些？(NLP场景)
> **A**:
> NLP不像CV有天然增强，要用特定策略：
> 1. **BPE/Dropout子词正则**：训练时每步BPE切分结果略不同，等价增强词表示
> 2. **Replace Token 5-15%随机替换**：BERT MLM思路，随机换近义词/[MASK]/保持
> 3. **Shuffle N-gram**：句子内3-gram顺序打乱置换（语法保留但不重复记忆模板）
> 4. **Back-Translation回译**：目标语言→翻译回源语言→平行语料×2倍
> 5. **MixUp文本**：隐藏层线性插值 h=λh1+(1-λ)h2, y=λy1+(1-λ)y2（Embedding之后加）

---

## 四、位置编码（5题）

### Q34. 正弦位置编码公式？有什么性质？可学习PE vs 正弦PE对比
> **A**:
> 公式 (Vaswani论文原始):
> ```
> PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
> PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
> 性质:
>   ① PE(pos+k) 可以用 PE(pos) 的线性变换表示 → 天然学相对位置
>   ② 固定函数无参数，序列多长都能用(天生外推)
> ```
> 对比表：
> | | 正弦Sinusoid | 可学习Learned | RoPE旋转 |
> |-|-------------|-------------|---------|
> | 外推 >训练长度 | ✅ 直接用 略掉点 | ❌ 训练外随机初始化 崩 | ✅ + NTK缩放 几乎无掉 |
> | 参数量 | 0 无 | N_max × d 1M+ | 0 计算中产生 |
> | 相对位置 | ✅ 隐式(线性组合可表示) | ❌ 靠训练碰 | ✅ 显式完美相对 |
> | 现在主流 | 老模型基线 | BERT类 <512 | **LLM事实标准** |

### Q35. RoPE旋转位置编码原理？为什么LLaMA/现在LLM全用它？
> **A**:
> 核心直觉：**不给Embedding加位置向量，而是旋转Q/K向量的角度** → 点积结果自动包含相对位置差！
>
> 复数形式证明（面试说这个加分）：
> ```
> 把 q 和 k 每两维组成复数: q_i = q_2i + j·q_2i+1
> 旋转m位: f(q,m) = q · e^{j·m·θ_i}  (模不变，角度转mθ_i)
> <f(q,m), f(k,n)> = Re( q_m · conj(k_n) )
>                  = Re( q·e^{jmθ} · conj(k·e^{jnθ}) )
>                  = Re( q·conj(k) · e^{j(m-n)θ} )
> ✅ 只和相对位置差 m-n 有关！和m,n绝对值无关
> ```
>
> LLM选它三大原因：
> 1. **外推极佳**：训练长度4K → 推理用NTK-aware RoPE θ缩放α → 直接16K不用微调
> 2. **数值稳定**：不引入额外参数/加法，只是旋转乘三角函数
> 3. **实现极简**：在attention函数最后Q,K乘旋转矩阵，10行代码

### Q36. ALiBi (Attention with Linear Biases)是什么？对比RoPE？
> **A**:
> ALiBi = 不加任何位置编码！在Attention分数上直接加线性惩罚距离：
> ```python
> # 在 scores = QK^T/√d 之后直接加
> bias = - (np.arange(N) - np.arange(N)[:,None]) * slope[head_idx]
> scores = scores + bias  # 位置越远罚越多! 第i头slope=m=2^{-8i/h}
> ```
> 8头：头0远距离罚最轻m=2^-0=1，头7罚最重m≈0.004 → 近的头看局部，远的头看全局
>
> 对比RoPE：
> - 外推能力 **ALiBi > RoPE**，10K→200K长文本ALiBi困惑度涨的最少
> - ALiBi实现最简单，连cos/sin都不调用
> - 多模态/需要显式位置的场景 RoPE更成熟，ALiBi是纯文本LLM(MPT/BLOOM)的最爱

---

## 五、工程部署（5题）

### Q37. KV Cache显存怎么估算？LLaMA-7B生成长度4096要多少显存？
> **A**:
> 公式：`KV_Cache_Byte = 2 × L × B × N_seq × d × bytes_per_param`
> 2 是因为K+V两个矩阵；L=层数；B=Batch；N_seq=已生成长度；d=hidden_dim；FP16=2字节
>
> LLaMA-7B例：L=32, d=4096, N=4096, B=1 (单Batch), FP16
> ```
> = 2 × 32 × 1 × 4096 × 4096 × 2 B
> = 2 × 32 × 2 × 4096² B
> = 128 × 16 MB = 2048 MB = 2 GB  单样本!
> Batch=32: 2GB × 32 = **64 GB** ← 这就是为什么A100才敢开大Batch
> ```
> 省显存：MQA共享K/V → 省8倍 = 256MB / 样本，或GQA 4倍省

### Q38. Transformer推理服务怎么做动态批处理Dynamic Batching？
> **A**:
> KV Cache是状态性的，静态Batcher不适合：
>
> 动态批3步：
> 1. 独立维护每个请求的past_key_values状态（字典{req_id: KVCache列表}）
> 2. 每10ms窗口把所有"正在生成"的请求的Step t新Token拼一个大Batch，**共享一次前向**
> 3. 返回结果按请求ID拆回去，更新各自Cache
>
> 吞吐：vLLM PagedAttention (把KV Cache分页管理类似OS虚拟内存)能做到 **23x吞吐量提升** 同延迟，工业部署事实标准

### Q39. 量化INT8/INT4部署Transformer？注意点？
> **A**:
> LLM量化核心挑战：激活值离群点(outlier)大，直接对称量化掉点严重
>
> 三大量化方法：
> - **PTQ 后训练** (GPTQ/AWQ)：少量校准数据(128样本)搜最优量化网格，4bit掉点<1%，推荐首选
> - **QAT 训练感知**：训练时fake量化，效果最好但费算力
> - **SmoothQuant**：把激活的离群点"平滑迁移"到权重，两者都好量化
>
> 经验：7B模型INT4 GPTQ ≈ FP16 99%效果，**从13GB显存压到~3.5GB**，单张消费级3090可以跑

### Q40. Beam Search怎么支持Batch？不同长度怎么处理？
> **A**:
> 朴素每条独立Beam慢，统一矩阵化：
> - Beam=K的Batch=B，实际凑成一个Batch=B×K维张量跑前向，输出用拆分
> - 不同长度的句子 pad到相同长度(特殊)

---

## 六、高频扩展题

### Q41. Transformer为什么在CV/NLP/语音三大领域都通吃？统一架构魔力在哪？
> **A**:
> 两大根本原因：
> 1. **归纳偏置弱=表示能力强**：CNN偏图像(2D网格)，RNN偏序列(1D)，Transformer几乎零归纳偏置，只要数据是"集合+关系"都能用，跨模态天然对齐空间一致
> 2. **Attention是通用关系建模算子**：CV里Patch关系，NLP里词间关系，语音里帧的关系，本质都是"元素两两相关性→加权求和"，Attention算子数学上严格通用。硬件(GPU/NPU/TPU)为Transformer做特殊优化，规模效应滚雪球

### Q42. BERT和GPT结构差别？(Encoder-only vs Decoder-only)
> **A**:
> | 模型 | 结构 | 注意力Mask | 任务 | 代表 |
> |-----|------|----------|-----|-----|
> | **BERT** | 纯Encoder堆叠 | 无因果Mask，全可见 | 双向理解(分类/实体/抽取) | BERT/RoBERTa/ERNIE |
> | **GPT** | 纯Decoder堆叠 | **有因果下三角Mask** | 自回归生成(对话/写作/代码) | GPT-3/LLaMA/ChatGLM |
> | T5/BART | Encoder-Decoder原始 | Encoder无 + Decoder有 | 序列到序列(翻译/摘要) | T5/BART/Google Gemini |

> 追问：「为什么现在大模型都是Decoder-only？」→ 简答①实证效果好：同等参数量和数据下Decoder-only生成质量>Encoder-Decoder ②训练简单：只用自注意力一套模块，架构统一好训 ③零样本能力更强(可能和因果Mask的Icl学习方式有关)

### Q43. T5用的是"Prefix LM Mask"还是Causal LM？区别？
> **A**:
> - **Causal LM (GPT)**：下三角0/1全Mask，任何位置i只看≤i，生成用
> - **Prefix LM (T5/UL2)**：前P个位置(Prefix输入) **双向可见**，后G个位置(生成) 因果Mask，这样Encoder-  Decoder结构可以压成单模型做翻译/摘要，输入部分双向看全

### Q44. 为什么Transformer要Share Embedding和Generator权重？省多少参数？
> **A**:
> 权重共享：`Generator.Linear.weight = Embedding.weight`，两者都是V×d矩阵一模一样
> 省了V·d参数量，词表V=3万 d=512时省**15M参数≈20%总参数量**
>
> 理论动机(Press & Wolf 2017)：词嵌入空间和输出分类空间本质是同一个语义空间，共享权重让"输入词向量"和"输出词分类向量"对齐，减少两套参数导致的语义不一致

### Q45. Multi-Head Attention头数是不是越多越好？h=128极端情况？
> **A**:
> 不是越多越好，存在最优d_k(通常64)：
> - 头太多→单头d_k太小→64→16→4，子空间太小装不下语义信息，表达能力退化
> - 头太少→d_k太大→内积方差大，softmax容易饱和
> - 经验：d_k = 32/64/128效果都不错，通常d_k固定64，维度d=512→h=8，d=768→h=12，d=4096→h=32，保证d_k≈64-128