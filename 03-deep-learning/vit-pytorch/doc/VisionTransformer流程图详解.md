# VisionTransformer 流程图详解

> 位置: 03-deep-learning/vit-pytorch/doc/
> 配套文档: VisionTransformer架构与实现.md | VisionTransformer性能优化重难点.md | VisionTransformer面试题汇总.md

---

## 一、ViT完整推理流程

### 1.1 端到端推理流程图

```mermaid
flowchart TD
    subgraph Input[输入层]
        A[224×224×3 RGB图像] --> A1[数据预处理]
        A1 --> A2[归一化 mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]]
    end

    subgraph PatchEmbedding[Patch Embedding层]
        A2 --> B1[图像分块: 16×16 Patch]
        B1 --> B2[得到 14×14=196个Patch]
        B2 --> B3[每个Patch展平: 16×16×3=768维]
        B3 --> B4[Linear投影: 768→768维]
        B4 --> B5[LayerNorm归一化]
    end

    subgraph TokenProcess[Token预处理]
        B5 --> C1[生成 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token: 1×768]
        C1 --> C2[拼接: CLS + 196 Patch = 197个Token]
        C2 --> C3[加可学习Position Embedding: 197×768]
        C3 --> C4[Dropout正则化]
    end

    subgraph TransformerEncoder[Transformer Encoder ×12层]
        C4 --> D1[第1层 Encoder]
        D1 --> D2[第2层 Encoder]
        D2 --> D3[...]
        D3 --> D4[第12层 Encoder]
    end

    subgraph EncoderLayer[单层Encoder内部结构]
        E0[输入: X] --> E1[LayerNorm前归一化 Pre-LN]
        E1 --> E2[Multi-Head Self-Attention 12头]
        E2 --> E3[Dropout]
        E3 --> E4[残差连接: X + Attention输出]
        E4 --> E5[LayerNorm前归一化]
        E5 --> E6[FFN: Linear→GELU→Dropout→Linear]
        E6 --> E7[Dropout]
        E7 --> E8[残差连接: 输出]
    end

    subgraph ClassificationHead[分类头]
        D4 --> F1[取第0个Token: <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>输出]
        F1 --> F2[LayerNorm]
        F2 --> F3[Linear: 768→1000类]
        F3 --> F4[Softmax得到概率分布]
    end

    F4 --> Output[输出: 1000维分类概率向量]
```

### 1.2 关键数据形状变化表

| 阶段 | 输入形状 | 操作 | 输出形状 | 说明 |
|-----|---------|------|---------|------|
| 原始图像 | [B, 3, 224, 224] | 数据预处理 | [B, 3, 224, 224] | Batch, Channel, H, W |
| Patch分块 | [B, 3, 224, 224] | Rearrange分块+展平 | [B, 196, 768] | 14×14=196个Patch |
| Linear投影 | [B, 196, 768] | Linear 768→768 | [B, 196, 768] | Patch Embedding |
| 加CLS Token | [B, 196, 768] | Concat CLS | [B, 197, 768] | 第0位是CLS |
| 加位置编码 | [B, 197, 768] | +PE | [B, 197, 768] | 逐元素相加 |
| Encoder×12 | [B, 197, 768] | 12层Transformer | [B, 197, 768] | 形状不变 |
| 取CLS输出 | [B, 197, 768] | 切片[:,0,:] | [B, 768] | 全局特征向量 |
| MLP分类头 | [B, 768] | LN+Linear | [B, 1000] | ImageNet分类 |

---

## 二、Multi-Head Self-Attention 详细流程

### 2.1 自注意力时序图

```mermaid
sequenceDiagram
    participant X as 输入X [B, 197, 768]
    participant LN as LayerNorm
    participant Proj as Q/K/V Linear投影
    participant Split as 拆分成12头
    participant Attn as Scaled Dot-Product Attention
    participant Concat as 多头结果拼接
    participant OutProj as 输出Linear投影

    X->>LN: Pre-LN归一化
    LN->>Proj: Linear ×3 生成Q/K/V
    Proj->>Proj: Q: [B, 197, 768]
    Proj->>Proj: K: [B, 197, 768]
    Proj->>Proj: V: [B, 197, 768]

    Proj->>Split: Reshape拆多头
    Split->>Split: Q: [B, 12, 197, 64]
    Split->>Split: K: [B, 12, 197, 64]
    Split->>Split: V: [B, 12, 197, 64]

    Split->>Attn: Scaled Dot-Product
    Note over Attn: scores = Q·K^T / √64<br/>scores = softmax(scores)<br/>output = scores·V

    Attn->>Concat: 注意力输出 [B, 12, 197, 64]
    Concat->>Concat: Reshape → [B, 197, 768]
    Concat->>OutProj: Linear投影 768→768
    OutProj-->>X: 残差连接到输入
```

### 2.2 单头注意力计算流程

```mermaid
flowchart LR
    subgraph 输入
        Q[Q: 197×64]
        K[K: 197×64]
        V[V: 197×64]
    end

    Q --> M1[MatMul: Q · K^T]
    K --> M1
    M1 --> S1[197×197 注意力分数矩阵]

    S1 --> D1[Scale除以√64=8]
    D1 --> S2[缩放后的分数]

    S2 --> SM[Softmax沿最后一维]
    SM --> W[注意力权重: 每行和=1]

    W --> M2[MatMul: 权重 · V]
    V --> M2
    M2 --> O[输出: 197×64]
```

---

## 三、ViT训练完整流程图

### 3.1 预训练流程 (JFT-300M/ImageNet-21K)

```mermaid
flowchart TD
    Start[开始预训练] --> Init[模型初始化]
    Init --> Init1[权重初始化: trunc_normal std=0.02]
    Init --> Init2[Position Embedding: 插值初始化]
    Init --> Init3[CLS Token: 正态初始化]

    Init --> EP[Epoch循环 ×300]
    EP --> B[Batch循环]
    B --> D1[数据增强: MixUp+CutMix+RandAugment]
    D1 --> D2[图像随机裁剪224×224]
    D2 --> D3[随机水平翻转]

    D3 --> FWD[前向传播]
    FWD --> LOSS[计算CrossEntropy Loss]
    LOSS --> LabelSmooth[标签平滑 ε=0.1]

    LabelSmooth --> BACK[反向传播计算梯度]
    BACK --> Clip[梯度裁剪 max_norm=1.0]
    Clip --> OPT[AdamW优化器更新]

    OPT --> Sched[学习率调度]
    Sched --> S1[Warmup前10000步线性上升]
    S1 --> S2[Cosine余弦退火衰减]

    OPT --> Eval{每1000步评估?}
    Eval -->|是| Acc[验证集Top1/Top5准确率]
    Acc --> Save[保存Best Checkpoint]
    Save --> B
    Eval -->|否| B

    B --> EPFinish{Epoch完成?}
    EPFinish -->|是| EP
    EPFinish -->|否| Done[预训练完成]
```

### 3.2 微调分类流程

```mermaid
flowchart TD
    Load[加载预训练权重] --> Freeze[冻结策略选择]

    Freeze --> F1[线性探测: 冻结Backbone只训MLP Head]
    Freeze --> F2[部分微调: 冻结前6层训后6层+Head]
    Freeze --> F3[全量微调: 解冻全部参数]

    F1 --> LR1[学习率: Head=1e-3]
    F2 --> LR2[学习率: 后6层=5e-5 Head=1e-3]
    F3 --> LR3[学习率: 分层衰减 LR × 0.65^层数]

    LR1 --> Train[微调训练 ×50 Epochs]
    LR2 --> Train
    LR3 --> Train

    Train --> Aug[数据增强: 轻量级AutoAugment]
    Aug --> Reg[正则化]
    Reg --> R1[DropPath随机深度率=0.1]
    Reg --> R2[Weight Decay=0.05]
    Reg --> R3[Stochastic Depth]

    Train --> Test[测试集评估]
    Test --> Compare[对比: 线性探测 vs 微调 vs 全量]
```

---

## 四、ViT变体结构对比流程图

### 4.1 ViT vs Swin Transformer 结构差异

```mermaid
flowchart LR
    subgraph ViT[标准ViT 单尺度]
        V1[224×224图像] --> V2[16×16 Patch 196个]
        V2 --> V3[全局自注意力 196²=38416次计算]
        V3 --> V4[单尺度输出 1/16降采样]
    end

    subgraph Swin[Swin Transformer 分层金字塔]
        S1[224×224图像] --> S2[4×4 Patch 3136个]
        S2 --> S3[Stage1 窗口注意力 7×7窗口]
        S3 --> S4[Patch Merging 2×2合并 → 1/8降采样]
        S4 --> S5[Stage2 窗口注意力]
        S5 --> S6[Patch Merging → 1/16降采样]
        S6 --> S7[Stage3 窗口+Shift窗口交替]
        S7 --> S8[Patch Merging → 1/32降采样]
        S8 --> S9[Stage4 窗口注意力]
        S9 --> S10[4级特征图 C2-C5 类似ResNet]
    end
```

### 4.2 Swin Shifted Window 工作机制

```mermaid
flowchart TD
    subgraph Layer1[第2l层: 标准窗口划分]
        L1A[图像8×8] --> L1B[划分2×2=4个窗口 每个4×4]
        L1B --> L1C[每个窗口内独立自注意力 16²计算 ×4]
    end

    subgraph Layer2[第2l+1层: Shifted偏移窗口]
        L2A[图像8×8] --> L2B[向右下偏移2×2像素]
        L2B --> L2C[形成3×3=9个不规则窗口]
        L2C --> L2D[Cyclic Shift循环移位]
        L2D --> L2E[重新规整为2×2窗口]
        L2E --> L2F[计算后反向Cyclic Shift还原]
        L2F --> L2G[跨窗口信息流动实现]
    end

    L1C -->|信息传递| L2B
    L2G -->|信息传递| NextL[下一层标准窗口]
```

---

## 五、ViT部署推理流程

### 5.1 ONNX导出与部署

```mermaid
flowchart TD
    PT[PyTorch ViT-B/16 state_dict] --> Load[加载模型结构+权重]
    Load --> Eval[model.eval()切换推理模式]
    Eval --> Dummy[生成Dummy输入: 1×3×224×224]

    Dummy --> Export[torch.onnx.export]
    Export --> P1[opset_version=17]
    Export --> P2[dynamic_axes={batch维度}]
    Export --> P3[do_constant_folding=True]

    P1 --> Check[onnx.checker.check_model]
    P2 --> Check
    P3 --> Check

    Check --> Simplify[onnxsim: 常量折叠+图简化]
    Simplify --> OPT[onnxoptimizer: 算子融合优化]

    OPT --> ORT[ONNX Runtime Inference]
    ORT --> EP1[CPU: XNNPACK/MKL-DNN EP]
    ORT --> EP2[GPU: CUDA/TensorRT EP]
    ORT --> EP3[移动端: NNAPI/CoreML EP]

    EP1 --> Benchmark[性能基准测试: 延迟/吞吐/准确率]
    EP2 --> Benchmark
    EP3 --> Benchmark
```

### 5.2 端侧部署量化流程

```mermaid
flowchart TD
    FP32[FP32 ONNX模型] --> Calib[校准数据集准备: 1000张ImageNet]
    Calib --> RunFP32[跑FP32推理 记录每层激活值范围]
    RunFP32 --> Histogram[收集权重和激活的直方图分布]

    Histogram --> MinMax[方案A: 简单MinMax对称量化]
    Histogram --> KL[方案B: KL散度优化非对称量化]

    MinMax --> QConfig[生成量化配置表: scale+zero_point]
    KL --> QConfig

    QConfig --> INT8[导出INT8量化模型]
    INT8 --> Compare[对比FP32 vs INT8]
    Compare --> Acc[准确率差异 <1% 可接受]
    Compare --> Speed[CPU推理速度 3~8x 加速]
    Compare --> Size[模型大小 75%压缩]
```

---

## 六、注意力可视化流程

```mermaid
flowchart TD
    Img[输入测试图像] --> FWD[前向传播]
    FWD --> Hook[Hook注册: 保存每层Attention权重]
    Hook --> AttnWeights[得到12层×12头的注意力图: L×H×N×N]

    AttnWeights --> Method1[方法1: CLS→Patch注意力]
    Method1 --> Extract[取第0行: CLS关注哪些Patch]
    Extract --> Avg[多层多头平均聚合]
    Avg --> Reshape[从196→14×14热力图]
    Reshape --> Upsample[上采样到224×224]
    Upsample --> Overlay[与原图叠加可视化]

    AttnWeights --> Method2[方法2: Attention Rollout]
    Method2 --> Recursive[递归计算残差+注意力矩阵乘积]
    Recursive --> Token2Token[得到最终Token×Token注意力]
    Token2Token --> Same[同上生成热力图]

    AttnWeights --> Method3[方法3: 层次t-SNE聚类]
    Method3 --> TSNE[768维特征→2D降维]
    TSNE --> Clusters[相似概念Patch聚在一起]
```