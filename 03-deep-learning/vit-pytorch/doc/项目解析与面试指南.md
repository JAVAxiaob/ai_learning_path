# ViT PyTorch 视觉Transformer解析

> 位置: 03-deep-learning/vit-pytorch/vit_pytorch/
> 简历推荐: 5星 | 岗位: CV/多模态算法工程师

---

## 一、ViT架构流程图

```mermaid
flowchart LR
    A[224×224×3 Image] --> B[切成16×16 Patch=14×14=196个]
    B --> C[每个Patch展平16×16×3=768维 → Linear投影到D=768]
    C --> D[前面加一颗 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 分类Token → N=197个Token]
    D --> E[加可学习Position Embedding 197×768]
    E --> F[Transformer Encoder ×12层 每层MHA+FFN+残差LN]
    F --> G[取第0个Token = <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 输出向量 768维]
    G --> H[MLP Head LayerNorm → Linear 768→1000类]
    H --> I[分类概率]
```

## 二、核心代码

```python
# vit_pytorch/vit.py 简化版
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, heads=12, mlp_dim=3072, channels=3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 14²=196
        patch_dim = channels * patch_size ** 2          # 3×16×16=768

        # Patch Embedding: 等价于 Kernel=Stride=Patch的Conv2d
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 可学习PE
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                      # CLS Token
        self.transformer = Transformer(dim, depth, heads, mlp_dim)                 # 纯Encoder
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [B, 196, D]
        b, n, _ = x.shape
        # 前面加CLS Token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 197, D]
        x += self.pos_embedding[:, :(n + 1)]    # 加位置编码
        x = self.transformer(x)                 # 12层Encoder
        x = x[:, 0]                             # 取CLS的第0个输出
        return self.mlp_head(x)                 # 分类
```

## 三、ViT vs ResNet 对比 + Swin升级点

| 维度 | ResNet-50 | ViT-B/16 | Swin-T |
|-----|----------|---------|--------|
| 参数 | 25.6M | 86M | 28M |
| ImageNet Top1 | 76.1% | 77.9% (JFT300M预训练→88.5%) | 81.3% |
| 复杂度 | O(nhw) 线性 | O(n²·d) hw²平方 | 窗口内O(n) 分层金字塔 |
| 小数据集泛化 | ✅好 归纳偏置强 | ❌差 需要大数据 | ✅中等 混合CNN+Transformer优点 |
| 下游密集预测 | ✅FPN/特征金字塔天生 | ❌只有单尺度输出 分割检测难 | ✅4级降采样金字塔 和ResNet一样接FPN |

> Swin三大创新点: ① 4级分层降采样(像ResNet C2-C5) ② Shifted Window MSA窗口局部自注意力(速度×N倍) ③ 连续Block窗口偏移 跨窗口信息流动

## 四、简历黄金句式

| 写法 |
|-----|
| 「复现ViT-B/16，ImageNet-1K Top1 Acc=77.9%；1%标签小样本迁移学习对比ResNet50 Acc↑6.3%，验证大模型小样本泛化优势」 |
| 「Swin-T复现+目标检测Faster-RCNN：COCO mAP 48.2，同等参数量比ResNet-50 backbone高7.1 mAP」 |
| 「MAE自监督预训练ViT：75%图像Patch随机Mask，重建原图像；10%标签微调比随机初始化Acc+12.8%」 |

## 五、面试题

**Q ViT为什么要加<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token？不用怎么做？**
> A: CLS Token作为整张图片的全局表示放在序列最前面，最后分类直接用这个位置输出不用做全局池化。替代方案：最后一层所有Token做Global Average Pooling (Swin Transformer就是这么做的)。

**Q 位置编码为什么ViT选可学习不用正弦？RoPE vs ALiBi？**
> A: 作者实验两者效果差不多，可学习实现简单。缺点：训练长度外推差，推理超长度位置随机初始化。RoPE旋转：Q/K乘复数旋转矩阵，相对位置直接体现在内积，当前LLM主流长文本外推最好。ALiBi无位置编码，Attention分数减去|i-j|·m线性惩罚，简单+外推极佳。

**Q 为什么ViT需要超大规模数据(JFT-300M/3B)才好？**
> A: ViT归纳偏置弱(没有CNN的平移不变性/局部性)，从小数据学不好这些先验；只有超大规模数据下才能从数据里学到比CNN手工归纳偏置更好的规律，发挥Transformer的表示上限。