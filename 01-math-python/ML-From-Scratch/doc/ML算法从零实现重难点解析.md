# ML From Scratch 从零实现重难点解析

> 位置: 01-math-python/ML-From-Scratch/doc/
> 配套文档: ML算法从零实现详解.md | ML算法从零实现重难点解析.md | ML算法面试题汇总.md

---

## 一、数值稳定性重难点

### 1.1 Softmax 上溢出/下溢出问题 (必考高频)

```python
# ❌ 朴素实现 数值爆炸!
def bad_softmax(x):
    exp_x = np.exp(x)  # x=1000 → np.exp(1000)=inf 上溢出
    return exp_x / np.sum(exp_x)  # inf/inf = nan 全废!

# ✅ 经典技巧: 每一项减去max(x) 不改变结果 但避免溢出
def good_softmax(x):
    x_shifted = x - np.max(x)  # max(x_shifted)=0 → exp(0)=1 不会上溢出
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x) + 1e-12)  # 分母+epsilon 避免除0下溢出
```

### 1.2 Sigmoid 溢出 + 交叉熵损失合并

```python
# ❌ 分开算: Sigmoid→ BCE 梯度消失/数值不稳
def bad_bce(logits, y):
    p = 1 / (1 + np.exp(-logits))  # logits<-700 → exp(-700)溢出成0; p=1 → log(0)=-inf
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

# ✅ LogSumExp 合并公式 数值稳健 (面试要求能手写推导)
def good_bce_with_logits(logits, y):
    # 数学等价: BCE(Sigmoid(x), y)
    # = max(x,0) - x*y + log(1 + exp(-|x|))
    return np.mean(
        np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    )
```

---

## 二、梯度下降 核心参数解析

### 2.1 BGD/SGD/Mini-Batch SGD 收敛性对比

| 算法 | 每步更新数据 | 收敛速度 | 噪声 | 跳出局部最优 | 适合数据量 |
|-----|------------|---------|-----|------------|----------|
| BGD全批量 | N条全部 | 慢 平滑 | 无 单调下降 | ❌ 容易 | <1万条 |
| SGD单条 | 1条 | 快抖动 | 大震荡 | ✅ 噪声帮跳出 | 全场景 |
| Mini-Batch⭐ | 32-256条 | **平衡** | 中等噪声 | ✅ 平衡 | **生产默认** |

### 2.2 学习率选择黄金法则

```
太大: loss震荡 甚至发散 NaN
太小: loss下降极慢 1000epoch不动
调参公式:
  ✅ Linear Scaling Rule: Batch ×k → LR ×k
  ✅ Adam 默认 1e-3 不用调就能跑
  ✅ SGD 默认 1e-2 + Momentum 0.9
  ✅ 必须配合 Warmup 前1000步线性从0升到目标LR
```

---

## 三、线性模型 数学推导关键步骤

### 3.1 最小二乘闭式解 (XᵀX)⁻¹Xᵀy 推导 + 复杂度

```
L(W) = ||XW - y||²   RSS残差平方和
∂L/∂W = 2Xᵀ(XW - y)  对W求导 令=0 →
XᵀXW = Xᵀy            正规方程 Normal Equation →
W* = (XᵀX)⁻¹ Xᵀ y     ⚠️ 复杂度O(n_features³) 维度不能大
```

💡 维度>1万时不要用闭式解! 用梯度下降迭代, 每次O(n×d)比d³更划算.

```python
# ✅ 加L2正则岭回归避免XᵀX奇异不可逆
def ridge_regression_closed_form(X, y, lambda_=1e-4):
    d = X.shape[1]
    I = np.eye(d)
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y  # 加lambdaI 保证可逆
```

---

## 四、SVM 支持向量机重难点

### 4.1 硬间隔 vs 软间隔 公式差异

```
硬间隔Hard Margin SVM: 假设数据完美线性可分
  min (1/2)||W||²
  s.t. yᵢ(Wᵀxᵢ+b) ≥ 1  ∀i    所有样本都在间隔外 → 异常点直接崩!

软间隔Soft Margin SVM (生产用✅): 加松弛变量ξ容忍错分
  min (1/2)||W||² + C Σξᵢ   C越大越怕错分→越容易过拟合
  s.t. yᵢ(Wᵀxᵢ+b) ≥ 1 - ξᵢ
       ξᵢ ≥ 0
C超参: C→∞ 等价硬间隔; C小 允许错分→更稳泛化好
```

### 4.2 核函数选择经验

| 核函数 | 公式 | 场景 |
|-------|------|-----|
| 线性核Linear | xᵢ·xⱼ | 特征数>样本数, 文本TF-IDF高维首选 ⭐快 |
| RBF高斯核⭐ | exp(-γ||xᵢ-xⱼ||²) | 通用默认 中小数据集 非线性 |
| 多项式Poly | (γxᵢ·xⱼ+r)^d | 图像像素 次选 |
| Sigmoid核 | tanh(γxᵢ·xⱼ+r) | 特定MLP近似场景 |

---

## 五、决策树 信息增益/增益率/基尼 三指标

### 5.1 三大分裂准则对比

| 准则 | 公式 | 偏好 | 代表算法 |
|-----|------|-----|---------|
| 信息增益 ID3 | Gain = H(D) - Σ|Dᵥ|/|D|·H(Dᵥ) | ❌ 偏好取值多的特征 独热编码炸 | ID3 |
| 增益率 C4.5 | GainRatio = Gain / IV(a)  惩罚多值 | 修正偏好 但更喜欢不平衡划分 | C4.5 |
| 基尼指数⭐CART | Gini=1-Σpₖ²  分类; 回归MSE | **生产默认** 实现快 数值稳 | CART/XGBoost |

### 5.2 CART 剪枝避免过拟合

```
前剪枝Pre-pruning: 分裂前先校验增益>阈值才分, 早停简单粗暴
后剪枝Post-pruning⭐: 先全树深分裂到纯 → 自底向上合并
  代价复杂度剪枝CCP: 损失 = α·叶子数 + Σ叶节点MSE/Gini
  α调参: α=0 全树; α大 砍成1个根节点树桩
```

---

## 六、K-Means聚类 坑点避坑

### 6.1 K值选择 + 初始化敏感两大致命问题

```
❌ 坑1: 随机初始化→20次跑出20种不同结果!
✅ 解法:
  1. K-Means++初始化: 第一个中心随机, 后续离已选中心越远概率越高
  2. 多跑n_init=10次, 取SSE最小的那次作为最终结果

❌ 坑2: K值怎么选? 不知道分几类
✅ 解法:
  肘法则Elbow: K=1~10画SSE折线图, 拐点位置=最优K
  轮廓系数Silhouette: [-1,1]越接近1越好, 找最高值点
```

---

## 七、PCA 主成分分析 推导三步核心

```
Step 1: 数据零均值化 X_centered = X - mean(X, axis=0)  必须做!
Step 2: 协方差矩阵 Σ = (1/n) Xᵀ X   [d×d]
Step 3: Σ的特征值分解 WᵀΣW = Λ → 取前K大特征值对应特征向量作投影矩阵

💡 解释方差比例Explained Variance Ratio = Σ前K个λ / Σ所有λ
  生产选K标准: 保留95%方差比例即可
```

---

## 八、面试代码实现TOP 5 (能手写!)

| 算法 | 必写核心部分 | 行数 |
|-----|------------|-----|
| 1. 线性回归梯度下降 | W -= lr * (2/n)Xᵀ(XW-y) | 20行 |
| 2. 逻辑回归 | Sigmoid + BCE求导公式 | 25行 |
| 3. 决策树CART | 递归分裂 + 最佳分裂点遍历 | 50行 |
| 4. KNN | 距离计算 + 多数投票 | 20行 |
| 5. 朴素贝叶斯 | 先验P(y) + 条件似然P(x|y)高斯 | 25行 |