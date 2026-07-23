# 🔷 技术方向1：数学与Python基础

## 1.1 知识点展开详解

### 1.1.1 线性代数（AI的数学语言）

| 知识点 | 展开内容 | 直觉理解 |
|--------|---------|----------|
| 矩阵乘法 | `(AB)_{ij} = Σ_k A_{ik}·B_{kj}`，注意`AB ≠ BA`（不满足交换律） | 矩阵乘法 = 向量的线性变换 + 信息聚合。Transformer 中的 Q·Kᵀ 就是计算"注意力权重" |
| 转置与逆 | `Aᵀ`：行列互换；`A⁻¹`：满足 `AA⁻¹ = I` 的矩阵 | 转置用于向量相似度计算（点积）；逆矩阵用于最小二乘求解线性回归 |
| 特征值/特征向量 | `Av = λv`，其中 v 是特征向量，λ 是特征值 | 找到"变换后方向不变"的向量 → PCA 降维的核心 |
| 向量点积/叉积 | 点积 `a·b = |a||b|cosθ`（相似度）；叉积 `a×b`（垂直向量，面积） | Embedding 相似度 = 点积；注意力分数 = Softmax(Q·Kᵀ) |
| 张量（Tensor） | 标量(0维) → 向量(1维) → 矩阵(2维) → 3维张量（RGB图像）→ 4维张量（batch图像） | PyTorch/TensorFlow 的基本数据结构。形状操作（reshape/permute/squeeze）是日常操作 |

**线性代数与Java/Android结合点**：
- Android 中 `Matrix` 类做图像变换（缩放、旋转）→ 本质就是 2D 矩阵乘法
- Java 二维数组模拟矩阵 → 理解 NumPy `ndarray` 的形状概念
- RecyclerView 列表布局 → 理解"维度"概念：每个item是向量，列表是矩阵

### 1.1.2 概率统计（不确定性建模）

| 知识点 | 展开内容 | 直觉理解 |
|--------|---------|----------|
| 条件概率 | `P(A|B) = P(A∩B) / P(B)` | 已知B发生后，A的概率。贝叶斯推断的基础 |
| 贝叶斯定理 | `P(A|B) = P(B|A)·P(A) / P(B)` | 先验 P(A) + 观测证据 P(B|A) → 更新后验概率。垃圾邮件分类经典应用 |
| 高斯分布 | `f(x) = (1/√(2π)σ)·exp(-(x-μ)²/2σ²)` | 自然界大量数据服从正态分布。权重初始化、噪声建模 |
| softmax | `softmax(x_i) = exp(x_i) / Σ_j exp(x_j)` | 将任意实数向量归一化为概率分布（和为1）。多分类输出层 |
| 交叉熵损失 | `H(p,q) = -Σ_i p_i·log(q_i)`，其中p是真实分布(one-hot)，q是预测分布(softmax输出) | 衡量两个概率分布的"距离"。比 MSE 更适合分类问题（梯度更稳定） |

**交叉熵 vs MSE 直觉**：
- 分类问题：正确类别预测概率0.01时，交叉熵惩罚很大（-log(0.01)≈4.6），而MSE只有(1-0.01)²≈0.98
- 这就是为什么分类用交叉熵，回归用MSE

### 1.1.3 梯度下降（优化核心算法）

**核心公式**：
```
θ_{t+1} = θ_t - η·∇L(θ_t)
```

其中：
- `θ`：模型参数（权重/偏置）
- `η`：学习率（超参数，太大震荡，太小收敛慢）
- `∇L(θ)`：损失函数对参数的梯度（指向损失增加最快的方向）

**梯度下降变体**：

| 方法 | 说明 | 优缺点 |
|------|------|--------|
| BGD（批量） | 每次迭代用全部样本计算梯度 | 稳定但慢，大数据集不可行 |
| SGD（随机） | 每次迭代用1个样本 | 快但震荡大，引入噪声有助于跳出局部最优 |
| Mini-batch SGD | 每次用一小批样本（32/64/128） | 平衡性能与速度，**实际最常用** |
| Momentum | 加入"惯性"：`v = β·v + (1-β)·∇L` | 加速收敛，抑制震荡 |
| Adam | Momentum + RMSProp + 偏置校正 | **默认首选优化器** |

**学习率调度**：
- 固定学习率：简单但后期可能在最优附近震荡
- 指数衰减：`lr = lr0 · γ^(epoch/step)`
- 余弦退火：`lr = lr_min + 0.5·(lr_max-lr_min)·(1+cos(π·T_cur/T_max))`
- Warm-up：前几步先用小学习率，避免初期不稳定

**梯度消失/爆炸问题**：
- 深层网络中，链式法则导致梯度连乘 → 接近0（消失）或无穷大（爆炸）
- 解决方案：ReLU激活（正区间梯度=1）、BatchNorm、残差连接、权重初始化（He/Xavier）、梯度裁剪

### 1.1.4 NumPy 与 Pandas

**NumPy核心操作**：

| 操作 | 说明 | 场景 |
|------|------|------|
| `np.dot(A, B)` / `np.matmul(A, B)` | 矩阵乘法（注意：3D+时matmul会智能处理batch维度） | 神经网络前向传播 |
| `np.einsum('ij,jk->ik', A, B)` | Einstein求和约定，灵活表达复杂张量操作 | Transformer 注意力计算 `np.einsum('bti,bui->btu', Q, K)` |
| 广播机制 | 形状不同的数组在满足条件时自动扩展维度 | `arr - arr.mean(axis=0)` 每行减去均值 |
| `reshape` / `transpose` / `squeeze` | 形状操作 | 图像数据 `(H,W,3) → (1,H,W,3)` 增加batch维度 |

**Pandas核心操作**：

| 操作 | 说明 |
|------|------|
| `df.isnull().sum()` | 统计每列缺失值数量 |
| `df.fillna(df.median())` | 用中位数填充缺失值 |
| `pd.get_dummies(df['category'])` | One-Hot 编码 |
| `df.groupby('col').transform('mean')` | 分组后转换（不聚合，保持原始行数） |
| `df.merge(other_df, on='key')` | 类似 SQL JOIN |
| `(df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())` | Min-Max 归一化到 [0,1] |

---

## 1.2 GitHub项目推荐

### 📦 项目1：numpy-ml（从零实现机器学习算法）

- **GitHub链接**：https://github.com/ddbourgin/numpy-ml
- **下载命令**：`git clone --depth 1 https://github.com/ddbourgin/numpy-ml.git`
- **解压到目录**：`D:\ai_learning\math_python\numpy-ml\`
- **Star数**：约18k · 最后更新：2025
- **技术栈**：纯Python + NumPy（无深度学习框架）
- **项目简介**：用纯NumPy从零实现几乎所有主流ML算法（线性回归、决策树、SVM、CNN、LSTM、Transformer等）。不依赖PyTorch/TensorFlow，数学公式与代码一一对应，是理解算法原理的绝佳项目。
- **学习价值**：
  - 理解每个算法的数学本质（不是调用sklearn.fit这么简单）
  - Python + NumPy 的工程化能力
  - 对后续用 Java/DJL 或 TF.js 复现模型打下基础
- **代码阅读路线**：
  - `numpy_ml/linear_models/least_squares.py`（最小二乘 → 线性回归）
  - `numpy_ml/neural_nets/layers.py`（CNN/LSTM层的NumPy实现）
  - `numpy_ml/neural_nets/transformers.py`（Transformer的NumPy实现 ⭐）
- **与传统开发结合点**：
  - Android：NumPy数组操作 → 理解 Kotlin `Array<List<Float>>` 多维数组
  - Java：NumPy的向量化思维 → 优化 Java 循环（`for` → 流操作/数组向量化）

### 📦 项目2：handson-ml2（《Hands-On ML》第二版代码）

- **GitHub链接**：https://github.com/ageron/handson-ml2
- **下载命令**：`git clone --depth 1 https://github.com/ageron/handson-ml2.git`
- **解压到目录**：`D:\ai_learning\math_python\handson-ml2\`
- **Star数**：约28k · 最后更新：2024
- **技术栈**：Python + NumPy + Pandas + scikit-learn + TensorFlow
- **项目简介**：《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》第二版官方配套代码。包含16章完整的Jupyter Notebook，覆盖从数据处理到深度学习的完整流程。
- **学习价值**：
  - 工业界标准的ML Pipeline实践
  - Pandas/NumPy数据处理真实案例
  - 与第三章"深度学习"无缝衔接
- **代码阅读路线**：
  - `02_end_to_end_machine_learning_project.ipynb`（完整ML项目流程 ⭐）
  - `04_training_linear_models.ipynb`（线性代数在ML中的应用）
  - `05_support_vector_machines.ipynb`（SVM数学原理）

### 📦 项目3：linear-algebra-visualizations（线性代数可视化教学）

- **GitHub链接**：https://github.com/3b1b/manim （3Blue1Brown 数学动画引擎）
- **下载命令**：`git clone --depth 1 https://github.com/3b1b/manim.git`
- **解压到目录**：`D:\ai_learning\math_python\manim\`
- **Star数**：约60k · 最后更新：2025
- **技术栈**：Python + NumPy（数学计算）+ Cairo（渲染）
- **项目简介**：3Blue1Brown 制作数学动画的引擎。通过代码生成数学可视化动画，是理解"线性代数的几何意义"的绝佳工具。
- **学习价值**：
  - 建立"数学直觉"：矩阵乘法=空间变换，特征向量=变换的"主轴"
  - Python + NumPy 生成复杂数学对象的工程能力
- **代码阅读路线**：
  - `manimlib/mobject/mobject.py`（理解面向对象 + 数学对象建模）
  - `example_scenes.py`（数学可视化脚本示例）

### 📦 项目4：pandas-practice-projects（Pandas实战项目集）

- **GitHub链接**：https://github.com/guipsamora/pandas_exercises
- **下载命令**：`git clone --depth 1 https://github.com/guipsamora/pandas_exercises.git`
- **解压到目录**：`D:\ai_learning\math_python\pandas_exercises\`
- **Star数**：约15k · 最后更新：2024
- **技术栈**：Python + Pandas + NumPy + Matplotlib
- **项目简介**：100+ 个 Pandas 练习题，从基础操作到复杂数据处理。每个练习有真实数据集，是掌握 Pandas 的最快路径。
- **学习价值**：
  - Pandas 数据清洗/转换/聚合的完整技能
  - 特征工程的基础数据操作能力
  - 与 Java 结合：理解 Apache Spark DataFrame（概念类似但分布式）

### 📦 项目5：pytorch-lightning-examples（PyTorch Lightning示例）

- **GitHub链接**：https://github.com/Lightning-AI/pytorch-lightning
- **下载命令**：`git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git`
- **解压到目录**：`D:\ai_learning\math_python\pytorch-lightning\`
- **Star数**：约28k · 最后更新：2025
- **技术栈**：Python + PyTorch（此项目为后续深度学习方向打基础）
- **项目简介**：PyTorch Lightning 是 PyTorch 的高层封装，标准化训练循环。虽然主要用于深度学习，但其示例中的 NumPy/Python 工程化模式值得学习。
- **学习价值**：
  - 大型 Python 项目的代码组织（Module/Callback/Logger 分离）
  - 为方向3（深度学习）做铺垫

---

### 项目清单汇总表

| 项目名 | 技术栈 | 难度 | 预计学习时长 | 核心学习点 | git clone 命令 |
|--------|--------|------|-------------|-----------|---------------|
| numpy-ml | Python+NumPy | ⭐⭐⭐ | 15-20h | 算法原理 + 向量化思维 | `git clone --depth 1 https://github.com/ddbourgin/numpy-ml.git` |
| handson-ml2 | Python+sklearn+TF | ⭐⭐ | 20-25h | 工业界ML Pipeline | `git clone --depth 1 https://github.com/ageron/handson-ml2.git` |
| manim（3B1B） | Python+NumPy+Cairo | ⭐⭐⭐⭐ | 10-15h | 数学直觉 + 可视化 | `git clone --depth 1 https://github.com/3b1b/manim.git` |
| pandas_exercises | Python+Pandas | ⭐ | 10h | 数据处理实战 | `git clone --depth 1 https://github.com/guipsamora/pandas_exercises.git` |
| pytorch-lightning | Python+PyTorch | ⭐⭐⭐ | 10h | 大型Python项目架构 | `git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git` |

---

## 1.3 完整可运行代码示例

### 代码1：纯Python实现矩阵乘法（不依赖NumPy）

```python
def matrix_multiply(A, B):
    """纯Python实现矩阵乘法，返回结果矩阵C = A × B"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError(
            f"矩阵维度不兼容: A({rows_A}x{cols_A}) × B({rows_B}x{cols_B})"
        )
    
    # 初始化结果矩阵为全零
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # 三重循环实现矩阵乘法
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


if __name__ == "__main__":
    # 测试：模拟一个简化的神经网络层计算
    # 输入向量 X (1x3)，权重矩阵 W (3x2)，偏置 b (1x2)
    X = [[1.0, 2.0, 3.0]]
    W = [[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]]
    b = [[0.1, 0.2]]
    
    # Y = X · W + b （注意：偏置广播，这里手动扩展）
    XW = matrix_multiply(X, W)
    Y = [[XW[0][j] + b[0][j] for j in range(2)]]
    
    print(f"输入 X = {X}")
    print(f"权重 W = {W}")
    print(f"X · W = {XW}")
    print(f"Y = X·W + b = {Y}")
    print(f"\n✅ 矩阵乘法验证成功！（这就是神经网络层的核心）")
```

**运行方法**：保存为 `matrix_multiply.py`，执行 `python matrix_multiply.py`

---

### 代码2：NumPy实现梯度下降求解线性回归

```python
import numpy as np

def gradient_descent_linear_regression(X, y, lr=0.01, epochs=1000):
    """
    使用梯度下降求解线性回归 y = X·w + b
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    n_samples, n_features = X.shape
    
    # 参数初始化
    w = np.zeros(n_features)  # 权重
    b = 0.0                    # 偏置
    
    loss_history = []
    
    for epoch in range(epochs):
        # 1. 前向传播：预测值 ŷ = X·w + b
        y_pred = np.dot(X, w) + b
        
        # 2. 计算 MSE 损失：L = (1/n)·Σ(y_pred - y)²
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)
        
        # 3. 反向传播：计算梯度
        # ∂L/∂w = (2/n)·Xᵀ·(y_pred - y)
        # ∂L/∂b = (2/n)·Σ(y_pred - y)
        dw = (2.0 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (2.0 / n_samples) * np.sum(y_pred - y)
        
        # 4. 参数更新：w ← w - lr·dw
        w = w - lr * dw
        b = b - lr * db
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss = {loss:.6f} | w = {np.round(w, 3)} | b = {b:.3f}")
    
    return w, b, loss_history


if __name__ == "__main__":
    # 构造模拟数据：y = 3·x1 - 2·x2 + 1 + noise
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100个样本，2个特征
    true_w = np.array([3.0, -2.0])
    true_b = 1.0
    y = np.dot(X, true_w) + true_b + np.random.randn(100) * 0.1  # 加噪声
    
    print("=" * 60)
    print("梯度下降求解线性回归")
    print(f"真实参数: w = {true_w}, b = {true_b}")
    print("=" * 60)
    
    w, b, losses = gradient_descent_linear_regression(X, y, lr=0.01, epochs=1000)
    
    print("\n" + "=" * 60)
    print(f"学习参数: w = {np.round(w, 3)}, b = {np.round(b, 3)}")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"✅ 梯度下降收敛成功！")
    print("=" * 60)
```

**运行方法**：`pip install numpy` → 保存为 `gd_linear_regression.py` → `python gd_linear_regression.py`

---

### 代码3：Pandas特征工程完整Pipeline

```python
import pandas as pd
import numpy as np

def build_feature_pipeline(df):
    """完整的特征工程Pipeline：缺失值处理 → 编码 → 归一化 → 特征交叉"""
    
    # 步骤1：缺失值处理
    # 数值型用中位数，类别型用众数（或单独"Unknown"类别）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    category_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    for col in category_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")
    
    # 步骤2：类别特征编码（One-Hot）
    if len(category_cols) > 0:
        df = pd.get_dummies(df, columns=category_cols, drop_first=True)
    
    # 步骤3：数值特征归一化 (Min-Max 到 [0,1])
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:  # 避免除零
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # 步骤4：特征交叉（简单示例：数值特征两两相乘）
    numeric_cols_after = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols_after) >= 2:
        for i in range(min(2, len(numeric_cols_after))):
            for j in range(i+1, min(3, len(numeric_cols_after))):
                col_i, col_j = numeric_cols_after[i], numeric_cols_after[j]
                df[f"{col_i}_x_{col_j}"] = df[col_i] * df[col_j]
    
    return df


if __name__ == "__main__":
    # 构造模拟数据
    np.random.seed(42)
    data = {
        "age": [25, 30, np.nan, 35, 40, 28, np.nan, 45, 38, 32],
        "income": [50000, 60000, 75000, np.nan, 80000, 55000, 70000, 90000, 65000, 58000],
        "city": ["Beijing", "Shanghai", "Beijing", np.nan, "Shenzhen",
                 "Beijing", "Shanghai", "Shenzhen", np.nan, "Beijing"],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("原始数据 (前5行):")
    print(df.head())
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print("=" * 60)
    
    df_processed = build_feature_pipeline(df.copy())
    
    print("\n处理后数据:")
    print(df_processed.head())
    print(f"\n✅ 特征工程Pipeline完成！特征数量: {len(df_processed.columns)}")
    print("=" * 60)
```

---

## 1.4 面试题库（数学与Python基础）

### 📝 基础理论题（10题）

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 1.1 | 为什么矩阵乘法不满足交换律（AB ≠ BA）？请用空间变换的直觉解释 | 简单 | ⭐⭐⭐ |
| 1.2 | 什么是特征值和特征向量？PCA 为什么要用它们？ | 中等 | ⭐⭐⭐ |
| 1.3 | 交叉熵损失函数为什么适合分类问题？和 MSE 相比有什么优势？ | 中等 | ⭐⭐⭐⭐ |
| 1.4 | 梯度下降中，学习率太大或太小分别会导致什么问题？ | 简单 | ⭐⭐⭐⭐ |
| 1.5 | SGD、Mini-batch GD、BGD 的区别？实际中为什么 Mini-batch 最常用？ | 中等 | ⭐⭐⭐ |
| 1.6 | 什么是梯度消失/爆炸？为什么深层网络容易出现这个问题？ | 中等 | ⭐⭐⭐⭐ |
| 1.7 | Adam 优化器相比 SGD 有什么改进？它的两个动量分别是什么？ | 中等 | ⭐⭐⭐ |
| 1.8 | softmax 的数学公式是什么？为什么输出的是概率分布？ | 简单 | ⭐⭐⭐⭐ |
| 1.9 | NumPy 的 `np.dot` 和 `np.matmul` 有什么区别？ | 简单 | ⭐⭐ |
| 1.10 | Pandas 的 `groupby().agg()` 和 `groupby().transform()` 有什么区别？ | 中等 | ⭐⭐⭐ |

### 💻 Python/NumPy代码题（10题）

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 1.11 | 用纯Python实现向量点积函数 `dot_product(a, b)`，不使用NumPy | 简单 | ⭐⭐⭐ |
| 1.12 | 用NumPy实现ReLU激活函数：`relu(x) = max(0, x)`，要求支持向量/矩阵输入 | 简单 | ⭐⭐⭐⭐ |
| 1.13 | 用NumPy实现softmax函数，并处理数值溢出问题（减去最大值） | 中等 | ⭐⭐⭐⭐ |
| 1.14 | 给定一个4维张量 `(batch, height, width, channels)`，将其转换为 PyTorch 格式 `(batch, channels, height, width)` | 中等 | ⭐⭐⭐ |
| 1.15 | 用NumPy实现一个简化版的卷积操作：`conv2d(input, kernel)`，假设 padding=0, stride=1 | 中等 | ⭐⭐⭐ |
| 1.16 | 用 Pandas 统计DataFrame每列的缺失值数量，并返回一个排序后的Series | 简单 | ⭐⭐⭐ |
| 1.17 | 用NumPy实现标准化：`(x - mean) / std`，要求沿指定维度计算 | 简单 | ⭐⭐⭐ |
| 1.18 | 用纯Python实现最小二乘法求解线性回归的闭式解：`w = (XᵀX)⁻¹Xᵀy`（可用NumPy的矩阵逆） | 中等 | ⭐⭐⭐ |
| 1.19 | 用NumPy实现交叉熵损失计算：输入为预测概率(batch, num_classes)和真实标签(batch,) | 中等 | ⭐⭐⭐⭐ |
| 1.20 | 用NumPy实现BatchNorm的前向传播：`y = γ·(x - μ)/√(σ²+ε) + β`，训练模式使用batch的均值方差 | 中等 | ⭐⭐⭐ |

### 🔗 与传统开发结合题（5题）

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 1.21 | Android 中的 `android.graphics.Matrix` 类做图像变换，其本质是什么数学操作？和 NumPy 的矩阵乘法有什么联系？ | 中等 | ⭐⭐⭐ |
| 1.22 | 用 Java 实现一个简化的矩阵乘法类 `Matrix`，提供 `multiply(Matrix other)` 方法 | 中等 | ⭐⭐⭐ |
| 1.23 | 前端用 JavaScript 实现 softmax 函数，输入为数字数组，输出为概率分布数组 | 简单 | ⭐⭐ |
| 1.24 | 解释 NumPy 的"广播机制"，并类比到 Java 的数组操作/流操作 | 中等 | ⭐⭐ |
| 1.25 | 用 Kotlin 实现一个简单的线性回归预测类：输入特征数组，输出预测值（假设权重已知） | 简单 | ⭐⭐⭐ |

---

> ✅ **方向1（数学与Python基础）学习完成自检清单**：
> - [ ] 能手动计算 2×2 矩阵乘法
> - [ ] 能解释交叉熵为什么比MSE更适合分类
> - [ ] 能独立写出梯度下降的伪代码
> - [ ] 能用 NumPy 实现基本张量操作
> - [ ] 能用 Pandas 完成基本数据清洗