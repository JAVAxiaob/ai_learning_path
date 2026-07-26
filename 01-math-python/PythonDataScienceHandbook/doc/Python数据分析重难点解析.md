# Python 数据分析与可视化 重难点解析

> 位置: 01-math-python/PythonDataScienceHandbook/doc/
> 配套文档: Python数据分析与可视化实战.md | Python数据分析重难点解析.md | Python数据分析面试题汇总.md

---

## 一、NumPy 性能优化底层原理

### 1.1 为什么NumPy比纯Python快100倍? 四大原因

| 优化点 | Python List慢 | NumPy NDArray快 |
|-------|--------------|----------------|
| 内存布局 | 链表指针 每个元素PyObject分散内存 指针跳 | **连续内存块** C级数组 dtype固定 缓存命中率×10 |
| 类型一致 | 任意类型混合 每次操作类型检查 | 同dtype 一次类型检查 跳过大量Python解释器 |
| 向量化运算 | 要写Python for循环 1次循环=解释器开销 | **C级SIMD矢量化** 一条CPU指令算16个数 |
| 并行BLAS | 自己写并行 复杂度高 | dot/matmul自动调用OpenBLAS/MKL多线程并行 |

### 1.2 必掌握10个广播原则 (避坑核心)

```python
# 广播4条铁则:
# 1) 从右往左对齐维度形状, 不足的左边补1
# 2) 维度相同 或 一方=1 才能广播
# 3) 维度=1的那维自动复制到匹配对方大小
# 4) 任何维度既不等也≠1 → ValueError: operands could not be broadcast

A = np.random.rand(64, 1, 32, 1)     # (64,1,32,1)
B = np.random.rand(1, 16, 1, 48)     # ( 1,16, 1,48)
C = A + B                            # ✅ 结果Shape = (64,16,32,48)

X = np.random.rand(10, 3)            # 10样本 3特征
mean = X.mean(axis=0, keepdims=True) # keepdims=True 保持(1,3)不丢维度
X_centered = X - mean                # 才能正确广播到(10,3)!  keepdims=False就变(3,)也能广播但坑多
```

---

## 二、Pandas 性能杀手 常见反模式

### 2.1 绝对禁止 ❌ iterrows() 循环遍历行

```python
df = pd.DataFrame({"a": np.random.rand(100_000), "b": np.random.rand(100_000)})

# ❌ 最蠢写法: 98秒 慢×1300
total = []
for idx, row in df.iterrows():        # iterrows每行转Series 巨慢!
    total.append(row["a"] + row["b"])

# ⚠️ 一般写法: 2.1秒 apply每调用一次Python函数
total = df.apply(lambda row: row.a + row.b, axis=1)

# ✅ 向量化NumPy: 7.4ms **快13000倍**
total = df["a"].values + df["b"].values

# ✅✅ 最快eval方法: 复杂表达式numba编译: 2.1ms
total = df.eval("a + b")
```

### 2.2 5个Pandas最佳实践

| 实践 | 正确做法 | 错误做法 慢×N |
|-----|---------|------------|
| 选列 | `df[['col1','col2']]` 用列表 | 多次df.col1 df.col2链式 |
| 过滤 | 布尔索引 `df[df.age>18]` | 先取再for判断 |
| 分组后操作 | groupby.agg/transform 内置 | groupby后自己for循环 |
| 拼接大表 | pd.concat一次性 传列表+ignore_index | 循环df=df.append() 旧O(N²) |
| 读大文件CSV | read_csv(chunksize=10万) + `dtype=`指定类型省内存 | 直接read_csv全读爆内存OOM |

### 2.3 内存优化: 数值列类型压缩

```python
# dtypes默认 float64/int64 → 下转型压缩 节省75%内存
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == 'int':
            mx, mn = df[col].max(), df[col].min()
            # 按数值范围选最小够用的类型 int8/16/32/64
            if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)  # -128~127
            elif mn > -32768 and mx < 32767:
                df[col] = df[col].astype(np.int16)
        elif col_type == np.float64:
            df[col] = df[col].astype(np.float32)  # 精度够用场景 降×2一半
        elif col_type == 'object' and df[col].nunique()/len(df)<0.5:
            df[col] = df[col].astype('category')  # 低基数字符串→category省80%
    return df
# 结果: 10GB CSV → 读完变成 2.3GB Pandas DataFrame
```

---

## 三、Matplotlib/Seaborn 可视化最佳实践

### 3.1 五大高频图对应场景

| 图表类型 | 用途场景 | 用什么函数 |
|---------|---------|----------|
| **折线图**⭐ | 时间序列趋势 连续变化 | `sns.lineplot`/`plt.plot` |
| **柱状图**⭐ | 分类对比/排名TopN | `sns.barplot`/`ax.bar` |
| **散点图** | 两连续变量相关性+气泡=第三维 | `plt.scatter`/`sns.scatter` |
| **直方图+KDE** | 单变量分布偏态/长尾 | `sns.histplot(kde=True)` |
| **热力图**⭐ | 相关性矩阵/交叉表 | `sns.heatmap(df.corr())` |

### 3.2 中文字体5行必写模板 (90%人被方块乱码卡)

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # 负号不是方块
plt.rcParams['figure.dpi'] = 120  # 默认72太糊 改120/150
import seaborn as sns
sns.set_style("whitegrid", {"font.sans-serif": ['SimHei']})  # seaborn也设置中文
```

---

## 四、缺失值与异常值 处理策略

### 4.1 缺失值处理决策树

```
缺失占比?
  ├─ >60% → ❌ 直接删列 df.drop(cols, axis=1)
  ├─ 20%~60% → 加"是否缺失"新列 + 填充:
  │     数值列: 中位数fillna(df[col].median()) 比均值抗异常
  │     分类列: 众数df[col].fillna(df[col].mode()[0]) 或 ML预测填
  └─ <20% → 同上填充; 或行少删行dropna()
  ⚠️ 严禁瞎填0/空串: 引入严重分布偏移!
```

### 4.2 异常值IQR箱线法 + 3σ 选择

```python
def remove_outliers_iqr(df, col, k=1.5):
    # 箱线法IQR 非参数 更抗极端值 不要求正态分布
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k*iqr, q3 + k*iqr  # k=1.5默认, k=3极端异常
    return df[(df[col] >= lower) & (df[col] <= upper)]

def remove_outliers_3sigma(df, col):
    # 3σ正态法 只适用于高斯分布变量 收入等长尾变量别用
    mu, sigma = df[col].mean(), df[col].std()
    return df[(df[col] >= mu-3*sigma) & (df[col] <= mu+3*sigma)]
```

---

## 五、特征工程 10大常用变换

| 变换 | 适用列 | 效果 |
|-----|-------|-----|
| StandardScaler z-score | 线性模型/NN/SVM/KNN/PCA | ⭐必须否则尺度乱 |
| MinMaxScaler [0,1] | 图像/神经网络sigmoid输入 | 范围固定但敏感异常值 |
| RobustScaler 四分位缩放 | 有大量异常值列 | IQR去极值 稳 |
| np.log1p(x)对数 | 收入/金额等长尾右偏列 | 长尾变正态 线性模型提1-5%点 |
| Box-Cox/Yeo-Johnson | 多变量变换 目标接近正态 | 调λ参数自动最优 |
| One-Hot pd.get_dummies | 低基数类别<10取值 | 线性模型标配 树模型不用也行 |
| TargetEncode 目标编码 | 高基数类别邮编/行业>50种 | 用均值Y编码 XGBoost提分 |
| Binning分箱pd.cut/qcut | 连续变量非线性关系 | 决策边界灵活 欠拟合→线性模型 |
| 交叉特征col_a*col_b | 特征工程调参手动造 | LR/NB提分; 树模型自动找不用 |
| Count频数编码 | 唯一值很多但非类别 | 先groupby.size map上去替换原值 |