# PythonDataScienceHandbook 解析

> 📍 位置: `01-math-python/PythonDataScienceHandbook/notebooks/`
> ⭐ 简历推荐: ⭐⭐⭐ | 🎯 岗位: 数据分析/数据科学

---

## 一、模块总览

```
Notebooks 核心章节 (对应5大工具库):
├── Ch01 IPython: 魔术命令/%timeit性能测速/%debug调试
├── Ch02 NumPy: 100道练习必备 (02.02数组基础/02.03通用函数ufunc/02.04聚合/02.05广播)
├── Ch03 Pandas: 数据清洗+EDA (03.01对象/03.02索引/03.05缺失值/03.08聚合GroupBy/03.07Merge连接)
├── Ch04 Matplotlib/Seaborn: 可视化 折线/散点/误差棒/密度/分布/热力图/多子图
└── Ch05 Scikit-learn: 特征工程/模型验证/朴素贝叶斯/SVM/随机森林/PCA+流形/KMeans
```

## 二、Pandas 高频20个操作速查 (面试必背)

```python
import pandas as pd
# 1. 读写
df = pd.read_csv('data.csv', encoding='utf-8', parse_dates=['date_col'])
df.to_parquet('clean.parquet', index=False)

# 2. 缺失值处理 (面试必考3种)
df.isnull().sum().sort_values(ascending=False)  # 每列缺失数
df['col'].fillna(df['col'].median(), inplace=True)  # 数值型:中位数/分类:众数
df.dropna(thresh=len(df)*0.7, axis=1, inplace=True)  # 缺失>30%的列直接删

# 3. 分组聚合 (高频!)
result = df.groupby(['city','category'], as_index=False) \
    .agg(total_sales=('amount','sum'),
         avg_price=('price','mean'),
         sku_count=('sku_id','nunique')) \
    .sort_values('total_sales', ascending=False) \
    .head(10)

# 4. 表连接 (SQL的JOIN)
pd.merge(left, right, on='user_id', how='left')   # left/inner/outer/right
pd.concat([df1, df2], axis=0, ignore_index=True)  # 纵向拼接 (行对齐)

# 5. 时间序列
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
monthly = df['sales'].resample('M').sum()   # 日→月聚合
df['rolling_7d'] = df['sales'].rolling(7).mean()  # 7日滑动平均
```

## 三、简历黄金句式

| 句式 |
|-----|
| 「用Pandas完成500万条电商订单数据清洗：缺失值5种策略对比、异常值IQR/ZScore联合检测、OneHot+Label编码，最终特征维度从58→42，下游XGBoost模型AUC+0.023」 |
| 「Seaborn + Matplotlib搭建数据分析仪表盘：销量热力图/用户RFM三维散点/漏斗转化多子图组合，洞察出复购率Top3客群特征，业务方据此运营ROI提升28%」 |
| 「NumPy广播机制深度优化：对比朴素Python循环1000×加速；用ufunc向量化操作替代显式循环，300万条数据KNN距离计算从18s→0.32s (56×加速)」 |

## 四、高频面试题

**Q: Pandas的loc vs iloc vs ix区别？**
> A: iloc用**整数位置**索引 ([0:5]取前5行, 左闭右开)，loc用**标签名**索引 (df.loc['2024-01':'2024-06'] 包含两端)，ix(已废弃) 混合模式推荐不要用。

**Q: GroupBy + transform 和 agg 的区别？**
> A: agg 返回聚合后的**压缩结果** (n行GroupBy后返回K行, K=组数)；transform返回**原形状广播回去** (和原df行数一样, 适合加一列"该组均值"做新特征, 不用再merge回来)。