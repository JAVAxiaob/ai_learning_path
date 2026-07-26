# Python 数据分析与可视化 面试题汇总 (30题)

> 位置: 01-math-python/PythonDataScienceHandbook/doc/
> 配套文档: Python数据分析与可视化实战.md | Python数据分析重难点解析.md | Python数据分析面试题汇总.md

---

## 一、NumPy 基础（8题）

### Q1. 为什么NumPy比Python纯List快100倍? 连续内存/C级类型/向量化SIMD/BLAS并行四要素

### Q2. NDArray广播(Broadcasting)4条规则: 右对齐/维相等或1/1复制/否则报错 能手算两个shape广播后的结果

### Q3. np.reshape vs np.resize vs np.transpose vs np.reshape(-1,1) vs view vs copy 哪些改数据是视图哪些是复制

### Q4. keepdims=True 为什么建议永远开? mean/sum降维后形状不对导致后续广播的坑

### Q5. fancy indexing布尔索引 vs 整数索引 vs 切片: 哪些返回视图哪些返回副本? 性能差多少

### Q6. np.dot vs np.matmul vs @运算符 vs np.multiply 区别? 矩阵乘法逐元素乘法标量积

### Q7. einsum爱因斯坦求和: "ij,jk->ik" = 矩阵乘法, 能手写3种常见einsum表达式(转置/点积/批处理)

### Q8. np.random.seed(42) 可复现实验重要性: 同seed保证每次跑相同随机数/划分相同训练集

---

## 二、Pandas 性能与技巧（8题）

### Q9. 行遍历4种写法性能排序: itertuples > apply > iterrows > 纯Python for + iloc 大数量级差1000倍! 什么时候用哪种

### Q10. 内存优化: float64→float32 int64→int8 object→category 三步通常压到原内存的20%怎么写函数

### Q11. merge 四种连接: left/right/inner/outer 与SQL JOIN对应; on参数key多对多笛卡尔爆炸怎么避免

### Q12. groupby分组后: agg多列多函数 vs transform广播结果 vs apply自定义, 三者区别与场景

### Q13. 滚动窗口rolling(window=7).mean() vs expanding expanding()累计统计: 时间序列特征工程常用7/30日均线

### Q14. 缺失值处理策略决策树: >60%删列;<20%中位数/众数填;中间填+加缺失布尔列;严禁瞎填0!

### Q15. 大CSV文件10GB读不进内存? 三种解法: read_csv(chunksize=)分块/dtype指定类型预先压缩/换Dask/Parquet列式存储

### Q16. Pivot透视表 vs Stack/Unstack多层索引 vs Melt逆透视宽表转长表: 能手写把销售表从宽表变成长表格式

---

## 三、可视化（6题）

### Q17. 5大常用图选择判断: 时间趋势→折线/分类对比→柱状/相关→散点/分布→直方图/交叉表→热力图

### Q18. Matplotlib vs Seaborn vs Plotly 三者关系? Seaborn是高级API封装Matplotlib统计图表, Plotly交互式

### Q19. 中文字体乱码5行标准配置模板: font.sans-serif字体列表+axes.unicode_minus=False负号+rcParams['figure.dpi']=120

### Q20. plt.subplots(2,3,figsize=(15,10)) 多子图布局; ax对象比pyplot plt直接函数更可控为什么?

### Q21. 箱线图BoxPlot 5数汇总: min/Q1/median/Q3/max 离群点IQR k=1.5倍怎么判异常

### Q22. 可视化误导避坑: Y轴不从0开始放大差异/双Y轴刻度陷阱/饼图超过5块看不清/气泡不按面积误导

---

## 四、特征工程与预处理（8题）

### Q23. StandardScaler vs MinMaxScaler vs RobustScaler 3种标准化选择策略表

### Q24. 为什么One-Hot不可以直接在高基数类别(>50种邮编/商品ID)上用? 维度爆炸→TargetEncoder目标编码均值编码解法

### Q25. 类别不平衡处理: SMOTE上采样(制造少数样本) vs 下采样(丢多数) vs class_weight损失加权 优劣对比

### Q26. 异常值3σ vs IQR箱线法选择: 正态分布用3σ, 长尾非参用IQR k=1.5; 金融风控异常要不要直接删? 保留作为特征

### Q27. 对数变换log1p为什么对收入/金额等长尾右偏分布有效? 线性模型前提假设误差正态

### Q28. 交叉特征组合: age_group × income_bucket 手动造特征 线性模型/NB提分; 为什么树模型不用手动交叉?

### Q29. 训练/测试集特征预处理陷阱: fit_transform只能对train做! test必须用train的scaler.transform, 否则数据泄漏!

### Q30. 数据泄漏Top5场景: 1.test数据参与fit 2.时间切分用随机切分 3.目标编码编码了测试集Y 4.标准化用全量数据 5.特征构造包含未来信息