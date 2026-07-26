# ML From Scratch 面试题汇总 (30题)

> 位置: 01-math-python/ML-From-Scratch/doc/
> 配套文档: ML算法从零实现详解.md | ML算法从零实现重难点解析.md | ML算法从零实现面试题汇总.md

---

## 一、基础概念（7题）

### Q1. 最小二乘法闭式解推导: W*=(XᵀX)⁻¹Xᵀy 全步骤手写

### Q2. 梯度下降的三类变体 BGD/SGD/Mini-Batch对比表 收敛/噪声/适用数据规模

### Q3. Softmax上溢出下溢出怎么修? 为什么减去max(x)结果不变(数学推导)

### Q4. BCEWithLogits比分开Sigmoid+BCE数值更稳的原理: max(x,0)-xy+log(1+exp(-|x|))推导

### Q5. 为什么要特征标准化StandardScaler? 不标准化对梯度下降/SVM/逻辑回归/KNN/树模型的影响分别是什么

### Q6. 训练/验证/测试 三份数据集切分 7:2:1 vs 交叉验证 什么时候用哪种验证法

### Q7. 过拟合vs欠拟合诊断曲线: train_loss vs val_loss随epoch变化 三大典型图形分析

---

## 二、线性模型 & SVM（7题）

### Q8. L1(Lasso) vs L2(Ridge)正则化6维度对比: 稀疏性/抗共线性/导数/特征选择/计算/场景

### Q9. 逻辑回归为什么用Sigmoid不用阶跃函数? 最大似然估计MLE推导求导梯度

### Q10. 多分类策略: OvR一对多 vs OvO一对一 对比优劣+适用场景

### Q11. SVM为什么叫最大间隔分类器? min 1/2||W||²约束条件的几何意义

### Q12. 软间隔SVM松弛变量ξ+C惩罚参数 调大C和调小C对过拟合的影响

### Q13. 核函数Kernel Trick 为什么不用算φ(x)高维映射就能算内积? 核函数Mercer条件

### Q14. 线性核SVM vs 逻辑回归 怎么选? 90%场景两者差不多选谁? 小样本谁更稳?

---

## 三、树模型 & 集成学习（8题）

### Q15. 三大分裂准则: 信息增益(ID3)/增益率(C4.5)/基尼指数(CART) 公式+偏好对比

### Q16. CART剪枝: 前剪枝早停 vs 后剪枝CCP代价复杂度 剪枝α超参怎么调

### Q17. 随机森林 vs GBDT vs XGBoost vs LightGBM 四大集成框架对比表

### Q18. Bagging并行 vs Boosting串行 核心区别: 样本/树独立与否 降低方差还是偏差

### Q19. XGBoost为什么快? 二阶泰勒展开/直方图算法/缺失值自动处理/列块并行Cache感知

### Q20. LightGBM GOSS单边梯度采样 + EFB互斥特征捆绑 两大创新原理详解

### Q21. 缺失值处理 为什么XGBoost/LGB不用手动填NA? 稀疏分裂方向自动学习机制

### Q22. 类别特征Category: One-Hot vs TargetEncoding vs 树模型原生Category 优劣对比

---

## 四、聚类 & 降维 & 概率模型（8题）

### Q23. K-Means两大痛点: 初始化敏感 + K值难选 解法K-Means++/肘法则/轮廓系数

### Q24. K-Means vs GMM高斯混合模型: 硬聚类vs软聚类, 什么时候必须用GMM(非球形簇)

### Q25. DBSCAN密度聚类 vs K-Means: 任意形状簇/自动找异常点/不用指定K 缺点是高维失效

### Q26. PCA推导核心三步骤: 零均值化→协方差矩阵→特征值分解 为什么必须中心化数据?

### Q27. t-SNE vs PCA 可视化降维: t-SNE为什么画2D散点图效果好但只能可视化/训练O(N²)慢死

### Q28. 朴素贝叶斯 为什么"朴素"? 条件独立假设 P(x|y)=ΠP(xᵢ|y) 效果居然还不错的原因

### Q29. 拉普拉斯平滑Laplace Smoothing α=1: 避免零概率 P(xᵢ|y)=(count+α)/(N+αK) 必要性

### Q30. EM算法GMM参数估计: E-step算后验γ / M-step更新μΣπ, 为什么能保证收敛到局部最优?