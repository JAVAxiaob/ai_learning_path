# 🤖 02 - 机器学习基础 章节导览

> AI核心基础知识：从线性模型到集成学习，从监督学习到聚类降维。**面试必考+传统数据项目必备**
> 预计学习周期：2周 (14天) | 目标掌握度：⭐⭐⭐⭐ L4熟练级（面试高频）
> 配套项目路径：`../../02-machine-learning/` / `../../01-math-python/ML-From-Scratch/`

---

## 📚 本章节文件索引

| 文件名 | 内容重点 | 学习优先级 |
|-------|---------|-----------|
| **README.md** (本文) | 本章路线 + 算法选型指南 | ⭐⭐⭐ 先读我 |
| **监督学习算法详解.md** | 线性/逻辑回归/决策树/随机森林/XGBoost/SVM 原理+公式+优缺点 | ⭐⭐⭐⭐⭐ 面试半壁江山 |
| **无监督学习与评估方法.md** | K-Means/PCA降维 + 准确率/精确率/召回率/F1/AUC混淆矩阵 | ⭐⭐⭐⭐⭐ 必考评估指标 |
| **可解释性机器学习.md** | SHAP/LIME/PDP/ALE 金融医疗合规必备，高薪算法岗核心 | ⭐⭐⭐⭐ 高阶加分 |
| **代码实战.md** | sklearn 房价预测/XGBoost/KMeans 完整可运行代码 | ⭐⭐⭐⭐ 必做 |
| **面试题库.md** | 80道ML高频面试题+标准答案（含决策树/XGBoost推导） | ⭐⭐⭐⭐⭐ 必背 |
| **GitHub项目推荐.md** | interpretable-ml-book/sklearn示例等项目 | ⭐⭐⭐ 配套做 |

---

## 🧠 机器学习全景知识图谱

```mermaid
graph TD
    A[机器学习] --> B[监督学习 Supervised 有标签y]
    A --> C[无监督学习 Unsupervised 无标签]
    A --> D[模型评估 + 可解释性]

    %% 监督学习
    B --> B1[线性模型 家族]
    B1 --> B1a[线性回归 f(x)=wx+b 回归问题]
    B1 --> B1b[逻辑回归 Sigmoid(wx+b) 二分类]
    B --> B2[树模型 家族 ⭐面试高频]
    B2 --> B2a[决策树 ID3/C4.5/CART 信息增益/基尼系数]
    B2 --> B2b[随机森林 Bagging + Bootstrap + 特征袋装]
    B2 --> B2c[XGBoost/LightGBM  Boosting ⭐⭐⭐⭐⭐]
    B --> B3[SVM 支持向量机]
    B3 --> B3a[硬间隔/软间隔 + 核技巧 RBF核]

    %% 无监督
    C --> C1[聚类 K-Means/层次/DBSCAN]
    C --> C2[降维 PCA/TSNE/UMAP]
    C --> C3[关联规则 Apriori FP-Growth 购物篮]

    %% 评估 + 可解释
    D --> D1[分类指标: 准确率/精确率/召回率/F1/混淆矩阵]
    D --> D2[排序指标: AUC/ROC/PR曲线]
    D --> D3[全局解释: Permutation重要性 + PDP/ALE图]
    D --> D4[局部解释: SHAP Shapley值 + LIME ⭐⭐⭐⭐⭐]
```

---

## 🔥 面试必考 Top 5 算法（按出现频率排序）

### 🥇 TOP 1: XGBoost / LightGBM 梯度提升树 (出现率 90%)

> 面试只要问"你用过什么机器学习模型？"，90% 会接着追问XGBoost。**必须会背下面6点**

| 维度 | XGBoost 核心要点（必背） |
|-----|-------------------------|
| **目标函数** | 正则化的损失函数：`Obj = L(θ) + Ω(θ) = Σloss(ŷ,y) + γT + 0.5λΣw²` <br> L是损失函数，Ω是树复杂度惩罚（叶子数T+叶子权重L2） |
| **分裂依据** | 二阶梯度近似 + 贪心枚举每个特征切分点：最大增益Gain = 左分数+右分数-父分数-λ |
| **为什么比GBDT快** | ① 直方图分箱(Histogram) O(d×K×n)，不用O(d×n²)枚举所有点 <br> ② 按层增长Level-wise + 特征并行 <br> ③ 缓存优化 + 列块存储 |
| **正则化手段** | ① 目标函数加 γ(叶子数) + λ(权重L2) <br> ② shrinkage学习率 η(每棵树贡献缩) <br> ③ 列抽样/行抽样 <br> ④ 早停Early Stop |
| **XGB vs LGB区别** | LightGBM: ① Histogram + 差分包加速 <br> ② Leaf-wise 按增益最大叶子增长 (XGB是Level-wise) <br> ③ 类别特征CatBoost直接处理 <br> ④ 相同精度快2~5倍 |
| **为什么用在表格数据** | 表格数据Tabular Data中XGB/LGB几乎一直霸榜Kaggle。对缺失值/异常值/特征尺度不敏感，少调参 |

> 💰 **简历黄金句式**: `基于XGBoost搭建信用卡欺诈检测模型，IV特征筛选+Woe编码+5折交叉验证+贝叶斯超参调优，AUC从0.89提升到0.974，每月识别欺诈交易挽回损失120万+`

### 🥈 TOP 2: 决策树 (基础中的基础，出现率 80%)

| 算法 | 分裂准则 | 支持任务 | 树类型 |
|-----|---------|---------|--------|
| ID3 | 信息增益 `Gain = H(D) - H(D|A)` (偏向多取值特征) | 仅分类 | 多叉树 |
| C4.5 | 信息增益率 (Gain / 固有值 IV)，修正多取值特征偏置 | 分类+回归 | 多叉树 |
| **CART** ⭐⭐⭐⭐⭐ | 分类用**基尼系数** `Gini = 1-Σp²` <br> 回归用**MSE方差减少** | 分类+回归 | **二叉树** (sklearn/XGB用) |

> 面试题高频：为什么XGB用CART二叉树不用ID3多叉？→ 二叉树特征可以反复用！多叉树一个特征用完就下一层，浪费

### 🥉 TOP 3: 评估指标 (每面必问，出现率 100%)

| 场景 | 推荐指标 | 公式 | 注意坑 |
|-----|---------|------|-------|
| **分类** 均衡样本 (正负各50%) | Accuracy 准确率 | `(TP+TN) / 总样本` | 样本不平衡不要用！(99%正样本瞎猜都99%准) |
| **分类** 不平衡 且 重视Precision (误判代价高 如癌症检测) | Precision 精确率 `TP/(TP+FP)` | 预测为正的里面，真的正的比率 | 癌症不能误诊→要Precision高 |
| **分类** 不平衡 且 重视Recall (漏检代价高 如欺诈识别) | Recall 召回率 `TP/(TP+FN)` | 真实为正的里面，被找出来的比率 | 欺诈不能漏→要Recall高 |
| **F1分数** | 精确率+召回率 调和平均 | `2PR/(P+R)` | P和R此消彼长，要综合看F1 |
| **排序/阈值不敏感** | **AUC-ROC** ⭐⭐⭐⭐⭐ | ROC曲线下面积 = 随机抽1正1负，正样本预测分>负样本的概率 | 面试最爱考定义 |
| **回归** | RMSE / MAE / R² | RMSE放大错误惩罚 | R²=1最好，<0模型还不如直接猜均值 |

> 🏆 AUC-ROC 直觉理解面试标准答案：`"AUC等于从所有正样本中随机抽1个，从所有负样本中随机抽1个，模型给正样本打分 > 给负样本打分的概率。0.5=瞎猜，1=完美，0.9+是好模型"`

### 🏅 TOP 4: 逻辑回归 vs 线性回归 (60%)

| 维度 | 线性回归 Linear Regression | 逻辑回归 Logistic Regression |
|-----|---------------------------|----------------------------|
| **任务类型** | 回归：预测连续值 (房价/年龄) | 二分类：预测0-1概率 (是否逾期/点击) |
| **公式** | `ŷ = w·x + b` | `ŷ = σ(w·x + b) = 1 / (1 + e^-(wx+b))` |
| **损失函数** | MSE均方误差 `Σ(ŷ-y)² / n` | 交叉熵损失 (负对数似然) `-Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]` |
| **为什么不用MSE做分类** | 逻辑回归+MSE是非凸函数，容易陷入局部最优；交叉熵是凸函数，梯度下降能找到全局最优 |
| **和SVM区别** | 逻辑回归是似然概率，能输出概率值；SVM是几何间隔最大，输出0/1分类 |

### 🏅 TOP 5: SVM 支持向量机 (40%)

核技巧 Kernel Trick 一句话理解：
> `"SVM把数据映射到高维空间(让数据线性可分)，但通过核函数直接在原空间算高维内积，不用显式升维→省计算量。"`
> 常用核：RBF高斯核 `K(x,z) = exp(-γ||x-z||²)` 万能核，数据量不大默认首选。

---

## 🛠️ sklearn 30行代码 = 完整ML项目模板 (复制即用)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

# 1. 读数据
df = pd.read_csv("data.csv")
X, y = df.drop("target", axis=1), df["target"]

# 2. 自动特征预处理 Pipeline
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# 3. 模型 Pipeline (预处理→模型)
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
])

# 4. 训练 + 5折交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"5折交叉验证 AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# 5. 测试集评估
pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
print(f"测试集 AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# 6. (可选) GridSearch 超参数调优
param_grid = {"model__max_depth": [3,5,7], "model__n_estimators": [100,300,500]}
gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}, Best AUC: {gs.best_score_:.4f}")
```

---

## 🎯 章节结业自测

达到 7/10 就合格：

- [ ] 能说出XGBoost比传统GBDT的3个改进点
- [ ] 能画出混淆矩阵，说出P/R/F1/AUC定义
- [ ] 能解释信息增益/基尼系数区别，CART树是什么
- [ ] 能解释过拟合的5种解决方法：正则/早停/数据增强/剪枝/集成
- [ ] 能解释什么是Bagging vs Boosting，随机森林 vs XGBoost区别
- [ ] 能独立用sklearn Pipeline完成完整分类/回归项目
- [ ] 能解释什么是SHAP值，3个公理是什么
- [ ] Permutation Importance vs Feature Importance哪个更靠谱？为什么
- [ ] 不平衡数据处理：过采样/欠采样/类权重/SMOTE 4种方法
- [ ] 训练-验证-测试 为什么要三切分？验证集和测试集的区别

---

## ⚠️ 常见面试踩坑点

1. **只会调库说不清原理** → 面试官问"XGB的分裂公式？" 答不上来 = 淘汰
2. **样本不平衡还用Accuracy吹** → 99%样本是0，模型全猜0，Accuracy99% = 废模型
3. **把测试集指标当最优调模型** → 数据泄漏！只能在验证集上调参，测试集只看一次
4. **Feature Importance迷信内置Gini** → 一定要用Permutation打乱特征比较，公平