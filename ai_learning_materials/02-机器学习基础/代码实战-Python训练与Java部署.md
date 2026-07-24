# 💻 02 - 机器学习基础 - 代码实战：Python训练 + Java部署

---

## 代码1：Python sklearn 完整训练评估 Pipeline（60行工业级）

训练多模型对比 + 交叉验证 + 网格搜索超参。面试直接写这版最加分。

```python
# pip install scikit-learn numpy pandas
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import numpy as np


def evaluate(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "准确率 Accuracy": accuracy_score(y_test, y_pred),
        "精确率 Precision": precision_score(y_test, y_pred),
        "召回率 Recall": recall_score(y_test, y_pred),
        "F1 分数": f1_score(y_test, y_pred),
        "⭐ AUC-ROC": roc_auc_score(y_test, y_pred_proba),
    }
    # 5折交叉验证F1（避免测试集偶然性）
    cv5_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    print(f"\n{'='*55}\n【模型】{name}\n{'='*55}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  5折CV F1 = {cv5_f1.mean():.4f} ± {cv5_f1.std():.4f}")
    print(f"  混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 乳腺癌二分类任务 - sklearn 多模型基线对比")
    print("=" * 60)
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"样本 {X.shape[0]} | 特征 {X.shape[1]} | 正类比例 {y.mean():.1%}")

    # StratifiedKFold 分层划分 保持正负比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练{len(y_train)} 测试{len(y_test)}")

    # ⭐ 多模型Pipeline（LR需要标准化！RF/GBDT树模型不需要）
    models = {
        "逻辑回归 LR": Pipeline([
            ("scaler", StandardScaler()),   # 线性模型必须标准化
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "随机森林 RF": RandomForestClassifier(n_estimators=200, max_depth=8,
                                               random_state=42, n_jobs=-1),
        "GBDT": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                            max_depth=4, random_state=42),
    }

    for name, m in models.items():
        evaluate(m, name, X_train, y_train, X_test, y_test)

    # ================ ⭐ GBDT 网格搜索调超参 ================
    print(f"\n\n{'='*60}\n🔍 网格搜索: GBDT 最优参数\n{'='*60}")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.1, 0.2],
        'subsample': [0.8, 1.0],  # 样本随机采样比例，防过拟合
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                        param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"✅ 最优参数: {grid.best_params_}")
    print(f"✅ 最优 5折 CV AUC: {grid.best_score_:.4f}")
    final_auc = roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test)[:, 1])
    print(f"✅ 测试集 AUC（从未见过的数据）: {final_auc:.4f}")
    print("\n🎉 sklearn 完整 Pipeline 完成！(正常运行 AUC 应 > 0.98)")
```

---

## 代码2：Java DJL 加载 ONNX 模型推理 ⭐（传统开发最关心的！）

> Python训练好的sklearn/XGBoost模型，怎么在Spring Boot服务里用？→ **转ONNX格式 → Java用DJL加载ONNX Runtime推理**
> 
> ⚠️ **重中之重：训练端的StandardScaler均值/标准差必须保存下来，Java端做完全一样的标准化！否则 Training-Serving Skew = 上线效果暴跌。**

### Step A（Python端导出 ONNX + 标准化参数 JSON）
```python
""" 运行环境: pip install skl2onnx onnxruntime json """
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
import json, pickle

data = load_breast_cancer()
X, y = data.data, data.target

# ⭐ Pipeline 完整保存: [标准化 + GBDT 模型]
pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", GradientBoostingClassifier(n_estimators=150))])
pipe.fit(X, y)

# ========== 1. 导出 ONNX（Java端用这个 .onnx 文件）==========
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = to_onnx(pipe, initial_types=initial_type)
with open("models/breast_cancer_pipeline.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ ONNX 模型已保存: models/breast_cancer_pipeline.onnx")

# ========== 2. 同时单独导出 StandardScaler 参数
#   (当Java端只转模型不含scaler时用，必须严格一致！)
scaler_params = {
    "means": pipe['scaler'].mean_.tolist(),
    "stds":  pipe['scaler'].scale_.tolist(),  # scale_ = 无偏标准差
    "feature_names": data.feature_names.tolist()
}
with open("models/scaler_params.json", "w", encoding="utf-8") as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print("✅ 标准化参数已保存: models/scaler_params.json")
print(f"   样本0 预测概率正类: {pipe.predict_proba(X[:1])[0,1]:.3f}")
```

### Step B（Java Spring Boot 端加载 ONNX 推理）
```java
// pom.xml 依赖（只加这3个）:
// <dependency>
//   <groupId>ai.djl</groupId><artifactId>api</artifactId><version>0.27.0</version>
// </dependency>
// <dependency>
//   <groupId>ai.djl.onnxruntime</groupId><artifactId>onnxruntime-engine</artifactId><version>0.27.0</version>
// </dependency>
// <dependency><groupId>com.google.code.gson</groupId><artifactId>gson</artifactId></dependency>

package com.ai.ml;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.google.gson.Gson;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class GBDTOnnxInference {

    // 从Python导出的 scaler_params.json 读取（⭐保证和训练端100%一致）
    static class ScalerParams { float[] means; float[] stds; }
    private static final ScalerParams SCALER;
    static {
        try (Reader r = Files.newBufferedReader(Paths.get("models/scaler_params.json"))) {
            SCALER = new Gson().fromJson(r, ScalerParams.class);
        } catch (IOException e) { throw new RuntimeException(e); }
    }
    private static final List<String> CLASS_NAMES = Arrays.asList("良性", "恶性");

    // Translator: 输入 Java float[] → ONNX模型张量；输出张量 → 人类可读字符串
    public static class CancerTranslator implements Translator<float[], String> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[] features) {
            NDManager mgr = ctx.getNDManager();
            NDArray arr = mgr.create(features).reshape(1, features.length);
            // ⚠️ 必须和sklearn StandardScaler完全一致的标准化！
            NDArray mean = mgr.create(SCALER.means);
            NDArray std  = mgr.create(SCALER.stds);
            NDArray normalized = arr.sub(mean).div(std);  // (x - mean) / std
            return new NDList(normalized);
        }
        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            // ONNX Pipeline输出: 第0个是类别, 第1个是概率
            NDArray probs = list.size() > 1 ? list.get(1) : list.get(0);
            float[] p = probs.toFloatArray();
            int predClass = p[0] > p[1] ? 0 : 1;
            return String.format("预测=%s, 恶性概率=%.2f%%, 良性概率=%.2f%%",
                CLASS_NAMES.get(predClass), p[1]*100, p[0]*100);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(60));
        System.out.println("☕ Java Spring Boot 服务: sklearn→ONNX 推理");
        System.out.println("=".repeat(60));

        Criteria<float[], String> criteria = Criteria.builder()
            .setTypes(float[].class, String.class)
            .optModelPath(Paths.get("models"))
            .optModelName("breast_cancer_pipeline")  // breast_cancer_pipeline.onnx
            .optTranslator(new CancerTranslator())
            .optEngine("OnnxRuntime")
            .build();

        // ⭐ Predictor 线程安全，整个服务复用一个就够了
        try (ZooModel<float[], String> model = criteria.loadModel();
             Predictor<float[], String> predictor = model.newPredictor()) {

            System.out.println("✅ ONNX 模型加载成功: " + model.getName());
            System.out.println("   输入形状: " + model.describeInput());
            System.out.println("   输出形状: " + model.describeOutput());

            float[][] samples = { /* 良性模式 */ new float[30], /* 恶性模式 */ new float[30] };
            Arrays.fill(samples[0], 0.5f); Arrays.fill(samples[1], 2.5f);
            for (int i = 0; i < samples.length; i++) {
                String result = predictor.predict(samples[i]);
                System.out.printf("\n样本 %d → %s", i+1, result);
            }

            // 性能：Spring Boot 生产 10K QPS 级别的基线
            int N = 1000;
            long t0 = System.currentTimeMillis();
            for (int i = 0; i < N; i++) predictor.predict(samples[0]);
            long cost = System.currentTimeMillis() - t0;
            System.out.printf("\n\n⚡性能: %d次推理 耗时%dms = %.2f ms/次 = %.0f QPS\n",
                N, cost, cost*1.0/N, N*1000.0/cost);
        }
        System.out.println("\n🎉 Java ONNX 推理成功！ (单线程约 2000~5000 QPS)");
    }
}
```

> ✅ **Java部署核心价值**：和你现有Spring Boot服务无缝集成，不用额外维护Python推理服务。单实例2000+QPS足够大部分业务场景。

---

## 代码3：XGBoost4J Java 原生推理（不用转ONNX，性能更快）

如果训练端就是XGBoost，直接用 `xgboost4j` 更简单：

```xml
<!-- pom.xml -->
<dependency>
    <groupId>ml.dmlc</groupId><artifactId>xgboost4j_2.12</artifactId><version>2.0.3</version>
</dependency>
```

```java
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import java.util.Map;

public class XgboostJavaDemo {
    public static void main(String[] args) throws Exception {
        // Python: model.save_model("xgboost.model") → Java 直接加载同一个文件
        Booster booster = XGBoost.loadModel("models/xgboost.model");

        // 构造 DMatrix: row-major 展开的 float 数组 + 行数 + 列数
        float[] data = {1.2f, -0.3f, 0.5f, /* 30维特征 */};
        DMatrix dmat = new DMatrix(data, 1, 30);

        // 预测: float[nRows][1] = 正类概率
        float[][] predicts = booster.predict(dmat, false, 0);
        float pMalignant = predicts[0][0];
        System.out.printf("XGBoost4J 原生推理: 恶性概率=%.2f%%%n", pMalignant*100);

        // 特征重要性 (上线Debug为什么这么预测用)
        Map<String, Integer> importance = booster.getFeatureScore("", "weight");
        System.out.println("特征重要性 Top-5: " + importance);
    }
}
```

---

## 代码4：K折 Target Encoding 正确写法（不泄漏！）

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def target_encode_kfold(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         cat_col: str, target_col: str, n_folds=5):
    """⭐工业界标准写法：K折 CV Target Encoding，100% 无标签泄漏"""
    oof_train = np.zeros(len(train_df))  # Out-of-Fold
    test_enc = 0
    # 全局先验均值，解决冷启动（新类别没见过用它兜底）
    global_mean = train_df[target_col].mean()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_means = []
    for tr_idx, val_idx in kf.split(train_df):
        # 每一折：只用训练折的数据算 groupby 均值 → 映射到验证折
        tr = train_df.iloc[tr_idx]
        fold_mean = tr.groupby(cat_col)[target_col].mean()
        fold_means.append(fold_mean)
        # 验证折编码 + fillna(global_mean) 兜底没见过的类别
        oof_train[val_idx] = train_df.iloc[val_idx][cat_col].map(fold_mean).fillna(global_mean)
        # 测试集：每折的均值先累加，最后取平均（多数投票防过拟合）
        test_enc += test_df[cat_col].map(fold_mean).fillna(global_mean)

    test_enc /= n_folds  # 测试集 = 5折均值的平均
    return oof_train, test_enc.values


if __name__ == "__main__":
    tr = pd.DataFrame({"city": ["北京","上海","北京","广州","上海","北京"]*100,
                       "click": [1,0,1,0,1,0]*100})
    te = pd.DataFrame({"city": ["北京","深圳","广州"]})  # 深圳=未见过类别冷启动
    tr_enc, te_enc = target_encode_kfold(tr, te, "city", "click")
    print(f"训练集前3个编码: {tr_enc[:3].round(3)}")
    print(f"测试集编码 (深圳=全局均值兜底): {te_enc.round(3)}")
    print("✅ K折 Target Encoding 完成，无泄漏！")
```

> 🏆 面试加分：被问"如何处理高基数类别特征？"，直接答这个方法，面试官瞬间知道你做过真实项目。