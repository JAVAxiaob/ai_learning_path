# 🔷 技术方向5：Java后端AI服务

## 5.1 核心技术概述

### Java AI推理方案对比

| 方案 | 说明 | 优势 | 适用场景 |
|------|------|------|----------|
| **DJL (Deep Java Library)** | Amazon开源，多引擎支持 | **最推荐**，统一API，支持PyTorch/TF/ONNX/MXNet | 生产环境服务 |
| **ONNX Runtime Java** | Microsoft官方Java绑定 | 与PyTorch训练无缝 | ONNX格式模型 |
| **TensorFlow Java** | Google官方Java API | TF模型原生支持 | TensorFlow SavedModel |
| **ProcessBuilder调用Python** | 子进程方式 | 灵活，直接调用Python | 原型验证/非性能关键 |
| **gRPC跨语言服务** | Java-client + Python-server | 语言解耦 | 大型模型服务 |

### DJL核心架构

```
               ┌─────────────────────────────────────────────────────┐
               │                   DJL API Layer                      │
               │  Criteria / Predictor / Translator / Model Zoo      │
               └────────────┬─────────────────┬──────────────────────┘
                            │                 │
              ┌─────────────▼──────┐  ┌──────▼──────────────────┐
              │ PyTorch Engine     │  │ ONNX Runtime Engine     │
              │ (torch C++ binding)│  │ (onnxruntime C++)       │
              └────────────────────┘  └─────────────────────────┘
                            │                 │
              ┌─────────────▼─────────────────▼───────────────┐
              │           预训练模型 (ONNX / TorchScript)      │
              └────────────────────────────────────────────────┘
```

**DJL核心代码模式**：
```java
// 1. 定义模型加载配置
Criteria<float[], String> criteria = Criteria.builder()
    .setTypes(float[].class, String.class)
    .optModelPath(Paths.get("models/"))        // 本地模型目录
    .optModelName("breast_cancer")              // 模型文件名（不含后缀）
    .optTranslator(new CustomTranslator())      // 输入/输出转换
    .optEngine("OnnxRuntime")                   // 指定引擎: PyTorchEngine/OnnxRuntime/TensorFlow
    .build();

// 2. 加载模型（一次性操作）
try (ZooModel<float[], String> model = criteria.loadModel();
     Predictor<float[], String> predictor = model.newPredictor()) {
    
    // 3. 推理（Predictor线程安全，可复用）
    float[] input = new float[30];  // 30维特征
    String result = predictor.predict(input);
    // 批量推理: List<String> results = predictor.batchPredict(inputs);
}
```

**Translator的关键作用**：
```java
public class CustomTranslator implements Translator<float[], String> {
    // 预处理: Java输入 → NDArray（模型输入格式）
    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        NDManager manager = ctx.getNDManager();
        NDArray features = manager.create(input).reshape(1, 30);
        // ⚠️ 必须与Python训练端的StandardScaler参数完全一致
        NDArray mean = manager.create(new float[] {123.68f, ...});
        NDArray std = manager.create(new float[] {58.39f, ...});
        return new NDList(features.sub(mean).div(std));
    }
    // 后处理: NDArray（模型输出） → Java输出
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        float[] probs = list.get(0).toFloatArray();
        return String.format("预测类别: %d (置信度%.2f%%)", 
            probs[1] > probs[0] ? 1 : 0, Math.max(probs[0], probs[1]) * 100);
    }
}
```

### Spring Boot集成模式

```
POST /api/predict
    ↓
Spring MVC Controller
    ↓
@Service PredictorService  (持有ZooModel + Predictor实例)
    ↓
DJL Translator.processInput()
    ↓
模型推理
    ↓
DJL Translator.processOutput()
    ↓
返回JSON响应
```

### Java与Python的协同模式

| 模式 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **同一进程推理** | DJL/ONNX Runtime Java直接推理 | 零IPC开销，部署简单 | 大模型堆内存压力 |
| **子进程调用** | ProcessBuilder启动Python脚本 | 灵活，Python生态完整 | 进程启动开销，数据序列化 |
| **gRPC服务** | Python启动gRPC server，Java作为client | 语言解耦，可独立扩缩 | 网络开销，需要服务治理 |
| **REST API** | FastAPI提供HTTP接口，Java用OkHttp/Feign调用 | 最简单，部署灵活 | 延迟最高 |

---

## 5.2 GitHub项目推荐

| 项目名 | 链接 | 核心学习点 | clone命令 |
|--------|------|-----------|-----------|
| djl-demo | github.com/deepjavalibrary/djl-demo | **DJL官方示例** - Spring Boot/模型加载/批量推理 | `git clone --depth 1 https://github.com/deepjavalibrary/djl-demo.git` |
| djl | github.com/deepjavalibrary/djl | DJL核心库源码学习 | `git clone --depth 1 https://github.com/deepjavalibrary/djl.git` |
| langchain4j | github.com/langchain4j/langchain4j | **Java版LangChain** - LLM应用开发 | `git clone --depth 1 https://github.com/langchain4j/langchain4j.git` |
| onnxruntime-java | github.com/microsoft/onnxruntime | ONNX Runtime Java API示例 | `git clone --depth 1 https://github.com/microsoft/onnxruntime.git` |
| spring-ai | github.com/spring-projects/spring-ai | **Spring官方AI项目** - 向量数据库/LLM集成 | `git clone --depth 1 https://github.com/spring-projects/spring-ai.git` |

---

## 5.3 Java完整示例：Spring Boot + DJL推理服务

### pom.xml 依赖

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>ai.djl</groupId>
        <artifactId>api</artifactId>
        <version>0.27.0</version>
    </dependency>
    <dependency>
        <groupId>ai.djl.onnxruntime</groupId>
        <artifactId>onnxruntime-engine</artifactId>
        <version>0.27.0</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

### Java完整代码

```java
package com.ai.mlservice;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

// ===== Translator: 输入/输出转换 =====
class ClassificationTranslator implements Translator<float[], Map<String, Object>> {
    private static final float[] MEANS = {14.13f, 19.29f, 91.97f, 654.89f, 0.10f, 0.10f, 0.09f, 0.05f,
        0.18f, 0.06f, 0.40f, 1.22f, 2.87f, 40.34f, 0.01f, 0.03f, 0.04f, 0.01f, 0.02f, 0.00f,
        16.27f, 25.68f, 107.26f, 880.58f, 0.13f, 0.25f, 0.27f, 0.11f, 0.29f, 0.08f};
    private static final float[] STDS = {3.52f, 4.30f, 24.30f, 351.92f, 0.01f, 0.05f, 0.08f, 0.04f,
        0.03f, 0.01f, 0.28f, 0.55f, 2.03f, 45.51f, 0.01f, 0.02f, 0.03f, 0.01f, 0.01f, 0.00f,
        4.83f, 6.15f, 33.60f, 569.36f, 0.02f, 0.16f, 0.21f, 0.07f, 0.06f, 0.02f};

    @Override
    public NDList processInput(TranslatorContext ctx, float[] features) {
        NDManager mgr = ctx.getNDManager();
        NDArray input = mgr.create(features).reshape(1, 30);
        NDArray mean = mgr.create(MEANS);
        NDArray std = mgr.create(STDS);
        return new NDList(input.sub(mean).div(std));
    }
    @Override
    public Map<String, Object> processOutput(TranslatorContext ctx, NDList list) {
        float[] probs = list.get(0).toFloatArray();
        Map<String, Object> result = new HashMap<>();
        result.put("predictedClass", probs[1] > probs[0] ? 1 : 0);
        result.put("confidence", Math.max(probs[0], probs[1]));
        result.put("benignProb", probs[0]);
        result.put("malignantProb", probs[1]);
        return result;
    }
}

// ===== 推理服务: 单例持有模型实例 =====
@Service
class InferenceService implements AutoCloseable {
    private final ZooModel<float[], Map<String, Object>> model;
    private final Predictor<float[], Map<String, Object>> predictor;
    private final ConcurrentHashMap<String, Long> latencyStats = new ConcurrentHashMap<>();

    public InferenceService() throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<float[], Map<String, Object>> criteria = Criteria.builder()
            .setTypes((Class<float[]>) (Class<?>) float[].class, 
                      (Class<Map<String, Object>>) (Class<?>) Map.class)
            .optModelPath(Paths.get("models/"))
            .optModelName("breast_cancer")  // 加载 models/breast_cancer.onnx
            .optTranslator(new ClassificationTranslator())
            .optEngine("OnnxRuntime")
            .build();
        
        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
        System.out.println("✅ DJL模型加载成功，输入形状: " + model.describeInput());
    }

    public Map<String, Object> predict(float[] features) throws TranslateException {
        long t0 = System.nanoTime();
        Map<String, Object> result = predictor.predict(features);
        long latencyMs = (System.nanoTime() - t0) / 1_000_000;
        result.put("latencyMs", latencyMs);
        return result;
    }

    public List<Map<String, Object>> batchPredict(List<float[]> batch) throws TranslateException {
        return predictor.batchPredict(batch);
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
    }
}

// ===== Controller: REST API =====
@RestController
@RequestMapping("/api")
class PredictionController {
    private final InferenceService inferenceService;

    public PredictionController(InferenceService inferenceService) {
        this.inferenceService = inferenceService;
    }

    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestBody Map<String, Object> body) {
        @SuppressWarnings("unchecked")
        List<Number> featureList = (List<Number>) body.get("features");
        float[] features = new float[30];
        for (int i = 0; i < Math.min(30, featureList.size()); i++) {
            features[i] = featureList.get(i).floatValue();
        }
        try {
            Map<String, Object> result = inferenceService.predict(features);
            result.put("status", "success");
            return result;
        } catch (TranslateException e) {
            Map<String, Object> err = new HashMap<>();
            err.put("status", "error");
            err.put("message", e.getMessage());
            return err;
        }
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "service", "ML-inference");
    }
}

@SpringBootApplication
public class MlServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(MlServiceApplication.class, args);
        System.out.println("✅ Spring Boot + DJL 推理服务已启动");
        System.out.println("   POST /api/predict  - 单样本推理");
        System.out.println("   GET  /api/health   - 健康检查");
    }
}
```

**运行测试**：
```bash
curl -X POST http://localhost:8080/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [12.3, 15.8, 79.5, 450.0, 0.095, 0.078, 0.045, 0.025, 0.17, 0.06, 0.25, 0.8, 1.8, 22.5, 0.006, 0.018, 0.022, 0.008, 0.015, 0.002, 14.2, 20.5, 92.0, 600.0, 0.12, 0.15, 0.12, 0.05, 0.25, 0.07]}'
```

---

## 5.4 面试题库

### 📝 理论题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 5.1 | DJL的Criteria、ZooModel、Predictor、Translator分别是什么职责？ | 中 | ⭐⭐⭐⭐ |
| 5.2 | Java推理和Python训练的特征处理必须完全对齐。常见偏差来源有哪些？ | 中 | ⭐⭐⭐⭐ |
| 5.3 | DJL多引擎支持（PyTorch/ONNX/TF）的优缺点对比？ | 中 | ⭐⭐⭐ |
| 5.4 | Spring Boot中加载AI模型应该用什么生命周期管理？Bean单例/原型？为什么？ | 中 | ⭐⭐⭐⭐ |
| 5.5 | Java调用Python的三种方式（ProcessBuilder/gRPC/REST）对比？ | 中 | ⭐⭐⭐ |
| 5.6 | 大模型推理对JVM堆内存/GC的影响？有哪些优化策略？ | 难 | ⭐⭐⭐ |
| 5.7 | 模型版本管理：线上服务如何平滑切换模型版本？（A/B测试、灰度发布） | 中 | ⭐⭐⭐ |
| 5.8 | 高并发推理场景：如何设计线程池/批量推理/队列缓冲？ | 难 | ⭐⭐⭐ |

### ☕ Java代码题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 5.9 | 用DJL写一个完整推理服务：Criteria定义 + Translator实现 + main方法测试 | 中 | ⭐⭐⭐⭐ |
| 5.10 | Java实现ONNX Runtime Java加载图像分类模型推理（参考ONNX官方示例） | 中 | ⭐⭐⭐ |
| 5.11 | Spring Boot + DJL集成：@Service持有模型实例 + @RestController暴露API | 中 | ⭐⭐⭐⭐ |
| 5.12 | Java用ProcessBuilder调用Python脚本完成推理，写一个完整示例 | 简 | ⭐⭐ |
| 5.13 | 用LangChain4j实现一个简单的LLM对话服务（Java原生实现） | 简 | ⭐⭐⭐ |

### 🔧 架构设计题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 5.14 | 设计企业级ML推理平台：模型注册中心、多模型管理、A/B测试、监控告警 | 难 | ⭐⭐⭐ |
| 5.15 | 设计高并发图像识别服务：队列缓冲+批处理+多模型实例+容错降级 | 难 | ⭐⭐⭐ |

---

> ✅ **方向5（Java后端AI服务）学习完成自检清单**：
> - [ ] 能用DJL加载ONNX模型并完成推理
> - [ ] 能写出完整的Spring Boot + DJL推理服务
> - [ ] 理解Java与Python特征处理对齐的重要性
> - [ ] 掌握Java多引擎选择策略（DJL/ONNX Runtime/TF Java）