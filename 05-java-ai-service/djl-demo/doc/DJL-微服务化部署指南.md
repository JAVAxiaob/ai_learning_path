# DJL 微服务化部署与Spring Cloud集成指南

> 位置: 05-java-ai-service/djl-demo/doc/
> 基础文档: 05-java-ai/djl-demo/doc/（DJL核心概念+面试题+重难点）
> 本篇重点: Spring Cloud微服务化 / Docker/K8s部署 / 与业务Feign调用 / 多模型服务

---

## 🎯 微服务架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DJL AI服务 微服务架构图                              │
│                                                                             │
│   前端/客户端                                                               │
│    ▲                                                                        │
│    │ 1. HTTPS请求                                                           │
│    ▼                                                                        │
│   Spring Cloud Gateway (鉴权JWT/限流/路由/灰度发布)                           │
│    │ 2. 路由 ai-djl-service → 负载均衡轮询N个Pod                              │
│    ▼                                                                        │
│   ┌─────────────────────┐      ┌─────────────────────┐                      │
│   │  DJL AI Service Pod1│      │  DJL AI Service PodN│  K8s HPA自动弹性扩缩  │
│   │  Spring Boot 3.x    │ ...  │  Spring Boot 3.x    │  GPU指标: utilization │
│   │  DJL Spring Starter │      │  DJL Spring Starter │  自定义: QPS/延迟   │
│   └─────────────────────┘      └─────────────────────┘                      │
│         │          ▲                                                         │
│    3.Feiyn调用    │4. 返回推理结果                                            │
│         ▼          │                                                         │
│   ┌─────────────────────────┐   ┌──────────────────────┐                    │
│   │ 业务服务(订单/商品/HR)   │   │   Nacos注册中心      │                    │
│   └─────────────────────────┘   └──────────────────────┘                    │
│                                                                             │
│   配置中心: Nacos / Spring Cloud Config (模型路径/EP/线程池参数动态配置)       │
│   链路追踪: SkyWalking / Tempo / Zipkin (TraceId贯通业务→AI推理)             │
│   监控告警: Prometheus + Grafana + 飞书/企微机器人                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 1. Maven Spring Cloud 集成依赖

```xml
<parent>
  <groupId>org.springframework.cloud</groupId>
  <artifactId>spring-cloud-starter-parent</artifactId>
  <version>2023.0.x</version> <!-- 匹配Spring Boot 3.2+ -->
  <relativePath/>
</parent>

<dependencies>
  <!-- 1. DJL Spring Boot Starter（核心，自动装配Model/Predictor）-->
  <dependency>
    <groupId>ai.djl.spring</groupId>
    <artifactId>djl-spring-boot-starter</artifactId>
    <version>0.26.0</version>
  </dependency>
  <!-- 2. DJL ONNX Runtime Engine（生产首选，CPU/GPU通吃）-->
  <dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.28.0</version>
    <classifier>linux-x86_64-gpu</classifier>
  </dependency>
  <!-- 3. Spring Cloud 基础 -->
  <dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
  </dependency>
  <dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
  </dependency>
  <dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
  </dependency>
  <dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-circuitbreaker-resilience4j</artifactId>
  </dependency>
  <!-- 4. 可观测性 -->
  <dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
  </dependency>
</dependencies>
```

---

## 🧩 2. 多模型统一服务接口设计（BERT/BGE/YOLO/OCR都跑在同一个微服务）

```java
/**
 * AI推理服务统一接口：
 * 业务侧不用管DJL内部Model/Predictor/NDArray细节，
 * 只传模型ID+JSON输入=拿JSON输出
 */
@RestController
@RequestMapping("/api/v1/ai")
public class DjlInferenceController {

    @Autowired ModelRouterService modelRouter;  // 多模型路由：按modelId找对应Predictor

    /** ✨ 通用推理端点：1个接口跑N个模型 */
    @PostMapping("/inference/{modelId}")
    @Time(value = "ai_inference_latency", extraTags = {"model", "{modelId}"}) // Micrometer埋点
    @CircuitBreaker(name = "aiInference", fallbackMethod = "inferenceFallback")  // Resilience4j熔断
    @RateLimiter(name = "aiQps")
    public <T, R> ApiResponse<R> inference(
            @PathVariable String modelId,
            @RequestBody @Valid InferenceRequest<T> req,
            @RequestHeader(required = false) String traceId) {
        long start = System.nanoTime();
        try {
            Predictor<T, R> predictor = modelRouter.getPredictor(modelId);
            R result = predictor.predict(req.getInput());
            return ApiResponse.ok(result);
        } catch (Exception e) {
            Metrics.counter("ai.inference.errors", "model", modelId).increment();
            log.error("模型{}推理失败 traceId={}", modelId, traceId, e);
            throw new AiInferenceException(e.getMessage());
        } finally {
            Metrics.timer("ai.inference.latency", "model", modelId)
                   .record(System.nanoTime()-start, TimeUnit.NANOSECONDS);
        }
    }

    /** 熔断降级：返回缓存默认值或转CPU备用模型 */
    public <T, R> ApiResponse<R> inferenceFallback(String modelId, InferenceRequest<T> req,
                                                    String traceId, CallNotPermittedException e) {
        log.warn("{}熔断触发，降级返回兜底 traceId={}", modelId, traceId);
        return ApiResponse.degraded("AI服务繁忙，稍后重试");
    }
}
```

```java
@Service
public class ModelRouterService {
    /** key=模型ID, value=DJL Predictor线程安全池化实例 */
    private final ConcurrentHashMap<String, Predictor<?, ?>> predictors = new ConcurrentHashMap<>();
    @Autowired ApplicationContext ctx;

    @PostConstruct
    public void initModels() throws ModelException, IOException {
        // 支持的模型清单：可以Nacos动态配置热更新
        List<ModelDefinition> models = loadFromNacosConfig();
        for (ModelDefinition def : models) {
            Criteria<?, ?> criteria = buildCriteria(def);
            Model model = Model.newInstance(def.getId());
            model.load(criteria);
            Predictor<?, ?> predictor = model.newPredictor();
            predictors.put(def.getId(), predictor);
            log.info("✅ 模型{}加载成功 type={} url={}", def.getId(), def.getType(), def.getUrl());
        }
    }

    /** ✅ 动态热加载：Nacos配置变更→无损加/卸载模型 */
    @NacosConfigListener(dataId = "ai-models.yaml", group = "DEFAULT_GROUP", timeout = 5000)
    public synchronized void onModelsConfigChanged(String newConfig) {
        List<ModelDefinition> latest = parseConfig(newConfig);
        // 卸载不再需要的模型
        predictors.keySet().removeIf(id -> {
            boolean remove = !latest.stream().anyMatch(d -> d.getId().equals(id));
            if (remove) { predictors.get(id).close(); predictors.get(id).getModel().close(); }
            return remove;
        });
        // 加载新增模型
        for (ModelDefinition def : latest) {
            if (!predictors.containsKey(def.getId())) {
                predictors.put(def.getId(), createPredictor(def));
            }
        }
    }

    @SuppressWarnings("unchecked")
    public <I, O> Predictor<I, O> getPredictor(String modelId) {
        Predictor<I, O> p = (Predictor<I, O>) predictors.get(modelId);
        if (p == null) throw new IllegalArgumentException("不支持的模型：" + modelId);
        return p;
    }
}
```

---

## 🔧 3. Nacos动态配置示例（无需重启加模型）

`ai-models.yaml`（Nacos Config中维护）：
```yaml
ai:
  models:
    - id: bge-large-zh           # 中文Embedding向量模型
      type: embedding
      engine: onnxruntime
      url: s3://ai-models/bge-large-zh-v1.5.zip
      translator: ai.djl.modality.nlp.translator.BertEmbeddingTranslator
      batch-size: 32
      threads: 8
      instance-pool: 4           # Predictor池大小，对应并发

    - id: yolov8s-detection      # 目标检测模型
      type: cv
      engine: onnxruntime
      url: s3://ai-models/yolov8s.onnx.zip
      translator: ai.djl.modality.cv.translator.YoloV8TranslatorFactory
      threshold: 0.25
      image-size: 640

    - id: ocr-ppocrv4           # PaddleOCR中英文识别
      type: ocr
      engine: paddlepaddle
      url: s3://ai-models/ppocr-v4.zip
      translator: ai.djl.modality.cv.translator.PpocrV4RecognitionTranslator
```
→ 改完配置发布：DJL服务自动监听Nacos事件，10秒内无损加载新模型，不用重启Pod💯

---

## 🐳 4. Docker 镜像构建最佳实践（多层+缓存+GPU/CUDA）

### 4.1 Dockerfile-CPU（x86通用服务器）
```dockerfile
# ✨ 多阶段构建：Maven缓存层 + JRE运行层 镜像仅180MB
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn -B -q dependency:go-offline  # 依赖层独立缓存，代码变了不用重下依赖
COPY src ./src
RUN mvn -B -q -DskipTests package

# 运行阶段：JRE 17 Alpine（轻量）
FROM eclipse-temurin:17-jre-alpine
RUN apk add --no-cache libgomp1 zlib-dev  # OpenMP/ONNX Runtime需要的动态库
WORKDIR /app
COPY --from=builder /app/target/ai-djl-service.jar /app/app.jar
ENV JAVA_OPTS="-XX:MaxRAMPercentage=75.0 -XX:+UseG1GC -Dorg.bytedeco.javacpp.maxbytes=4G"
# ✅ DJL模型缓存：/root/.djl.ai 单独挂载Volume，Pod重启不用重复下载几百MB模型
VOLUME ["/root/.djl.ai/cache"]
EXPOSE 8080
ENTRYPOINT exec java $JAVA_OPTS -jar app.jar
```

### 4.2 Dockerfile-GPU（NVIDIA CUDA 12.x）
```dockerfile
# 基于NVIDIA官方CUDA基础镜像，自带cuDNN驱动
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
# 安装JDK 17
RUN apt-get update && apt-get install -y openjdk-17-jre-headless wget
COPY target/ai-djl-service.jar /app/app.jar
# NVIDIA要求环境变量
ENV NVIDIA_VISIBLE_DEVICES=all \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    DJL_ENGINE_CACHE_DIR=/models/.djl-cache
VOLUME ["/models"]
EXPOSE 8080
HEALTHCHECK --interval=10s CMD curl -f http://localhost:8080/actuator/health || exit 1
ENTRYPOINT ["java","-Xmx12G","-Dai.djl.default_engine=OnnxRuntime","-jar","/app/app.jar"]
```

---

## ☸️ 5. Kubernetes 部署 + HPA GPU自动扩缩容

### 5.1 Deployment YAML：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-djl-service
spec:
  replicas: 3
  selector: {matchLabels: {app: ai-djl-service}}
  template:
    metadata: {labels: {app: ai-djl-service}}
    spec:
      runtimeClassName: nvidia  # ✅ GPU节点必须指定nvidia runtime
      containers:
      - name: ai-djl
        image: registry.company.com/ai/djl-service:v1.2
        ports: [{name: http, containerPort: 8080}]
        resources:
          requests: {cpu: "8", memory: "16Gi", nvidia.com/gpu: "1"}  # 1张GPU卡
          limits:   {cpu: "16", memory: "32Gi", nvidia.com/gpu: "1"}
        volumeMounts:
        - name: djl-cache
          mountPath: /root/.djl.ai/cache  # 模型缓存共享盘，不用重复下载
        readinessProbe:
          httpGet: {path: /actuator/health, port: 8080}
          initialDelaySeconds: 60  # 模型加载慢，首启给60秒
        livenessProbe:
          httpGet: {path: /actuator/health/liveness, port: 8080}
          periodSeconds: 30
        lifecycle:
          preStop:
            exec:
              # ✅ 优雅下线：等30秒处理完在途请求再关Pod，防止502
              command: ["sh", "-c", "sleep 30 && curl -X POST http://localhost:8080/actuator/shutdown"]
      volumes:
      - name: djl-cache
        hostPath: {path: /data/djl-cache, type: DirectoryOrCreate}
      terminationGracePeriodSeconds: 120  # 配合preStop给足2分钟优雅下线
```

### 5.2 HPA按GPU利用率扩容：
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: ai-djl-hpa}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: ai-djl-service}
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu   # ✅ GPU利用率>70%自动扩容
      target: {type: Utilization, averageUtilization: 70}
  - type: Pods
    pods:
      metric: {name: ai_inference_qps_per_pod}  # 自定义指标：单Pod QPS>50
      target: {type: AverageValue, averageValue: "50"}
  behavior:
    scaleUp:   {stabilizationWindowSeconds: 60,  policies: [{type: Percent, value: 100, periodSeconds: 60}]}
    scaleDown: {stabilizationWindowSeconds: 300, policies: [{type: Pods, value: 2, periodSeconds: 120}]}
```

---

## 🔗 6. 业务服务Feign客户端调用示例

**订单服务调用AI OCR识别发票：**
```java
// 订单服务侧 Feign Client（不用知道DJL，纯HTTP调用AI微服务）
@FeignClient(name = "ai-djl-service", configuration = FeignRetryConfig.class)
public interface AiDjlClient {
    @PostMapping("/api/v1/ai/inference/ocr-ppocrv4")
    ApiResponse<OcrResult> recognizeInvoice(@RequestBody InferenceRequest<ImageInput> req);
}
```

```java
// 订单业务中用：
@Service
public class InvoiceService {
    @Autowired AiDjlClient aiDjlClient;

    public InvoiceData ocrAndParseInvoice(MultipartFile invoiceFile) throws IOException {
        byte[] imgBytes = invoiceFile.getBytes();
        String base64 = Base64.getEncoder().encodeToString(imgBytes);
        InferenceRequest<ImageInput> req = new InferenceRequest<>(new ImageInput(base64, "OCR识别发票"));
        OcrResult ocr = aiDjlClient.recognizeInvoice(req).getData();
        return parseOcrToInvoiceData(ocr);
    }
}
```

---

## ⚡ 7. 生产级性能优化&监控建议

| 优化项 | 建议配置 | 效果 |
|---|---|---|
| **Predictor池化** | `model.newPredictor(poolSize)` poolSize=2×GPU SM数 | 高并发不用争锁，吞吐量+80% |
| DJL **批处理自动Batching** | 开启`opt.dynamic_batching=true` + `opt.batch_size=32` | GPU场景快4-10倍 |
| NDArray **内存池参数** | `-Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=16G` | 防止Native OOM，减少GC |
| **模型预热WarmUp** | `@PostConstruct`里先拿假数据跑10次predict | 首请求从300ms→10ms，消除冷启动 |
| **监控4大黄金指标** | Prometheus+Grafana仪表盘：GPU%/单请求延迟P99/推理QPS/错误率% | 故障秒级发现 |
| **Resilience4j配置** | 熔断：100个请求错误率>30%→开30秒；限流：200QPS/实例 | 雪崩保护稳定不挂 |
| **SkyWalking Trace贯通** | DJL Agent自动Span：`Model.load/Predictor.predict`耗时 | 业务→AI全链路慢调用定位 |

> 📌 完整DJL面试题25题+重难点详解：见 `05-java-ai/djl-demo/doc/` 同级目录
