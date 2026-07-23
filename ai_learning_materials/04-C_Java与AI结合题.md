# 04-C - 面试题库：Java与AI结合题 (35题)

> 核心：用Java实现AI推理服务，这是你的主战场！

---

## 第一部分：ONNX Runtime Java基础题 (15题)

### 难度: 简单

**1. 用Java调用ONNX模型做图像分类（完整代码）**
``java
import ai.onnxruntime.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

// Maven依赖:
// <dependency>
//   <groupId>com.microsoft.onnxruntime</groupId>
//   <artifactId>onnxruntime</artifactId>
//   <version>1.17.0</version>
// </dependency>

public class ImageClassifier implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public ImageClassifier(String modelPath) throws OrtException {
        // 1. 创建环境（整个应用只需创建一次）
        this.env = OrtEnvironment.getEnvironment();

        // 2. 创建Session（可配置线程数、GPU等）
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(4);  // 内部操作并行数
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session = env.createSession(modelPath, opts);

        // 3. 获取输入输出信息
        this.inputName = session.getInputNames().iterator().next();
        System.out.println("模型输入: " + session.getInputInfo());
        System.out.println("模型输出: " + session.getOutputInfo());
    }

    public float[] classify(float[][][][] preprocessedImage) throws OrtException {
        // 4. 创建输入Tensor (NCHW格式: batch, channels, height, width)
        // preprocessedImage 形状: [1, 3, 224, 224]
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, preprocessedImage);

        // 5. 执行推理
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, inputTensor);

        try (OrtSession.Result result = session.run(inputs)) {
            // 6. 获取输出（形状: [1, 1000]）
            float[][] output = (float[][]) result.get(0).getValue();
            return output[0];  // 返回第一张图片的分类结果
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
        env.close();
    }

    public static void main(String[] args) throws Exception {
        try (ImageClassifier classifier = new ImageClassifier("mobilenet.onnx")) {
            float[][][][] image = preprocessImage("test.jpg");  // 需要你实现预处理
            float[] probabilities = classifier.classify(image);
            
            // 找到Top-3类别
            int topK = 3;
            Integer[] indices = new Integer[probabilities.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            Arrays.sort(indices, (a, b) -> Float.compare(probabilities[b], probabilities[a]));
            
            for (int i = 0; i < topK; i++) {
                System.out.printf("Rank %d: Class %d (%.4f)%n", i+1, indices[i], probabilities[indices[i]]);
            }
        }
    }

    // 关键：图像预处理必须与训练时完全一致
    private static float[][][][] preprocessImage(String imagePath) {
        // TODO: 加载图片 -> resize -> RGB -> 归一化(除以255,减去均值,除以标准差) -> NCHW
        // 注意: 这是最容易出错的地方！
        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (ImageNet标准)
        return null;
    }
}
``

**2. ONNX Runtime的基本架构组件有哪些？**
- OrtEnvironment：全局环境，整个应用一个实例
- OrtSession：模型会话，每个模型一个，加载模型文件
- OnnxTensor：输入/输出的数据容器（支持多种数据类型和维度）
- SessionOptions：配置选项（优化级别、线程数、执行提供器如CUDA/TensorRT）
- 你的应用场景：一个Spring服务可能有多个OrtSession（多个模型）

**3. 如何配置ONNX Runtime的线程优化？**
``java
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
opts.setIntraOpNumThreads(4);      // 单个算子内部的并行线程数（矩阵运算的并行）
opts.setInterOpNumThreads(2);      // 多个独立算子之间的并行线程数
opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

// 选择CPU/GPU执行提供器
opts.registerOpenVINOExecutionProvider();  // Intel硬件加速
// opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING);
``
- 线程数设置原则：总线程数 <= CPU核心数，避免过度调度
- 对小模型/小batch，减少线程数反而更快（避免线程调度开销）

**4. 支持的Tensor数据类型有哪些？怎么选择？**
- 支持: FLOAT、DOUBLE、INT8、INT16、INT32、INT64、BOOL、STRING
- 选择策略：
  - 量化模型用INT8（更小更快，Android/边缘端推荐）
  - CPU推理常用FLOAT32
  - GPU推理可考虑FLOAT16（但Java端CPU场景不常用）
- 你的注意：数据类型必须与模型导出时一致，否则推理报错

**5. OrtSession应该是单例还是每次请求创建？**
- 应该是单例（或Bean单例）：Session创建涉及模型加载，开销大（秒级）
- 线程安全：OrtSession.run() 是线程安全的，多线程可共享
- 推荐：应用启动时创建，容器关闭时销毁
- Spring Boot配置示例：
``java
@Configuration
public class ModelConfig {
    @Bean(destroyMethod = "close")
    public ImageClassifier imageClassifier() throws OrtException {
        return new ImageClassifier("models/mobilenet.onnx");
    }
}
``

### 难度: 中等

**6. 处理ONNX的动态batch和动态输入尺寸**
``java
public class DynamicInference {
    // 对于动态batch模型（模型定义时batch= -1）
    public float[][] batchInference(float[][][][] batchImages) throws OrtException {
        // batchImages: [N, 3, 224, 224] 其中N可变
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, batchImages);
        try (OrtSession.Result result = session.run(
            Collections.singletonMap(inputName, inputTensor))) {
            return (float[][]) result.get(0).getValue();  // [N, 1000]
        }
    }

    // 对于动态尺寸模型（如文本模型，序列长度可变）
    public float[] dynamicSequenceInference(long[] inputIds) throws OrtException {
        // inputIds: [seq_len] -> 需要包装成 [1, seq_len]
        long[][] batchInput = new long[][] { inputIds };
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, batchInput);
        
        try (OrtSession.Result result = session.run(
            Collections.singletonMap("input_ids", inputTensor))) {
            float[] output = (float[]) result.get(0).getValue();
            return output;
        }
    }
}
``

**7. 如何实现图像预处理（与Python/PyTorch完全对齐）？**
``java
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;

public class ImagePreprocessor {
    // ImageNet预训练模型的标准参数
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};  // R,G,B
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    
    public static float[][][][] preprocess(String imagePath, int targetSize) throws Exception {
        BufferedImage img = ImageIO.read(new File(imagePath));
        
        // 1. Resize到目标尺寸（这里简化，实际应用用Image.getScaledInstance或ImageJ）
        BufferedImage resized = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(img.getScaledInstance(targetSize, targetSize, 0), 0, 0, null);
        
        // 2. HWC -> CHW, 归一化 (224,224,3) -> (3,224,224)
        float[][][][] result = new float[1][3][targetSize][targetSize];  // NCHW
        
        for (int h = 0; h < targetSize; h++) {
            for (int w = 0; w < targetSize; w++) {
                int rgb = resized.getRGB(w, h);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;
                
                // 减去均值, 除以标准差
                result[0][0][h][w] = (r - MEAN[0]) / STD[0];
                result[0][1][h][w] = (g - MEAN[1]) / STD[1];
                result[0][2][h][w] = (b - MEAN[2]) / STD[2];
            }
        }
        return result;
    }
}
// 易错点：1)通道顺序(RGB vs BGR)  2)归一化参数  3)像素值是否除以255
// 如果Java结果和Python不同，优先检查这三个点！
``

**8. Spring Boot封装推理服务REST API**
``java
@RestController
@RequestMapping("/api/model")
public class ModelController {
    private final ImageClassifier classifier;
    
    @Autowired
    public ModelController(ImageClassifier classifier) {
        this.classifier = classifier;
    }
    
    // 同步推理接口
    @PostMapping("/classify")
    public ResponseEntity<ClassificationResult> classify(@RequestParam("file") MultipartFile file) {
        try {
            // 1. 读取并预处理图片
            BufferedImage img = ImageIO.read(file.getInputStream());
            float[][][][] input = ImagePreprocessor.preprocess(img, 224);
            
            // 2. 执行推理
            float[] probabilities = classifier.classify(input);
            
            // 3. 后处理（取Top-K）
            List<Category> topCategories = PostProcessor.getTopK(probabilities, 5);
            
            return ResponseEntity.ok(new ClassificationResult("success", topCategories));
        } catch (Exception e) {
            return ResponseEntity.status(500)
                .body(new ClassificationResult("error: " + e.getMessage(), null));
        }
    }
    
    // 模型元信息接口
    @GetMapping("/info")
    public ModelInfo getModelInfo() {
        return new ModelInfo("MobileNetV3", 224, 1000, "ImageNet", classifier.getLatencyP50());
    }
}

// 数据类
class ClassificationResult {
    String status;
    List<Category> categories;
    // getters...
}

// 注意: 文件上传需要在application.yml配置:
// spring.servlet.multipart.max-file-size=10MB
// spring.servlet.multipart.max-request-size=10MB
``

**9. 实现异步推理接口（避免长请求阻塞Tomcat线程）**
``java
@RestController
@RequestMapping("/api/async")
public class AsyncModelController {
    private final ImageClassifier classifier;
    private final ExecutorService inferenceExecutor;
    
    @Autowired
    public AsyncModelController(ImageClassifier classifier) {
        this.classifier = classifier;
        // 专用推理线程池，隔离业务线程
        this.inferenceExecutor = new ThreadPoolExecutor(
            4, 8, 60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(100),
            new ThreadPoolExecutor.CallerRunsPolicy()  // 队列满时由调用线程执行
        );
    }
    
    @PostMapping("/classify")
    public CompletableFuture<ResponseEntity<ClassificationResult>> classifyAsync(
            @RequestParam("file") MultipartFile file) {
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                float[][][][] input = ImagePreprocessor.preprocess(file, 224);
                float[] probabilities = classifier.classify(input);
                List<Category> cats = PostProcessor.getTopK(probabilities, 5);
                return ResponseEntity.ok(new ClassificationResult("success", cats));
            } catch (Exception e) {
                return ResponseEntity.status(500)
                    .body(new ClassificationResult("error: " + e.getMessage(), null));
            }
        }, inferenceExecutor);  // 使用专用线程池
    }
}

// 配置: spring.mvc.async.request-timeout=30000
``

**10. 实现模型推理的延迟监控和指标上报**
``java
@Component
public class MonitoredImageClassifier {
    private final ImageClassifier classifier;
    private final MeterRegistry meterRegistry;
    
    // 用Micrometer记录（Prometheus标准）
    private final AtomicReference<Double> p50Latency = new AtomicReference<>(0.0);
    private final AtomicReference<Double> p99Latency = new AtomicReference<>(0.0);
    private final ConcurrentLinkedDeque<Long> recentLatencies = new ConcurrentLinkedDeque<>();
    
    public float[] classify(float[][][][] input) throws Exception {
        long start = System.nanoTime();
        try {
            float[] result = classifier.classify(input);
            long latencyMs = (System.nanoTime() - start) / 1_000_000;
            recordLatency(latencyMs);
            return result;
        } catch (Exception e) {
            meterRegistry.counter("inference.errors", "model", "mobilenet").increment();
            throw e;
        }
    }
    
    private void recordLatency(long latencyMs) {
        recentLatencies.offer(latencyMs);
        if (recentLatencies.size() > 1000) {
            recentLatencies.poll();
        }
        // 更新百分位数
        List<Long> sorted = new ArrayList<>(recentLatencies);
        Collections.sort(sorted);
        p50Latency.set((double) sorted.get((int)(sorted.size() * 0.5)));
        p99Latency.set((double) sorted.get((int)(sorted.size() * 0.99)));
        
        // 上报到Micrometer
        meterRegistry.gauge("inference.latency.p50", p50Latency, AtomicReference::get);
        meterRegistry.gauge("inference.latency.p99", p99Latency, AtomicReference::get);
    }
}

// 在application.yml暴露指标:
// management.endpoints.web.exposure.include=health,metrics,prometheus
// 访问 http://localhost:8080/actuator/prometheus 查看
``

### 难度: 困难

**11. 实现动态加载/切换模型（热更新）**
``java
@Component
public class ModelRouter {
    private final Map<String, ImageClassifier> modelPool = new ConcurrentHashMap<>();
    private volatile String activeModel;
    
    // 初始化加载默认模型
    @PostConstruct
    public void init() throws Exception {
        loadModel("v1.0", "models/mobilenet_v1.onnx");
        activeModel = "v1.0";
    }
    
    // 加载新版本模型
    public synchronized void loadModel(String version, String modelPath) throws Exception {
        if (!modelPool.containsKey(version)) {
            ImageClassifier newClassifier = new ImageClassifier(modelPath);
            modelPool.put(version, newClassifier);
            System.out.println("已加载模型版本: " + version);
        }
    }
    
    // 切换激活模型
    public void switchModel(String version) {
        if (modelPool.containsKey(version)) {
            activeModel = version;
            System.out.println("已切换到模型: " + version);
        }
    }
    
    // 推理请求路由到当前激活模型
    public float[] classifyWithActiveModel(float[][][][] input) throws Exception {
        ImageClassifier classifier = modelPool.get(activeModel);
        if (classifier == null) throw new IllegalStateException("No active model");
        return classifier.classify(input);
    }
    
    // 管理接口: 获取所有版本
    public List<String> getAvailableVersions() {
        return new ArrayList<>(modelPool.keySet());
    }
    
    // 管理接口: 卸载模型（需谨慎，避免影响正在进行的推理）
    public synchronized void unloadModel(String version) throws Exception {
        if (!version.equals(activeModel) && modelPool.containsKey(version)) {
            modelPool.get(version).close();
            modelPool.remove(version);
        }
    }
}

// 在@Configuration类中用@Scheduled可实现定时检查新模型文件
``

**12. 实现模型A/B测试（流量分配）**
``java
@Component
public class ABTestingRouter {
    private final ModelRouter modelRouter;
    
    // 版本 -> 流量比例 (0-1)
    private final Map<String, Double> trafficAllocation = new ConcurrentHashMap<>();
    
    public void setAllocation(String version, double ratio) {
        trafficAllocation.put(version, ratio);
    }
    
    public float[] classifyWithAB(String requestId, float[][][][] input) throws Exception {
        // 根据请求ID哈希分配（保证同一请求稳定到同一版本）
        int hash = Math.abs(requestId.hashCode() % 100);
        double random = hash / 100.0;
        
        String selectedVersion = selectVersion(random);
        System.out.println("请求 " + requestId + " 路由到模型 " + selectedVersion);
        
        // 记录请求用于后续分析
        recordRequest(requestId, selectedVersion);
        
        // 调用对应模型推理
        ImageClassifier classifier = modelRouter.getModel(selectedVersion);
        return classifier.classify(input);
    }
    
    private String selectVersion(double random) {
        double cumulative = 0;
        for (Map.Entry<String, Double> entry : trafficAllocation.entrySet()) {
            cumulative += entry.getValue();
            if (random < cumulative) return entry.getKey();
        }
        return trafficAllocation.keySet().iterator().next();
    }
}

// 后续可配合数据分析: 比较不同版本的业务指标(CTR/转化率等)
// 统计显著性检验 (t检验/卡方检验) 决定是否全量
``

**13. 使用DJL (Deep Java Library) 做推理**
``java
// DJL是AWS推出的Java深度学习框架，支持多种后端(PyTorch/TensorFlow/MXNet/ONNX)
// 优点: 1)纯Java API，更符合Java习惯  2)模型Zoo，无需手动转换格式

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.ndarray.*;
import ai.djl.translate.*;

public class DjlImageClassifier {
    // 定义输入输出处理 Pipeline
    private static final Translator<Image, Classifications> translator =
        Translator.builder()
            .transform(new Resize(224, 224))
            .transform(new ToTensor())
            .transform(new Normalize(new float[]{0.485f, 0.456f, 0.406f},
                                    new float[]{0.229f, 0.224f, 0.225f}))
            .optSynsetArtifactName("synset.txt")
            .optApplySoftmax(true)
            .build();
    
    public Classifications classify(String imagePath) throws Exception {
        // 1. 定义模型位置
        Criteria<Image, Classifications> criteria = Criteria.builder()
            .setTypes(Image.class, Classifications.class)
            .optModelUrls("file:///models/resnet50")  // 本地路径或URL
            .optTranslator(translator)
            .optProgress(new ProgressBar())
            .build();
        
        // 2. 加载模型 + 推理 (try-with-resources自动管理)
        try (ZooModel<Image, Classifications> model = criteria.loadModel();
             Predictor<Image, Classifications> predictor = model.newPredictor()) {
            
            Image img = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
            return predictor.predict(img);
        }
    }
}

// DJL与ONNX Runtime选择:
// - 已有ONNX模型: 直接用ONNX Runtime (轻量)
// - 需要多种框架/模型Zoo: 选择DJL (功能丰富)
// - 团队Java背景强: 选择DJL (API更友好)
``

**14. 调用外部LLM API (OpenAI兼容) 的Java实现**
``java
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

@Component
public class LlmService {
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String apiKey;
    private final String apiUrl = "https://api.openai.com/v1/chat/completions";
    
    // 简单的LLM缓存
    private final LoadingCache<String, String> responseCache;
    
    public LlmService(@Value("") String apiKey) {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        this.objectMapper = new ObjectMapper();
        this.apiKey = apiKey;
        
        // Caffeine缓存: 10分钟过期, 最大10000条
        this.responseCache = Caffeine.newBuilder()
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .maximumSize(10000)
            .build(CacheLoader.from(this::callLlmInternal));
    }
    
    // 非流式调用
    public String chat(String userPrompt) throws Exception {
        return responseCache.get(userPrompt);  // 缓存优先
    }
    
    private String callLlmInternal(String userPrompt) throws Exception {
        Map<String, Object> requestBody = Map.of(
            "model", "gpt-3.5-turbo",
            "messages", List.of(
                Map.of("role", "system", "content", "You are a helpful assistant"),
                Map.of("role", "user", "content", userPrompt)
            ),
            "temperature", 0.7,
            "max_tokens", 500
        );
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Authorization", "Bearer " + apiKey)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(requestBody)))
            .timeout(Duration.ofSeconds(30))
            .build();
        
        HttpResponse<String> response = httpClient.send(request, 
            HttpResponse.BodyHandlers.ofString());
        
        // 解析JSON (省略错误处理)
        Map<String, Object> json = objectMapper.readValue(response.body(), Map.class);
        List<Map<String, Object>> choices = (List) json.get("choices");
        Map<String, Object> message = (Map) choices.get(0).get("message");
        return (String) message.get("content");
    }
    
    // 流式调用(SSE) - 给前端更好的体验
    public Flux<String> chatStream(String userPrompt) {
        // 实际实现: 使用Spring WebClient + SSE
        // webClient.post()
        //   .body(...)
        //   .retrieve()
        //   .bodyToFlux(String.class)
        //   .map(...) // 解析SSE格式 "data: {...}"
        return Flux.empty();  // 此处简化
    }
}

// 关键配置在 application.yml:
// llm.api.key=sk-your-api-key-here
// llm.api.timeout=30s
// 生产环境建议: 1)重试机制(Resilience4j)  2)熔断/降级  3)token成本监控
``

**15. 实现Java版向量检索 (用本地库或调用服务)**
``java
import java.util.*;
import java.io.*;

// 方案一: 调用Milvus/Pinecone等向量数据库服务 (生产推荐)
@Component
public class VectorRetriever {
    private final EmbeddingService embeddingService;  // 负责把文本转成向量
    private final MilvusClient milvusClient;          // Milvus向量数据库客户端
    
    public List<RetrievedDoc> search(String query, int topK) {
        // 1. 查询向量化
        float[] queryVector = embeddingService.embed(query);
        
        // 2. 向量检索
        List<String> docIds = milvusClient.search(queryVector, topK);
        
        // 3. 获取原文
        return getDocumentsByIds(docIds);
    }
    
    // 新增文档索引
    public void indexDocument(String docId, String content) {
        float[] vector = embeddingService.embed(content);
        milvusClient.insert(docId, vector, Map.of("content", content));
    }
}

// 方案二: 纯Java实现简单向量检索 (学习用, 不推荐生产)
@Component
public class SimpleVectorRetriever {
    private final List<float[]> vectors = new ArrayList<>();
    private final List<String> documents = new ArrayList<>();
    private final EmbeddingService embeddingService;
    
    public List<RetrievedDoc> search(String query, int topK) {
        float[] qv = embeddingService.embed(query);
        qv = normalize(qv);  // 归一化后点积 = 余弦相似度
        
        // 计算与所有文档的相似度 (线性扫描, 文档>10万时应换成ANN)
        List<ScoredDoc> scored = new ArrayList<>();
        for (int i = 0; i < vectors.size(); i++) {
            float[] dv = vectors.get(i);
            float similarity = dotProduct(qv, dv);
            scored.add(new ScoredDoc(i, similarity));
        }
        scored.sort((a, b) -> Float.compare(b.score, a.score));
        
        // 返回Top-K
        List<RetrievedDoc> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, scored.size()); i++) {
            int idx = scored.get(i).docIndex;
            result.add(new RetrievedDoc(documents.get(idx), scored.get(i).score));
        }
        return result;
    }
    
    private float[] normalize(float[] v) {
        float sum = 0;
        for (float x : v) sum += x * x;
        float norm = (float) Math.sqrt(sum);
        for (int i = 0; i < v.length; i++) v[i] /= norm;
        return v;
    }
    
    private float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
    
    private static class ScoredDoc { int docIndex; float score; }
}
