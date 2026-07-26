# DJL Java深度学习部署 重难点解析

> 位置: 05-java-ai/djl-demo/doc/
> 配套文档: DJL-Java深度学习部署实战.md | DJL深度学习部署重难点解析.md | DJL深度学习部署面试题汇总.md

---

## 一、DJL引擎架构6层抽象 (解决Java深度学习碎片化)

```mermaid
flowchart TD
    subgraph 用户代码层 纯Java零C++
        APP["业务代码SpringBoot Service<br/>Predictor.predict(img) 纯Java调用"]
    end

    subgraph DJL API层 引擎无关一套API
        APP --> MODEL["Model 模型加载抽象<br/>model.load(\"PyTorch/resnet18\")<br/>不管是PyTorch/ONNX/TensorFlow/MXNet接口一样"]
        MODEL --> PRED["Predictor 推理器<br/>线程安全的输入→输出 自动批处理"]
        PRED --> CRIT["Criteria 配置<br/>.optEngine(\"PyTorch\")<br/>.optModelUrls(\"s3://models/r18.zip\")<br/>.optTranslator(translator 前后处理)"]
        CRIT --> TRANS["Translator 翻译器<br/>preProcess Java对象→NDArray张量<br/>postProcess NDArray→Java对象"]
    end

    subgraph Engine Provider SPI 插件化引擎
        TRANS --> SPI["Java SPI 自动发现META-INF/services<br/>运行时加载哪个引擎依赖JAR就用哪个"]
    end

    subgraph 具体引擎实现 8种Engine
        SPI --> ONNX_RT["✅ ONNX Runtime 跨平台推荐⭐生产常用<br/>CPU: onnxruntime-engine-1.17.jar<br/>GPU: onnxruntime-gpu-engine.jar cuda12x"]
        SPI --> PYTORCH["✅ PyTorch Engine JNI绑定<br/>torchscript/trace模型 原版pth模型不能直接读!"]
        SPI --> TF["✅ TensorFlow 2.x SavedModel格式"]
        SPI --> MXNET["MXNet (D2L教材配套入门用)"]
        SPI --> TENSORRT["TensorRT Engine 极致GPU 调参复杂"]
        SPI --> FASTTEXT["FastText 文本分类轻量专用"]
        SPI --> XGBOOST["XGBoost GBDT ML模型专用 DJL统一接口"]
        SPI --> LIGHTGBM["LightGBM Engine 同上"]
    end

    subgraph 底层原生库
        ONNX_RT --> NATIVE["~/.djl.ai/cache 首次启动自动下载<br/>对应平台Linux/Windows/Mac原生so/dll/框架<br/>不用手动装CUDA/CuDNN!"]
    end
```

> 🏆 生产90%场景选型: **ONNX Runtime Engine** = 跨平台最强兼容 + 无需JNI折腾 + 速度和原生接近

---

## 二、NDArray 核心数据结构 (NumPy Java版重写)

### 2.1 NDManager 生命周期 内存管理 (内存泄漏第一大杀手!)

```
NDArray 是off-heap堆外内存 不受JVM GC管! 用完必须手动close!
DJL解决: try-with-resources + NDManager作用域树形管理
```

```java
// ✅ 最佳实践 永远用try-with-resources
try (NDManager manager = NDManager.newBaseManager()) {  // 1. 创建作用域
    NDArray array = manager.create(new float[]{1,2,3,4}, new Shape(2,2));  // 2. 在作用域里创建
    NDArray result = array.relu().matMul(array.transpose());
    System.out.println(result.toDebugString(/*True打印数值*/));
} // 3. 作用域结束自动close所有NDArray 堆外内存立刻释放!

// ❌ 危险: 方法返回NDArray 外面拿出去调用方忘了close = 显存泄漏 跑几小时OOM!
public NDArray badMethod() { return manager.create(...); }  // 💥 内存泄漏!
// ✅ 正确: 返回复制到JVM的基本类型 / 封装到自定义POJO detach
public float[][] goodMethod() {
    try (NDManager m = NDManager.newBaseManager()) {
        return m.ones(new Shape(2,3)).toFloatArray(); // 转成堆内float[]安全返回
    }
}
```

NDManager 父子层级:
```
BaseManager根作用域 (大模型加载到根 跟着Model生命周期)
  ├─ SubPredictor1 子Manager (每推理1次Predictor自动开子)
  │    └─ NDArrays 中间变量 推理完自动释放
  ├─ SubPredictor2 子Manager
  │    └─ ...
  ⚠️ 别在子manager里创建传到父Manager外面! 会提前被close成AlreadyClosed
```

---

## 三、Translator 前后处理 80%推理Bug都在这

### 3.1 图像分类ImageClassification Translator 黄金模板

```java
// ✅ ResNet50 标准预处理 和PyTorch torchvision完全对齐 精度0损失!
Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
    // 1. 解码: 自动处理jpg/png/bmp 颜色空间RGB/BGR别搞反
    .optFlag(Image.Flag.COLOR)
    // 2. Resize保持比例+CenterCrop 224×224 和训练时严格一致! 不一致精度掉爆
    .addTransform(new Resize(256, Image.Interpolation.BILINEAR))
    .addTransform(new CenterCrop(224, 224))
    // 3. ToTensor: HWC uint8 [0,255] → CHW float32 [0,1]
    .addTransform(new ToTensor())
    // 4. Normalize ImageNet 均值标准差 和PyTorch官方保持一致!
    .addTransform(new Normalize(
        new float[]{0.485f, 0.456f, 0.406f},  // mean R,G,B
        new float[]{0.229f, 0.224f, 0.225f})) // std R,G,B
    // 5. 后处理: Top-5分类 Softmax + 映射到ImageNet 1000类标签synset.txt
    .optApplySoftmax(true)
    .optSynsetArtifactName("synset.txt") // 放在Model.zip内部resources下
    .optTopK(5)
    .build();
```

> 🔴 **致命坑80%**: 训练和推理预处理不一致!!
> - 训练用了RandomResizedCrop 推理忘了CenterCrop 精度掉15%
> - 颜色空间RGB/BGR搞反了 OpenCV默认是BGR DJL Image是RGB 别直接喂!
> - Normalize的mean/std不是ImageNet 用了自己数据集的也要一模一样

---

## 四、并发性能与线程池调优

### 4.1 Predictor 线程安全策略

```
❌ 错误 1个Predictor多线程共享: 内部状态有Tensor 线程冲突崩溃
✅ 正确 Predictor随用随new 极轻量级: Model是线程安全大对象 共享1个
Model model = Model.newInstance("resnet18");  // 单例 整个App只加载1次

@Bean public Model resnetModel() { return Model.newInstance("r18"); }  // Spring单例

@GetMapping("/predict")
public Result predict(BufferedImage img) {
    // ✅ 每个请求 new Predictor try-with-resources自动close 无状态
    try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
        return predictor.predict(img).toResult();
    }
}
```

### 4.2 批量推理 性能提升

```java
// ✅ 批处理 Predictor 批量predict 小图从10ms/张→批8张30ms=3.75ms/张 快2.6倍
List<Image> batch = Arrays.asList(img1, img2, img3, img4);
List<Classifications> results = predictor.batchPredict(batch);  // 一次Kernel Launch

// 服务端动态攒批 (参考vLLM Continuous Batching 简化版)
int MAX_BATCH = 8, MAX_WAIT_MS = 5;
LinkedBlockingQueue<Request> q = new LinkedBlockingQueue<>();
// 攒批线程: 攒够8个 or 等5ms超时 就跑一次batchPredict 吞吐量翻倍
```

---

## 五、生产部署 6种服务方式对比

| 部署方式 | 延迟 | 吞吐 | 运维复杂度 | 推荐场景 |
|---------|-----|------|----------|---------|
| **内嵌Spring Boot Jar** | 低直连 | 中等CPU | ★最简单 | 中小项目 小流量 |
| **Spring Boot + GPU ONNX** | 低 | 高A100×20 | ★★ | ⭐生产主流首选 |
| **DJL Serving 独立模型服务** gRPC | 中网络R/T | 很高动态批 | ★★★ | 多模型微服务解耦 |
| **DJL Spark 分布式批推理** | 高 | 吞吐量百万级 | ★★★★ | 海量离线样本夜间批 |
| **Android端侧** aar包 | 低本地 | 手机端 | ★★ | App离线推理 |
| **AWS SageMaker DJL容器** | - | 自动扩缩容 | ★ | 云上全托管SageMaker LMI容器大模型70B |

Spring Boot 内嵌启动Docker多阶段构建:
```dockerfile
# 阶段1 拉模型到镜像内
FROM curlimages/curl AS model-download
RUN curl -O https://resources.djl.ai/model/resnet18_v1.zip

# 阶段2 运行镜像
FROM amazoncorretto:17-alpine
COPY target/app.jar /app/
COPY --from=model-download /home/curl_user/*.zip /opt/models/
# GPU版 基础镜像换nvidia/cuda:12.1.0-runtime-ubuntu22.04 + 装JDK17
ENV DJL_CACHE_DIR=/opt/djl-cache
EXPOSE 8080
ENTRYPOINT ["java", "-Xms4g", "-Xmx8g", "-Dai.djl.default_engine=OnnxRuntime", \
            "-jar", "/app/app.jar"]
```

---

## 六、常见错误TOP 10排查清单

| 错误症状 | 排查项 (按概率) |
|---------|--------------|
| `IllegalStateException: NDArray已关闭` | 跨Manager作用域泄漏 子Manager创建外面用了 |
| 推理精度比Python低10%+ | 1. 预处理是否对齐? Resize方式/Normalize值/颜色RGB<br/>2. input dtype是不是FP32变了FP16精度 |
| 启动就报UnsatisfiedLinkError找不dll | 1. 引擎JAR有没有带? onnxruntime-engine.jar<br/>2. cache目录.djl.ai权限? 手动设置-Djava.io.tmpdir |
| CPU跑不满利用率低 | 1. intra_op_num_threads 设CPU核数 .optOption("intra_op_num_threads","16")<br/>2. 并行独立请求开多线程别串行 |
| GPU跑了但是utilization只有10% | 1. CPU预处理瓶颈 解图片Resize慢 → 开独立线程池<br/>2. Batch太小1张 → 动态攒批到8 |
| 多线程OOM堆外内存爆 | Predictor必须try-with-resources! 不要静态字段持有大NDArray |
| ONNX转完模型Shape不匹配 | 先Netron看onnx输入是不是和Translator生成的shape完全一样 |
| 中文乱码/标签读synset.txt错 | 文件UTF-8无BOM 每行一个标签别带空行 |
| 第一次启动巨慢2分钟 | .djl.ai/cache没预热 首次下载JNI引擎 ~500MB; Dockerfile提前烘好缓存 |
| PyTorch加载pth文件报错 | 必须先Python端torch.jit.trace保存成TorchScript格式.pt .pth原始权重DJL读不了! |