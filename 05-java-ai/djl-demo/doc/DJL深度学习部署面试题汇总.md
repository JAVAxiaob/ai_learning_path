# DJL深度学习部署 面试题汇总 (25题)

> 位置: 05-java-ai/djl-demo/doc/
> 配套文档: DJL-Java深度学习部署实战.md | DJL深度学习部署重难点解析.md | DJL深度学习部署面试题汇总.md

---

## 一、架构与基础（7题）

### Q1. DJL解决Java AI什么痛点? 8种Engine SPI插件化, 不用写JNI/C++跨平台一次代码PyTorch/ONNX/TF/MXNet通吃（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：
DJL (Deep Java Library) 是AWS开源的Java深度学习框架，核心解决**Java生态与AI框架割裂**的5大痛点：
- 痛点①：Java服务要接PyTorch/TF模型必须手写JNI/C++封装，维护成本高、崩溃排查难
- 痛点②：每种引擎一套API，换引擎全量重写代码
- 痛点③：跨平台(Windows/Linux/macOS/Aarch64)原生库打包噩梦
- 痛点④：Java堆内存 vs 模型大张量堆外内存管理混乱
- 痛点⑤：缺少Spring Boot微服务级别的生产级最佳实践

DJL用**Engine SPI插件化架构**统一8种引擎API：
```
         ┌───────────────────────────────────┐
         │    DJL Core API (Criteria/Model)  │  用户代码只写这一层
         └─────────────┬─────────────────────┘
                       │ SPI接口
     ┌─────────┬───────┼───────┬──────────┐
     ▼         ▼       ▼       ▼          ▼
 PyTorch   ONNX    TensorFlow  MXNet   TensorRT ... (8种Engine Provider JAR)
     │         │       │       │          │
     └──各自封装原生C++库，自动下载匹配OS/Arch的native so/dll/dylib
```

2. **对比表格**：
| 方案 | 写JNI代码 | 跨引擎复用 | 跨平台打包 | GPU支持 | 生产稳定性 |
|-----|----------|-----------|-----------|---------|-----------|
| 手写JNI封装 | ✅ 几千行C++ | ❌ 每引擎重写 | ❌ 自编译各平台 | ⚠️ 自写CUDA调用 | ❌ JVM崩溃无日志 |
| DJL | ❌ 零JNI | ✅ 同一套API | ✅ 自动native下载 | ✅ CUDA自动识别 | ✅ 百万级生产验证 |

3. **代码示例**：切换引擎只需改pom依赖，Java代码**零修改**：
```xml
<!-- 用ONNX Runtime引擎 -->
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.27.0</version>
</dependency>
<!-- 换成PyTorch引擎只需替换上面这个dependency，下面Java代码不变 -->
```
```java
// 通用API，不依赖任何具体引擎
Criteria<Image, Classifications> criteria = Criteria.builder()
    .setTypes(Image.class, Classifications.class)
    .optModelUrls("file:///models/resnet18") // 不用指定.engine！
    .optTranslator(ImageClassificationTranslator.builder().build())
    .build();
```

4. **常见坑点/面试追问**：
- 追问：「8种Engine是哪些？」→ 答：PyTorch、ONNX Runtime、TensorFlow、MXNet、TensorRT、PaddlePaddle、TFLite、GGUF(LLM)
- 坑：同项目不能混放两个Engine JAR的不同大版本（类冲突），用djl-bom统一管理版本

---

### Q2. Engine Provider SPI加载机制: 为什么只改pom依赖换engine JAR就能无缝切引擎 代码一行不变? META-INF/services自动发现原理（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：基于Java标准`ServiceLoader` SPI (Service Provider Interface) 机制，实现**面向接口编程 + 运行时动态发现实现类**：
```
DJL启动流程：
1. ClassLoader扫描所有JAR的 META-INF/services/ai.djl.engine.EngineProvider 文件
2. 每个Engine JAR在这个文件里写自己的Provider全限定类名：
   onnxruntime-engine JAR里写: ai.djl.onnxruntime.engine.OnnxRuntimeEngineProvider
   pytorch-engine JAR里写:     ai.djl.pytorch.engine.PtEngineProvider
3. ServiceLoader遍历所有Provider，调用`.getEngine()`返回Engine单例
4. DJL Core按优先级排序，选排名第1的Engine作为默认引擎
```

2. **对比表格**：
| SPI关键点 | 作用 | 没有的后果 |
|----------|------|-----------|
| META-INF/services文件 | 声明"这个JAR里有Engine实现" | ServiceLoader找不到引擎，抛EngineNotFoundException |
| EngineProvider接口 | 统一工厂方法Engine getEngine() | 每个引擎初始化方式不同(CUDA设备/ native库加载)没法统一 |
| Engine.getRank()优先级数字 | 同classpath多个引擎时谁优先 | ONNX和PyTorch都在不知道选谁，行为不确定 |

3. **代码示例**：手写一个自定义Engine Provider的最小骨架：
```java
// 1. 实现EngineProvider接口
public class MyCustomEngineProvider implements EngineProvider {
    @Override
    public Engine getEngine() {
        // 懒加载单例：加载native .so/.dll，初始化CUDA context，只执行一次
        return MyCustomEngine.INSTANCE;
    }
    @Override
    public int getRank() { return 10; } // 数字越小越优先，ONNX是10，PyTorch是5
}
// 2. 在resources/META-INF/services/ai.djl.engine.EngineProvider里写一行：
//    com.yourcompany.MyCustomEngineProvider
```

4. **常见坑点/面试追问**：
- 追问：「怎么强制指定Engine不按Rank？」→ 答：Criteria.optEngine("OnnxRuntime") 显式指定引擎名，跳过SPI排序
- 坑：Spring Boot FatJar打包时用了`spring-boot-maven-plugin`的zip布局，META-INF/services可能被覆盖，用`ServicesResourceTransformer`合并
- 坑：Java Module Path (JPMS) 下要用`provides ai.djl.engine.EngineProvider with ...`在module-info.java声明，不是靠META-INF

---

### Q3. 90%生产用哪个Engine? ONNX Runtime原因: 跨平台/不用trace/速度快/C++原生绑定稳定（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：**90%以上Java生产部署选择ONNX Runtime作为DJL的Engine**，不是PyTorch也不是TF，四大技术原因：

| 维度 | ONNX Runtime | PyTorch Engine (JNI) | TensorFlow Engine |
|-----|-------------|---------------------|------------------|
| **跨平台** | ✅ Win/Linux/macOS/ARM64/Android/iOS 所有官方预编译 | ⚠️ Linux x86_64好，ARM/Win常缺包 | ⚠️ TF2以后停止维护Java JNI |
| **模型格式** | ✅ 所有主流框架(PyTorch/TF/Paddle/MXNet)都能export ONNX，统一格式 | ❌ 必须Python端做torch.jit.trace脚本化，动态图控制流(trace不支持if/for)会挂 | ❌ 必须SavedModel格式，冻结变量麻烦 |
| **推理速度** | ✅ 内部400+算子图优化(常量折叠/算子融合/内存复用) + CUDA/TensorRT EP一键切，快2-5倍 | ⚠️ 原生LibTorch，无自动图优化 | ❌ TF1遗留架构慢 |
| **生产稳定性** | ✅ C API是纯C稳定ABI，JNA直接绑，崩溃少（JNA比JNI安全） | ❌ JNI绑C++ ABI，LibTorch版本不兼容直接SIGSEGV JVM崩溃无堆栈 | ❌ Java JNI层3年没更新 |

2. **代码示例**：开启ONNX Runtime的CUDA + TensorRT执行提供者(EP)：
```java
Criteria<Image, Classifications> criteria = Criteria.builder()
    .setTypes(Image.class, Classifications.class)
    .optEngine("OnnxRuntime")  // 强制选ONNX，虽然默认就是它
    .optModelUrls("file:///models/resnet50-onnx/")
    .optDevice(Device.gpu(0))
    // 🔥 生产必加：执行提供者顺序 TensorRT(最快) -> CUDA -> CPU(兜底)
    .optOption("executionMode", "ORT_SEQUENTIAL")
    .optOption("optimizationLevel", "ORT_ENABLE_ALL")
    .optOption("ep.executionProvider", "Tensorrt")  // TensorRT EP开
    .optOption("ep.cuda.deviceId", "0")             // CUDA EP兜底
    .build();
```

3. **常见坑点/面试追问**：
- 追问：「PyTorch Engine什么时候用？」→ 答：模型里有Python自定义算子(torch.autograd.Function)，ONNX不支持导出时，迫不得已用PyTorch Engine + TorchScript自定义算子注册
- 坑：ONNX opset版本必须和转换时一致，opset太低(<=11)很多现代算子(Swish/GELU/RoPE)没有
- 坑：生产环境CPU必须开MKL-DNN/OpenVINO EP，不要用默认CPU EP，慢3倍

---

### Q4. Model大对象 vs Predictor轻量对象 共享策略: Model全局Spring单例1次加载权重 / Predictor每个请求new无状态 try用完关（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL对象生命周期两层次架构，**必须严格遵守，否则内存泄漏+并发崩溃**：

| 对象 | 重量级/轻量 | 线程安全 | 内部持有什么 | 生命周期管理 |
|-----|------------|---------|-------------|------------|
| **Model** | 🏋️ 重量级（100MB-几十GB权重） | ✅ 线程安全设计 | 反序列化权重、Engine graph、KV Cache内存池 | **全局单例！App启动加载1次，进程退出才close** |
| **Predictor** | 🪶 轻量级（几KB对象） | ❌ 非线程安全！ | 本次推理的临时NDArray、中间Tensor workspace | **每次请求new一个，try-with-resources用完关** |

> 关键理解：Model = 模型权重的**只读共享**内存；Predictor = 每次推理的**独立工作区**，类比数据库连接池：DataSource是单例（像Model），Connection每次请求借用完归还（像Predictor）

2. **代码示例**：Spring Boot标准写法（生产级模板）：
```java
@Configuration
public class DjlConfig {
    // ✅ Model 单例 启动时加载
    @Bean(destroyMethod = "close")
    public Model resnetModel() throws IOException, ModelException {
        Criteria<Image, Classifications> criteria = Criteria.builder()
            .setTypes(Image.class, Classifications.class)
            .optModelUrls("file:///opt/models/resnet50_v2")
            .optTranslator(translator())
            .build();
        return criteria.loadModel(); // 加载权重只做1次！
    }

    @Bean
    public ImageClassificationTranslator translator() {
        return ImageClassificationTranslator.builder()
            .addTransform(new Resize(256)).addTransform(new CenterCrop(224, 224))
            .addTransform(new ToTensor()).addTransform(new Normalize(mean, std))
            .optApplySoftmax(true).build();
    }
}

@Service
public class ClassificationService {
    private final Model model;

    public ClassificationService(Model model) { this.model = model; }

    public Classifications predict(BufferedImage image) {
        // ✅ Predictor 每次请求new，try自动close释放推理临时内存
        try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(image);
        } catch (TranslateException | IOException e) {
            throw new RuntimeException("推理失败", e);
        }
    }
}
```

3. **常见坑点/面试追问**：
- ❌ 致命错误1：把Predictor做成@Singleton单例，100并发请求进来 → 内部工作区被写坏，输出错乱+偶发native崩溃
- ❌ 致命错误2：每次请求都Model.newInstance()重新加载权重 → OOM，每次加载等10秒
- 追问：「Predictor能池化复用吗？」→ 答：DJL 0.23+内部已经做了Predictor内存池（NDManager工作区），new Predictor实际开销极低（<1μs），不需要用户自己做池，老版本可以用`PredictorPool`
- 追问：「多GPU怎么玩？」→ 答：每个GPU一个Model单例（Model绑Device），请求按GPU ID轮询路由

---

### Q5. Criteria 配置模型六大要素: optEngine/optModelUrls/optModelName/optTranslator前后处理/optDevice GPU0/optOption线程数（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：Criteria是DJL的**模型装载构建器**（Builder模式），六大必填/关键要素缺一不可，缺一个要么报错要么性能拉胯：

| # | 要素方法 | 作用 | 不填的后果 | 生产示例 |
|---|---------|------|-----------|---------|
| 1 | `setTypes(I, O)` | 输入输出类型约束（泛型擦除后运行期校验） | Translator类型不匹配抛ClassCastException | `setTypes(Image.class, Classifications.class)` |
| 2 | `optModelUrls()` | 模型所在URL：file:///s3://hdfs://，可以是目录或zip | 不知道去哪里找模型文件 | `optModelUrls("s3://my-bucket/models/yolov5s-v3/")` |
| 3 | `optModelName()` | 模型文件名(不带后缀)，区分同目录多个模型 | 目录下有resnet18.onnx和resnet50.onnx不知道加载谁 | `optModelName("resnet50_v2_int8")` |
| 4 | `optTranslator()` | ⚠️ 最核心！前后处理逻辑，决定80%精度 | 输入输出张量对不上模型期望shape，直接推理错或报错 | 自定义`YoloV5Translator` |
| 5 | `optDevice()` | 指定CPU/GPU号 | 有GPU默认用第0块，但多卡场景必须显式指定 | `optDevice(Device.gpu(1))` 用第2张卡 |
| 6 | `optOption()` | 引擎底层参数：线程数/EP/精度/内存策略 | CPU推理单线程跑，多核机器只用到1核=浪费87.5%性能 | `optOption("intra_op_num_threads", "16")` |

2. **代码示例**：生产级Criteria完整配置（YOLOv5目标检测场景）：
```java
Criteria<Image, DetectedObjects> criteria =
    Criteria.builder()
        .setTypes(Image.class, DetectedObjects.class)
        // 1.引擎：ONNX Runtime显式指定
        .optEngine("OnnxRuntime")
        // 2.模型URL：本地文件系统路径
        .optModelUrls("file:///opt/models/yolov5s-v6.1-onnx/")
        // 3.模型名：加载yolov5s.onnx (模型名不带后缀)
        .optModelName("yolov5s")
        // 4.Translator：自定义YOLO前后处理
        .optTranslator(
            YoloV5Translator.builder()
                .optImgSize(640)
                .optThreshold(0.25f)  // 置信度阈值
                .optNmsThreshold(0.45f) // NMS IoU阈值
                .optSynsetArtifactName("coco.names") // 标签文件在zip里
                .build())
        // 5.设备：GPU 0号卡
        .optDevice(Device.gpu(0))
        // 6.引擎参数：ONNX 关键性能调优
        .optOption("intra_op_num_threads", "16")   // CPU算子内部并行线程
        .optOption("inter_op_num_threads", "4")    // 算子间并行
        .optOption("optimizationLevel", "ORT_ENABLE_ALL") // 开全部图优化
        .optOption("memoryPatternOptimization", "true") // 内存复用优化
        .build();
```

3. **常见坑点/面试追问**：
- 追问：「optProgress是做啥的？」→ 答：大模型下载(>1GB)显示进度条回调，用户体感好
- 坑：optModelUrls写的是相对路径"./models"，在Tomcat/Spring Boot里user.dir是bin目录，不是jar所在目录，必须绝对路径或classpath:前缀
- 坑：多模型同JVM时，每个Criteria必须显式optEngine，不然一个用PyTorch一个用ONNX，SPI优先级混乱可能串了

---

### Q6. NDManager作用域内存管理堆外内存 为什么不交给JVM GC? try-with-resources自动close 父子树形继承（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：NDArray（张量）是存在**堆外Native内存**里的，不受JVM GC管理，原因三大点：
- ① **GC可达性分析延迟**：1GB的Tensor用完了，GC要等下次YGC/FGC才回收，期间堆外占满OOM
- ② **大对象不能频繁GC**：几百MB的张量如果放堆内，每分配一个触发Old Gen GC，Stop-The-World几百毫秒，服务抖动
- ③ **Native引擎需要物理连续内存**：JVM堆是分代的，内存不连续，GPU CUDA/Runtime要求连续物理页做DMA传输

**解决方案**：树形继承的NDManager作用域管理 + try-with-resources RAII确定性释放：
```
          NDManager (System/全局) - App启动创建
            └── NDManager (Model级别) - Model加载时创建，绑Model生命周期
                  └── NDManager (Predictor级别) - 每次推理创建，推理完关
                        └── NDManager (用户自定义子作用域) - 大临时张量及时关
```
父Manager.close()会递归关闭所有子Manager，保证不会泄漏。

2. **代码示例**：try-with-resources正确管理NDArray作用域：
```java
// ❌ 错误写法：临时大张量没关，堆外泄漏
public NDArray badCompute(NDManager parentManager, NDArray input) {
    NDManager tmp = parentManager.newSubManager();
    NDArray huge = tmp.create(new Shape(1, 1024, 1024, 256), DataType.FLOAT32); // 1GB
    return input.add(huge); // huge忘记关！tmp没关！每次调用泄漏1GB
}

// ✅ 正确写法：try-with-resources 离开块自动close
public NDArray goodCompute(NDManager parentManager, NDArray input) {
    try (NDManager tmp = parentManager.newSubManager()) {
        try (NDArray huge = tmp.create(new Shape(1, 1024, 1024, 256), DataType.FLOAT32)) {
            // 计算中间结果
            NDArray partial = input.add(huge);
            // ⚠️ 关键：要return的张量，必须attach到父Manager！不然离开try会被关
            return partial.detach(); // detach把NDArray从tmp移到parentManager
        }
    }
}
```

3. **常见坑点/面试追问**：
- 🔥 No.1内存泄漏原因：**忘记detach要返回的NDArray**，结果一离开try块就被关了，下次用的时候NDArray已经closed抛IllegalStateException
- 追问：「怎么排查堆外泄漏？」→ 答：① DJL自带JMX MXBean：`ai.djl:type=MemoryUsage` 看`allocatedDirectBytes` ② Linux `pmap -x <pid>` 看anon段持续增长 ③ NDManager.debugDump()打印树形结构看哪些Manager没关
- 追问：「和Netty PooledByteBuf原理一样吗？」→ 答：思想完全相同，都是堆外内存+引用计数+手动/作用域释放，区别是DJL用树形父子继承管理，Netty用引用计数retain/release

---

### Q7. NDArray与Apache Commons Math Vector/Jama对比? 为什么DJL重新实现一套: GPU支持/张量广播/C运算符速度/LAPACK MKL原生快（⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：NDArray是DJL的多维张量类，对标NumPy ndarray/PyTorch Tensor，功能远超传统Java数学库。三大根本原因DJL不使用Commons-Math/Jama：

| 维度 | DJL NDArray (PyTorch/ONNX Backend) | Apache Commons Math 3 RealVector | Jama Matrix |
|-----|----------------------------------|--------------------------------|-------------|
| **维度支持** | ✅ 0D标量~N维张量（图像4D NCHW、视频5D） | ❌ 只能1D向量 | ❌ 只能2D矩阵 |
| **GPU加速** | ✅ 自动调用CUDA cuBLAS/cuDNN，A100速度快500倍 | ❌ 纯CPU计算 | ❌ 纯CPU计算 |
| **广播Broadcasting** | ✅ 完全兼容NumPy规则(右对齐/维数1可广播) | ❌ 必须等长向量，长度不等抛Exception | ❌ 必须严格相同维度 |
| **底层加速** | ✅ C++后端+MKL-DNN/OpenBLAS/LAPACK原生绑定向量指令AVX512 | ❌ Java三重for循环纯解释执行 | ❌ Java for循环，无SIMD |
| **性能对比** | [1000×1000]矩阵乘法 ~0.5ms | ~600ms (Java循环) 慢1200× | ~200ms 慢400× |
| **与模型集成** | ✅ 零拷贝直接传给Engine推理 | ❌ 要转double[]再拷贝给native层 | ❌ 同上 |

2. **代码示例**：广播机制一行完成批归一化(BN)，Commons-Math要写几十行for：
```java
// NDArray：4维图像Batch做BN，自动广播[1,C,1,1]到[B,C,H,W]
try (NDManager manager = NDManager.newBaseManager()) {
    NDArray x = manager.randomNormal(new Shape(32, 64, 224, 224)); // Batch图像
    NDArray gamma = manager.ones(new Shape(1, 64, 1, 1));  // 缩放因子 C=64
    NDArray beta  = manager.zeros(new Shape(1, 64, 1, 1)); // 偏移
    NDArray mean = x.mean(new int[]{0, 2, 3}, true); // 保留维度广播
    NDArray var  = x.variance(new int[]{0, 2, 3}, true);
    // 🔥 一行广播完成BN，gamma自动从[1,64,1,1]扩到[32,64,224,224]
    NDArray x_bn = gamma.mul(x.sub(mean).div(var.sqrt().add(1e-5f))).add(beta);
}
```

3. **常见坑点/面试追问**：
- 追问：「NDArray和TensorFlow Java的Tensor区别？」→ 答：DJL NDArray是跨引擎的（在OnnxRuntime里和PyTorch里都能用），TF Tensor绑死TF引擎
- 坑：`toFloatArray()` 会把GPU上的NDArray拷贝回CPU内存，大张量千万不要在推理热路径调
- 坑：`getFloat(long... indices)` 逐个取元素是JNI调用，每个元素1μs。100万元素要1秒！必须批量处理。

---

## 二、核心API与Translator前后处理（6题）

### Q8. Translator 前后处理两阶段 preProcess(BufferedImage→NDArray) / postProcess(NDArray→Java POJO): 80%精度Bug在预处理与训练不一致怎么排查?（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：Translator是DJL的**前后处理适配层接口**，是用户代码和模型之间的"翻译官"，核心两个方法必须严格对齐Python训练时的逻辑：

```
┌──────────────┐   preProcess    ┌─────────────┐   Engine推理   ┌──────────────┐   postProcess   ┌──────────────┐
│  Java业务输入 │ ==============> │ 模型期望NDArray │ =============> │ 模型输出NDArray │ ===============> │ Java业务POJO  │
│ BufferedImage │  (JVM→Native)   │ [B,C,H,W] FP32 │   (Forward)    │ [B,1000] Scores │  (Native→JVM)    │ Classifications│
└──────────────┘                 └─────────────┘                └──────────────┘                 └──────────────┘
```

**为什么80%精度Bug在预处理？** 因为模型权重是对的，但输入值分布错了1个标准差，模型输出就全乱了：
- 训练时是 Normalize(mean=[0.485,...])，推理时忘了/用反了 → 输入差2.5倍标准差 → Top1准确率从76%掉到20%
- 训练时RGB顺序，OpenCV读的是BGR没转 → 通道反过来 → 准确率掉到10%以下
- 训练时除以255归一化到[0,1]，推理用的[0-255]整值 → 输入大255倍 → Softmax输出全错

2. **排查方法论（面试必背）**：
```
Step 1: 拿**同一张测试图片**，分别在Python训练代码 + Java DJL代码 跑一次
Step 2: 在preProcess结束后，把NDArray保存成.npy文件 (NDArray.save())
Step 3: 在Python里把同一个图训练时的预处理结果也保存.npy
Step 4: np.testing.assert_allclose(java_npy, python_npy, rtol=1e-3, atol=1e-5)
       - 如果这一步报错 → 预处理Bug！看是哪一步的值差了
       - 如果这一步相等 → 那问题在postProcess或模型转换
Step 5: 再比模型输出NDArray，Python vs Java也要1e-3以内一致
```

3. **代码示例**：自定义一个Debug版Translator，打印预处理各阶段统计量：
```java
public class DebugResNetTranslator implements Translator<Image, Classifications> {
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager manager = ctx.getNDManager();
        NDArray array = input.toNDArray(manager, Image.Flag.COLOR); // HWC uint8 [0-255]
        System.out.println("1.原始图像: shape=" + array.getShape() + 
                          " min=" + array.min() + " max=" + array.max());
        
        array = array.getNDArrayInternal().resize(256, 256, true); // Resize256
        array = NDArrays.centerCrop(array, new Shape(224, 224));    // CenterCrop224
        
        array = array.toType(DataType.FLOAT32, false);
        array = array.div(255.0f); // ToTensor: /255 归一化
        System.out.println("2./255后: mean=" + array.mean() + " std=" + array.std());
        
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std  = {0.229f, 0.224f, 0.225f};
        array = array.sub(manager.create(mean)).div(manager.create(std)); // Normalize
        System.out.println("3.Normalize后: mean=" + array.mean() + " std=" + array.std());
        
        array = array.transpose(2, 0, 1); // HWC → CHW (PyTorch通道序)
        array = array.expandDims(0);      // +Batch维 → [1,3,224,224]
        return new NDList(array);
    }
    
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities = list.get(0).softmax(0); // 模型输出[1000] Softmax
        return new Classifications(ctx.getModel().getArtifact("synset.txt"), probabilities, 5);
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 Top1 坑：**通道顺序搞反**！OpenCV.imread() 默认BGR，ImageIO.read() 默认RGB，两者读同一张图数组差2个通道
- 🔥 Top2 坑：**Normalize顺序错**！应该先/255再减mean除以std，有人直接在[0-255]范围减mean → 结果差255倍
- 追问：「postProcess输出和Python差0.1%正常吗？」→ 答：rtol<1e-3（千分之一）以内正常，因为浮点数累加顺序不一样；超过1%一定有Bug
- 追问：「为什么要转CHW？」→ 答：PyTorch/ONNX的BatchNorm/Conv算子权重是(C, 3, k, k)，期望输入特征图通道在前，OpenCV/TensorFlow默认HWC在后

---

### Q9. ResNet50标准训练预处理四件套: Resize256+CenterCrop224 / ToTensor HWC→CHW [0-255]→[0-1] / Normalize mean=[0.485,0.456,0.406] std=[0.229,...] / RGB顺序; 少一个精度掉15%（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：这是**ImageNet-1K官方训练时用的预处理流水线**，所有基于ImageNet预训练的模型（ResNet/ViT/EfficientNet等）必须原封不动抄下来，差一个步骤精度直接掉10-30%。四件套的作用分别是：

| 步骤 | 操作 | 原理 | 少了的后果 |
|-----|------|------|-----------|
| ① Resize(256) | 把短边缩放到256，长边等比例 | 统一图片长宽比，原图300×400 → 256×341 | 少了直接缩到224，物体比例变形 +10%误差 |
| ② CenterCrop(224,224) | 从正中心裁224×224正方形 | 训练时也是从256图随机Crop，验证/推理用CenterCrop等价"平均"了四个角+中心5个TTA | 用Resize直接硬拉224 → 目标不在正中心 +8%误差 |
| ③ ToTensor | HWC uint8 [0-255] → CHW float32 [0,1] | /255归一化数值范围 + 转PyTorch的CHW通道序 | 少÷255 → 输入数值大255倍，BN层均值直接炸 |
| ④ Normalize | 每个通道减均值除标准差：x'=(x-μ)/σ | 把ImageNet全量图片的像素分布拉到N(0,1)，匹配训练时模型见过的分布 | 少这步 → 输入分布和训练差2.5σ → 准确率76%→60% |

> mean/std的来源：在ImageNet 128万张训练图片上，统计R通道所有像素均值=0.485，标准差0.229，G=0.456/0.224，B=0.406/0.225，**是数据集统计值不是魔法数**！自己的数据集必须重新算自己的mean/std

2. **对比表格**：少每一步对ImageNet Top1的影响（ResNet50 PyTorch官方权重）：
| 情况 | Top1准确率 | 相对下降 |
|-----|-----------|---------|
| ✅ 四件套全对 | 76.13% | 基线 |
| ❌ 少Normalize只用[0-1] | 61.2% | -14.9% 暴跌 |
| ❌ 少÷255（输入0-255范围） | 0.1% 完全错 | -76% 崩盘 |
| ❌ HWC不转CHW直接喂 | 12.4% 乱猜 | -63% |
| ❌ BGR顺序不转RGB | 52.8% | -23% |
| ❌ 直接Resize(224,224)不用Crop | 73.2% | -2.9% |

3. **代码示例**：DJL ImageClassificationTranslator完整正确配置（工业生产用标准写法）：
```java
ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
    // ① Resize短边到256
    .addTransform(new Resize(256, true)) // true=保持纵横比，短边对齐
    // ② CenterCrop 224×224
    .addTransform(new CenterCrop(224, 224))
    // ③ ToTensor: /255 + HWC→CHW (ToTensor Transform内部两步都做了!)
    .addTransform(new ToTensor())
    // ④ Normalize: ImageNet标准均值和方差
    .addTransform(new Normalize(
        new float[]{0.485f, 0.456f, 0.406f},  // RGB mean
        new float[]{0.229f, 0.224f, 0.225f})) // RGB std
    .optFlag(Image.Flag.COLOR)          // 用RGB三通道！
    .optApplySoftmax(true)              // 模型输出的是logits，外面再套Softmax
    .optSynsetArtifactName("synset.txt") // 标签名
    .optTopK(5)                          // 返回Top5结果
    .build();
```

4. **常见坑点/面试追问**：
- 追问：「自己的医疗/工业数据集训练的模型，还能用ImageNet的mean/std吗？」→ 答：**绝对不行**！CT图像是灰度的，数值范围-1024~+3071，分布完全不一样，必须自己重算：
```python
# Python端算自己数据集的mean/std（面试写代码）
mean = 0.0; std = 0.0; n = 0
for img_path in all_train_images:
    img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
    mean += img.mean(axis=(0,1))
    std += img.std(axis=(0,1))
    n += 1
mean /= n; std /= n
print(f"自己数据集 mean={mean}, std={std}")
```
- 追问：「训练用RandomResizedCrop + RandomHorizontalFlip，推理为什么不用？」→ 答：训练是数据增强增加多样性，推理要**确定性结果** + 对齐验证集预处理（验证集用的就是Resize+CenterCrop），TTA时可以五Crop/十Crop再平均提0.5-1个点

---

### Q10. Translator为什么optSynsetArtifactName要放Model.zip里? 模型+标签+配置打包一个zip包, 版本统一不会标签和模型对不上（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL的**模型打包规范**要求所有部署相关物料（权重文件+标签+预处理配置+超参JSON）打包成**一个ZIP文件**，作为不可分割的整体版本管理。核心要解决**"模型v3权重配了v2标签表"**这类配置漂移的版本不一致问题：

```
标准Model.zip结构（面试要会画）：
my-model-v3.2.zip
├── resnet50-v3.2.onnx        ← ① 模型权重/结构（唯一对应版本）
├── synset.txt                ← ② 标签列表（1000类，一行一个，必须和训练时class_to_idx对应）
├── preprocess_config.json    ← ③ 预处理配置（mean/std/img_size，让Translator能读，硬编码Java少）
├── calib_data.cache          ← ④ (可选) INT8量化校准缓存
└── META-INF/
    └── MANIFEST.MF           ← ⑤ 模型元数据（版本号/训练时间/指标Top1=78.3%）
```

> 类比：Docker Image把OS+JDK+App+配置打成一个不可变镜像，版本号v1.2.3对应所有内容；Model.zip就是AI领域的"Docker镜像"

2. **对比表格**：标签放ZIP里 vs 标签写Java代码里 vs 放数据库：
| 方案 | 版本一致性 | 回滚旧版本 | 多模型部署 | 热更新 |
|-----|-----------|-----------|-----------|-------|
| ✅ **标签放Model.zip** | 💯 天然一致，zip是原子单元 | 一键换zip就好 | 每个模型自带标签互不干扰 | ✅ 换zip重启或热加载 |
| ❌ Java代码里写死List | 💥 发布新模型还要改Java代码重新发版 | 改代码回滚，容易错 | 10个模型10套代码 | ❌ 必须重发JAR |
| ❌ 数据库存标签名 | ⚠️ 要人工维护版本对应 | 容易配错v3模型对应v2标签表 | 查库有性能开销 | ✅ 但一致性风险大 |

3. **代码示例**：自定义Translator加载Model.zip内的synset.txt和preprocess_config.json：
```java
public class ConfigurableTranslator implements Translator<Image, Classifications> {
    private float[] mean; private float[] std; private int imgSize;
    private List<String> synset;

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Model model = ctx.getModel();
        // 🔥 读取ZIP里的配置文件（不是读文件系统!）
        try (InputStream is = model.getArtifactAsStream("preprocess_config.json")) {
            JsonObject cfg = JsonParser.parseReader(new InputStreamReader(is)).getAsJsonObject();
            this.imgSize = cfg.get("img_size").getAsInt();
            this.mean = gson.fromJson(cfg.get("mean"), float[].class);
            this.std = gson.fromJson(cfg.get("std"), float[].class);
        }
        // 读取标签文件 synset.txt 在zip里
        this.synset = Utils.readLines(model.getArtifact("synset.txt"), StandardCharsets.UTF_8);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager m = ctx.getNDManager();
        NDArray arr = input.toNDArray(m)
            .getNDArrayInternal().resize(imgSize, imgSize, true)
            .toType(DataType.FLOAT32, false).div(255f)
            .sub(m.create(mean)).div(m.create(std))
            .transpose(2,0,1).expandDims(0);
        return new NDList(arr);
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray prob = list.getSingletonOrThrow().softmax(-1);
        return new Classifications(synset, prob, 5);
    }
}
```

4. **常见坑点/面试追问**：
- 追问：「怎么自定义Artifact？我想放YOLO的anchor.txt进去」→ 答：Model.getArtifact("yolo_anchors.txt")直接读InputStream，任意文件都可以塞ZIP里
- 坑：synset.txt标签顺序错！比如训练时0=cat, 1=dog，synset里写反了→所有分类结果标签名对不上，准确率数值对但人类看错
- 坑：大模型zip>100MB，Maven/Git仓库放不下，要用独立的对象存储S3/MinIO，DJL的optModelUrls("s3://bucket/models/v3/")自动下

---

### Q11. 自定义YOLOv5目标检测Translator难点: postProcess里NMS非极大值抑制IoU阈值0.5+置信度阈值0.25手写字节流Java版 NMS box筛选重复框（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：YOLOv5输出张量形状是 `[1, 25200, 85]` → `[Batch, Anchor数, 4(xywh) + 1(obj_conf) + 80(class_scores)]`，postProcess三步曲：
- ① **阈值过滤**：obj_conf × class_score > 0.25 才是候选框（筛掉95%低质量框）
- ② **坐标还原**：模型输出的是Sigmoid归一化相对于网格的偏移，要还原回原图640×640坐标，xywh→xyxy（左上右下）
- ③ **NMS非极大值抑制**：同一类的重复候选框，IoU(交并比) > 0.5的只留置信度最高的那个，去掉重叠的

| 后处理步骤 | 难点 | 错了的表现 |
|-----------|------|-----------|
| 85维解析 | 前4是xywh，第5是objectness，后80是类别score要×第5 | 所有框都在左上角0,0点，没检测 |
| 坐标还原 | YOLOv5用的是cxcywh格式+Sigmoid偏置，公式: bx = 2σ(tx) - 0.5 + grid_x，易写错 | 检测框偏移半个网格，检测不准 |
| NMS | 按类别分组，每组独立NMS；手写循环慢，25200框Java for循环500ms | 同一物体重复10个框，漏检/误检 |

2. **对比表格**：手写NMS vs DJL内置NMS vs ONNX图内NMS：
| 方案 | 速度(25200框) | 精度 | 可维护性 | 推荐 |
|-----|--------------|------|---------|------|
| ONNX图里加NMS节点(推荐) | <1ms (GPU) | 同Python官方 | ✅ 不用写Java代码 | ⭐⭐⭐⭐⭐ |
| DJL NDArray向量化NMS | ~10ms (CPU) | 同Python | ✅ 无for循环 | ⭐⭐⭐⭐ |
| Java手写三重for循环 | ~500ms | 同Python | ❌ 难写易出Bug | ⚠️ 面试才手写 |

3. **代码示例**：Java版向量化高效NMS（面试白板写，要求看懂）：
```java
// IoU计算：两个矩形框交并比 (x1,y1,x2,y2)格式
public static float bboxIou(float[] a, float[] b) {
    float xx1 = Math.max(a[0], b[0]), yy1 = Math.max(a[1], b[1]);
    float xx2 = Math.min(a[2], b[2]), yy2 = Math.min(a[3], b[3]);
    float inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
    float union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return inter / union; // 交并比：交集/并集
}

// NMS 主函数（置信度排序+贪心选框）
public static List<int[]> nms(List<float[]> boxes, float[] scores, float iouThresh) {
    // 1. 按置信度从大到小排序，取索引
    Integer[] order = IntStream.range(0, boxes.size()).boxed().toArray(Integer[]::new);
    Arrays.sort(order, Comparator.comparingDouble(i -> -scores[i]));
    
    boolean[] suppressed = new boolean[boxes.size()];
    List<int[]> keep = new ArrayList<>();
    
    for (int i = 0; i < order.length; i++) {
        int idx = order[i];
        if (suppressed[idx]) continue; // 已经被抑制的跳过
        keep.add(new int[]{idx}); // 高分框留下
        // 和后面所有框比IoU，重叠大的删掉
        for (int j = i+1; j < order.length; j++) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            if (bboxIou(boxes.get(idx), boxes.get(jdx)) > iouThresh) {
                suppressed[jdx] = true; // IoU大的被抑制掉
            }
        }
    }
    return keep;
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：**NMS没按类别分组**！猫框和狗框IoU大也被抑制掉了，猫和狗叠在一起只检测到一个 → 必须先按class_id分组，每组独立做NMS
- 追问：「NMS阈值怎么选？」→ 答：检测严格场景(医疗)IoU=0.3(少重复)；宽松场景(自动驾驶怕漏检)=0.6-0.7，通用coco标准0.5
- 追问：「Soft-NMS是什么？」→ 答：不是直接suppress=false删掉，而是把重叠框的score×(1-IoU)衰减，密集人群场景少漏检，COCO mAP高1-2个点

---

### Q12. NDArray的broadcast广播 vs numpy规则是否一致? 从右向左对齐/维等或1 原理一致; shape不匹配抛IllegalArgumentException（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：**DJL NDArray广播规则和NumPy 100%完全一致**，面试直接按NumPy广播答就行。两大规则：
```
规则1: 两个数组shape从右端开始向左对齐，维度不相等时，小的一边补前导1
例:  A [3, 1, 4]  → 左端补1到4维? 不，左端补! 所以A是[3,1,4]右端是4
     B    [5, 1]  → 左端补1前导 → [1,5,1]
对齐后:
  A = [3, 1, 4]
  B = [1, 5, 1]
规则2: 每个维度上要么相等，要么其中一个是1 → 可广播
     3 vs 1? ✅ A是3，B是1 → 广播后3
     1 vs 5? ✅ A是1，B是5 → 广播后5
     4 vs 1? ✅ A是4，B是1 → 广播后4
✅ 最终广播shape = [3, 5, 4]!
```

2. **对比表格**：常见合法/非法广播组合（面试高频判断）：
| A shape | B shape | 广播后shape | 合法? | 说明 |
|---------|---------|------------|-------|------|
| `[5,3]` | `[3]` | `[5,3]` | ✅ | B左端补1→[1,3]，5vs1和3vs3都OK |
| `[1,3,256,256]` | `[3,1,1]` | `[1,3,256,256]` | ✅ | 逐维比：1vs1 3vs3 256vs1 256vs1 ✅ |
| `[B,1000]` | `[1000]` | `[B,1000]` | ✅ | 分类logits减标量偏置的标准模式 |
| `[8,3,64,64]` | `[64,1]` | ❌ | ❌ | 右端64vs64 OK，3vs1 OK，往左8vs(没)补1 OK，但倒数第2维 64 vs 64 OK？哦这个合法→ [8,3,64,64] |
| `[4,3]` | `[4]` | ❌ | ❌ | 右端3vs4 不相等且都不是1 → IllegalArgument！ |

3. **代码示例**：DJL广播和NumPy结果对比，数值完全相等：
```python
# NumPy端验证广播语义
import numpy as np
A = np.random.randn(2, 1, 4).astype(np.float32)  # [2,1,4]
B = np.random.randn(3, 1).astype(np.float32)     # [3,1]
C_np = A + B
print(C_np.shape)  # (2, 3, 4) ✓
```
```java
// DJL端广播结果完全一致
try (NDManager m = NDManager.newBaseManager()) {
    NDArray A = m.randomNormal(new Shape(2, 1, 4));
    NDArray B = m.randomNormal(new Shape(3, 1));
    NDArray C = A.add(B); // 广播自动触发!
    System.out.println(C.getShape()); // (2, 3, 4) ✓ 和NumPy一样
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：有人记错方向，从左向右对齐shape → 全错，**一定是从右向左**！想象写数字个位对齐，不会从高位对齐
- 追问：「为什么要有广播？」→ 答：① 代码简洁，不用手动expandDims+tile复制 ② 性能好，广播不实际复制内存，只是在算子遍历时逻辑多循环一次，节省内存100倍(不用实际存[2,3,4]的B副本)
- 追问：「NDArray和PyTorch Tensor广播不一致的情况？」→ 答：0.20+版本DJL完全对齐PyTorch/NumPy，极少差异；老版本对0维scalar处理略不同

---

### Q13. ModelZoo内置模型和自定义加载区别: ModelZoo.loadModel(criteria)自动下载DJL官方模型/自己的s3:// / hdfs:// / file://本地路径optModelUrls()（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL两种模型加载方式，适用于不同场景：

| 方式 | API | 模型来源 | 适用场景 | 版本控制 |
|-----|-----|---------|---------|---------|
| **ModelZoo内置模型** | `ModelZoo.loadModel(criteria)` | DJL官方S3仓库，已预训练好的CV/NLP模型100+ | Demo/快速验证/教学 | DJL发版时冻结版本，一般比官方SOTA落后3-6个月 |
| **自定义URL加载** | `criteria.optModelUrls(urls).build().loadModel()` | **任意URL**：<br>• `file:///本地绝对路径`<br>• `s3://` AWS S3 / MinIO对象存储<br>• `hdfs://` Hadoop HDFS<br>• `http(s)://` HTTP静态服务器 | ✅ 生产环境！自己训练的模型 | 用户自己控制，v1/v2/v3全自己管 |

> 底层实现：ModelZoo.loadModel内部也是拼了DJL官方S3的optModelUrls路径，本质是同一个加载器，ModelZoo是封装好的"快捷方式"

2. **对比表格**：生产环境自定义URL三大存储方案选型：
| 存储方案 | optModelUrls示例 | 优点 | 缺点 | 推荐场景 |
|---------|-----------------|------|------|---------|
| **本地文件系统** | `file:///opt/ai/models/resnet-v3/` | 速度最快，IO 1GB/s | 多Pod部署每台机器要放一份，模型更新难 | 单实例服务/Docker镜像COPY进镜像内 |
| **对象存储S3/MinIO** | `s3://ai-models-prod/yolov5/v6.1-20240520/` | ⭐ 生产首选，版本化，权限IAM控制，多Pod共享 | 首次下载要等，几百MB模型3-10秒 | K8s多副本部署，模型多迭代频繁 |
| **HDFS** | `hdfs://namenode:8020/models/vllm-70b/` | 大数据公司已有HDFS集群，不用额外出钱 | 小文件IO慢，Java客户端配置复杂 | Hadoop生态公司 |

3. **代码示例**：生产环境MinIO(S3兼容)加载自定义模型标准写法：
```java
// 配置MinIO/S3的ak/sk/endpoint（不要硬编码写代码，走配置中心!）
System.setProperty("ai.djl.s3.endpoint", "https://minio-internal.corp.com:9000");
System.setProperty("ai.djl.s3.accessKey", s3AccessKey);
System.setProperty("ai.djl.s3.secretKey", s3SecretKey);
System.setProperty("ai.djl.repository.zoo.location", // 缓存目录: K8s用hostPath共享
    "/data/.djl.ai/model-cache/"); // 多个Pod共享同节点缓存，只下载1次!

Criteria<Image, DetectedObjects> criteria = Criteria.builder()
    .setTypes(Image.class, DetectedObjects.class)
    .optEngine("OnnxRuntime")
    // 🔥 生产自定义模型：S3路径，带版本号v6.1-20240520
    .optModelUrls("s3://ai-models-prod/computer-vision/yolov5/v6.1-20240520/")
    .optModelName("yolov5s") // 加载目录下yolov5s.onnx
    .optTranslator(new YoloV5Translator(640, 0.25f, 0.45f))
    .optDevice(Device.gpu(0))
    .optProgress(new ProgressBar()) // 首次下载显示进度条
    .build();
// 第一次调用：S3下载3个文件到本地缓存 → 第二次调用直接读缓存
Model model = criteria.loadModel();
```

4. **常见坑点/面试追问**：
- 🔥 坑：S3模型更新了同一个路径下的文件（比如覆盖了onnx），DJL缓存不失效! → **必须模型版本号放URL路径里**，`v6.1/` → `v6.2/` 强制触发重新下载
- 追问：「加载进度条怎么实现？」→ 答：optProgress传ProgressBar接口实现，里面有`onProgress(long downloaded, long total)`回调，K8s部署看日志知道卡在哪
- 追问：「怎么离线加载？完全内网无外网环境」→ 答：① 提前在有网机器下载好模型ZIP和native引擎JAR ② COPY到镜像内 ③ optModelUrls写file://绝对路径 ④ 设置`ai.djl.offline=true` 禁止任何网络请求

---

## 三、性能与并发（6题）

### Q14. Predictor.predict vs Predictor.batchPredict: Batch=8同一次GPU Kernel Launch小图 3.75ms/张 vs 单张10ms/张 2.6倍吞吐量提升（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：batchPredict把N张输入拼成一个大Batch，做**一次前向传播**输出N个结果，节省了大量GPU Kernel Launch开销和CUDA上下文切换。GPU是SIMT架构（单指令多线程），Batch越大越能占满SM流多处理器。

```
单张predict走8次流程 vs batch=8一次流程：
8次predict:  [Launch]→[计算]→[拷贝] 重复8次 = 8*(1ms launch + 9ms compute + 1ms copy) = 88ms 总计
1次batch=8:  [Launch]→[计算]→[拷贝] 1次 = 1ms + 29ms compute(多3.2x不是8x有开销) + 1ms = 30ms → 3.75ms/张
```

2. **对比表格**：BatchSize对ResNet50性能影响(A10 GPU ONNX FP16)：
| Batch Size | 总耗时ms | 每张耗时ms | 吞吐量QPS | GPU SM利用率 |
|-----------|---------|-----------|----------|------------|
| 1 (单predict) | 9.8ms | 9.8ms | 102 | 32% 浪费! |
| 2 | 12.1ms | 6.05ms | 165 | 51% |
| 4 | 17.2ms | 4.3ms | 232 | 73% |
| **8 (最优)** | **30.1ms** | **3.76ms** | **266** | **94% 满载** |
| 16 | 56.8ms | 3.55ms | 281 | 98% |
| 32 | 110ms | 3.44ms | 290 | 99% 但P99延迟×11，实时性不行 |

> 结论：**实时服务选Batch=8，吞吐优先选16-32**，超过32延迟线性涨收益递减

3. **代码示例**：Spring WebFlux响应式批量端点 + batchPredict：
```java
@PostMapping(value = "/batch-classify", produces = MediaType.APPLICATION_JSON_VALUE)
public Mono<List<Classifications>> batchClassify(@RequestBody List<MultipartFile> files) {
    // 1. 并行加载所有图片
    List<Image> images = files.parallelStream()
        .map(f -> {
            try { return ImageFactory.getInstance().fromInputStream(f.getInputStream()); }
            catch (IOException e) { throw new RuntimeException(e); }
        }).toList();
    
    // 2. 一次batchPredict，GPU一次Kernel Launch
    return Mono.fromCallable(() -> {
        try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.batchPredict(images); // 🔥 关键API
        }
    }).subscribeOn(Schedulers.boundedElastic());
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：输入图片大小不一直接扔batchPredict，内部Pad到最大尺寸+Mask，轻则速度变慢重则报错 → 提前统一Resize到相同尺寸
- 追问：「batchPredict会OOM吗？」→ 答：会的，Batch=64 640×640 YOLOv5 FP16会占~12GB显存，小Batch逐步试找最优值，用CUDA OOM保护catch OutOfMemoryError降级小Batch
- 追问：「CPU推理也有收益吗？」→ 答：有但小一些，Batch=4约1.5倍提升，CPU主要瓶颈在线程数（见Q16），不在Kernel Launch

---

### Q15. 动态攒批Continuous Batching: 队列+超时5ms或满8张触发batchPredict; 什么时候加? GPU利用率<50%浪费的时候加（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：**动态攒批（也叫Micro-Batching/Request Batching）** 是在服务端加一个请求队列，不立刻做推理，而是**等5ms时间窗口** 或者 **攒够8张**，然后一起batchPredict，用低5ms延迟换2-3倍吞吐量提升。这个是vLLM/LLM推理的标配，CV场景也同样适用。

```
动态攒批流程：
  请求1→入队(队列: [1])  启动计时器
  请求2→入队(队列: [1,2])
  请求3→入队(队列: [1,2,3])
  ... 5ms时间到 or 累计满8个
  → 触发: batchPredict([1,2,3,4,5]) → 5个结果分别回给5个请求
```

2. **对比表格**：动态攒批 vs 静态固定Batch vs 单张predict：
| 方案 | 平均延迟 | 吞吐量 | 适合QPS | 代码复杂度 |
|-----|---------|-------|---------|----------|
| 单张predict | 10ms | 100 QPS | <30 QPS低峰 | 最简单 |
| 静态固定Batch | 35ms±10ms抖动大 | 250 | 恒定高QPS(>200) | 中 |
| **动态攒批(推荐)** | **13ms(只加5ms窗口)** | **230** | **任意QPS，波峰波谷自适应** | 中(加Disruptor队列) |

> **什么时候加？** 先压测看GPU利用率：
> - GPU利用率 <40%：**一定要加！** 加了白赚2-3倍吞吐量，延迟只涨30%
> - 40%-70%：可选加，成本敏感值得加
> - >80%接近满载：不用加，加了延迟反而涨吞吐不变

3. **代码示例**：LMAX Disruptor高性能环形队列实现攒批（生产级）：
```java
@Component
public class BatchingInferenceService {
    private final Model model;
    private final RingBuffer<InferenceEvent> ringBuffer;
    private final int maxBatch = 8;
    private final long timeoutNs = 5_000_000L; // 5ms超时

    @PostConstruct
    public void init() {
        // 1. 启动Disruptor消费者线程，专门负责攒批
        Thread.startVirtualThread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                List<InferenceEvent> batch = new ArrayList<>(maxBatch);
                long deadline = System.nanoTime() + timeoutNs;
                
                // 2. 最多等5ms or 攒满8个，哪个先到触发
                while (batch.size() < maxBatch && System.nanoTime() < deadline) {
                    InferenceEvent evt = ringBuffer.next(0, 1, TimeUnit.MICROSECONDS);
                    if (evt != null) batch.add(evt);
                }
                
                if (batch.isEmpty()) continue;
                
                // 3. 一次batchPredict处理所有！
                try (Predictor<Image, Classifications> p = model.newPredictor()) {
                    List<Image> imgs = batch.stream().map(e -> e.image).toList();
                    List<Classifications> results = p.batchPredict(imgs);
                    // 4. 每个结果返回给对应CompletableFuture
                    for (int i = 0; i < batch.size(); i++) {
                        batch.get(i).result.complete(results.get(i));
                    }
                }
            }
        });
    }

    // 对外API：返回CompletableFuture异步非阻塞
    public CompletableFuture<Classifications> classifyAsync(Image img) {
        CompletableFuture<Classifications> cf = new CompletableFuture<>();
        ringBuffer.publishEvent((evt, seq) -> {
            evt.image = img; evt.result = cf;
        });
        return cf;
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：攒批的Consumer线程挂了（抛异常），所有CompletableFuture永远不complete → 请求全超时！必须加try-catch + UncaughtExceptionHandler + 熔断重启线程
- 追问：「超时时间怎么选？」→ 答：目标延迟P99的1/3~1/2。目标P99=30ms → 超时10ms；目标P99=100ms → 超时30ms
- 追问：「LLM动态批和CV的区别？」→ 答：LLM是Continuous Batching(迭代级插空)，每生成1个Token插入新请求；CV是Micro Batching(请求级攒批)，整个推理完才下一批，LLM粒度细多了

---

### Q16. ONNX Runtime intra_op_num_threads内部算子并行线程数: 16核CPU设置成16而不是默认1! 90%人漏调性能×8差距（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：ONNX Runtime有两个线程池参数，99%的Java部署只改第一个就够了：

| 参数 | 含义 | 默认值 | 正确设置 | 影响 |
|-----|------|-------|---------|------|
| **intra_op_num_threads** | 单个算子内部并行线程数（如Conv/MatMul内部for循环拆多少线程跑） | **=1** 串行!!! | =物理CPU核数（k8s requests.cpu数） | **性能差8-16倍!** |
| inter_op_num_threads | 算子间并行（图里两个独立算子同时跑） | CPU核数 | 2-4够了 | 影响小，10%左右 |

> 默认值坑：ONNX Runtime JAVA绑定默认intra_op_num_threads=1，这是给移动端(Android单CPU核)设计的默认值！x86服务器16核机器用默认1=浪费15/16的算力

2. **对比表格**：ResNet50 CPU推理不同线程数性能（AWS c5.4xlarge 16核，FP32）：
| intra_op_num_threads | 单张耗时 | 吞吐量QPS | 加速比 |
|---------------------|---------|----------|-------|
| 1 (默认坑值!) | 482ms | 2.1 | 1.0x 基线 |
| 2 | 253ms | 4.0 | 1.9x |
| 4 | 131ms | 7.6 | 3.7x |
| 8 | 72ms | 13.9 | 6.7x |
| **16 (正确设置)** | **59ms** | **17.0** | **8.2x 🚀** |
| 32 (超线程HT) | 56ms | 17.8 | 8.6x 边际收益递减 |

3. **代码示例**：ONNX Runtime CPU/GPU三大关键性能参数正确配置（Spring Boot）：
```java
@Bean
public Model onnxResnetModel() throws Exception {
    int cpuCores = Runtime.getRuntime().availableProcessors(); // 16
    
    Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class)
        .optEngine("OnnxRuntime")
        .optModelUrls("file:///models/resnet50")
        .optDevice(Device.cpu()) // CPU场景，GPU场景CUDA会自动接管
        // 🔥🔥🔥 最重要的性能调优，一行不能少！
        .optOption("intra_op_num_threads", String.valueOf(cpuCores)) // 算子内并行
        .optOption("inter_op_num_threads", "4")                     // 算子间并行
        .optOption("optimizationLevel", "ORT_ENABLE_ALL")           // 开全部图优化：常量折叠/算子融合/死代码消除
        .optOption("memoryPatternOptimization", "true")             // 内存复用减少malloc
        .optOption("arenaExtendStrategy", "kNextPowerOfTwo")        // 内存池策略，减少碎片
        .optTranslator(resnetTranslator())
        .build();
    return criteria.loadModel();
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：K8s里设了CPU limits=4，但DJL读的是宿主机/proc/cpuinfo的96核！设intra=96疯狂线程切换上下文，反而变慢 → 必须从Downward API拿limits值，不是Runtime.getRuntime()
- 追问：「GPU推理要不要设intra？」→ 答：不用设，设了也不生效，算子都在GPU跑，CPU只做调度1-4线程够了
- 追问：「和MKL-DNN OMP_NUM_THREADS区别？」→ 答：本质一样，ONNX的intra_op_num_threads会覆盖OpenMP的OMP_NUM_THREADS环境变量，优先用optOption设置更可移植

---

### Q17. CPU/GPU选型: 10QPS以内小流量(ResNet) CPU 4核就够省成本; 100+QPS大吞吐必须A10/A100 GPU（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：选型三要素：**延迟要求 × QPS吞吐 × 预算成本**。GPU算力是CPU的10-100倍，但有固定成本（卡本身+PCIe开销），低QPS下GPU的利用率太低，成本是CPU几倍。

```
成本曲线(ResNet50推理 单模型)：
QPS  | CPU 4核成本 | A10 GPU成本 | GPU划算?
  5  | ¥200/月(共享)  | ¥3000/月    | ❌ 浪费15倍钱
 20  | ¥800/月(4台)   | ¥3000/月    | ❌ 还是CPU省
 80  | ¥3200/月(16台) | ¥3000/月    | ⚖️ 差不多
200  | ¥8000/月(40台) | ¥3000/月    | ✅ GPU开始省
1000 | ¥40000/月      | ¥6000(A10*2)| ✅ GPU巨省 6.6倍
```

2. **对比表格**：CPU vs GPU vs 边缘NPU选型决策矩阵：
| 场景 | 建议硬件 | 典型机型 | 单ResNet QPS | 成本/千QPS/月 | 备注 |
|-----|---------|---------|-------------|-------------|------|
| **<10 QPS 小服务** | CPU 4核 | 通用云主机c5.large | 8-15 | ¥200 | ⭐ 90%内部管理系统选这个 |
| 10-100 QPS 中流量 | CPU 16核 or T4 | c5.4xlarge / T4 | 17 / 600 | ¥350-400 | 看延迟要求 |
| **100+ QPS 高吞吐** | **A10 / L20 GPU** | g5.xlarge A10 24GB | 3500 (FP16) | ¥80 | ⭐ 线上大流量标配 |
| >10K超大流量 | A100 80GB ×2 | p4d.24xlarge | 25000 | ¥60 | BERT/ViT大模型 |
| 边缘端/端侧Jetson | Xavier / Orin NX | Jetson Orin NX 16GB | 200 | 硬件一次性 | 工业摄像头本地推理 |

3. **代码示例**：CPU场景做低成本优化（CPU推理的性能补全）：
```java
// CPU场景三大补刀优化，性能再提2-3倍，接近低端GPU
// 1. INT8量化模型：QAT训练后导出INT8 ONNX，比FP32快3-4倍，精度掉<1%
Criteria.builder()
    .optModelName("resnet50_int8_qat") // INT8版模型
    .optOption("executionMode", "ORT_SEQUENTIAL")  // CPU上SEQUENTIAL比PARALLEL快
    .optOption("intra_op_num_threads", "16")
    .optOption("ep.executionProvider", "OpenVINO_CPU") // 🔥 开OpenVINO EP，Intel CPU再提1.5-2倍！
    // 2. 开启Graph优化缓存，第一次编译图存磁盘，以后启动毫秒
    .optOption("saveModelFormat", "ORT")
    .optOption("sessionConfigEntry", "optimization_pre_model_file_path:/cache/resnet_opt.onnx")
    // 3. JVM参数：-XX:+UseVectorApiIntrinsics --add-modules jdk.incubator.vector 启用AVX512向量
    .build();
```

4. **常见坑点/面试追问**：
- 🔥 坑：买了A100 80GB跑ResNet50小模型，QPS只比A10高20% → 浪费！A100算力强但单模型kernel占不满，要用MIG（多实例GPU）切7份，或并发跑8个不同模型才能吃满
- 追问：「P99延迟要求<20ms选什么？」→ 答：直接上GPU，CPU即使16核P99也在50ms+，抖动大，GC一跑就破100ms
- 追问：「BatchNorm在CPU推理特别慢怎么优化？」→ 答：训练后做BN折叠（BatchNorm folding到Conv权重），ONNX的optimizationLevel=ALL会自动做这件事，速度提升~20%

---

### Q18. 堆外内存泄漏两大典型原因: 1. NDArray跨Manager作用域没关 try-with-resources漏写2. 静态字段static缓存大NDArray一直存活（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL堆外内存泄漏=**NDArray被创建了，但是对应的NDManager.close()没被调用，native free()没执行**，JVM堆里只有一个小对象指针，GC管不到，但Native端占了几GB内存，最后OOM kill进程。

**两大典型原因占95%泄漏：**
| # | 泄漏原因 | 占比 | 泄漏速度 | 怎么查 |
|---|---------|------|---------|-------|
| 1 | **try-with-resources漏写** | 60% | 每次请求泄漏几十MB，几小时OOM | grep代码找`newSubManager()`没有紧跟try |
| 2 | **static静态集合缓存NDArray** | 35% | 慢慢涨，几天OOM | `HeapDump`查`ai.djl.ndarray.NDArray`谁引用它 |
| 3 | 其他(NDManager跨线程传/attach attach错父) | 5% | 不确定 | NDManager.debugDump()打印树形结构 |

2. **对比表格**：JVM堆OOM vs DJL堆外OOM的现象区别：
| 现象 | JVM堆内存OOM | 堆外Native内存泄漏 |
|-----|-------------|------------------|
| `jmap -histo` | 大对象占满堆 | 堆使用正常，看不出问题 |
| `top/htop` RES | 和Xmx差不多 | RES远大于Xmx，差值就是泄漏的堆外 |
| `OutOfMemoryError`消息 | `Java heap space` | 无Java Error，Linux OOM Killer直接杀进程dmesg日志有 |
| GC日志 | Full GC连续，Old Gen 99% | GC健康，Young Gen正常回收 |
| **触发点** | 请求量大时立刻 | 运行几小时/几天后 |

3. **代码示例**：泄漏的写法 vs 安全的写法 + 监控告警：
```java
// ❌ 泄漏版：60%开发踩过的坑
@Service
public class LeakyService {
    // 2. 致命坑：static缓存NDArray！GC不会回收，永久存活
    private static final Map<String, NDArray> BAD_CACHE = new ConcurrentHashMap<>(); 
    
    public float badCalc(String key, NDArray input) {
        // 1. 致命坑：newSubManager()后没try-with-resources，close永远不执行
        NDManager tmp = input.getManager().newSubManager();
        NDArray hugeTensor = tmp.create(new Shape(1, 64, 512, 512)); // 64MB！
        float result = input.dot(hugeTensor).sum().getFloat();
        
        BAD_CACHE.put(key, hugeTensor); // 2.缓存NDArray，永远不释放
        
        return result; // tmp.close()没调用，64MB泄漏！下次又64MB
    }
}

// ✅ 安全版：try-with-resources + NDArray不能跨线程传 + 缓存存byte[]不存NDArray
@Service
public class SafeService {
    // 缓存要序列化后存byte[]或float[]，不要存NDArray引用！
    private final LoadingCache<String, float[]> safeCache = CacheBuilder.newBuilder()
        .maximumSize(1000).expireAfterWrite(1, TimeUnit.HOURS).build(...);
    
    public float goodCalc(String key, NDArray input) {
        // try-with-resources：方法退出自动close tmp和hugeTensor！
        try (NDManager tmp = input.getManager().newSubManager();
             NDArray hugeTensor = tmp.create(new Shape(1, 64, 512, 512))) {
            
            NDArray dot = input.dot(hugeTensor);
            float result = dot.sum().getFloat();
            // 缓存只存Java数组，不存NDArray
            safeCache.put(key, dot.toFloatArray());
            
            return result;
        } // 🔥 退出块自动：hugeTensor.close() → tmp.close() → 64MB立即释放
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：生产环境`-XX:MaxDirectMemorySize`没设，默认等于Xmx，堆外泄漏到物理内存OOM Killer杀进程，Java没日志 → 必设MaxDirectMemorySize，超了抛Java OOM有堆栈
- 追问：「怎么监控告警？」→ 答：① Micrometer埋点`MeterRegistry.gauge("djl_direct_memory_bytes", allocatedDirectBytes)` ② 阈值: 当前>峰值70%告警 ③ 每小时增长>500MB告警
- 追问：「NDArray的detach()和close()区别？」→ close()立刻释放内存，detach()是从当前Manager解绑挂到父Manager，推迟释放时机，不是释放！

---

### Q19. DJL Serving gRPC独立服务 vs Spring内嵌FatJar: 多模型/多团队共用/微服务解耦选Serving; 中小项目简单快速选内嵌Jar（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL两种部署拓扑结构，类比数据库的Embedded SQLite vs 独立MySQL服务：

```
【方案A: Spring内嵌FatJar (90%中小项目选)】
  业务Spring Boot App.jar
  ├── 业务代码: OrderService / UserController
  └── DJL Model (直接在JVM内加载权重, 同进程, 零网络)
       └── ONNX Runtime native .so
  
  优点：部署简单(java -jar一键), 无RPC延迟(<1ms), 事务内调用
  缺点：模型和业务耦合, 多模型JVM堆+堆外爆20GB+, 业务升级要重启模型
```

```
【方案B: DJL Serving gRPC独立服务 (平台级选)】
  Gateway/Istio
    ├── Service A (Spring, 业务代码) ──gRPC──┐
    ├── Service B (Node.js Python) ──HTTP───┤  跨语言通用
    └── Service C (Go)         ──gRPC──────┘
                                          ▼
                              DJL Serving (独立Deployment 10 Pod)
                              ├── 模型1: ResNet (2副本)
                              ├── 模型2: YOLOv5 (3副本 GPU)
                              └── 模型3: BERT-base (5副本)
  优点：模型独立扩缩容, 多团队共用, 版本灰度/回滚, 业务代码不用带模型JAR
  缺点：多一层gRPC 1-3ms延迟, 部署多套K8s YAML
```

2. **对比表格**：
| 维度 | Spring内嵌FatJar | DJL Serving 独立服务 |
|-----|----------------|-------------------|
| **项目规模** | 1-3个模型, <1000DAU 小项目 | 10+模型, 多团队共用 平台级 |
| **启动/部署** | ✅ java -jar 1步 | ❌ 要装Serving + 模型注册 + gRPC客户端 |
| **调用延迟** | ✅ <1ms 同内存 | ⚠️ 2-5ms 加gRPC网络 |
| **扩缩容** | ❌ 业务和模型绑定一起扩 | ✅ ResNet CPU扩8 Pod，YOLO GPU扩3 Pod独立 |
| **内存效率** | ❌ 3模型各占3GB JVM，共9GB | ✅ 3模型同一Serving共享，共4GB |
| **多语言支持** | ❌ 只有Java用 | ✅ gRPC: Java/Go/Python/Node.js/C++ |
| **模型热更新** | ❌ 重启JAR (停机10s+) | ✅ 模型新版本注册，5%流量灰度，不停机 |
| **运维复杂度** | ✅ 几乎零 | ⚠️ 加了一个服务 = 监控/告警/日志翻倍 |

3. **代码示例**：两种部署的客户端调用示例对比：
```java
// 【方案A: Spring 内嵌】最简单，直接@Autowired Model
@RestController
public class InlineController {
    private final Model resnet;
    @PostMapping("/classify")
    public Classifications classify(@RequestParam MultipartFile img) throws Exception {
        try (Predictor<Image, Classifications> p = resnet.newPredictor()) {
            return p.predict(ImageFactory.getInstance().fromInputStream(img.getInputStream()));
        }
    }
}

// 【方案B: DJL Serving gRPC客户端】1. 依赖djlserving-client 2. protobuf
@Service
public class ServingClient {
    private final PredictServiceGrpc.PredictServiceBlockingStub stub;
    
    public Classifications classifyServing(byte[] imgBytes) {
        PredictRequest req = PredictRequest.newBuilder()
            .putInputs("data", TensorShape.newBuilder()
                .putStringMap("Content-Type", "image/jpeg")
                .setBody(ByteString.copyFrom(imgBytes)).build())
            .setModelName("resnet-v3").setModelVersion("3.2.1").build();
        
        PredictResponse resp = stub.withDeadlineAfter(50, TimeUnit.MILLISECONDS).predict(req);
        // 解析Classifications Protobuf
        return parseClassificationsFromResponse(resp);
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：DJL Serving开启多模型时，所有模型用一个intra_op_num_threads全局配置，CPU争用互相影响 → 每个模型单独Serving Deployment隔离
- 追问：「什么场景必须选Serving？」→ 答：① 要支持Node.js/Python前端直接调模型（没有Java服务）② 多团队不同服务共用大模型(不想复制10份权重) ③ 高频灰度A/B测试模型版本
- 追问：「内嵌Jar的OOM怎么规避？」→ 答：Spring Boot 3+ 虚拟线程+单独写`-XX:MaxDirectMemorySize=12G`，堆内存Xmx=4G够了，大部分内存是堆外给模型用的

---

## 四、模型格式转换与避坑（6题）

### Q20. PyTorch原始pth文件DJL能直接读吗? ❌ 必须Python端torch.jit.trace(model, dummy_input).save("resnet18.pt")保存TorchScript格式! 原始state_dict权重DJL识别不了（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：PyTorch三种保存格式，只有TorchScript能被DJL PyTorch Engine加载：
| 格式 | 保存方式 | 内容 | DJL可读取 |
|-----|---------|------|----------|
| state_dict.pth ❌ | `torch.save(model.state_dict(), "a.pth")` | 只存权重Tensor字典，**没有模型结构代码** | ❌ 100%不行，Java不知道怎么拼ResNet/Bottleneck结构 |
| **TorchScript Trace** ✅ | `torch.jit.trace(model, dummy).save("b.pt")` | 记录算子+权重的**静态计算图**，不依赖Python代码 | ✅ DJL标准加载方式 |
| TorchScript Script ✅ | `torch.jit.script(model).save("c.pt")` | 带控制流if/for的动态图脚本化（比Trace支持更多） | ✅ 优先用Script比Trace更全 |

> 为什么state_dict不行？`state_dict = {"conv1.weight": Tensor, "bn1.running_mean": Tensor...}` 只有权重数值，没有"第一层是Conv2d 3→64 7×7 stride2"这种结构信息，DJL是纯Java环境没法执行model.py里的Python类定义。

2. **对比表格**：Trace vs Script两种TorchScript转换方式：
| 方案 | 原理 | 支持if/for控制流 | 常见失败点 | 转换成功率 |
|-----|------|-----------------|-----------|----------|
| **torch.jit.trace** | 喂1个假输入，把实际跑过的算子记录下来成静态图 | ❌ if分支只记录走到的那条，else丢了 | 输入动态shape N可变的模型Trace出来固定死 | 80% 简单模型(ResNet/ViT)一次过 |
| **torch.jit.script** | 静态分析Python AST语法树，把整个Module编译成TorchScript IR | ✅ 支持for/if/while/dict/list | 不支持部分Python类型(kwargs**/lambda) | 95% 改改代码都能过 |

3. **代码示例**：PyTorch转TorchScript生产级步骤（Python端）：
```python
import torch
import torchvision.models as models

# 1. 加载PyTorch模型 + state_dict权重
model = models.resnet18(weights=None)
checkpoint = torch.load("resnet18_state_dict_only.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()  # ！！！必须eval！Dropout/BatchNorm要推理模式，不然转出来BN running_mean错

# 2. 构造dummy假输入，shape和实际输入必须完全一致！
# batch=1, 3通道RGB, 224×224 → dtype FP32
dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# 3. Script优先，不行再Trace
try:
    scripted = torch.jit.script(model)  # 优先：保留控制流
    print("✅ Script成功")
except Exception as e:
    print(f"⚠️ Script失败{e}, 用Trace")
    scripted = torch.jit.trace(model, dummy, strict=True)  # strict=True查错！
    # 动态shape模型要加check_trace: check_inputs=[(dummy1,),(dummy2,)]

# 4. 验证转换前后输出一致！(最重要的一步，80%人跳过踩坑)
with torch.no_grad():
    out_pytorch = model(dummy)
    out_scripted = scripted(dummy)
torch.testing.assert_close(out_pytorch, out_scripted, rtol=1e-3, atol=1e-5)
print(f"✅ 转换验证通过，最大误差={(out_pytorch-out_scripted).abs().max().item():.2e}")

# 5. 保存 + 放到Model.zip里（synset.txt一起打包）
scripted.save("resnet18.pt")
# 最后: zip resnet18-v1.zip resnet18.pt synset.txt preprocess_config.json
```

4. **常见坑点/面试追问**：
- 🔥 坑：转换时是model.train()模式，BatchNorm running_mean/var不对，DJL推理准确率掉30% → 必须先`model.eval()`再转
- 追问：「转TorchScript报TracerWarning: Cannot iterate over a tuple of Tensors什么意思？」→ 答：Trace模式下for循环被展开成N个固定算子，可变长度N会丢失，换成torch.jit.script就能保留循环逻辑
- 追问：「自定义算子torch.autograd.Function转Script失败怎么办？」→ 答：Python写的自定义算子TorchScript支持很差，要么① 重写用标准PyTorch算子组合 ② 转ONNX用自定义算子注册 ③ 迫不得已用PyTorch Engine + 注册C++自定义算子JNI

---

### Q21. ONNX转换前后不一致np.testing.assert_allclose Python vs DJL推理误差: 常见原因1.dtype隐式转FP16 2.预处理Normalize顺序错 3.RGB/BGR颜色通道颠倒（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：转换验证三部曲：Python端PyTorch输出A → Python端ONNX Runtime输出B → Java DJL端输出C，三者必须误差<1e-3。不一致90%是**预处理/数据类型**不是模型权重的问题：

| 不一致原因 | 误差量级 | 怎么验证？ | 修复方法 |
|-----------|---------|-----------|---------|
| ① **FP32→FP16隐式强转** | 1e-2 ~ 5e-2 5%误差 | 打印out.dtype看是float32还是float16 | `do_constant_folding=True, opset=17+fp32` 显式dtype |
| ② 预处理错：Normalize顺序/均值错 | 1e-1 ~ 1.0 天差地别 | 保存preprocessed_input.npy Python/Java对比 | 四件套严格对齐，不要自己写代码转 |
| ③ **通道反了：RGB vs BGR** | 0.5~1.0 完全乱分类 | save image C×H×W看第一个通道均值R vs B | OpenCV读的图必须cvtColor COLOR_BGR2RGB |
| ④ NCHW vs NHWC通道序 | 1.0 完全错 | 检查model.inputs[0].shape ONNX输入 | `transpose([0,3,1,2])`显式转 |
| ⑤ opset版本低算子语义变 | 1e-3~1e-2 | 用opset 17/18/19新版 | 转换指定`opset_version=17` |
| ⑥ Transformer/LayerNorm epsilon错 | 1e-3 | 打印LN eps值 | 默认1e-5 vs 1e-6不同 |

2. **排查顺序优先级（面试必背）**：
```
Step 1: 模型权重/算子本身对不对？
  → 喂全0 dummy输入，Python ONNX vs Java DJL输出必须误差<1e-5
  → 这一步都不对 = 转换错了（opset/dtype）
  → 这一步对了 = 模型转换没问题，问题在预处理/后处理！

Step 2: 预处理输入对不对？
  → Java: preProcess之后NDArray.save("java_input.npy")
  → Python: 同样预处理，np.save("python_input.npy")
  → np.testing.assert_allclose(java, python, rtol=1e-4)
  → 不对 = 预处理Bug，90%在这里

Step 3: 后处理解析对不对？
  → 保存模型原始output raw scores，两边先比scores
  → 再比Softmax之后的Classifications结果
```

3. **代码示例**：全链路误差校验的Python+Java联合调试脚本：
```python
# ======= Python端：生成三个ground truth npy文件 =======
import onnxruntime as ort, numpy as np, torch
# 1. 原始PyTorch模型输出
model = models.resnet18(pretrained=True).eval()
img_pil = Image.open("test_cat.jpg").convert("RGB")
img_tensor = valid_transform(img_pil).unsqueeze(0) # [1,3,224,224] 标准预处理
with torch.no_grad():
    out_pt = model(img_tensor).numpy()
np.save("01_pytorch_output.npy", out_pt)
np.save("02_preprocessed_input.npy", img_tensor.numpy()) # 重点：保存预处理结果！

# 2. ONNX Runtime Python输出
sess = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
out_ort = sess.run(None, {"input": img_tensor.numpy()})[0]
np.save("03_onnx_python_output.npy", out_ort)

np.testing.assert_allclose(out_pt, out_ort, rtol=1e-3, atol=1e-5)
print("✅ PyTorch vs ONNX Python 一致")
```
```java
// ======= Java DJL端：读取相同npy做误差校验 =======
@Test
void verifyModelConversion() throws Exception {
    // 1. 直接加载Python预处理好的npy喂模型，跳过Java预处理干扰
    try (NDManager m = NDManager.newBaseManager()) {
        NDArray pythonPreprocessed = m.load("02_preprocessed_input.npy").get(0);
        System.out.println("Java读Python预处理: shape=" + pythonPreprocessed.getShape()
            + " mean=" + pythonPreprocessed.mean() + " std=" + pythonPreprocessed.std());
        
        // 2. 构造假Translator，直接用Python预处理好的NDArray
        Translator<Void, NDArray> rawTranslator = new Translator<>() {
            @Override
            public NDList processInput(TranslatorContext ctx, Void v) {
                return new NDList(pythonPreprocessed); // 直接喂！
            }
            @Override
            public NDArray processOutput(TranslatorContext ctx, NDList list) {
                return list.get(0);
            }
        };
        
        try (Model model = Model.newInstance("test");
             Predictor<Void, NDArray> predictor = model.newPredictor(rawTranslator)) {
            model.load(Paths.get("models/resnet18-onnx/"));
            NDArray javaOutput = predictor.predict(null);
            
            // 3. 和Python ONNX输出比
            NDArray pythonOnnxOut = m.load("03_onnx_python_output.npy").get(0);
            float maxError = javaOutput.sub(pythonOnnxOut).abs().max().getFloat();
            System.out.println("最大误差: " + maxError); // 必须 < 1e-3
            assertTrue("最大误差超阈值: " + maxError, maxError < 0.001f);
            System.out.println("✅ Java DJL vs Python ONNX 完全一致！");
        }
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：有人在Java预处理先Normalize再/255，顺序反了差255倍！顺序必须是：**1.转float 2. /255到[0,1] 3.减mean 4.除std**
- 追问：「误差1e-4以内但不是0是正常的吗？」→ 答：完全正常，CPU MKL vs Java LAPACK算子累加顺序不同导致浮点舍入误差，rtol<1e-3分类任务准确率没差别；BBox检测误差<1像素可接受
- 追问：「大模型LLaMA-7B转完误差1e-2算正常吗？」→ 答：FP16下算正常（FP16只有3位有效小数位），FP32仍需<5e-3；可通过opset升级+LayerNorm epsilon对齐降误差

---

### Q22. 生产部署Docker GPU镜像基础镜像选什么? nvidia/cuda:12.1.0-runtime-ubuntu22.04 + 手动装JDK17 别用alpine glibc兼容坑多（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：GPU镜像选型三要素：① CUDA版本要和DJL/PyTorch编译时匹配 ② 基础镜像系统glibc兼容ONNX Runtime/原生库 ③ JDK版本=17+（虚拟线程+ZGC）。**不要用Alpine**，十大坑：

| 镜像 | 大小 | CUDA支持 | glibc兼容 | 推荐度 | 坑点 |
|-----|-----|---------|----------|--------|-----|
| ✅ nvidia/cuda:12.1.0-runtime-ubuntu22.04 | ~2GB | 完美Ampere/Hopper | 完美2.35+ | ⭐⭐⭐⭐⭐ 生产首选 | 体积大但零坑 |
| nvidia/cuda:12.x.x-base-ubuntu22.04 | ~800MB | 不自带cuDNN | 好 | ⭐⭐⭐ 自己装cuDNN 2GB | 要手动装cuDNN麻烦 |
| ❌ alpine + zulu jdk alpine | ~500MB | 几乎不能用 | musl不兼容glibc! | ❌ 踩坑王 | 一堆NoSuchMethodError: memcpy@GLIBC_2.xx 调不通native |
| openjdk:17-jdk-slim-bullseye | ~600MB | ❌ 无CUDA | 好 | ⭐⭐ 纯CPU推理 | CPU场景可选，GPU场景还得装CUDA |

2. **对比表格**：DJL各Engine版本要求的CUDA最低版本（0.27.0）：
| DJL Engine | 要求CUDA版本 | 对应cuDNN | 要求NVIDIA Driver >= |
|-----------|------------|-----------|-------------------|
| PyTorch 2.1 Engine | CUDA 12.1 | cuDNN 8.9 | 530.30.02 |
| ONNX Runtime GPU EP | CUDA 11.8+ | cuDNN 8.9 | 450.80.02+ |
| TensorRT 8.6 EP | CUDA 12.1 | cuDNN 8.9 | 530+ |

3. **代码示例**：生产级GPU Dockerfile完整模板：
```dockerfile
# 基础镜像：官方CUDA Runtime + Ubuntu 22.04（glibc 2.35）
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="ai-platform@corp.com"

# 1. 不交互装apt，加清华源国内加速
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        # JDK 17 必需 + curl wget + tzdata + ca-certificates
        openjdk-17-jre-headless curl tzdata ca-certificates \
        # ONNX/MKL要求的libgomp OpenMP
        libgomp1 libstdc++6 libgcc1 && \
    rm -rf /var/lib/apt/lists/*

# 2. 🔥 预烘DJL缓存！！！解决首次启动下载500MB引擎问题（见Q23）
#    提前用curl把对应版本的native引擎下到正确目录
ENV DJL_CACHE_DIR=/root/.djl.ai/cache \
    DJL_VERSION=0.27.0 \
    AI_DJL_OFFLINE=true  # 禁止运行时再联网下

RUN mkdir -p ${DJL_CACHE_DIR} && \
    # 预下载 ONNX Runtime GPU 1.17 native (500MB)
    curl -fsSL https://djl-ai.s3.amazonaws.com/publish/cache/onnxruntime/${DJL_VERSION}/onnxruntime-cuda-12.1-linux-x86_64.jar \
    -o ${DJL_CACHE_DIR}/onnxruntime-cuda.jar && \
    # 预下载 PyTorch 2.1 GPU native (2GB)
    curl -fsSL https://djl-ai.s3.amazonaws.com/publish/cache/pytorch/${DJL_VERSION}/pytorch-cu121-linux-x86_64.jar \
    -o ${DJL_CACHE_DIR}/pytorch-cu121.jar

# 3. App用户非root，安全
RUN groupadd appuser && useradd -m -u 1000 -g appuser appuser
WORKDIR /app

# 4. JVM参数：ZGC低延迟 + 堆外内存16GB上限（模型权重走堆外）
ENV JAVA_OPTS=" \
    -XX:+UseZGC \
    -Xms2g -Xmx4g \
    -XX:MaxDirectMemorySize=16g \
    -XX:+ExitOnOutOfMemoryError \
    -Dfile.encoding=UTF-8 \
    -Dai.djl.default_engine=OnnxRuntime \
    -Djava.util.concurrent.ForkJoinPool.common.parallelism=16"

COPY target/ai-service.jar /app/app.jar

USER appuser
EXPOSE 8080

# 5. 健康检查 + 启动
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/app.jar --spring.profiles.active=prod"]
```

4. **常见坑点/面试追问**：
- 🔥 坑：用了-XX:+UseG1GC + 大堆外内存，DirectByteBuffer回收被Full GC挡住 → 用ZGC/Shenandoah并发GC或设-XX:+DisableExplicitGC
- 追问：「ARM Graviton服务器镜像选什么？」→ 答：`nvidia/cuda:12.1.0-runtime-ubuntu22.04`支持arm64架构，JDK装openjdk-17-jre-headless arm64版，DJL ONNX Engine原生支持aarch64
- 追问：「镜像大3GB，Harbor仓库拉取慢怎么办？」→ 答：① 基础镜像做Harbor缓存 ② 模型权重不放镜像里，启动时S3 mount到PVC ③ 用Slim Builder多阶段构建，JRE+cuDNN裁剪能压到1.1GB

---

### Q23. 首次启动2分钟下载500MB原生引擎jar慢死怎么解决? Dockerfile提前烘缓存 预下载到/root/.djl.ai/cache 打镜像内 启动毫秒级（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：DJL Engine JAR是双层结构：
```
onnxruntime-engine-0.27.0.jar = 几百KB 纯Java代码(包装逻辑)
    └── 首次调用Engine.getEngine()时，从S3下载对应平台的500MB原生JAR
        └── native JAR = .so/.dll/.dylib CUDA/MKL/算子 存在/root/.djl.ai/cache
```
首次启动500MB+1GB(PyTorch)国内S3慢→加载2-10分钟。三大解决方案，**Docker预烘缓存是生产必做**。

2. **对比表格**：三种预加载方案对比：
| 方案 | 启动时间 | 复杂度 | 适用场景 |
|-----|---------|-------|---------|
| ① Dockerfile RUN curl烘缓存到镜像 | ⭐ 1秒（镜像内） | 低，加5行Dockerfile | ⭐⭐⭐⭐⭐ 生产级首选，K8s标准 |
| ② K8s Init Container + PVC共享缓存 | ⭐ 10秒（节点间共享） | 中 | 模型多、经常换版本的K8s集群 |
| ③ 离线环境打fat jar，把native JAR一并mvn依赖 | 3秒 | 中 | 内网完全离线无外网 |
| ❌ 不做，等启动时动态下 | 2-10分钟 不可接受 | - | 仅本地开发 |

3. **代码示例**：
**方案A Dockerfile预烘(标准)** 见Q22 Dockerfile模板，主要两行关键：
```dockerfile
ENV DJL_CACHE_DIR=/root/.djl.ai/cache AI_DJL_OFFLINE=true
# 提前curl下载native jar到cache目录
RUN curl -fsSL https://djl-ai.s3.amazonaws.com/publish/cache/onnxruntime/${DJL_VERSION}/onnxruntime-cuda-12.1-linux-x86_64.jar \
    -o ${DJL_CACHE_DIR}/onnxruntime-cuda.jar
```
DJL启动时先检查cache目录有没有native JAR，有直接解压用，不联网。

**方案B Init Container PVC共享(进阶)**：
```yaml
# K8s deployment InitContainer: 只在节点第一次时下载，后续Pod复用PVC
initContainers:
- name: djl-cache-warmup
  image: curlimages/curl:8.5.0
  command: ["sh", "-c", "
    mkdir -p /cache && \
    [ ! -f /cache/onnxruntime-cu12-0.27.jar ] && \
      curl -L $DJL_CACHE_URL/onnxruntime-cu12.jar -o /cache/onnxruntime-cu12-0.27.jar || true
  "]
  volumeMounts: [{name: djl-cache, mountPath: /cache}]
containers:
- name: ai-app
  env: [{name: AI_DJL_CACHE_DIR, value: /cache}]
  volumeMounts: [{name: djl-cache, mountPath: /cache}]
volumes:
- name: djl-cache
  hostPath: {path: /data/djl-shared-cache, type: DirectoryOrCreate} # 节点级共享
```

4. **常见坑点/面试追问**：
- 🔥 坑：K8s每次Recreate Pod，emptyDir cache没了→第一次启动又要等！→ 必须hostPath或PVC
- 追问：「怎么验证已经用了缓存不是重新下？」→ 答：日志看`Found cached engine: ...`没有`Downloading: ...https://djl-ai...500MB`的字样
- 追问：「DJL版本升级要重新下吗？」→ 答：要，不同版本DJL的native引擎哈希不同，目录结构带版本号，0.26/0.27分开存，不会冲突

---

### Q24. 怎么监控DJL线上性能? Micrometer打点predict秒数 / GPU利用率nvidia-smi / NDManager堆外内存 MXBean / 业务准确率漂移监控（⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：AI服务监控=**系统资源监控 + 模型性能监控 + 业务指标监控**三层，三层都要有，AI服务最容易漏第③层准确率漂移（Data Drift）。

```
监控三层金字塔：
┌───────────────────────────────────────────────┐
│ ③ 业务层监控（AI特有）                         │
│   准确率漂移(P99置信度下降>10%告警)             │
│   Top5分类类别分布变化(数据分布Drift)           │
│   YOLO检测框面积/类别占比变化                   │
├───────────────────────────────────────────────┤
│ ② 模型性能监控（DJL引擎层）                    │
│   单predict P50/P95/P99延迟/错误率            │
│   batchPredict BatchSize分布/队列等待时间       │
│   NDManager堆外内存占用(泄漏告警)              │
│   Engine版本/模型版本号                        │
├───────────────────────────────────────────────┤
│ ① 系统资源监控（通用）                         │
│   GPU利用率/显存占用(>90%告警)                 │
│   CPU Load/堆内存/FGC频率/线程数              │
│   K8s Pod重启次数/QPS/4xx/5xx错误率            │
└───────────────────────────────────────────────┘
```

2. **对比表格**：关键指标 + Prometheus + Grafana告警阈值：
| 指标名 | PromQL | 告警阈值 | 等级 |
|-------|--------|---------|-----|
| P99推理延迟 | `histogram_quantile(0.99, rate(djl_predict_seconds_bucket[5m]))` | >50ms 持续5分钟 | P1 |
| GPU SM利用率 | `nvidia_gpu_utilization{gpu=0}` | <30% (可缩容) / >95% (要扩容) | P2 |
| 堆外内存占用 | `djl_direct_memory_bytes{app=xx}` | >80% MaxDirectMemory 或 1h涨2GB | P0 泄漏！ |
| 推理错误率 | `rate(djl_predict_failed_total[5m]) / rate(djl_predict_total[5m])` | >1% | P1 |
| **业务Top1置信度均值** | `avg(djl_top1_confidence_overall)` | 比上周均值掉>8% → Drift可能！ | P1 AI特有 |
| 输出类别分布熵 | `djl_prediction_distribution_entropy` | 熵突然升高=模型突然随机输出 | P1 |

3. **代码示例**：Micrometer埋点 + 堆外MXBean监控(生产标准写法)：
```java
@Configuration
public class DjlMetricsConfig {
    @Bean
    public MeterBinder djlMemoryMetrics() {
        return registry -> {
            // 1. 堆外内存MXBean: DJL自带JMX MBean
            try {
                MBeanServer mBeanServer = ManagementFactory.getPlatformMBeanServer();
                ObjectName name = new ObjectName("ai.djl:type=MemoryUsage");
                Gauge.builder("djl.direct.memory.bytes", () -> {
                        try { return (Long) mBeanServer.getAttribute(name, "AllocatedDirectBytes"); }
                        catch (Exception e) { return 0L; }
                    })
                    .description("DJL堆外Native内存总占用(字节)")
                    .baseUnit(BaseUnits.BYTES).register(registry);
            } catch (MalformedObjectNameException ignore) {}
        };
    }
}

// AOP切面：对所有predict调用自动打点（不用改业务代码）
@Aspect
@Component
public class PredictMetricsAspect {
    private final MeterRegistry registry;
    @Around("@annotation(ai.djl.inference.Predictor) || execution(* com.corp..*Service.predict(..))")
    public Object aroundPredict(ProceedingJoinPoint pjp) throws Throwable {
        long start = System.nanoTime();
        Timer.Sample sample = Timer.start(registry);
        String modelName = pjp.getTarget().getClass().getSimpleName();
        try {
            Object result = pjp.proceed();
            sample.stop(Timer.builder("djl.predict.seconds")
                .tag("model", modelName).tag("status", "SUCCESS")
                .publishPercentileHistogram().register(registry));
            return result;
        } catch (Exception e) {
            sample.stop(Timer.builder("djl.predict.seconds")
                .tag("model", modelName).tag("status", "FAILED").register(registry));
            Counter.builder("djl.predict.failed.total").tag("model", modelName).register(registry).increment();
            throw e;
        }
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：只看延迟/QPS，上线2周后训练数据和线上数据分布变了（比如上线前训练的都是白天图片，上线后晚上图暴增）→ 模型准确率从90%掉到60%没人发现！→ **必须做业务准确率监控：随机抽10%请求离线打标比对 + 置信度分布漂移检测**
- 追问：「GPU显存怎么监控？」→ 答：部署`dcgm-exporter` DaemonSet，Prometheus拿`DCGM_FI_DEV_FB_USED`(帧缓存显存)，`FB_FREE < 2GB`就告警，下一次推理OOM直接Kill
- 追问：「nvidia-smi GPU-Util很高(98%)但吞吐上不去为啥？」→ 答：用`nsys profile`看，90%概率是小Batch，GPU在等CPU喂数据，PCIe带宽瓶颈。加动态攒批/大Batch让GPU真正忙起来算

---

### Q25. A/B新模型灰度方案: 两个Model Bean分别加载v1/v2版本 注入到RouterService Istio流量2%切新模型→指标错误率P95延迟正常→100%切（⭐⭐⭐⭐⭐）

**【标准答案】**

1. **定义/原理**：AI模型灰度=**流量分阶段切分 + 双模型并行预测 + 指标自动比对 + 自动回滚**四步法，AI和普通SWE灰度不同点：代码对但模型可能对部分类别错，要同时看**错误率、P95延迟、分类别准确率**三个维度。

```
灰度推进节奏（风险控制）：
Day 0: v1=100% (当前线上稳定版)
Day 1: v2=2%  → 只跑内部/QA白名单流量，看有没有NaN/崩溃
Day 2: v2=5%  → 真实小流量，比对A/B准确率误差<0.5%
Day 4: v2=20% → P99延迟 < 1.2 × v1 P99
Day 7: v2=50% → 分类别Precision/Recall看长尾类不掉
Day 10: v2=100% v1下掉，留3天可回滚版本
```

2. **对比表格**：三种灰度切流粒度适用场景：
| 切流粒度 | 实现方式 | 优点 | 缺点 | 适用场景 |
|---------|---------|------|------|---------|
| **流量百分比Istio** | VirtualService 权重路由 | ✅ 不用改业务代码，全局生效 | ❌ 同一请求只进一个模型，看不到AB对比 | 大部分场景首选 |
| **Shadow暗线双发** | 100%请求同时跑v1和v2，v2结果不返回只对比指标 | ✅ 可离线计算两模型一致性 | ❌ 双倍GPU算力贵 | 大版本升级（ViT→Swin） |
| **User ID哈希** | 固定用户百分比永远看v2 | ✅ 同一用户体验一致，AB实验科学 | ❌ 实现复杂 | 用户行为模型/推荐系统 |

3. **代码示例**：Spring RouterService + 双Model Bean + 动态权重路由：
```java
@Configuration
public class AbTestConfig {
    // v1=当前线上版本 Model单例1
    @Bean("modelV1") @Primary
    public Model modelV1() throws Exception {
        return Criteria.builder()
            .optModelUrls("s3://my-models/resnet/v1.8.3-prod/")
            .optModelName("resnet_v1_8_3_int8").build().loadModel();
    }
    // v2=新模型 单例2 两个模型权重分别加载，互不影响
    @Bean("modelV2")
    public Model modelV2() throws Exception {
        return Criteria.builder()
            .optModelUrls("s3://my-models/resnet/v2.0.1-candidate/")
            .optModelName("resnet_v2_0_1").build().loadModel();
    }
}

@Service
public class AbRouterService {
    private final Model modelV1; private final Model modelV2;
    private final AtomicInteger trafficPercentage = new AtomicInteger(2); // 2%流量先v2
    private final ThreadLocalRandom rand = ThreadLocalRandom.current();
    private final MeterRegistry meterRegistry;

    public Classifications routeAndPredict(Image img) {
        // 1. 动态路由：2%概率走v2，Apollo配置中心可热更trafficPercentage不用重启
        Model pickedModel; String version;
        if (rand.nextInt(100) < trafficPercentage.get()) {
            pickedModel = modelV2; version = "v2";
        } else {
            pickedModel = modelV1; version = "v1";
        }
        
        // 2. 预测并打点
        Timer.Sample s = Timer.start(meterRegistry);
        try (Predictor<Image, Classifications> p = pickedModel.newPredictor()) {
            Classifications result = p.predict(img);
            // 3. 🔥 A/B监控：按版本号打Tag对比Top1置信度分布
            DistributionSummary.builder("djl.ab.top1.confidence")
                .tag("version", version).register(meterRegistry)
                .record(result.getTopK().get(0).getProbability());
            return result;
        } catch (Exception e) {
            Counter.builder("djl.ab.error.total").tag("version", version).register(meterRegistry).increment();
            throw e;
        } finally {
            s.stop(Timer.builder("djl.ab.predict.latency").tag("version", version)
                .publishPercentileHistogram().register(meterRegistry));
        }
    }
    
    // Apollo配置中心热更新灰度比例，0-100
    @ApolloConfigChangeListener("abtest.traffic")
    public void onChange(ConfigChangeEvent e) {
        int p = Integer.parseInt(e.getChangedValue("abtest.v2.traffic.percent"));
        trafficPercentage.set(p);
        log.warn("🔥 灰度比例已更新: v2={}%", p);
    }
}
```

4. **常见坑点/面试追问**：
- 🔥 坑：v1/v2加载顺序+Bean名不一致，@Primary搞反，100%流量直接切到v2上了=全量上线翻车 → 灰度前单元测试：1000次调用，验证20次走v2 980次走v1，比例正确
- 追问：「Shadow双发怎么保证不影响用户？」→ 答：v2结果只用CompletableFuture异步跑，结果写到Kafka做离线对比，在线Response永远用v1的
- 追问：「v2错误率高了怎么自动回滚？」→ 答：Prometheus Alert Rule告警→钉钉/飞书通知 + Webhook调用Apollo改百分比→0%；或用Argo Rollouts自动基于指标分析回滚
- 追问：「两个Model Bean同JVM加载显存会不会炸？」→ 答：可能！v1+v2各占5GB，A10=24GB撑住；A10=10GB两个模型就爆了→要么Serving方案分开部署，要么只CPU做Shadow