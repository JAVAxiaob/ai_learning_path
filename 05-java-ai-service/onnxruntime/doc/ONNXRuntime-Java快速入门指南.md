# ONNX Runtime Java 快速入门指南

> 位置: 05-java-ai-service/onnxruntime/doc/
> 配套: ONNXRuntime-Java推理面试题15题.md
> 对比文档: DJL深度学习部署实战.md (05-java-ai/djl-demo/doc/)

---

## 📦 1. Maven依赖引入

### 1.1 CPU版本（通用服务器/本地开发，~15MB）
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.20.1</version> <!-- 2025年最新稳定版 -->
</dependency>
```

### 1.2 GPU CUDA版本（NVIDIA显卡，~200MB含CUDA库）
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime-gpu</artifactId> <!-- 自动带CUDA12/cuDNN9 -->
    <version>1.20.1</version>
    <classifier>linux-x86_64-gpu</classifier> <!-- Linux服务器 -->
    <!-- <classifier>win-x64-gpu</classifier>  Windows开发 -->
</dependency>
```
> ⚠️ 注意：CPU版和GPU版不要同时引入，会冲突选其中一个！

---

## 🚀 2. 三步Hello World推理（MNIST手写数字识别）

```java
import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

public class OrtHelloWorld {
    public static void main(String[] args) throws OrtException {
        // ✅ Step 1: 创建Environment（JVM全局单例！！！绝对不能重复new，放@Bean单例）
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // ✅ Step 2: 创建Session（加载模型，重量级对象，必须池化！生产请用Apache Pool2）
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT); // 开最高图优化
        opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors());  // CPU核数=算子并行度
        OrtSession session = env.createSession("models/mnist-8.onnx", opts);

        // 打印模型输入输出信息（部署前必看，知道input name/shape）
        System.out.println("模型输入：" + session.getInputInfo());  // {Input3: TensorInfo shape=[N,1,28,28] float32}
        System.out.println("模型输出：" + session.getOutputInfo()); // {Plus214_Output_0: shape=[N,10] float32 概率}

        // ✅ Step 3: 构造输入→推理→取出结果
        float[] pixelData = new float[1 * 28 * 28];  // 一张MNIST 28×28手写图灰阶值
        // 像素填充（代码省略：ImageIO读图片→28×28 resize→转0-1灰阶）
        long[] inputShape = new long[]{1, 1, 28, 28}; // N=1, C=1灰度, H=28, W=28

        // ✨ 必须try-with-resources自动close释放Native内存！
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pixelData), inputShape);
             OrtSession.Result output = session.run(Collections.singletonMap("Input3", inputTensor))) {

            float[][] probabilities = (float[][]) output.get(0).getValue(); // [1][10]
            int predictedDigit = argmax(probabilities[0]);  // 找最大概率的类别号(0-9)
            System.out.println("预测数字：" + predictedDigit + "，置信度：" + probabilities[0][predictedDigit]);
        }

        // 服务关闭时记得显式关闭
        session.close();
        env.close();
    }
    static int argmax(float[] arr) { int idx=0; float m=arr[0]; for(int i=1;i<arr.length;i++) if(arr[i]>m){m=arr[i];idx=i;} return idx; }
}
```

---

## 🧩 3. Spring Boot生产标准模板（池化Session + Bean生命周期）

### 3.1 pom.xml + application.yml
```xml
<dependencies>
    <dependency>
        <groupId>com.microsoft.onnxruntime</groupId>
        <artifactId>onnxruntime-gpu</artifactId>
        <version>1.20.1</version>
    </dependency>
    <dependency> <!-- Session池化：Apache Commons Pool2 官方推荐-->
        <groupId>org.apache.commons</groupId>
        <artifactId>commons-pool2</artifactId>
    </dependency>
</dependencies>
```
```yaml
ai:
  onnx:
    bge-model-path: /data/models/bge-large-zh-v1.5_int8.onnx
    session-pool:
      max-total: 16    # CPU: 核数×2；GPU: 显存/单模型占用（如24G卡/3GB模型=8个）
      min-idle: 4
      max-wait-ms: 5000
```

### 3.2 配置类：Environment单例 + Session池工厂
```java
@Configuration
@ConfigurationProperties(prefix = "ai.onnx")
@Data
public class OrtPoolConfig {
    String bgeModelPath;
    SessionPool sessionPool = new SessionPool();
    @Data public static class SessionPool { int maxTotal=16; int minIdle=4; long maxWaitMs=5000; }

    // ✅ OrtEnvironment：JVM全局单例！destroyMethod指定Spring容器关闭时自动close
    @Bean(destroyMethod = "close")
    public OrtEnvironment ortEnvironment() { return OrtEnvironment.getEnvironment(); }

    // ✅ BGE Embedding模型 Session池
    @Bean
    public GenericObjectPool<OrtSession> bgeSessionPool(OrtEnvironment env) throws Exception {
        BasePooledObjectFactory<OrtSession> factory = new BasePooledObjectFactory<>() {
            @Override public OrtSession create() throws OrtException {
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setOptimizationLevel(ALL_OPT);
                opts.setIntraOpNumThreads(4);
                opts.setInterOpNumThreads(2);
                opts.addCUDA(0);  // 用GPU 0号卡，CPU场景注释掉这行自动CPU跑
                // TensorRT EP生产可加：opts.addTensorRT(new OrtTensorRTProviderOptions(), 1);
                return env.createSession(bgeModelPath, opts);
            }
            @Override public PooledObject<OrtSession> wrap(OrtSession s) { return new DefaultPooledObject<>(s); }
            @Override public void destroyObject(PooledObject<OrtSession> p) throws Exception { p.getObject().close(); }
        };
        GenericObjectPoolConfig<OrtSession> cfg = new GenericObjectPoolConfig<>();
        cfg.setMaxTotal(sessionPool.maxTotal);
        cfg.setMinIdle(sessionPool.minIdle);
        cfg.setMaxWait(Duration.ofMillis(sessionPool.maxWaitMs));
        cfg.setTestOnBorrow(true); // 借前校验Session状态
        return new GenericObjectPool<>(factory, cfg);
    }
}
```

### 3.3 业务Service：借/还Session + try-with-resources安全释放
```java
@Service
public class BgeEmbeddingService {
    @Autowired GenericObjectPool<OrtSession> bgePool;

    // 生成1024维BGE中文Embedding向量
    public float[] encode(String text) throws Exception {
        OrtSession session = null;
        long[] shape = {1, tokenIds.length};
        // 预处理tokenize→int[] ids(代码省略：用HuggingFace Tokenizers Java或Jieba+BPE自己实现)
        long[] inputIds = tokenize(text);
        long[] attentionMask = new long[inputIds.length]; Arrays.fill(attentionMask, 1L);

        try {
            session = bgePool.borrowObject(); // ✅ 从池借一个Session
            // ✨ try-with-resources自动关闭Tensor/Result Native内存！
            try (OnnxTensor ids = OnnxTensor.createTensor(env, inputIds, shape);
                 OnnxTensor mask = OnnxTensor.createTensor(env, attentionMask, shape);
                 OrtSession.Result r = session.run(Map.of("input_ids", ids, "attention_mask", mask))) {
                float[][][] lastHidden = (float[][][]) r.get(0).getValue(); // [1, seqLen, 1024]
                return meanPooling(lastHidden[0]); // 取CLS或均值池化得到1024维向量
            }
        } finally {
            if (session != null) bgePool.returnObject(session); // ✅ 还回池（无论成功异常）
        }
    }
}
```

---

## ⚡ 4. 关键性能优化清单（从1ms→0.2ms优化5倍）

| 优化项 | 做法 | 预期提升 |
|---|---|---|
| 1️⃣ 开`ALL_OPT`全部图优化 | `opts.setOptimizationLevel(ALL_OPT)` | 30% |
| 2️⃣ INT8量化模型 | 用`python onnxruntime.quantization.quantize_dynamic()`量化 | CPU快3.5倍，模型小4倍 |
| 3️⃣ **Batch推理批量合并请求** | 10个单请求拼成Batch=10一起跑 | GPU场景快4-8倍 |
| 4️⃣ Session池化（不用每个请求new） | Commons Pool2 预创建N个 | 100×启动开销没了 |
| 5️⃣ 算子线程数调优 | `setIntraOpNumThreads(N)`：N=CPU核数/2（8核设4） | 20% |
| 6️⃣ GPU CUDA EP / TensorRT EP | `opts.addCUDA()` / `addTensorRT(...)` | 快5-60倍 |
| 7️⃣ **IO Binding零拷贝**（GPU大张量）| GPU数组→ORT直接用指针，不拷贝回CPU | 大张量再省20-50%拷贝耗时 |
| 8️⃣ 直接用`FloatBuffer/IntBuffer`代替`float[]` | `OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), shape)` | 省一次JNI数组拷贝，10%+ |
| 9️⃣ 预热WarmUp | 启动后先跑100次空推理触发JIT/图编译 | 消除首请求300ms冷启动 |

---

## 🆚 5. 常见数据类型对应表（面试+开发都有用）

| Java类型 → OnnxTensor.createTensor参数 | ONNX类型 | Ort对应的OnnxJavaType枚举 |
|---|---|---|
| `float[]` / `FloatBuffer` | Tensor(float32) | `OnnxJavaType.FLOAT` |
| `double[]` / `DoubleBuffer` | Tensor(float64/double) | `OnnxJavaType.DOUBLE` |
| `long[]` / `LongBuffer` | Tensor(int64) | `OnnxJavaType.INT64`（BERT input_ids必用long！）|
| `int[]` / `IntBuffer` | Tensor(int32) | `OnnxJavaType.INT32` |
| `short[]` / `ShortBuffer` | Tensor(int16/INT8量化模型) | `INT16`/`INT8` |
| `boolean[]` | Tensor(bool) | `OnnxJavaType.BOOL`（attention mask其实一般用int64）|
| `String[]` → UTF-8 bytes | Tensor(string) | `OnnxJavaType.STRING` |
| `Map<K,V>` → 分类任务标签 | OnnxMap (稀有) | MapInfo |
| `List<T>` 动态列表 | OnnxSequence | SequenceInfo |

---

## ❌ 6. 5大生产常见坑（90%新手踩）

| 坑 | 症状 | 修复 |
|---|---|---|
| 🔴 `OrtEnvironment`每次请求new | 10分钟后JVM崩溃：JNI Reference溢出 | ✅ JVM单例，getEnvironment()只调用一次 |
| 🔴 OnnxTensor没close() | Native内存RSS暴涨→Linux OOM Killer杀进程 | ✅ 强制try-with-resources |
| 🔴 input_ids传了int[]而不是long[] | 形状对但推理结果全错/崩溃NPE | ✅ HuggingFace模型input_ids必传`long[]`！ |
| 🔴 Session池maxTotal太小 | 高并发请求排队超时503/504 | ✅ GPU按显存/CPU按核心数合理设置，加监控借池等待时长 |
| 🔴 CUDA装了但JVM找不到`libcudart.so` | UnsatisfiedLinkError加载GPU版失败 | ✅ `LD_LIBRARY_PATH`加CUDA路径或LD_PRELOAD |

---

## 📌 附：从PyTorch导出ONNX必用参数
```python
# PyTorch端导出代码（保证Java ORT推理速度拉满）
torch.onnx.export(
    model, dummy_input, "bge.onnx",
    export_params=True,
    opset_version=17,               # ✅ 选17+高版本，算子全且优化好
    do_constant_folding=True,       # ✅ 常量折叠省计算
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={                  # ✅ 动态batch/seq_len长度支持
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "last_hidden_state": {0: "batch", 1: "seq_len"}
    }
)
```
→ 导出后运行`python -m onnxruntime.quantization.quantize_dynamic bge.onnx bge_int8.onnx weight_type=QUInt8` → INT8直接省75%内存快3.5倍！
