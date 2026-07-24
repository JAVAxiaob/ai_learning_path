# TensorFlow Lite Android 解析

> 位置: 04-android-ai/tensorflow-examples/lite/examples/
> 简历推荐: 5星 | 岗位: Android AI工程师 (工业界最通用方案)

---

## 一、TFLite Examples模块速查

```
lite/examples/ 每个子目录 = 一个独立完整App (可以直接AS打开跑)
├── image_classification/      图像分类
│   ├── EfficientNet-Lite0/1/2/3/4 5档速度/精度梯度
│   └── MobileNetV2 1.0 224 基准
├── object_detection/          目标检测 SSD-MobileNet/YOLOv5n-tflite
├── pose_estimation/           MoveNet人体姿态 (Single/Lightning/Thunder 3档)
├── segmentation/              DeepLabV3 人像/场景20类分割
├── speech_commands/           语音命令 "OK Google"/10个词 Keyword Spotting
├── smart_reply/               对话智能回复 (NLClassifier)
├── gesture_classification/    6轴IMU传感器手势分类 (加速度+陀螺仪)
├── digit_classifier/          MNIST手写数字画板识别
├── style_transfer/            风格迁移 (把照片转梵高/毕加索风格)
└── audio_classification/      音频事件分类 (说话声/犬吠/警笛...)
```

## 二、高性能推理代码 (Options配置 = 面试逐行问)

```kotlin
// build.gradle:
implementation 'org.tensorflow:tensorflow-lite:2.16.1'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.16.1'
implementation 'org.tensorflow:tensorflow-lite-xnnpack:2.16.1'
implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'

// 代码: 最优Options配置 (不同手机跑不同硬件!)
val modelFile = MappedByteBuffer(FileUtil.loadMappedFile(context, "mobilenet_v2.tflite"))
val options = Interpreter.Options().apply {
    setNumThreads(Runtime.getRuntime().availableProcessors()/2)  // 1. 线程数=大核数
    setUseXNNPACK(true)                                          // 2. XNNPACK NEON优化引擎 (必开!)
    // 3. GPU委托 (注意: 部分模型不支持GPU可以fallback)
    runCatching {
        addDelegate(GpuDelegate(GpuDelegate.Options()
            .setPrecisionLossAllowed(true)       // FP32→FP16,精度损失可忽略速度翻倍
            .setQuantizedModelsAllowed(true)     // 量化模型上GPU跑
            .setInferencePreference(INFERENCE_PREFERENCE_SUSTAINED_SPEED)))
    }
    // 4. NNAPI委托: 高通Hexagon/联发科APU 长期后台低功耗场景首选
    runCatching {
        addDelegate(NnApiDelegate(NnApiDelegate.Options()
            .setExecutionPreference(PREFERENCE_LOW_POWER)))
    }
    setAllowFp16PrecisionForFp32(true)    // 5. FP32算子自动降FP16
    setUseBufferHandle(true)               // 6. 零拷贝AHardwareBuffer
}
val tflite = Interpreter(modelFile, options)

// 输入输出 (Support库自动处理: 尺寸缩放/Crop/Normalize/ByteBuffer!)
val imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .add(NormalizeOp(floatArrayOf(127.5f,127.5f,127.5f), floatArrayOf(127.5f,127.5f,127.5f)))
    .build()
var tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
tensorImage = imageProcessor.process(tensorImage)

val outputs = TensorBuffer.createFixedSize(intArrayOf(1,1001), DataType.FLOAT32)
tflite.run(tensorImage.buffer, outputs.buffer)  // 推理!
val labels = FileUtil.loadLabels(context, "labels.txt")
val topK = TensorLabel(labels, outputs).mapWithFloatValue
    .toList().sortedByDescending { it.second }.take(5)  // Top 5概率
```

## 三、Delegate性能对比表 (背下来!)

| Delegate | 跑在什么硬件 | 加速比 | 适用场景 | 坑 |
|---------|------------|-------|---------|-----|
| **XNNPACK (默认!)** | CPU ARM NEON / x86 AVX | 2~5x vs 单线程CPU | 兼容性要求高/小模型/跨平台 | 几乎无坑 |
| **GPU OpenGL/Vulkan** | Mali/Adreno移动GPU | 2~10x | CNN大模型/吞吐优先 | 部分模型算子不支持→自动fallback |
| **NNAPI HAL3+** | 厂商NPU/DSP: 高通Hexagon/联发科APU/华为达芬奇 | 5~20x + 省电60%+ | 后台长期运行/低功耗监控场景 | Android 10+ 每个厂商支持千差万别 |
| **Hexagon DSP** | 高通Hexagon V66+ | 6~15x + 最省电 | 高通机型占比高 (70%+国内机型) | libhexagon_interface.so要打包 |

## 四、简历黄金句式

| 写法 |
|-----|
| 「MobileNetV3垃圾分类TFLite部署：INT8量化+NNAPI委托，骁龙865 DSP推理延迟3ms，APK包体仅+5MB，日均调用量50W+，Acc 93.5%」 |
| 「MoveNet Lightning姿态估计33点实时追踪：XNNPACK4线程+FP16优化，骁龙778G中端机1080P@45fps，健身类App姿态指导准确率91%」 |
| 「TFLite模型热更新：CDN差分下载+MD5校验+灰度，新模型迭代无需发版，线上异常率从1.3%→0.07%」 |

## 五、面试题

**Q: INT8量化两种模式: Post-training vs Quantization-aware training区别？**
> A: PTQ训练后量化：①float32现成模型 ②拿1000张校准图跑一遍统计min/max+KL散度 ③直接INT8，1天搞定，精度降<1%。QAT感知量化：训练图里插入FakeQuant节点模拟量化误差反向传播，8bit实际训练，2~3周，精度几乎无损(<0.3%)。生产先试PTQ，精度不达标再QAT。

**Q: MobileNet V1/V2/V3/EfficientNet 选型思路？**
> A: 参数量和计算量从少到多排序: MobileNetV1 < MobileNetV2 < MobileNetV3-Large < EfficientNet-Lite0 < EfficientNet-Lite4。速度和Acc trade-off：① 超低端机必须MobileNetV2 1.0 ② 常规中低端机MobileNetV3-Large首选(多一个HardSwish激活+SE通道注意力，同速度Acc高2%) ③ 高端机对Acc极致要求 EfficientNet-Lite2/3/4。

**Q: 端侧包体优化策略？如何减小APK大小+5MB以内？**
> A: ① 模型量化INT8 = ÷4，FP16 = ÷2 ② 模型剪枝: 结构化剪通道剪70%参数 ③ Huffman编码权重再压30% ④ 分ABIs armeabi-v7a只打包32位(兼容99%)或arm64-v8a ⑤ Play Store App Bundle分架构下发 ⑥ 动态下发: 首次启动按需从CDN下载模型不打APK里。