# 11-04 Android端侧AI项目解析

> 📂 项目: `mlkit/` `tensorflow-examples/` `ncnn-android-yolov5/` `MNN/` `PaddleOCR/`
> ⭐ 简历推荐: ⭐⭐⭐⭐ | 🎯 岗位: Android开发(AI方向)、端智能工程师

---

## 一、端侧AI全景

### 1.1 为什么做端侧？ (VS云端AI)

| 维度 | 云端AI | 端侧AI |
|-----|--------|-------|
| 延迟 | 100~1000ms (网络) | 1~50ms (本地) |
| 隐私 | 数据上传，风险大 | 数据不出机，合规 |
| 离线 | 必须联网 | 离线可用 |
| 成本 | 服务器贵 | 免费利用用户算力 |
| 典型场景 | 超大模型、复杂推理 | 人脸/AR/实时特效/OCR |

### 1.2 主流端侧框架选型

| 框架 | 厂商 | 硬件支持 | 适用 | 代表App |
|-----|------|---------|-----|---------|
| **ML Kit** | Google | CPU/GPU/DSP | 快速上线，API直接用 | 谷歌翻译/Lens |
| **TFLite** | Google | CPU/GPU/NPU/TPU | 通用场景，社区大 | 大多数App |
| **NCNN** | 腾讯 | ARM NEON + Vulkan | 极致性能，包体小 | 微信/QQ |
| **MNN** | 阿里 | CPU/GPU/NPU | 训练+推理一体 | 支付宝拍立淘 |
| **Paddle Lite** | 百度 | CPU/NPU | 国产硬件友好 | 百度地图 |

---

## 二、ML Kit (Google官方方案)

### 2.1 能力地图

```
mlkit/android/vision/
├── barcode-scanning       条码/二维码
├── face-detection         人脸检测+33关键点
├── face-mesh-detection    468点人脸网格 (AR/美颜)
├── text-recognition       OCR 100+语言
├── image-labeling         图像分类标签
├── object-detection+track 物体检测+跟踪
├── pose-detection         33点人体姿态
└── selfie-segmentation    人像抠图/背景替换
```

### 2.2 5行代码实现人脸检测

```kotlin
// build.gradle
implementation 'com.google.mlkit:face-detection:16.1.6'

// 代码
val opt = FaceDetectorOptions.Builder()
    .setPerformanceMode(PERFORMANCE_MODE_FAST)
    .setContourMode(CONTOUR_MODE_ALL)
    .build()
val detector = FaceDetection.getClient(opt)
detector.process(image, rot)
    .addOnSuccessListener { faces ->
        for (f in faces) { f.boundingBox; f.headEulerAngleY; f.smilingProbability }
    }
```

### 2.3 实战: CameraX实时链路

```
CameraX (PreviewView)
    ↓ ImageAnalysis.Analyzer (每帧YUV回调)
    ↓ InputImage.fromMediaImage() → 注意旋转角度
    ↓ detector.process() 异步 (不能阻塞Analyzer线程)
    ↓ 坐标转换矩阵: 图像→预览View (镜像+旋转+缩放)
    ↓ 自定义OverlayView Canvas绘制人脸框/网格点
    ↓ 帧率节流: 每30ms最多处理1帧 (避免GPU占用过高)
```

> ✍️ **简历**: 「CameraX+ML Kit实现实时人脸AR特效，468点网格+33姿态点，1080P@30FPS稳定，NPU delegate延迟<15ms，中端机型功耗<200mA」

---

## 三、TFLite Android (工业界通用)

### 3.1 性能优化4大Delegate

```kotlin
val opt = Interpreter.Options().apply {
    setNumThreads(4)                         // 1. CPU线程池 = 大核数
    setUseXNNPACK(true)                       // 2. XNNPACK (Google NEON优化算子库,必开!)
    addDelegate(GpuDelegate(                  // 3. GPU委托 OpenGL/Vulkan
        GpuDelegate.Options()
            .setPrecisionLossAllowed(true)      // FP32→FP16, 速度翻倍,精度几乎不降
            .setInferencePreference(INFERENCE_PREFERENCE_SUSTAINED_SPEED)
    ))
    addDelegate(NnApiDelegate(                // 4. NNAPI → 高通Hexagon/联发科APU (最省电!)
        NnApiDelegate.Options().setExecutionPreference(PREFERENCE_LOW_POWER)
    ))
    setAllowFp16PrecisionForFp32(true)
}
val tflite = Interpreter(loadModelFile("mobilenet_v2.tflite"), opt)
```

| Delegate | 硬件 | 加速比 | 适合 |
|---------|-----|-------|-----|
| XNNPACK | CPU NEON | 2~5x | 小模型/兼容优先 |
| GPU | Mali/Adreno | 2~10x | CNN大模型/吞吐优先 |
| NNAPI | 厂商NPU/DSP | 5~20x + 省电60%+ | 后台常驻/低功耗 |

---

## 四、NCNN (腾讯极致性能方案)

### 4.1 项目结构 (YOLOv5安卓参考)

```
ncnn-android-yolov5/
├── jni/                             ← NDK C++核心层
│   ├── CMakeLists.txt                 链接 ncnn + vulkan + jnigraphics
│   ├── yolov5ncnn.cpp                 前处理/推理/后处理
│   ├── ndkcamera.cpp                  NDK Camera2纯C++取帧
│   └── yolo_jni.cpp                   Java↔C++ JNI桥
├── assets/
│   ├── yolov5s.param                  NCNN模型结构 (可读文本)
│   └── yolov5s.bin                    权重 (二进制)
└── java/.../YoloV5Ncnn.kt             JNI调用封装
```

### 4.2 NCNN推理三阶段

```cpp
// yolov5ncnn.cpp 核心
int detect(cv::Mat& rgb, vector<Object>& objects) {
    // 1. 前处理: LetterBox + Normalize + HWC→CHW
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, 640, 640);
    float norm[] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(0, norm);

    // 2. 推理 (自动CPU/Vulkan)
    ncnn::Extractor ex = net->create_extractor();
    ex.set_light_mode(true);          // 轻量: 不用的层立即释放
    ex.set_vulkan_compute(use_gpu);   // 开GPU
    ex.input("images", in);
    ncnn::Mat out; ex.extract("output", out);  // [25200, 85]

    // 3. 后处理: 解码 + NMS非极大值抑制
    generate_proposals(out, 0.25f, objects);
}
```

### 4.3 NCNN模型加速方案

| 手段 | 命令/做法 | 效果 |
|-----|---------|-----|
| **INT8对称量化** | quantize校准数据集1000张 | 模型÷4, 速度×2~4, Acc↓<1% |
| **FP16半精度** | ncnnoptimize --fp16 | 大小÷2, 速度×1.5~2 |
| **算子融合** | ncnnoptimize Conv+BN+ReLU | 临时内存减少 |
| **Winograd** | Conv3×3自动走F(6,3)算法 | 卷积FLOPs理论÷4 |

> ✍️ **简历**: 「NCNN+Vulkan落地实时安全帽检测：YOLOv5m INT8量化+Winograd，骁龙8 Gen2上1080P@87fps，模型28MB→7MB，准确率97.2%」

---

## 五、PaddleOCR + MNN

### PaddleOCR 三阶段流水线

```
PaddleOCR = DB文本检测 + CRNN文字识别 + 方向分类器
① Det: DB (Differentiable Binarization)   → 图中文字框定位
② Cls: 3分类(0°/180°)                     → 文字是否倒置,先转正
③ Rec: CRNN + CTC Loss                    → 每个框识别文字内容

安卓部署 (Paddle Lite):
  3个模型总大小: 3.7+9.5+1.5 = 14.7MB
  中端机型平均: 整图OCR ~400ms, 中文印刷体识别率>98%
```

---

## 六、简历亮点 + 面试题

### ✍️ 简历句式 (含量化!)

| 方向 | 写法 |
|-----|-----|
| TFLite | 「MobileNetV3垃圾分类:TFLite INT8量化+NNAPI委托,骁龙865 DSP延迟3ms,APK包体仅+5MB,模型Acc 93.5%」 |
| 包体优化 | 「6个端侧AI模型总大小286MB→67MB (结构化剪枝+INT8+Huffman), APK下载转化率↑18%」 |
| 热更新 | 「设计模型热更新系统:CDN差分下载+MD5校验+灰度发布,模型版本迭代无需发版,异常率从1.3%→0.07%」 |

### 🎯 面试题

**Q1: TFLite vs NCNN 怎么选？**
> A: TFLite优先(Google维护、文档全、社区大)，80%场景够用；NCNN当极致性能/不依赖Google服务/包体限制，Vulkan挖GPU更彻底；MNN适合阿里系技术栈。

**Q2: INT8量化为什么能提速？原理？**
> A: 公式 value = scale*(int8 - zero_point)。ARM NEON 1次能算16个INT8 MAC (乘加)，FP32只能算4个→理论4倍算力；同时内存读÷4→带宽瓶颈缓解。校准流程：准备数据集→跑推理记录每层激活min/max→KL散度找最优scale→量化存INT8。

**Q3: 模型热更新方案？**
> A: 三阶段: ①云端CDN放模型zip+版本+MD5；②启动时后台差分下载校验；③解密覆盖私有目录下次加载。兜底:加载失败fallback APK内置模型；灰度开关+一键回滚。

---

**下一篇**: 👉 [11-05 Java后端AI服务解析](11-05_Java后端AI服务解析.md)