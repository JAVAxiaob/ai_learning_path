# TFLite 推理流程图详解

> 位置: 04-android-ai/tensorflow-examples/lite/examples/

---

## 一、端到端推理完整流程

```mermaid
graph TD
    subgraph UI层
        A[PreviewView 相机预览]
        B[OverlayView 绘制结果]
    end
    
    subgraph CameraX层
        C[ImageAnalysis.Analyzer] --> D[YUV_420_888 帧数据]
        D --> E{帧节流 Throttle}
        E -->|时间差>30ms| F[处理帧]
        E -->|否则| G[丢弃帧]
    end
    
    subgraph 预处理层
        F --> H[旋转角度校正]
        H --> I[YUV→RGB转换]
        I --> J[LetterBox Resize]
        J --> K[归一化 /255]
        K --> L[HWC→NHWC TensorShape]
    end
    
    subgraph 推理层
        L --> M[TensorBuffer 输入]
        M --> N[Interpreter.runForMultipleOutputs]
        N --> O{Delegate选择}
        O -->|XNNPACK| P[CPU NEON加速]
        O -->|GPU| Q[OpenGL/Vulkan FP16]
        O -->|NNAPI| R[厂商NPU/DSP]
    end
    
    subgraph 后处理层
        P --> S[置信度过滤]
        Q --> S
        R --> S
        S --> T[NMS非极大值抑制]
        T --> U[坐标还原 LetterBox反变换]
        U --> V[结果列表]
    end
    
    subgraph 绘制层
        V --> W[坐标矩阵转换]
        W --> B
    end
    
    A --> C
```

## 二、关键节点详解

### 1. 帧节流机制
```kotlin
private var lastProcessTimeMs = 0L

fun analyze(image: ImageProxy) {
    val currentTime = System.currentTimeMillis()
    if (currentTime - lastProcessTimeMs < 33) {  // 30FPS阈值
        image.close()
        return
    }
    lastProcessTimeMs = currentTime
    // 处理帧...
}
```

### 2. Delegate三级降级策略
```kotlin
val options = Interpreter.Options().apply {
    setUseXNNPACK(true)  // 兜底方案
    runCatching { addDelegate(GpuDelegate(...)) }   // 优先GPU
    runCatching { addDelegate(NnApiDelegate(...)) } // 后台用NNAPI
}
```

### 3. 预处理流水线
```kotlin
val processor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .add(NormalizeOp(floatArrayOf(127.5f,127.5f,127.5f), floatArrayOf(127.5f,127.5f,127.5f)))
    .build()
```

## 三、数据流时序图

```mermaid
sequenceDiagram
    participant CameraX as CameraX
    participant Analyzer as ImageAnalysis
    participant Preprocess as 预处理
    participant TFLite as Interpreter
    participant Postprocess as 后处理
    participant UI as OverlayView

    CameraX->>Analyzer: onImageAvailable(image)
    Analyzer->>Analyzer: 帧节流检查
    alt 时间差>30ms
        Analyzer->>Preprocess: YUV_420_888
        Preprocess->>Preprocess: 旋转/Resize/归一化
        Preprocess->>TFLite: TensorBuffer
        TFLite->>TFLite: run(input, output)
        TFLite->>Postprocess: 原始输出
        Postprocess->>Postprocess: NMS+坐标还原
        Postprocess->>UI: 检测结果
        UI->>UI: Canvas绘制
    else 时间差<30ms
        Analyzer->>Analyzer: 丢弃帧
    end
```