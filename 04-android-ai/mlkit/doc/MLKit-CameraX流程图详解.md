# ML Kit + CameraX 流程图详解

> 位置: 04-android-ai/mlkit/android/

---

## 一、CameraX实时处理架构

```mermaid
graph TD
    subgraph CameraX层
        A[CameraX PreviewView] --> B[ImageAnalysis UseCase]
        B --> C[Analyzer回调]
        C --> D[YUV_420_888帧数据]
    end
    
    subgraph 帧节流层
        D --> E{距上次>30ms?}
        E -->|是| F[处理帧]
        E -->|否| G[丢弃帧]
    end
    
    subgraph ML Kit层
        F --> H[InputImage创建]
        H --> I[Detector.process]
        I --> J{异步回调}
        J -->|Success| K[检测结果]
        J -->|Failure| L[错误处理]
    end
    
    subgraph 绘制层
        K --> M[坐标转换矩阵]
        M --> N[OverlayView Canvas绘制]
    end
    
    style E fill:#f9f,stroke:#333,stroke-width:2px
```

## 二、CameraX配置流程

```mermaid
flowchart LR
    A[初始化CameraProvider] --> B[创建Preview UseCase]
    B --> C[创建ImageAnalysis UseCase]
    C --> D[设置Analyzer]
    D --> E[绑定到Lifecycle]
    E --> F[开始预览]
```

```kotlin
// CameraX配置示例
val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

cameraProviderFuture.addListener({
    val cameraProvider = cameraProviderFuture.get()
    
    val preview = Preview.Builder()
        .build()
        .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }
    
    val imageAnalysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()
        .also { it.setAnalyzer(cameraExecutor, MyAnalyzer()) }
    
    cameraProvider.bindToLifecycle(
        this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
}, ContextCompat.getMainExecutor(context))
```

## 三、ML Kit检测流程

```mermaid
sequenceDiagram
    participant CameraX as CameraX
    participant Analyzer as ImageAnalysis.Analyzer
    participant MLKit as FaceDetector
    participant UI as OverlayView
    
    CameraX->>Analyzer: analyze(ImageProxy)
    Analyzer->>Analyzer: 帧节流检查
    alt 时间差>30ms
        Analyzer->>MLKit: process(InputImage)
        MLKit->>MLKit: 异步推理
        Note right of MLKit: 在NPU/GPU上执行
        MLKit-->>Analyzer: onSuccess(faces)
        Analyzer->>UI: post { 绘制人脸框 }
        UI->>UI: Canvas.drawRect
    else 时间差<30ms
        Analyzer->>Analyzer: imageProxy.close()
    end
```

## 四、坐标转换流程

```mermaid
graph TD
    A[检测结果坐标] --> B[图像坐标系]
    B --> C[旋转角度校正]
    C --> D[缩放适配PreviewView]
    D --> E{前置相机?}
    E -->|是| F[水平镜像]
    E -->|否| G[直接输出]
    F --> H[View坐标系]
    G --> H
    H --> I[Canvas绘制]
```

```kotlin
// 坐标转换矩阵
val matrix = Matrix().apply {
    // 旋转
    postRotate(rotationDegrees.toFloat())
    // 缩放
    postScale(scaleX, scaleY)
    // 前置镜像
    if (isFrontCamera) postScale(-1f, 1f, centerX, centerY)
}
```

## 五、关键节点详解

### 1. 帧节流实现

```kotlin
class ThrottlingAnalyzer(
    private val intervalMs: Long = 33,
    private val callback: (ImageProxy) -> Unit
) : ImageAnalysis.Analyzer {
    
    private var lastProcessTime = 0L
    
    override fun analyze(image: ImageProxy) {
        val current = System.currentTimeMillis()
        if (current - lastProcessTime < intervalMs) {
            image.close()
            return
        }
        lastProcessTime = current
        callback(image)
    }
}
```

### 2. InputImage创建

```kotlin
// 关键：必须传入正确的旋转角度
val rotationDegrees = imageProxy.imageInfo.rotationDegrees
val inputImage = InputImage.fromMediaImage(
    imageProxy.image!!, 
    rotationDegrees
)
```

### 3. 检测结果处理

```kotlin
detector.process(inputImage)
    .addOnSuccessListener { faces ->
        for (face in faces) {
            val boundingBox = face.boundingBox
            val smileProbability = face.smilingProbability
            // 转换坐标并绘制
        }
    }
    .addOnFailureListener { e ->
        // 错误处理
    }
    .addOnCompleteListener {
        imageProxy.close()  // 必须释放!
    }
```