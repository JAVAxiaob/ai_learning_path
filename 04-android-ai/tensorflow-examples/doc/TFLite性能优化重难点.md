# TFLite 性能优化重难点分析

> 位置: 04-android-ai/tensorflow-examples/lite/examples/

---

## 一、核心性能优化手段

### 1. Delegate选型策略

| Delegate | 适用场景 | 加速比 | 坑点 |
|---------|---------|-------|-----|
| **XNNPACK** | 通用场景/兼容性优先 | 2~5x | 几乎无坑 |
| **GPU** | 大模型/高吞吐 | 2~10x | 部分算子不支持→fallback |
| **NNAPI** | 后台低功耗场景 | 5~20x | Android 10+，厂商差异大 |
| **Hexagon DSP** | 高通机型独占 | 6~15x | libhexagon_interface.so需打包 |

### 2. INT8量化优化

```python
# PTQ训练后量化流程
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 校准数据集 (关键!)
def representative_data_gen():
    for _ in range(1000):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
```

**量化关键点：**
- 校准数据必须是**真实场景分布**的图（500~2000张）
- 覆盖各种角度/光照/遮挡/类别
- 统计每层激活的min/max → KL散度找最优scale

### 3. 零拷贝优化

```kotlin
// Android 8.0+ 支持
val options = Interpreter.Options().apply {
    setUseBufferHandle(true)  // AHardwareBuffer直接传GPU
}
```

### 4. 帧节流实现

```kotlin
class ThrottlingAnalyzer(
    private val intervalMs: Long = 33,  // 约30FPS
    private val listener: (ImageProxy) -> Unit
) : ImageAnalysis.Analyzer {
    
    private var lastProcessTime = 0L
    
    override fun analyze(image: ImageProxy) {
        val current = System.currentTimeMillis()
        if (current - lastProcessTime < intervalMs) {
            image.close()
            return
        }
        lastProcessTime = current
        listener(image)
    }
}
```

## 二、常见坑点与解决方案

### 坑1：GPU Delegate算子不支持

**现象**：部分模型在某些机型上GPU推理失败

**解决方案**：
```kotlin
val options = Interpreter.Options().apply {
    runCatching {
        addDelegate(GpuDelegate(GpuDelegate.Options()
            .setPrecisionLossAllowed(true)))
    }
    // GPU失败自动fallback到CPU
}
```

### 坑2：旋转角度错误导致检测失败

**现象**：竖屏拍摄时检测不到物体

**解决方案**：
```kotlin
// 获取正确的旋转角度
val rotationDegrees = imageProxy.imageInfo.rotationDegrees
val inputImage = InputImage.fromMediaImage(image, rotationDegrees)
```

### 坑3：坐标转换错误

**现象**：检测框位置偏移

**解决方案**：
```kotlin
// 图像坐标 → View坐标转换矩阵
val matrix = Matrix().apply {
    // 旋转+缩放+镜像
    postRotate(rotationDegrees.toFloat())
    postScale(scaleX, scaleY)
    if (isFrontCamera) postScale(-1f, 1f)  // 前置镜像
}
```

### 坑4：内存泄漏

**现象**：App运行一段时间后OOM

**解决方案**：
```kotlin
// 及时释放资源
override fun onDestroy() {
    tflite?.close()
    imageProcessor?.let { /* 释放 */ }
    super.onDestroy()
}
```

## 三、性能监控指标

```kotlin
class PerformanceMonitor {
    private val inferenceTimes = mutableListOf<Long>()
    
    fun recordInferenceTime(ms: Long) {
        inferenceTimes.add(ms)
        if (inferenceTimes.size > 100) inferenceTimes.removeFirst()
    }
    
    fun getStats(): PerformanceStats {
        val avg = inferenceTimes.average()
        val p90 = inferenceTimes.sorted()[(inferenceTimes.size * 0.9).toInt()]
        return PerformanceStats(avg, p90, inferenceTimes.size)
    }
}
```