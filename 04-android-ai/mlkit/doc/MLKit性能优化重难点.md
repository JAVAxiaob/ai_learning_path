# ML Kit + CameraX 性能优化重难点分析

> 位置: 04-android-ai/mlkit/android/

---

## 一、性能优化策略

### 1. Delegate选型

| Delegate | 适用场景 | 加速比 | 兼容性 |
|---------|---------|-------|-------|
| **XNNPACK** | 通用场景 | 2~5x | 所有机型 |
| **GPU** | 大模型/高吞吐 | 2~10x | 部分老机型有bug |
| **NNAPI** | 后台低功耗 | 5~20x | Android 10+ |

```kotlin
// 配置Delegate优先级
val options = FaceDetectorOptions.Builder()
    .setPerformanceMode(PERFORMANCE_MODE_FAST)
    .build()
```

### 2. 帧节流优化

```kotlin
// 关键：控制处理帧率
private const val PROCESS_INTERVAL_MS = 33  // 约30FPS

fun analyze(image: ImageProxy) {
    val current = System.currentTimeMillis()
    if (current - lastProcessTime < PROCESS_INTERVAL_MS) {
        image.close()
        return
    }
    lastProcessTime = current
    // 处理帧...
}
```

### 3. 异步处理优化

```kotlin
// 不要阻塞Analyzer线程!
detector.process(inputImage)
    .addOnSuccessListener { faces ->
        // 在回调中处理结果
        overlayView.post { overlayView.drawResults(faces) }
    }
    .addOnCompleteListener { image.close() }
```

### 4. 资源管理优化

```kotlin
// Activity销毁时释放检测器
override fun onDestroy() {
    detector?.close()
    super.onDestroy()
}

// 及时关闭ImageProxy
detector.process(inputImage)
    .addOnCompleteListener { imageProxy.close() }
```

## 二、常见坑点

### 坑1：检测不到人脸

**现象**：竖屏拍摄时检测不到人脸

**原因**：旋转角度错误（80%的坑！）

**解决方案**：
```kotlin
// 必须从ImageInfo获取旋转角度
val rotationDegrees = imageProxy.imageInfo.rotationDegrees
val inputImage = InputImage.fromMediaImage(image, rotationDegrees)
```

### 坑2：检测框位置偏移

**现象**：检测框不在物体上

**原因**：坐标转换矩阵不正确

**解决方案**：
```kotlin
val matrix = Matrix().apply {
    // 1. 旋转
    postRotate(rotationDegrees.toFloat())
    // 2. 缩放适配
    postScale(scaleX, scaleY)
    // 3. 前置镜像
    if (isFrontCamera) postScale(-1f, 1f)
}
```

### 坑3：卡顿/发烫

**现象**：App运行一段时间后卡顿

**原因**：
1. 没有帧节流
2. 内存泄漏
3. 没有释放资源

**解决方案**：
```kotlin
// 帧节流 + 及时释放
fun analyze(image: ImageProxy) {
    if (System.currentTimeMillis() - lastProcessTime < 33) {
        image.close()
        return
    }
    lastProcessTime = System.currentTimeMillis()
    
    detector.process(inputImage)
        .addOnCompleteListener { image.close() }
}
```

### 坑4：异步回调线程问题

**现象**：在回调中更新UI崩溃

**原因**：回调不在UI线程

**解决方案**：
```kotlin
detector.process(inputImage)
    .addOnSuccessListener { faces ->
        // 切换到UI线程
        overlayView.post {
            overlayView.drawResults(faces)
        }
    }
```

## 三、性能监控

```kotlin
class PerformanceMonitor {
    private val inferenceTimes = mutableListOf<Long>()
    
    fun recordTime(ms: Long) {
        inferenceTimes.add(ms)
        if (inferenceTimes.size > 100) inferenceTimes.removeFirst()
    }
    
    fun getStats(): String {
        val avg = inferenceTimes.average().toInt()
        val p90 = inferenceTimes.sorted()[(inferenceTimes.size * 0.9).toInt()]
        return "Avg: $avg ms, P90: $p90 ms"
    }
}
```