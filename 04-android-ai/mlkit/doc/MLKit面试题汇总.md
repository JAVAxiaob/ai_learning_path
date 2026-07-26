# ML Kit + CameraX 面试题汇总

> 位置: 04-android-ai/mlkit/android/

---

## 一、基础概念

### Q1: ML Kit是什么？有哪些能力？

**A**: ML Kit是Google提供的端侧AI SDK，提供"开箱即用"的预训练模型：
- 视觉类：人脸检测/468点人脸网格/OCR/条码/姿态估计/人像分割
- NLP类：智能回复/翻译/语言识别/实体抽取
- GenAI：端侧Gemini Nano

### Q2: CameraX是什么？相比传统Camera API有什么优势？

**A**: CameraX是Jetpack组件，简化相机开发：
- 生命周期自动管理
- 统一API兼容各厂商
- 内置预览/拍照/分析UseCase
- 自动处理设备差异

### Q3: ImageAnalysis的STRATEGY有哪些？

**A**: 
- `STRATEGY_KEEP_ONLY_LATEST`: 只保留最新帧，丢弃旧帧
- `STRATEGY_BLOCK_PRODUCER`: 阻塞等待处理完再取新帧

---

## 二、集成实践

### Q4: CameraX如何绑定多个UseCase？

**A**: 
```kotlin
cameraProvider.bindToLifecycle(
    this, cameraSelector, preview, imageAnalysis, imageCapture)
```

### Q5: InputImage如何创建？需要注意什么？

**A**: 
```kotlin
// 关键：旋转角度必须正确
val rotationDegrees = imageProxy.imageInfo.rotationDegrees
val inputImage = InputImage.fromMediaImage(image, rotationDegrees)
```

### Q6: 为什么必须调用imageProxy.close()？

**A**: ImageProxy持有底层图像缓冲区，不释放会导致相机帧队列阻塞，最终相机卡死。

---

## 三、性能优化

### Q7: 帧节流的作用是什么？怎么实现？

**A**: 控制AI推理帧率，避免处理不过来导致卡顿发烫。实现：时间戳比较，小于阈值就丢弃。

### Q8: ML Kit支持哪些Delegate？如何选型？

**A**: 
- XNNPACK: 兼容性最好，默认推荐
- GPU: 大模型/高吞吐场景
- NNAPI: 后台低功耗场景

### Q9: 如何避免内存泄漏？

**A**: 
1. Activity销毁时close检测器
2. 每个process完成后close ImageProxy
3. 使用WeakReference避免生命周期问题

---

## 四、坐标转换

### Q10: 检测结果的坐标是什么坐标系？

**A**: 是原始图像的坐标系，需要转换到View坐标系。

### Q11: 坐标转换需要考虑哪些因素？

**A**: 
1. 旋转角度
2. 缩放比例（PreviewView尺寸 vs 图像尺寸）
3. 前置相机镜像翻转

### Q12: 为什么前置相机需要镜像？

**A**: 前置相机采集的图像是水平翻转的，检测结果也是基于翻转后的图像，所以绘制时需要再翻转回来。

---

## 五、错误排查

### Q13: 检测不到物体常见原因？

**A**: 
1. 旋转角度错误（80%的坑）
2. 图像格式不对（YUV vs RGB）
3. 检测器配置错误（FAST模式可能漏检）
4. 光线条件差

### Q14: 检测框位置偏移？

**A**: 
1. 没有进行坐标转换
2. 转换矩阵缺少旋转/缩放/镜像
3. 宽高比计算错误

### Q15: 异步回调崩溃？

**A**: 
1. 回调不在UI线程，直接更新UI
2. View已经销毁但回调还在执行
3. 使用WeakReference保护View引用

---

## 六、进阶问题

### Q16: ML Kit模型可以自定义吗？

**A**: 部分能力支持：
- Image Labeling / Object Detection支持AutoML Vision Edge
- 其他能力（人脸/OCR/分割）只能用官方模型

### Q17: 如何实现多任务同时检测？

**A**: 
1. 创建多个检测器
2. 在Analyzer中依次调用
3. 注意性能影响

### Q18: 离线模式支持吗？

**A**: 大部分能力支持离线，需要提前下载模型：
```kotlin
val modelManager = ModelManager.getInstance()
modelManager.download(model)
    .addOnSuccessListener { /* 下载完成 */ }
```