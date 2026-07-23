# 🔷 技术方向4：Android端侧AI开发

## 4.1 核心技术概述

### 端侧推理框架对比

| 框架 | 厂商 | 优势 | 适用场景 |
|------|------|------|----------|
| **TensorFlow Lite** | Google | 成熟稳定，文档完善 | 图像/视频/文本/音频 |
| **ML Kit** | Google | 开箱即用API | 常见CV/NLP任务 |
| **NCNN** | 腾讯 | 极致性能优化 | 复杂模型如YOLO |
| **ONNX Runtime** | Microsoft | PyTorch无缝 | 跨平台部署 |
| **PyTorch Mobile** | Meta | 训练端一致API | PyTorch模型原生部署 |

### TFLite核心流程

```
Python训练模型
  ↓ 导出
ONNX / SavedModel
  ↓ 转换
tflite_convert
  ↓ Android集成
assets/model.tflite
  ↓ Kotlin推理
Interpreter.run(input, output)
```

**Kotlin TFLite核心代码**：
```kotlin
// 加载模型
val modelBuffer = FileUtil.loadMappedFile(context, "model.tflite")
val options = Interpreter.Options().setNumThreads(4).addDelegate(GpuDelegate())
val interpreter = Interpreter(modelBuffer, options)

// 图像预处理（⚠️与Python训练端transform参数完全一致！）
fun preprocess(bitmap: Bitmap): ByteBuffer {
    val input = ByteBuffer.allocateDirect(4 * 1 * 224 * 224 * 3).order(ByteOrder.nativeOrder())
    val pixels = IntArray(224 * 224)
    bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
    for (p in pixels) {
        // ImageNet标准归一化: (pixel - mean) / std
        input.putFloat(((p shr 16 and 0xFF) - 123.68f) / 58.39f)
        input.putFloat(((p shr 8 and 0xFF) - 116.78f) / 57.12f)
        input.putFloat((p and 0xFF - 103.94f) / 57.38f)
    }
    return input
}

// 推理
val output = Array(1) { FloatArray(1001) }
interpreter.run(inputBuffer, output)

// 取top-3
val top3 = output[0].withIndex().sortedByDescending { it.value }.take(3)
```

### 模型量化

| 方法 | 精度 | 模型大小 | 速度 | 说明 |
|------|------|----------|------|------|
| FP32 | 全精度 | 100% | 1x | 基准 |
| FP16 | 半精度 | ~50% | ~1.5x | GPU友好 |
| **INT8 PTQ** | 训练后量化 | ~25% | **2-4x** | **最常用**，需校准数据 |
| INT8 QAT | 感知量化训练 | ~25% | 2-4x | 精度更高，需重新训练 |

**PyTorch导出量化TFLite**：
```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model = load_trained_model().eval()
# 训练后动态量化（Linear层）
model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# 导出TorchScript
scripted = torch.jit.script(model_int8)
optimized = optimize_for_mobile(scripted)
optimized.save("model_quantized.pt")
# Android依赖: implementation 'org.pytorch:pytorch_android:1.13.1'
```

### 硬件加速代理

```kotlin
val options = Interpreter.Options().apply {
    setNumThreads(4)  // CPU多线程
    // GPU代理: 适合浮点模型，速度3-7x
    addDelegate(GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true)))
    // NNAPI代理: Android系统级神经网络加速
    // addDelegate(NnApiDelegate())
}
val interpreter = Interpreter(modelBuffer, options)
```

**Pixel 7上MobileNet推理耗时对比**：
| 配置 | 耗时 | 加速比 |
|------|------|--------|
| CPU单线程 | 35ms | 1.0x |
| CPU 4线程 | 18ms | 1.9x |
| GPU代理 | 5ms | **7.0x** |
| NNAPI | 6ms | 5.8x |

---

## 4.2 GitHub项目推荐

| 项目名 | 链接 | 核心学习点 | clone命令 |
|--------|------|-----------|-----------|
| tensorflow/examples | github.com/tensorflow/examples | **TFLite Android官方示例** - 分类/检测/分割/风格迁移 | `git clone --depth 1 https://github.com/tensorflow/examples.git` |
| NCNN | github.com/Tencent/ncnn | 腾讯NCNN + Android Demo - 极致性能优化 | `git clone --depth 1 https://github.com/Tencent/ncnn.git` |
| tflite-support | github.com/tensorflow/tflite-support | TFLite Task API - Vision/NLP/Audio高层封装 | `git clone --depth 1 https://github.com/tensorflow/tflite-support.git` |
| MediaPipe | github.com/google/mediapipe | 手势/人脸/姿态识别端侧方案 | `git clone --depth 1 https://github.com/google/mediapipe.git` |
| PaddleOCR | github.com/PaddlePaddle/PaddleOCR | 端侧OCR完整实现 | `git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git` |

---

## 4.3 Kotlin完整示例：CameraX实时图像分类

```kotlin
package com.ai.tflite.classifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * Android依赖:
 *   implementation 'org.tensorflow:tensorflow-lite:2.15.0'
 *   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
 *   implementation 'org.tensorflow:tensorflow-lite-gpu:2.15.0'
 *   implementation "androidx.camera:camera-core:1.3.0"
 *   implementation "androidx.camera:camera-camera2:1.3.0"
 */
class ImageClassifier(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val inputSize: Int = 224
) {
    private val output = Array(1) { FloatArray(labels.size) }
    private val MEAN = floatArrayOf(123.68f, 116.78f, 103.94f)
    private val STD = floatArrayOf(58.39f, 57.12f, 57.38f)

    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (p in pixels) {
            input.putFloat(((p shr 16 and 0xFF) - MEAN[0]) / STD[0])
            input.putFloat(((p shr 8 and 0xFF) - MEAN[1]) / STD[1])
            input.putFloat((p and 0xFF - MEAN[2]) / STD[2])
        }
        input.rewind()
        
        output[0].fill(0f)
        interpreter.run(input, output)
        
        return output[0].withIndex().sortedByDescending { it.value }.take(3).map { labels[it.index] to it.value }
    }
}

class MainActivity : AppCompatActivity() {
    private val REQUEST_CODE = 1001
    private lateinit var classifier: ImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // 1. 加载TFLite模型（放在assets/mobilenet_v2.tflite）
        val modelBuffer = FileUtil.loadMappedFile(this, "mobilenet_v2.tflite")
        val interpreter = Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))
        val labels = FileUtil.loadLabels(this, "imagenet_labels.txt")
        classifier = ImageClassifier(interpreter, labels)
        println("✅ TFLite模型加载成功，类别数: ${labels.size}")
        
        // 2. 摄像头权限
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CODE)
        } else {
            startCamera()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder.surfaceProvider) }
            
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        val bitmap = yuvToBitmap(imageProxy)
                        imageProxy.close()
                        val t0 = System.currentTimeMillis()
                        val results = classifier.classify(bitmap)
                        val elapsed = System.currentTimeMillis() - t0
                        runOnUiThread {
                            tvResult.text = buildString {
                                append("推理: ${elapsed}ms\n\n")
                                results.forEachIndexed { i, (label, conf) ->
                                    append("${i+1}. $label: ${String.format("%.1f%%", conf*100)}\n")
                                }
                            }
                        }
                    }
                }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun yuvToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        val ySize = yBuffer.remaining()
        val nv21 = ByteArray(ySize + uBuffer.remaining() + vBuffer.remaining())
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vBuffer.remaining())
        uBuffer.get(nv21, ySize + vBuffer.remaining(), uBuffer.remaining())
        val out = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
            .compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        return android.graphics.BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    override fun onRequestPermissionsResult(rc: Int, perms: Array<String>, grants: IntArray) {
        super.onRequestPermissionsResult(rc, perms, grants)
        if (rc == REQUEST_CODE && grants[0] == PackageManager.PERMISSION_GRANTED) startCamera()
    }
}
```

---

## 4.4 面试题库

### 📝 理论题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 4.1 | TFLite Interpreter.run()的输入输出格式？为什么用ByteBuffer？ | 简 | ⭐⭐⭐ |
| 4.2 | PTQ训练后量化 vs QAT感知量化训练的区别？各自适用场景？ | 中 | ⭐⭐⭐⭐ |
| 4.3 | GPU代理/NNAPI代理/Hexagon DSP代理在TFLite中如何选择？ | 中 | ⭐⭐⭐ |
| 4.4 | 端侧推理的"训练-推理偏差"有哪些来源？如何检测和修复？ | 中 | ⭐⭐⭐ |
| 4.5 | CameraX的ImageAnalysis中YUV→RGB格式转换的性能优化方法？ | 中 | ⭐⭐⭐ |
| 4.6 | 模型热更新：如何从服务器下载新.tflite并动态替换？ | 中 | ⭐⭐⭐ |
| 4.7 | 端云协同架构：哪些任务适合端侧？哪些适合云端？切换策略？ | 中 | ⭐⭐⭐ |
| 4.8 | Android推理性能优化的N种方法？ | 中 | ⭐⭐⭐⭐ |

### 📱 Kotlin代码题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 4.9 | 写Kotlin函数：Bitmap→TFLite输入ByteBuffer（缩放224+ImageNet归一化） | 简 | ⭐⭐⭐⭐ |
| 4.10 | 写TFLite推理完整流程：加载模型→推理→取top-K | 中 | ⭐⭐⭐⭐ |
| 4.11 | 用ML Kit TextRecognition实现简化版实时OCR | 简 | ⭐⭐⭐ |
| 4.12 | 实现推理耗时统计工具：测量p50/p99延迟 | 简 | ⭐⭐⭐ |
| 4.13 | 端云协同：TFLite本地推理 + OKHttp云端API，根据置信度切换 | 中 | ⭐⭐⭐ |

### 🔧 架构设计题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 4.14 | 设计Android端AI推理框架：支持多模型、模型热更新、硬件加速自动选择、推理缓存 | 难 | ⭐⭐⭐ |
| 4.15 | 设计端云协同图像识别方案：端侧快速分类+云侧细粒度识别 | 难 | ⭐⭐⭐ |

---

> ✅ **方向4（Android端侧AI）学习完成自检清单**：
> - [ ] 能独立完成TFLite图像分类的完整Android项目
> - [ ] 理解INT8量化的流程和精度/大小的权衡
> - [ ] 能写出CameraX实时推理回调
> - [ ] 掌握GPU/NNAPI代理的配置方法