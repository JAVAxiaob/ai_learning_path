# 04-D - 面试题库：Android与AI结合题 (35题)

> 核心：在Android端部署和运行AI模型，你的第二主战场！

---

## 第一部分：TensorFlow Lite基础 (15题)

### 难度: 简单

**1. 在Android App中集成TensorFlow Lite做图像分类（完整Kotlin代码）**
``kotlin
// build.gradle 添加依赖:
// implementation 'org.tensorflow:tensorflow-lite:2.15.0'
// implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
// implementation 'org.tensorflow:tensorflow-lite-gpu:2.15.0' // GPU加速可选

// 1. 把模型文件(mobilenet_v1.tflite)放入 app/src/main/assets/
//    标签文件(labels.txt)也放入assets

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteImageClassifier(private val context: Context) {
    // TFLite解释器（整个App只用一个实例，避免重复加载模型）
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    
    // 模型输入配置
    private val inputImageWidth = 224
    private val inputImageHeight = 224
    private val inputImageChannels = 3
    
    // ImageNet预训练模型的归一化参数
    private val mean = floatArrayOf(127.5f, 127.5f, 127.5f)  // 255/2
    private val std = floatArrayOf(127.5f, 127.5f, 127.5f)
    
    init {
        try {
            // 1. 加载模型文件
            val tfliteModel = loadModelFile("mobilenet_v1.tflite")
            // 2. 配置解释器选项
            val options = Interpreter.Options()
                .setNumThreads(4)           // CPU线程数
                .setUseNNAPI(true)          // 使用Android NNAPI加速(>=API 27)
                .addDelegate(GpuDelegate()) // GPU加速(可选, 需GPU库)
            interpreter = Interpreter(tfliteModel, options)
            // 3. 加载标签文件
            labels = FileUtil.loadLabels(context, "labels.txt")
            println("模型加载成功, 输入形状: ")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    // 从assets加载模型文件为MappedByteBuffer
    private fun loadModelFile(modelFilename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    // 核心推理方法: 输入Bitmap -> 输出分类结果
    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        if (interpreter == null) return emptyList()
        
        // 1. 创建输入Tensor容器
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // 2. 图像预处理Pipeline: resize -> 归一化
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(mean, std))  // (pixel - mean) / std, 结果范围[-1, 1]
            .build()
        
        val processedImage = imageProcessor.process(tensorImage)
        
        // 3. 创建输出容器: [1, 1001] (MobileNet输出1001类概率)
        val probabilityBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, labels.size), DataType.FLOAT32
        )
        
        // 4. 执行推理 (核心一步！耗时: 几十到几百毫秒)
        interpreter?.run(processedImage.buffer, probabilityBuffer.buffer)
        
        // 5. 后处理: 应用softmax -> 映射到标签 -> 取Top-K
        val probabilityProcessor = TensorProcessor.Builder()
            .add(NormalizeOp(0.0f, 1.0f))  // 这里简化, MobileNet输出已是概率
            .build()
        
        val labeledProbability = TensorLabel(
            labels,
            probabilityProcessor.process(probabilityBuffer)
        ).mapWithFloatValue
        
        // 6. 取Top-5最可能的类别
        return labeledProbability.entries
            .sortedByDescending { it.value }
            .take(5)
            .map { it.key to it.value }
    }
    
    // 释放资源 (Activity onDestroy时调用)
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

// ============ 在Activity/Fragment中使用 ============
class ClassifyActivity : AppCompatActivity() {
    private lateinit var classifier: TFLiteImageClassifier
    private lateinit var binding: ActivityClassifyBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityClassifyBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        classifier = TFLiteImageClassifier(this)
        
        // 点击按钮: 从相机/相册获取图片后推理
        binding.btnClassify.setOnClickListener {
            val bitmap = getCurrentCameraFrame()  // 需要你实现: 相机预览或选择图片
            
            // 在后台线程执行推理(避免阻塞UI)
            lifecycleScope.launch(Dispatchers.Default) {
                val startTime = System.currentTimeMillis()
                val results = classifier.classify(bitmap)
                val latency = System.currentTimeMillis() - startTime
                
                // 切回主线程更新UI
                withContext(Dispatchers.Main) {
                    binding.tvLatency.text = "推理耗时: ms"
                    binding.tvResult.text = results.joinToString("\n") { 
                        "% - " 
                    }
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}
``

**2. TFLite vs ML Kit vs 云端推理，如何选型？**
- TFLite: 完全端侧运行，离线可用，隐私好，模型自选
  - 适用场景: 实时图像处理、离线功能、敏感数据
- ML Kit (Google): Google提供的封装API，开箱即用，模型已内置
  - 适用场景: 通用场景(人脸/OCR/条码/地标/文本识别)，不想自己折腾模型
- 云端推理: Java后端部署大模型，Android只负责API调用
  - 适用场景: 复杂任务(LLM问答、语音识别)、模型更新频繁
- 综合方案: 端云协同 - 轻模型TFLite快速响应，复杂请求云端处理

**3. Interpreter.Options的关键配置**
``kotlin
val options = Interpreter.Options().apply {
    setNumThreads(4)           // CPU线程数: 建议2-4，太多反而因调度开销变慢
    setUseNNAPI(true)          // Android NNAPI: 自动调用DSP/GPU/NPU (>=API 27)
    // setAllowFp16PrecisionForFp32(true)  // Float16代替Float32, 精度稍降,速度↑
    // addDelegate(XnnpackDelegate())      // XNNPACK: 优化的CPU推理后端
    // addDelegate(GpuDelegate())          // GPU推理: 精度兼容的模型才用
}
``

**4. 如何在Android中管理模型文件？**
- 方案一: 打包在 assets 目录
  - 优点: 开箱即用
  - 缺点: APK体积增大(模型通常几MB到几十MB)
- 方案二: 首次启动后从服务器下载(推荐)
  - 优点: APK小、可动态更新模型
  - 实现: WorkManager + DownloadManager
  - 注意: 校验文件完整性(MD5)，避免损坏的模型导致崩溃
- 模型更新策略: 检查远端版本号 → 差异下载 → 原子替换

**5. 模型输入/输出的维度格式**
- 图像输入常见格式: [batch, height, width, channels] = [1, 224, 224, 3] (NHWC)
- 注意: PyTorch是NCHW，TFLite是NHWC，转换时容易出错！
- 图像预处理的三要素:
  1. Resize到目标尺寸
  2. 归一化 (除以255, 或 (pixel - mean) / std)
  3. 数据类型转换 (UINT8 → FLOAT32, 或保持INT8量化模型)
- 检查方法: 用Python跑一张图，记录中间值，和Java/Kotlin结果对比，必须完全一致

### 难度: 中等

**6. 实现量化模型推理（速度更快、体积更小）**
``kotlin
// 量化模型: Float32 → Int8, 体积↓75%, 速度↑2-4x, 精度↓1-2%
// 注意: 量化模型的输入/输出可能是UINT8而非FLOAT32

class QuantizedClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    
    init {
        val options = Interpreter.Options().setNumThreads(4)
        // 量化模型无需GPU delegate (GPU支持的操作有限)
        interpreter = Interpreter(loadModelFile("mobilenet_quant.tflite"), options)
    }
    
    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        // 1. 准备输入 (UINT8类型, 形状 [1, 224, 224, 3])
        val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3).order(ByteOrder.nativeOrder())
        
        // 2. 直接从Bitmap填像素值 (0-255，量化模型不需要浮点归一化！)
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = bitmap.getPixel(x, y)
                inputBuffer.put(((pixel shr 16) and 0xFF).toByte())  // R
                inputBuffer.put(((pixel shr 8) and 0xFF).toByte())   // G
                inputBuffer.put((pixel and 0xFF).toByte())            // B
            }
        }
        inputBuffer.rewind()
        
        // 3. 准备输出 (UINT8 -> 需要反量化为概率)
        val outputBuffer = Array(1) { ByteArray(1001) }
        
        // 4. 推理
        interpreter?.run(inputBuffer, outputBuffer)
        
        // 5. 反量化 (从 UINT8 转回 float概率)
        // 从模型metadata获取zero_point和scale (或者硬编码已知值)
        val outputTensor = interpreter!!.getOutputTensor(0)
        val scale = outputTensor.quantizationParams().scale
        val zeroPoint = outputTensor.quantizationParams().zeroPoint.toInt()
        
        val probabilities = FloatArray(1001) { i ->
            val quantizedValue = outputBuffer[0][i].toInt() and 0xFF
            (quantizedValue - zeroPoint) * scale  // 反量化公式
        }
        
        // 6. 取Top-K
        return labels.indices
            .sortedByDescending { probabilities[it] }
            .take(5)
            .map { labels[it] to probabilities[it] }
    }
}
``

**7. 使用ML Kit做文字识别(OCR)**
``kotlin
// build.gradle:
// implementation 'com.google.android.gms:play-services-mlkit-text-recognition:19.0.0'
// 或者使用Chinese模型:
// implementation 'com.google.android.gms:play-services-mlkit-text-recognition-chinese:16.0.0'

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions

class MLKitOCR(private val context: Context) {
    // 获取中文OCR识别器实例 (单例)
    private val recognizer = TextRecognition.getClient(
        ChineseTextRecognizerOptions.Builder().build()
    )
    
    // 从Bitmap识别文字
    fun recognizeText(bitmap: Bitmap, onResult: (List<String>) -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)  // rotation: 0/90/180/270
        
        recognizer.process(image)
            .addOnSuccessListener { resultText ->
                // 解析识别结果 (包含文本、位置、置信度)
                val lines = mutableListOf<String>()
                
                for (block in resultText.textBlocks) {
                    for (line in block.lines) {
                        val lineText = line.text
                        // line.confidence: 置信度 (0-1)
                        // line.cornerPoints: 文字框的四个角坐标, 可用于UI高亮
                        // line.elements: 每个元素(通常是一个字符或单词)
                        lines.add(lineText)
                    }
                }
                
                onResult(lines)
            }
            .addOnFailureListener { e ->
                e.printStackTrace()
                onResult(emptyList())
            }
    }
    
    // 关闭资源
    fun close() {
        recognizer.close()
    }
}

// 实际场景应用: 证件识别(身份证/驾照)、购物小票识别、翻译App、笔记扫描
// 进阶技巧:
// - 用CameraX获取实时相机帧 (分析用例: ImageAnalysis.Analyzer)
// - 裁剪ROI区域(如只识别身份证文字区域), 提升速度和准确率
// - 对识别结果进行后处理: 正则表达式提取特定信息(身份证号/手机号)
``

**8. 使用CameraX实时获取相机帧 → 实时AI推理**
``kotlin
// build.gradle:
// implementation 'androidx.camera:camera-core:1.3.0'
// implementation 'androidx.camera:camera-camera2:1.3.0'
// implementation 'androidx.camera:camera-lifecycle:1.3.0'
// implementation 'androidx.camera:camera-view:1.3.0'

class RealTimeClassifyActivity : AppCompatActivity() {
    private lateinit var classifier: TFLiteImageClassifier
    private var lastClassifyTime = 0L
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        classifier = TFLiteImageClassifier(this)
        startCamera()
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // 1. 预览用例: 在PreviewView显示相机画面
            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
            
            // 2. 图像分析用例: 每一帧回调analyze()方法
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        // 节流: 每300ms最多推理一次(避免帧率过高)
                        val now = System.currentTimeMillis()
                        if (now - lastClassifyTime < 300) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        lastClassifyTime = now
                        
                        // ImageProxy → Bitmap
                        val bitmap = imageProxy.toBitmap()
                        
                        // 执行推理
                        val results = classifier.classify(bitmap)
                        
                        // 更新UI
                        runOnUiThread { updateUI(results) }
                        
                        imageProxy.close()  // 必须关闭, 否则阻塞后续帧
                    }
                }
            
            // 绑定到生命周期
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview, imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }
    
    // 注意事项:
    // - 推理耗时必须远小于帧间隔(通常16-33ms), 否则需要降低推理频率或用更轻量模型
    // - CameraX的image是YUV_420_888格式, 需要转RGB才能推理(上面的toBitmap已处理)
    // - 避免在UI线程做推理, 但CameraX默认在提供的executor执行, 建议用后台executor
    // - 手机竖屏时, 相机帧是横的(270°旋转), 需要旋转后再推理
}
``

**9. Android端AI推理性能优化要点**
``kotlin
// ============ 性能优化 Checklist ============

class PerformanceOptimizer {
    
    fun optimize1_ThreadConfiguration() {
        // 1. 线程数: 不是越多越好, 2-4线程通常最佳
        val options = Interpreter.Options()
        options.setNumThreads(Runtime.getRuntime().availableProcessors() / 2)
        // 因为CPU还有其他线程在跑(UI,系统), 预留一些核心
    }
    
    fun optimize2_ModelChoice() {
        // 2. 模型选择: 优先量化模型
        // Float32 (30MB) -> INT8量化 (7.5MB), 速度 200ms -> 80ms
        // 还可以选更小的架构: MobileNetV3-Small比ResNet50小10倍,快5倍
    }
    
    fun optimize3_BitmapEfficiency() {
        // 3. Bitmap优化: 避免不必要的拷贝
        //    直接在ByteBuffer中填充像素, 不经过ARGB_8888中转
        //    对于实时相机: 直接操作YUV数据而非转Bitmap
    }
    
    fun optimize4_InferenceFrequency() {
        // 4. 推理频率节流: 实时场景不需要每一帧都推理
        //    每秒3-5帧推理足够, 其余帧复用上次结果
    }
    
    fun optimize5_ModelWarmup() {
        // 5. 模型预热: 应用启动时跑一张空图
        //    第一次推理会很慢(几十ms额外开销), 因为JIT编译/资源加载
        val dummy = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        interpreter?.run(dummy.buffer, TensorBuffer.createFixedSize(intArrayOf(1, 1001), DataType.FLOAT32).buffer)
        // 预热后, 后续推理稳定
    }
    
    fun benchmarkModel() {
        // ============ 性能基准测试方法 ============
        // 测试多次取平均, 排除JIT编译导致的首次异常值
        val times = mutableListOf<Long>()
        for (i in 0 until 20) {
            val start = System.nanoTime()
            classifier.classify(testBitmap)
            if (i >= 5) times.add(System.nanoTime() - start)  // 忽略前5次预热
        }
        val avg = times.average() / 1_000_000  // 转ms
        val p99 = times.sorted()[(times.size * 0.99).toInt()] / 1_000_000
        Log.d("Benchmark", "平均: ms, P99: ms")
    }
}
// 最终目标: 推理耗时 < 100ms (普通手机), 高端手机 < 50ms
// 如果超过300ms, 用户会感觉到明显延迟
``

**10. 实现端云协同推理（本地初筛 + 云端精判）**
``kotlin
// 场景: 拍照搜商品. 手机端用轻模型快速做粗分类+提取特征, 
//       云端用大模型做精准识别 + 搜索商品数据库.

class HybridInferenceManager(val context: Context) {
    private val localClassifier = TFLiteImageClassifier(context)  // 端侧: MobileNet
    private val cloudApi = CloudInferenceService()  // 云端: ResNet + 商品搜索
    
    data class InferenceResult(
        val source: String,         // "LOCAL" or "CLOUD"
        val category: String,
        val confidence: Float,
        val cloudDetail: String? = null
    )
    
    fun hybridClassify(bitmap: Bitmap, onResult: (InferenceResult) -> Unit) {
        // Step 1: 本地快速推理 (<100ms), 给用户即时反馈
        val localResult = localClassifier.classify(bitmap).firstOrNull()
        val localCategory = localResult?.first ?: "unknown"
        val localConfidence = localResult?.second ?: 0f
        
        // 立即返回本地结果 (提升用户体验, 不卡UI)
        onResult(InferenceResult("LOCAL", localCategory, localConfidence))
        
        // Step 2: 根据置信度决定是否需要云端增强
        if (localConfidence < 0.7f) {
            // 置信度低: 异步请求云端更精确的识别
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val cloudResult = cloudApi.remoteClassify(bitmap)
                    // 云端结果更可信, 更新UI
                    withContext(Dispatchers.Main) {
                        onResult(cloudResult)
                    }
                } catch (e: Exception) {
                    // 网络异常: 继续使用本地结果
                }
            }
        }
        // 置信度足够高: 直接使用本地结果, 节约云端成本
    }
}

// 云端推理API实现
class CloudInferenceService {
    private val client = OkHttpClient()
    
    suspend fun remoteClassify(bitmap: Bitmap): TFLiteImageClassifier.InferenceResult {
        // 1. 图片压缩 (减少上传流量)
        val bytes = ByteArrayOutputStream().apply {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, this)
        }.toByteArray()
        
        // 2. 上传到Java后端
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "photo.jpg", RequestBody.create(bytes, "image/jpeg".toMediaType()))
            .build()
        
        val request = Request.Builder()
            .url("https://your-api-server.com/api/classify")
            .post(requestBody)
            .build()
        
        // 3. 解析返回结果
        val response = client.newCall(request).await()
        val json = JSONObject(response.body?.string() ?: "{}")
        return TFLiteImageClassifier.InferenceResult(
            source = "CLOUD",
            category = json.getString("category"),
            confidence = json.getDouble("confidence").toFloat()
        )
    }
}
// 端云协同优点: 
// - 延迟低 (本地100ms vs 云端1000ms)
// - 成本低 (高置信度场景不走云端)
// - 隐私好 (图片不上传高置信度场景)
// - 准确率可接受 (低置信度场景有云端兜底)
``

### 难度: 困难

**11. 实现端侧物体检测（识别多个物体+画框）**
``kotlin
// 模型: MobileNet SSD, YOLOv5-tflite, EfficientDet-Lite
// 输入: [1, 320, 320, 3] 或 [1, 640, 640, 3]
// 输出: detections = [boxes, classes, scores, num_detections]

import org.tensorflow.lite.Interpreter
import android.graphics.RectF
import android.graphics.Canvas
import android.graphics.Paint

class ObjectDetector(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val inputSize = 320
    private val labels = listOf("person", "bicycle", "car", /* ...共80类 */)
    private val confidenceThreshold = 0.5f
    
    init {
        val options = Interpreter.Options().setNumThreads(4)
        interpreter = Interpreter(loadModelFile("ssd_mobilenet_v1.tflite"), options)
    }
    
    data class Detection(
        val boundingBox: RectF,  // 物体框坐标 (相对比例 0-1)
        val classIndex: Int,
        val className: String,
        val confidence: Float
    )
    
    fun detect(bitmap: Bitmap): List<Detection> {
        // 1. 预处理: resize到模型输入大小
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // 2. 创建输入 Tensor
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resized.getPixel(x, y)
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        
        // 3. 准备输出容器: SSD模型通常输出4个Tensor
        // outputLocations: [1, 10, 4] - 10个物体, 每个4个坐标 (y1,x1,y2,x2)
        // outputClasses: [1, 10] - 10个物体的类别索引
        // outputScores: [1, 10] - 10个物体的置信度
        // numDetections: [1] - 实际检测到的物体数
        val outputMap = HashMap<Int, Any>()
        val outputLocations = Array(1) { Array(10) { FloatArray(4) } }
        val outputClasses = Array(1) { FloatArray(10) }
        val outputScores = Array(1) { FloatArray(10) }
        val numDetections = FloatArray(1)
        outputMap[0] = outputLocations
        outputMap[1] = outputClasses
        outputMap[2] = outputScores
        outputMap[3] = numDetections
        
        // 4. 推理
        interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        
        // 5. 解析结果
        val numDet = numDetections[0].toInt()
        val results = mutableListOf<Detection>()
        
        for (i in 0 until numDet) {
            val score = outputScores[0][i]
            if (score < confidenceThreshold) continue  // 过滤低置信度
            
            val classIdx = outputClasses[0][i].toInt()
            val box = outputLocations[0][i]  // [y1, x1, y2, x2], 相对坐标
            
            // 坐标转换: 模型输出是相对比例(0-1), 需要映射回原图尺寸
            val rectF = RectF(
                box[1] * bitmap.width,  // x1 (left)
                box[0] * bitmap.height, // y1 (top)
                box[3] * bitmap.width,  // x2 (right)
                box[2] * bitmap.height  // y2 (bottom)
            )
            
            results.add(
                Detection(
                    boundingBox = rectF,
                    classIndex = classIdx,
                    className = labels[classIdx],
                    confidence = score
                )
            )
        }
        
        // 6. NMS非极大值抑制 (去除重叠框, 简化: 这里只按置信度排序取Top-N)
        return results.sortedByDescending { it.confidence }.take(5)
    }
    
    // 绘制检测框到Canvas (在自定义View的onDraw调用)
    fun drawDetections(canvas: Canvas, detections: List<Detection>) {
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 8f
            textSize = 48f
        }
        for (d in detections) {
            canvas.drawRect(d.boundingBox, paint)
            canvas.drawText(" %",
                d.boundingBox.left, d.boundingBox.top - 10, paint)
        }
    }
}
// 应用场景: 实时人流统计、安全报警(检测危险物品)、仓储盘点、自动驾驶辅助
// 进阶: YOLOv5/YOLOv8导出为TFLite (NMS需特殊处理)
// 手机端目标检测性能: 中高端手机 50-150ms/帧, 可实现10-20 FPS实时检测
``

**12. 使用NDK直接调用C++模型库**
``cpp
// ============ C++ 端 (JNI层) ============
// Android NDK可以直接调用ONNX Runtime C++ API或TFLite C API
// 优点: 1) 避免Java层GC  2) 与现有C++代码库集成  3) 性能更稳定

#include <jni.h>
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapp_NativeClassifier_nativeClassify(
    JNIEnv* env, jobject, jfloatArray jInput) {
    
    // 1. 获取输入数据
    jfloat* input = env->GetFloatArrayElements(jInput, NULL);
    int inputSize = env->GetArrayLength(jInput);
    
    // 2. 推理 (此处使用已初始化的interpreter)
    // 实际代码: 复制input到input_tensor, interpreter->Invoke(), 读取output
    std::vector<float> output = runTFLite(input, inputSize);
    
    // 3. 返回结果到Java/Kotlin层
    jfloatArray result = env->NewFloatArray(output.size());
    env->SetFloatArrayRegion(result, 0, output.size(), output.data());
    env->ReleaseFloatArrayElements(jInput, input, 0);
    
    return result;
}
``

**13. MVVM架构 + AI推理的数据流设计**
``kotlin
// 在现代Android应用中, AI推理应该融入到MVVM架构中

// 1. Model层: 封装推理逻辑
data class ClassifyResult(val labels: List<String>, val latencyMs: Long, val source: String)

class TFLiteModel(val context: Context) {
    private val interpreter = Interpreter(loadModel())
    
    suspend fun classify(bitmap: Bitmap): ClassifyResult {
        // 实际推理
        val start = System.currentTimeMillis()
        val results = doClassifyInternal(bitmap)
        return ClassifyResult(results, System.currentTimeMillis() - start, "TFLITE")
    }
}

// 2. ViewModel层: 管理UI状态 + 业务逻辑
@HiltViewModel
class ClassifyViewModel @Inject constructor(
    private val model: TFLiteModel
) : ViewModel() {
    // UI状态
    private val _uiState = MutableStateFlow(ClassifyUiState())
    val uiState: StateFlow<ClassifyUiState> = _uiState.asStateFlow()
    
    fun onImageSelected(bitmap: Bitmap) {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            val result = model.classify(bitmap)
            _uiState.update { it.copy(isLoading = false, result = result) }
        }
    }
}

data class ClassifyUiState(
    val isLoading: Boolean = false,
    val result: ClassifyResult? = null
)

// 3. View层 (Activity/Fragment): 只负责UI渲染
@AndroidEntryPoint
class ClassifyFragment : Fragment() {
    private val viewModel: ClassifyViewModel by viewModels()
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        lifecycleScope.launch {
            viewLifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    // 根据state更新UI: loading动画/显示结果/错误提示
                    renderState(state)
                }
            }
        }
    }
}
// 设计优点: 关注点分离 → 可测试(ViewModel单元测试不需要UI)、可维护、可扩展
// 数据流: 用户操作 → ViewModel调用 → Model推理 → 返回结果 → 更新UI State → 渲染
``

### 难度: 中等

**14. AI推理的内存管理和对象复用**
``kotlin
// 在Android中, 频繁推理会产生大量临时对象, 导致GC卡顿.
// 关键策略: 预分配+复用对象

class OptimizedClassifier(context: Context) {
    private val interpreter: Interpreter
    
    // 预分配输入输出Buffer, 每次推理复用 (不重新创建)
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        .order(ByteOrder.nativeOrder())
    private val outputBuffer: TensorBuffer = TensorBuffer.createFixedSize(
        intArrayOf(1, 1001), DataType.FLOAT32
    )
    
    // 复用Bitmap配置 (不每次创建新Bitmap)
    private val resizedBitmap: Bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
    private val canvas: Canvas = Canvas(resizedBitmap)
    private val paint: Paint = Paint(Paint.FILTER_BITMAP_FLAG)
    
    fun classifyFast(inputBitmap: Bitmap): List<Pair<String, Float>> {
        // 1. 复用Canvas把输入Bitmap resize到固定大小
        canvas.drawBitmap(
            inputBitmap,
            Rect(0, 0, inputBitmap.width, inputBitmap.height),
            RectF(0, 0, 224, 224),
            paint
        )
        
        // 2. 复用inputBuffer (重置position, 不重新分配)
        inputBuffer.rewind()
        fillPixelsToBuffer(resizedBitmap, inputBuffer)
        
        // 3. 推理 (复用outputBuffer)
        interpreter.run(inputBuffer, outputBuffer.buffer.rewind())
        
        // 4. 解析结果
        return parseOutput(outputBuffer.floatArray)
    }
    
    // Activity切换横竖屏等配置变化时, 用ViewModel保留模型实例
    // 避免重复加载模型 (耗时几十到几百ms, 用户会明显感知卡顿)
}

// ViewModel保留模型实例:
@HiltViewModel
class SharedModelViewModel @Inject constructor(
    val classifier: TFLiteImageClassifier  // 配置变化时不会被销毁
) : ViewModel() {
    // 不需要额外代码, 默认的ViewModel生命周期就保证了这一点
}
``

**15. 端侧AI功能的离线/在线切换**
``kotlin
class OfflineFirstInferenceManager(
    private val localClassifier: TFLiteImageClassifier,
    private val remoteClassifier: CloudInferenceService,
    private val networkMonitor: NetworkMonitor
) {
    data class Result(val text: String, val isOffline: Boolean)
    
    suspend fun classify(bitmap: Bitmap): Result {
        return try {
            if (networkMonitor.isOnline()) {
                // 网络可用: 并行尝试, 谁先回使用谁 (带超时)
                withTimeoutOrNull(1500) {  // 1.5秒内没回来就用离线
                    val deferredRemote = async { remoteClassifier.classify(bitmap) }
                    val deferredLocal = async { localClassifier.classify(bitmap) }
                    
                    // 优先云端, 但有超时兜底
                    try {
                        val remote = deferredRemote.await()
                        Result(remote, false)
                    } catch (e: Exception) {
                        Result(deferredLocal.await(), true)
                    }
                } ?: Result(localClassifier.classify(bitmap), true)
            } else {
                // 离线: 直接用端侧模型
                Result(localClassifier.classify(bitmap), true)
            }
        } catch (e: Exception) {
            // 任何异常, 降级到本地
            Result(localClassifier.classify(bitmap), true)
        }
    }
}
// 产品价值: 给用户稳定的体验, 同时在网络好的时候提供更强大的能力
// 类似Google Photos: 离线也能搜索(端侧索引), 在线时用更强大的云端能力
``

---

## 第二部分：ML Kit + 端云协同 + 高级应用 (20题)

### 难度: 简单-中等（16-25题，每题含关键代码片段）

**16. ML Kit人脸检测+表情识别**
- 实现要点: FaceDetection.getClient() -> 检测到人脸 -> 提取landmark
- 结合点: 相机实时检测、美肤/贴纸应用、人脸登录

**17. ML Kit语音识别**
- 实现要点: SpeechRecognizer + 实时转文字
- 结合点: 语音助手、语音输入框、会议记录

**18. 离线翻译 (ML Kit Translation)**
- 实现要点: Translation.getClient(TranslateRemoteModel) -> 下载语言包
- 结合点: 旅行App、国际化应用

**19. Android端智能图片裁剪 (基于内容的裁剪)**
- 实现要点: 检测主体+计算最佳裁剪框
- 结合点: 相册编辑、社交App图片上传

**20. 手写数字识别 (CNN + TFLite)**
- 实现要点: Canvas画数字 -> 转28x28灰度图 -> LeNet推理
- 结合点: 笔记App、表单输入

**21. 文档扫描 (边缘检测+透视变换+OCR)**
- 实现要点: OpenCV or ML Kit + 手动实现透视变换
- 结合点: 办公App、票据识别

**22. Android端音频分类 (鸟鸣/枪声/婴儿哭声)**
- 实现要点: 音频特征(MFCC) -> CNN推理
- 结合点: 智能音箱、家庭安防、健康监测

**23. 实时风格迁移 (把相机变成油画/漫画风格)**
- 实现要点: 轻量级GAN模型 -> 实时推理
- 结合点: 相机滤镜、短视频应用

**24. 推荐系统的端侧排序 (Re-rank on-device)**
- 实现要点: 接收粗排结果 -> 端侧DNN精排 -> 返回排序后列表
- 结合点: 内容推荐、电商App、个性化首页

**25. 隐私计算 (数据不出端)**
- 实现要点: 在端侧完成特征提取和推理, 仅上传匿名结果
- 结合点: 健康类应用、用户画像、联邦学习客户端

### 难度: 中等-困难（26-35题，侧重架构设计和性能优化）

**26. 实现推理结果的缓存策略**
- 相同图片/输入 → 直接返回缓存结果
- 图片哈希 + LRU缓存
- 应用场景: 重复图片分类、OCR结果缓存

**27. 批量推理优化 (处理多张图片)**
- 把多张图片拼入一个batch, 提升GPU利用率
- 应用场景: 相册批量识别、图片集打标签

**28. 实现模型版本管理和热更新**
- WorkManager后台下载新模型 → 重启解释器
- 版本校验+回滚机制
- 应用场景: 应用不升级的情况下更新模型

**29. 端侧异常监测 (推理失败/性能下降)**
- 记录每次推理的延迟、准确率、内存
- 异常上报到监控系统(如Firebase/自研)
- 自动降级策略(模型太大→切更轻模型)

**30. 设计一个端-云混合的推荐系统架构**
- 端侧: 用户行为采集 + 轻量级排序模型
- 云端: 用户画像 + 粗排召回 + 大模型精排
- 数据流: Kafka + 特征平台 + 向量数据库

**31. 实现推理服务的A/B测试框架**
- 多版本模型共存
- 按用户ID/地域分流
- 指标收集+统计显著性
- Android端集成

**32. 端侧推理的能耗优化**
- 在低电量时降低推理频率
- 充电时才做批量任务
- 用WorkManager+约束调度
- 监控推理耗电量

**33. 模型失败案例分析系统**
- 自动记录推理失败(置信度低但人工标注正确的样本)
- 上传到平台用于模型迭代
- 持续学习闭环

**34. 实现一个离线智能助手**
- 端侧小LLM (如TinyLlama-1.1B量化到4bit)
- 结合本地知识库(RAG的向量索引也在端侧)
- 完全离线,响应即时
- 适用于隐私敏感场景

**35. 从0到1: Android端AI项目完整清单**
- 需求: 明确要解决的问题 → 定义成功指标
- 模型选型: 精度 vs 速度 vs 体积 trade-off
- 数据: 数据采集 → 标注 → 测试集
- 模型转换: PyTorch/TF → TFLite/ONNX → 验证精度对齐
- 集成: 依赖配置 → 模型加载 → 推理实现
- 优化: 性能基准测试 → 瓶颈分析 → 量化/多线程/NNAPI
- 测试: 单元测试 + 真机性能测试 + 回归测试(升级模型后测相同输入)
- 发布: 模型打包/下载策略 → 灰度发布 → A/B测试
- 监控: 失败率+延迟+成本监控 → 告警 → 迭代
