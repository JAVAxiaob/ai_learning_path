# NCNN Android YOLOv5 端侧部署解析

> 位置: 04-android-ai/ncnn-android-yolov5/
> 简历推荐: 4星 | 岗位: 端智能/Android AI工程师

---

## 一、项目结构

```
ncnn-android-yolov5/
├── app/src/main/jni/                   NDK C++核心
│   ├── CMakeLists.txt                    链接ncnn + vulkan + jnigraphics
│   ├── yolov5ncnn.cpp                    前处理/推理/后处理 (核心!)
│   ├── ndkcamera.cpp                     NDK Camera2 C++取帧
│   └── yolo_jni.cpp                      Java ↔ C++ JNI桥接
├── app/src/main/assets/
│   ├── yolov5s.param                   NCNN模型结构文本
│   └── yolov5s.bin                     NCNN权重二进制
└── app/src/main/java/.../
    ├── MainActivity.kt                 CameraX 预览SurfaceView
    └── YoloV5Ncnn.kt                   JNI接口: init/detect/close
```

## 二、推理三阶段核心C++代码

```cpp
// yolov5ncnn.cpp (代码脉络,面试要能说出来!)
int YoloV5Ncnn::detect(const cv::Mat& rgb, std::vector<Object>& objects) {
    // ======= Stage 1: 前处理 LetterBox + Normalize + HWC2CHW =======
    const int target_size = 640;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, target_size, target_size);
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};  // YOLOv5归一化 /255
    in.substract_mean_normalize(0, norm_vals);

    // ======= Stage 2: NCNN 推理 (CPU/Vulkan自动切换) =======
    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);            // 轻量模式: 用完的Blob立即释放,内存÷2
    ex.set_num_threads(4);              // CPU线程=大核数
    ex.set_vulkan_compute(use_gpu);     // Vulkan GPU开关
    ex.input("images", in);
    ncnn::Mat out;
    ex.extract("output", out);          // 输出 [25200, 85] = xywh + obj + 80类

    // ======= Stage 3: 后处理 NMS非极大值抑制 =======
    std::vector<Object> proposals;
    for (int i=0; i<out.h; i++) {
        const float* row = out.row(i);
        float objness = row[4];
        if (objness < prob_threshold) continue;   // 置信度过滤
        int label = max_arg(row+5, 80);
        float score = objness * row[5+label];
        if (score < prob_threshold) continue;
        // 还原坐标到原图 (反LetterBox padding)
        Object obj; decode_bbox(row, ...); obj.rect = ...;
        proposals.push_back(obj);
    }
    // NMS: IoU>0.45的重复框删掉只留最高的
    nms_sorted_bboxes(proposals, objects, nms_threshold);
    return 0;
}
```

## 三、NCNN性能优化6招

| 手段 | 命令/做法 | 效果 |
|-----|---------|-----|
| **INT8对称量化** | 工具: ncnn/build/tools/quantize + 1000张校准图 | 模型大小÷4,速度×2~4,精度↓<1% |
| **FP16半精度** | `ncnnoptimize yolov5s.param yolov5s.bin fp16.param fp16.bin 65536` | 模型÷2,速度×1.5~2 (ARMv8.2+FP16指令) |
| **算子融合** | ncnnoptimize自动: Conv+BN+ReLU/Concat+Permute融合 | 临时内存减少,快10~20% |
| **Winograd卷积** | Conv3×3自动F(6,3)算法 | 卷积FLOPs理论÷4 |
| **Vulkan GPU** | `ex.set_vulkan_compute(true)` | GPU加速2~10×, 大模型效果明显 |
| **LightMode内存复用** | `set_light_mode(true)` | 运行内存峰值÷2~3 |

## 四、简历黄金句式

| 写法 |
|-----|
| 「NCNN+Vulkan落地实时安全帽检测：YOLOv5m INT8量化+Winograd+算子融合，骁龙8 Gen2 1080P@87fps，模型28MB→7MB，工地实测准确率97.2%」 |
| 「端侧模型优化管线：结构化剪枝80%通道+INT8量化+权重Huffman编码，6个端侧模型总大小286MB→67MB，APK下载转化率+18%」 |
| 「NDK Camera2 C++全链路取帧→预处理→NCNN推理→后处理→JNI Java回调渲染，零拷贝AHardwareBuffer，端到端延迟<11ms」 |

## 五、面试题

**Q NCNN vs TFLite vs MNN 选型依据？**
> A: TFLite优先(Google维护+文档全+XNNPACK CPU快)，80%常规场景足够。NCNN选在：① 不依赖Google服务(GMS)/国内安卓环境 ② Vulkan GPU极致性能挖掘比TFLite成熟 ③ 包体要求极小(2MB以内)。MNN适合阿里技术栈+端侧微调训练一体化需求。

**Q INT8量化校准数据集要多少张？选什么样的图？**
> A: 500~2000张**真实场景分布**的图(不要用纯ImageNet随机图!)，覆盖各种角度/光照/遮挡/类别，量少不准、量多校准慢。校准目标是统计出每层激活的min/max→KL散度找最优scale让量化信息损失最小。

**Q 端侧App上线后AI崩溃怎么排查？**
> A: ① 崩溃日志 + 设备型号/CPU架构(ARMv7/v8/ARMv9) + GPU型号 ② 特定设备开CPU fallback路径绕过Vulkan bug机型 ③ 缩小模型 + 降低max_detector_num ④ A/B开关 新模型先灰度1%看崩溃率 ⑤ 采集异常输入样本回归到校准集重新量化。