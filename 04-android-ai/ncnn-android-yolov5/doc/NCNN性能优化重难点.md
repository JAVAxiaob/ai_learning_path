# NCNN YOLOv5 性能优化重难点分析

> 位置: 04-android-ai/ncnn-android-yolov5/

---

## 一、NCNN性能优化手段

### 1. 量化优化

```bash
# INT8量化命令
ncnnoptimize yolov5s.param yolov5s.bin yolov5s_int8.param yolov5s_int8.bin 65536

# FP16量化
ncnnoptimize yolov5s.param yolov5s.bin yolov5s_fp16.param yolov5s_fp16.bin 65536 1
```

| 量化类型 | 模型大小 | 速度提升 | 精度损失 |
|---------|---------|---------|---------|
| FP32(原始) | 100% | 1x | - |
| FP16 | 50% | 1.5~2x | <0.5% |
| INT8 | 25% | 2~4x | <1% |

### 2. Vulkan GPU加速

```cpp
// 启用Vulkan
ncnn::Extractor ex = Net->create_extractor();
ex.set_vulkan_compute(true);

// 检查是否支持Vulkan
if (ncnn::get_gpu_count() == 0) {
    // Fallback到CPU
    ex.set_vulkan_compute(false);
}
```

### 3. 算子融合优化

```bash
# 自动算子融合
ncnnoptimize input.param input.bin output.param output.bin 0
```

融合规则：
- Conv + BN + ReLU → 单个算子
- Concat + Permute → 合并
- BatchNorm折叠到Conv权重

### 4. Winograd卷积优化

```cpp
// ncnn自动启用Winograd
// Conv3x3自动使用F(6,3)算法
// 理论FLOPs ÷4
```

### 5. LightMode内存复用

```cpp
ex.set_light_mode(true);  
// 用完的Blob立即释放，运行内存峰值 ÷2~3
```

## 二、常见坑点

### 坑1：Vulkan内存分配失败

**现象**：部分低端机型GPU内存不足

**解决方案**：
```cpp
try {
    ex.set_vulkan_compute(true);
} catch (...) {
    ex.set_vulkan_compute(false);  // fallback CPU
}
```

### 坑2：模型加载失败

**现象**：`load_param`或`load_model`返回非0

**原因**：
- 文件路径错误
- .param和.bin版本不匹配
- 模型损坏

**解决方案**：
```cpp
int ret = Net->load_param(assetManager, "yolov5s.param");
if (ret != 0) {
    LOGE("load_param failed");
    return false;
}
```

### 坑3：检测框位置偏移

**现象**：检测框不在物体上

**原因**：LetterBox的padding没有正确还原

**解决方案**：
```cpp
// 后处理时反LetterBox
float scale = std::min((float)target_size/width, (float)target_size/height);
float pad_x = (target_size - width * scale) / 2;
float pad_y = (target_size - height * scale) / 2;

// 还原到原图坐标
float x = (bbox.x - pad_x) / scale;
float y = (bbox.y - pad_y) / scale;
```

### 坑4：JNI线程安全

**现象**：多线程调用崩溃

**解决方案**：
```cpp
// 每个线程独立Extractor
thread_local ncnn::Extractor ex = Net->create_extractor();
```

## 三、性能对比

| 配置 | 骁龙8 Gen2 | 骁龙778G | 麒麟990 |
|-----|-----------|---------|--------|
| CPU单线程 | ~45ms | ~85ms | ~95ms |
| CPU 4线程 | ~18ms | ~35ms | ~40ms |
| Vulkan GPU | ~8ms | ~18ms | ~22ms |
| INT8 + Vulkan | ~4ms | ~10ms | ~14ms |