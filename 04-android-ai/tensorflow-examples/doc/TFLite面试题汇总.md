# TFLite 面试题汇总

> 位置: 04-android-ai/tensorflow-examples/lite/examples/

---

## 一、基础概念

### Q1: TFLite是什么？和TensorFlow有什么关系？

**A**: TensorFlow Lite是Google专门为端侧部署设计的轻量级推理框架。它将TensorFlow训练的模型转换为.tflite格式，去除训练相关的OP，优化内存占用和推理速度。

### Q2: TFLite模型文件结构是什么样的？

**A**: `.tflite`文件是FlatBuffer格式，包含：
- Model: 模型元信息
- Subgraph: 计算图
- OperatorCode: 算子类型
- Tensor: 张量定义
- Buffer: 权重数据

### Q3: Interpreter和Model的关系？

**A**: Model是模型的静态表示，Interpreter是运行时执行引擎。一个Model可以创建多个Interpreter实例。

---

## 二、Delegate相关

### Q4: 4种Delegate的区别和选型策略？

**A**: 
- **XNNPACK**: CPU NEON优化，兼容性最好，2~5x加速，默认推荐
- **GPU**: OpenGL/Vulkan，2~10x加速，大模型/高吞吐场景
- **NNAPI**: 调用厂商NPU/DSP，5~20x加速+省电60%+，后台低功耗场景
- **Hexagon DSP**: 高通独占，6~15x加速，国内市场高通机型占比70%+

### Q5: Delegate fallback机制是怎样的？

**A**: 当某个Delegate初始化失败或推理失败时，自动降级到其他Delegate。最佳实践是三级降级：NNAPI → GPU → XNNPACK。

### Q6: GPU Delegate为什么会fallback到CPU？

**A**: ① 部分算子不支持GPU实现；② 模型输入输出格式不兼容；③ 驱动版本过低；④ 内存不足。

---

## 三、量化相关

### Q7: INT8量化的两种模式区别？

**A**:
| 特性 | PTQ (Post-training Quantization) | QAT (Quantization-aware Training) |
|-----|---------------------------------|-----------------------------------|
| 时机 | 训练后 | 训练中 |
| 精度损失 | <1% | <0.3% |
| 周期 | 1天 | 2~3周 |
| 校准数据 | 需要（500~2000张） | 不需要 |

### Q8: 量化校准数据集怎么选？

**A**: 需要**真实场景分布**的图，覆盖各种角度/光照/遮挡/类别，数量500~2000张。量少不准确，量多校准慢。

### Q9: INT8量化能带来什么收益？

**A**: 模型大小÷4，推理速度×2~4，内存带宽需求÷4。

---

## 四、性能优化

### Q10: 端侧推理性能优化有哪些手段？

**A**: 
1. 选对Delegate（XNNPACK/GPU/NNAPI）
2. INT8量化
3. 帧节流（处理不过来就丢帧）
4. 预处理移到C++（NDK libyuv）
5. 算子融合（Conv+BN+ReLU）
6. 零拷贝BufferHandle
7. 选轻量模型（MobileNetV3/EfficientNet-Lite）

### Q11: 为什么要做帧节流？

**A**: 相机输出30FPS，但AI推理可能跟不上。如果不节流，帧会排队，导致UI卡顿、手机发烫。节流策略：时间戳比较，小于阈值就丢弃。

### Q12: 预处理为什么要在C++做？

**A**: Kotlin/Java的Bitmap处理比C++慢10倍。用NDK的libyuv/libjpeg-turbo可以让预处理速度×5~10。

---

## 五、工程实践

### Q13: 模型热更新怎么做？

**A**: 
1. CDN差分下载
2. MD5校验完整性
3. 灰度发布（先推1%用户）
4. 失败自动回滚到旧版本

### Q14: 多模型并行推理怎么优化？

**A**: 
1. 模型并行：不同模型跑不同Delegate
2. 线程池管理：避免线程竞争
3. 内存复用：共享中间缓冲区

### Q15: 如何处理不同分辨率的输入？

**A**: 
1. LetterBox Resize（保持比例填充）
2. 后处理时还原坐标（反LetterBox）
3. 注意宽高比不一致带来的检测框偏移

---

## 六、错误排查

### Q16: 检测不到物体常见原因？

**A**: 
1. 旋转角度错误（80%的坑）
2. 输入图像归一化参数不对
3. 模型输入尺寸不匹配
4. Delegate初始化失败fallback到CPU但没日志

### Q17: 推理速度慢怎么排查？

**A**: 
1. 检查Delegate是否正确加载
2. 查看CPU/GPU占用率
3. 用TFLite Profiler定位瓶颈算子
4. 检查是否开启了不必要的调试选项

### Q18: OOM内存泄漏怎么排查？

**A**: 
1. 用Android Studio Profiler看内存曲线
2. 检查Interpreter是否及时close
3. 检查Bitmap是否及时recycle
4. 检查TensorBuffer是否复用