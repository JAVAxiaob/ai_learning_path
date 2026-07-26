# NCNN YOLOv5 面试题汇总

> 位置: 04-android-ai/ncnn-android-yolov5/

---

## 一、NCNN基础

### Q1: NCNN是什么？和TFLite/MNN相比有什么优势？

**A**: NCNN是腾讯开源的端侧深度学习推理框架。优势：
- Vulkan GPU支持最成熟
- 包体极小（核心库<2MB）
- 国内环境不依赖GMS
- 算子覆盖全面

### Q2: NCNN支持哪些硬件加速？

**A**: 
- CPU: ARM NEON / x86 SSE/AVX
- GPU: Vulkan / OpenGL ES
- DSP: 高通Hexagon（需额外配置）

### Q3: .param和.bin文件是什么？

**A**: 
- `.param`: 模型结构描述文件（文本格式）
- `.bin`: 权重数据文件（二进制格式）

---

## 二、推理流程

### Q4: NCNN推理的三个阶段是什么？

**A**: 
1. **前处理**: LetterBox Resize + Normalize + HWC2CHW
2. **推理**: Extractor.input -> extract
3. **后处理**: 置信度过滤 + NMS + 坐标还原

### Q5: Extractor是什么？为什么每个线程需要独立的Extractor？

**A**: Extractor是NCNN的推理执行器，包含中间状态和缓冲区。多线程共享会有竞态条件，所以每个线程需要独立创建。

### Q6: set_light_mode(true)有什么作用？

**A**: 轻量模式，用完的Blob立即释放，运行内存峰值降低2~3倍。

---

## 三、性能优化

### Q7: NCNN有哪些性能优化手段？

**A**: 
1. INT8/FP16量化
2. Vulkan GPU加速
3. 算子融合（Conv+BN+ReLU）
4. Winograd卷积优化
5. LightMode内存复用
6. 多线程并行

### Q8: INT8量化需要哪些步骤？

**A**: 
1. 准备500~2000张校准图片
2. 使用`ncnnoptimize`工具量化
3. 替换原模型文件

### Q9: Vulkan和OpenGL ES有什么区别？

**A**: 
- Vulkan是新一代图形API，更低开销
- 显式内存管理，性能更可控
- 多线程友好，可并行提交命令

---

## 四、工程实践

### Q10: NCNN模型如何热更新？

**A**: 
1. 从服务器下载新的.param和.bin
2. 验证文件完整性（MD5/SHA256）
3. 替换旧模型，重新初始化Net
4. 失败回滚到旧版本

### Q11: 如何处理不同分辨率的输入？

**A**: 
1. LetterBox保持宽高比缩放
2. 后处理时反LetterBox还原坐标
3. 注意padding的计算

### Q12: JNI调用需要注意什么？

**A**: 
1. 线程安全：每个线程独立Extractor
2. 资源释放：及时delete Net
3. 异常处理：catch所有可能的异常
4. JavaVM Attach：确保在正确的线程调用

---

## 五、选型对比

### Q13: NCNN vs TFLite vs MNN 如何选型？

**A**: 
- **TFLite**: 首选，Google维护，文档全，XNNPACK CPU快
- **NCNN**: 极致性能/Vulkan成熟/包体小，国内环境首选
- **MNN**: 阿里技术栈，端侧微调训练一体化

### Q14: 什么时候选择NCNN？

**A**: 
1. 不依赖GMS服务
2. Vulkan GPU极致性能需求
3. 包体要求极小（<2MB）
4. 腾讯系技术栈

---

## 六、错误排查

### Q15: 推理结果全是背景/检测不到物体？

**A**: 
1. 检查模型路径是否正确
2. 检查输入图像格式（RGB/BGR）
3. 检查归一化参数
4. 检查模型输入尺寸

### Q16: Vulkan推理崩溃？

**A**: 
1. 检查GPU是否支持Vulkan
2. 检查GPU内存是否充足
3. 添加try-catch fallback到CPU
4. 检查驱动版本

### Q17: 检测框位置不对？

**A**: 
1. 检查LetterBox的padding计算
2. 检查后处理的坐标还原
3. 检查图像旋转角度