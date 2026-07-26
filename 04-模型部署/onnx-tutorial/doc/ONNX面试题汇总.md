# ONNX模型部署面试题汇总 (35题)

> 位置: 04-模型部署/onnx-tutorial/doc/
> 配套文档: ONNX模型部署实战指南.md | ONNX流程图详解.md | ONNX性能优化重难点.md

---

## 一、ONNX基础（8题）

### Q1. 为什么需要ONNX标准? PyTorch直接部署不行吗？三大好处：统一IR/跨框架/多后端优化

### Q2. ONNX IR结构4要素: Graph/Node/Tensor/Initializer 详解

### Q3. PyTorch→ONNX导出15个核心参数: opset_version/dynamic_axes/do_constant_folding/export_params作用

### Q4. model.eval()为什么导出前必须调用？不调用精度掉10%+的原理是什么

### Q5. DummyInput假输入形状错了会怎么样? 为什么要严格对齐真实输入的shape/dtype/device

### Q6. onnxsim模型简化为什么必须跑? 10%模型体积减+30%速度提升的原理是啥

### Q7. Netron可视化工具怎么用？怎么查某层权重有没有正确导出

### Q8. onnx.checker.check_model()检查通过但跑不起来，5种典型根因

---

## 二、后端与性能（8题）

### Q9. 4大Execution Provider对比: CPU MLAS vs OpenVINO vs CUDA vs TensorRT 选型

### Q10. TensorRT EP 为什么首次启动慢30秒-2分钟？Engine缓存序列化怎么实现

### Q11. 动态shape三档参数: min/opt/max 怎么配置？不配置动态Batch直接炸的报错是什么

### Q12. SessionOptions.graph_optimization_level 4级(ORT_DISABLE→ALL)的优化内容

### Q13. intra_op_num_threads vs inter_op_num_threads 两个线程数区别？8核CPU怎么配最优

### Q14. OpenVINO EP 为什么Intel CPU比ORT CPU快3倍？原理算子融合/指令集优化

### Q15. CoreML/NPU/NNAPI三种端侧EP适用硬件和优缺点对比

### Q16. 为什么生产环境providers列表按顺序写TensorRT→CUDA→CPU三级Fallback降级

---

## 三、量化（7题）

### Q17. FP32/FP16/INT8三种精度对比表：显存/速度/精度/模型大小4个维度

### Q18. 动态量化Dynamic vs 静态量化Static vs 量化感知训练QAT 三种量化对比场景

### Q19. 静态量化校准集Calibration为什么必须是真实业务图？随机噪声校准直接报废模型的原理

### Q20. 校准样本500-1000张的依据是什么？多了浪费少了不准的统计学原理

### Q21. FP16量化后全NaN溢出怎么修？keep_io_types_float32输入输出保留FP32的坑

### Q22. MinMax校准 vs Entropy(熵)校准 vs Percentile校准 分布假设有何不同

### Q23. 量化后精度降2%+无法接受: 5个补救步骤(校准集/量化方法/敏感层回FP16/部分算子/QAT)

---

## 四、踩坑实战（7题）

### Q24. export成功数值不一致 assert_allclose FAIL: 6步排查法 eval/no_grad/dummy/seed/tolerance

### Q25. ViT/Deformable Attention算子不支持: 3个方案 替换标准算子/TensorRT插件/混合执行

### Q26. 不支持的算子CustomOp: 符号函数register_custom_op_symbolic写法示例

### Q27. NLP Transformer奇怪输出错位: position_ids没传导致的自动从0开始问题修复

### Q28. ORT推理反而比PyTorch慢2倍? 检查 intra_op_num_threads线程数是否默认=1

### Q29. 相同输入每次推理输出不同: eval()漏了? seed固定? 模型里有随机算子?

### Q30. Batch>1时结果错Batch=1时对: dynamic_axes Batch维度配置漏了

---

## 五、生产部署（5题）

### Q31. 动态Batching服务端批处理怎么实现？队列攒Batch ×1ms超时的实现代码

### Q32. Batch大小怎么调最优? 延迟/吞吐曲线拐点的经验值 (CV=32 LLM=8)

### Q33. GPU显存OOM优化6招: 小Batch/梯度检查点/FP16/INT8/多卡切片/梯度累积

### Q34. ONNX模型版本管理 & 灰度发布: A/B测试 5%流量切新模型 监控6指标无问题全量

### Q35. 可观测性生产监控仪表盘: P95延迟/GPU利用率/显存/QPS吞吐/错误率/首包延迟