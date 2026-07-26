# ONNX性能优化重难点解析

> 位置: 04-模型部署/onnx-tutorial/doc/
> 配套文档: ONNX模型部署实战指南.md | ONNX流程图详解.md | ONNX面试题汇总.md

---

## 一、ONNX Runtime 4大执行后端对比

### 1.1 CPU vs CUDA vs TensorRT vs OpenVINO 全面PK

| 后端Execution Provider | 硬件要求 | 相对速度CPU=1x | 模型大小 | 精度损失 | 启动速度 | 场景 |
|---------------------|---------|--------------|---------|---------|---------|-----|
| **CPU MLAS** | 通用x86/ARM | **1x 基线** | 原大小 | FP32无损 | 秒开⭐ | 开发调试/低并发服务 |
| **OpenVINO EP** | Intel CPU/核显 | **2.5-4x** | 原大小 | <0.3% | 秒开 | Intel服务器CPU首选⭐免费加速 |
| **CUDA EP** | 英伟达GPU | **8-15x** | 原大小 | <0.1% | 5-10s | GPU通用推理 |
| **TensorRT EP** | 英伟达GPU | **15-30x 最猛** | +Engine缓存 | FP16<0.5% | 首次编译慢30s-2min⭐ | 生产GPU服务 极致性能首选 |
| **CoreML EP** | Apple M系列 | **5-10x** | 原大小 | <0.2% | 秒开 | Mac端侧/iPhone部署 |
| **NNAPI EP** | Android NPU | **3-8x** | INT8量化 | <1% | 秒开 | 安卓端侧NPU |

> 🏆 生产黄金组合：GPU=TensoRT+FP16/INT8；CPU=OpenVINO；开发调试=CPU基线

---

## 二、TensorRT EP 极致性能优化技巧

### 2.1 三大坑 90%用户踩

| 坑 | 现象 | 解决代码 |
|----|------|---------|
| **首次启动编译30s-2min** | 第一次跑超级慢 | 首次启动后Engine序列化缓存到磁盘 |
| **动态shape支持差** | 动态Batch报错 | 必须指定min/opt/max三档shape范围 |
| **不支持的算子Fallback** | 子图切太碎反而慢 | 把不支持的算子替换成标准算子 |

**TensorRT EP 生产配置代码：**
```python
import onnxruntime as ort

so = ort.SessionOptions()
# 🚀 图优化开到最高
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "model_optimized.onnx"  # 保存优化后图

trt_options = {
    # 1. 精度 FP16 必开 A100/3090+ 原生TensorCore 加速×2
    "trt_fp16_enable": True,

    # 2. INT8量化校准 极致压缩
    "trt_int8_enable": True,
    "trt_int8_calibration_table_name": "calibration.cache",

    # 3. 动态shape 三档必设(动态Batch的模型没有这个会崩)
    "trt_min_shape_0": "1x3x224x224",
    "trt_opt_shape_0": "32x3x224x224",  # Batch=32最优
    "trt_max_shape_0": "128x3x224x224",

    # 4. Engine序列化缓存 避免每次重新编译(⭐最重要)
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_cache",  # 首编后存下来

    # 5. 工作显存
    "trt_max_workspace_size": 8 * 1024 * 1024 * 1024,  # 8GB
}

providers = [
    ('TensorrtExecutionProvider', trt_options),  # 优先TensorRT
    ('CUDAExecutionProvider', {"device_id": 0}), # Fallback到CUDA
    ('CPUExecutionProvider'),                    # 再Fallback到CPU
]

sess = ort.InferenceSession("model.onnx", sess_options=so, providers=providers)
```

---

## 三、量化 4方案选型 + 精度控制

### 3.1 量化方法选型树

```
                      原始模型 FP32 100%精度
                     /           |            \
                    /            |             \
          动态量化INT8      静态量化INT8        FP16半精度
         (权重量化)       (权重+全激活)        (权重+全部)
          /    \              /    \              |
      CPU场景   NLP首选    GPU首选   CPU CV场景   GPU TensorCore首选
      LSTM/Transformer ResNet/ViT 大模型10B+    精度无损几乎0损失

  精度损失:   <0.5%          0.5-1.5%       <0.1% 最佳精度
  加速比:   CPU×1.5-2.5   CPU×2.5-4/GPU×5-8   GPU×2 TensorCore
  模型大小: 100MB→25MB      100MB→25MB       100MB→50MB
```

### 3.2 静态量化校准数据集 黄金法则

> 🔴 **致命错误：用随机数做校准集 = 量化出来模型直接报废**

校准数据集5要求：
1. ✅ 必须是真实业务场景的样本(不是torch.randn！)
2. ✅ 500-1000张足够，多了浪费，少了统计不准
3. ✅ 覆盖长尾：各种角度/光照/类别均衡分布
4. ✅ 预处理(train_transform/eval_transform)和线上完全一致
5. ✅ 量化后必须跑完整验证集调参

```python
# 校准集读取示例 - 必须用真实图片
class CalibrationDataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self):
        self.data = load_real_images_from_disk(n=500)  # ⭐ 500张真图
    def get_next(self):
        return {"input": next(self.data)}  # 返回预处理好的张量

quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=CalibrationDataReader(),
    calibrate_method=CalibrationMethod.MinMax  # MinMax或Entropy二选一
)
```

---

## 四、导出ONNX 12个踩坑点 + 修复方案

| 坑编号 | 报错现象 | 根因 | 修复 |
|-------|---------|-----|------|
| #1 | 精度掉10%+，百思不解 | ❌ 忘写 model.eval()！Dropout/BN还在训练模式 | 导出前必须model.eval() |
| #2 | 数值不对，但只在Batch>1时错 | dynamic_axes配置漏了Batch维度 | dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}} |
| #3 | NLP模型奇怪输出错位 | 🪤 position_ids没传 自动从0开始 | 手动传position_ids给dummy input |
| #4 | opset算子不支持 prim::XXX | opset版本太老 | 升opset_version=19/20 新版支持更多算子 |
| #5 | 自定义算子Module报错 | 自己写的特殊算子不在ATen标准里 | a)符号函数注册 b)替换算子 c)自定义ORT Op |
| #6 | onnx.checker通过但sess.run错 | shape推理失败没报错 | 开dynamic_axes + simplify简化模型 |
| #7 | 相同输入每次输出不同 | 模型还有随机性：比如Dropout没关 | eval() + seed固定 + check_no_state_alter |
| #8 | ViT/Deformable Attention算子炸坑 | 多尺度DeformAttn ONNX不友好 | 替换标准算子 / TensorRT插件 / 保持PyTorch |
| #9 | FP16转换后全NaN溢出 | 某些算子数学范围超FP16(大Exp/Div) | keep_io_types_float32 设输入输出保留FP32 |
| #10 | BatchNorm折叠后精度差异 | 折叠BN有epsilon舍入误差 | do_constant_folding=True + tolerance加大atol=1e-2 |
| #11 | 导出成功 但特别慢 性能不如PyTorch | 🚨 onnxsim没跑 存在大量冗余计算 | 必须onnxsim简化模型：onnxsim.simplify() |
| #12 | ORT比PyTorch慢2倍 | intra_num_threads线程数默认1 | 开intra_op_num_threads=CPU核数 |

---

## 五、Batching & 吞吐优化

### 5.1 Batch大小延迟/吞吐Tradeoff曲线

```
GPU RTX 4090 + TensorRT INT8 + ViT-B
────────────────────────────────────────
Batch Size     延迟 p95 ms      吞吐 images/s
    1           1.2 ms             833
    8           1.8 ms            4,444
    32          4.5 ms            7,111 ⭐最佳性价比
    128         15.2 ms           8,421 最大吞吐
    256         29.1 ms           8,799
```

> 🏆 **黄金建议**：服务端不追求绝对最低延迟 → Batch=32-64吞吐最高最划算；实时线上接口推荐Batch=8延迟1.8ms用户无感。

### 5.2 动态Batching (服务端必备)

```python
# 简单动态批处理实现: 攒够8张或等1ms 就跑一次
batch_queue = Queue()
result_promises = {}

async def batch_infer_worker():
    while True:
        items = []
        # 攒最多8个 OR 等最多1ms超时就跑
        deadline = time.time() + 0.001
        while len(items)<8 and time.time()<deadline:
            try:
                items.append(batch_queue.get(timeout=0.0001))
            except Empty: break

        if items:
            batch_input = np.concatenate([x[1] for x in items], axis=0)  # 拼Batch
            outputs = session.run(None, {"input": batch_input})  # 一次推理
            for i, (req_id, _, promise) in enumerate(items):
                promise.set_result(outputs[0][i:i+1])  # 拆分结果回每个请求

# 动态Batching × 32Batch = 吞吐×3-5倍 延迟几乎不变
```

---

## 六、生产部署 性能监控清单

| 监控指标 | 建议阈值 | 工具 |
|---------|---------|------|
| P95延迟 | ViT <5ms / LLM 7B <20ms/token | Prometheus histogram |
| GPU利用率 | 维持65-95%，<50%说明Batch太小浪费卡 | nvidia-smi dmon |
| GPU显存占用 | 不要超过90%留余量 | DCGM exporter |
| CPU核利用率 | OpenMP绑核 numa绑定 100%/核 | htop |
| 模型加载首延迟 | <20s（缓存Engine后） | 日志 |
| 内存泄漏 | 24小时RSS增长<5% | memory_profiler |
| QPS吞吐压测 | 至少×1.5峰值容量 | locust / wrk |