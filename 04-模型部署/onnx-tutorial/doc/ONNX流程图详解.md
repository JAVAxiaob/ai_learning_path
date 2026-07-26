# ONNX模型部署流程图详解

> 位置: 04-模型部署/onnx-tutorial/doc/
> 配套文档: ONNX模型部署实战指南.md | ONNX性能优化重难点.md | ONNX面试题汇总.md

---

## 一、PyTorch模型 → ONNX 完整转换流程

```mermaid
flowchart TD
    Start[PyTorch模型: ResNet50.pt / ViT.pth]

    Start --> STEP1["步骤1: 模型准备<br/>model.eval() 切换推理模式<br/>⚠️必须关掉Dropout/BatchNorm的train模式"]
    STEP1 --> WARN1["⚠️ 常见坑点: 忘记model.eval()导出的模型精度掉5-10%"]

    STEP1 --> STEP2["步骤2: 构造Dummy Input假输入<br/>形状=[1,3,224,224] BatchSize×Channel×H×W<br/>dtype=float32 和真输入严格对齐"]
    STEP2 --> WARN2["⚠️ 常见坑点: DummyInput形状/dtype错 转出来模型运行不了"]

    STEP2 --> STEP3["步骤3: 导出torch.onnx.export()<br/>核心参数15个"]
    STEP3 --> KEYP["核心参数详解: opset_version=17<br/>dynamic_axes动态Batch/动态图片尺寸<br/>do_constant_folding=True常量折叠<br/>export_params=True导出权重"]

    KEYP --> STEP4["步骤4: 运行时Shape推理 & 验证"]
    STEP4 --> CHECK1["onnx.checker.check_model(model) 检查IR格式合法"]
    STEP4 --> CHECK2["Netron.app 可视化网络图 一眼看穿结构"]
    CHECK1 --> STEP5
    CHECK2 --> STEP5["步骤5: 数值一致性验证 (最关键)"]

    STEP5 --> ORT["pip install onnxruntime<br/>加载.onnx → sess.run(输入)"]
    ORT --> Compare["pytorch输出 vs onnx输出<br/>np.testing.assert_allclose<br/>atol=1e-3 rtol=1e-3<br/>✅ 差异<0.1%才算合格"]
    Compare --> SUCCESS["✅ 转换成功 模型.onnx"]
    Compare --> FAIL["❌ 差异大 debug: 开opset降级/dynamic_axes错/算子不支持"]
```

---

## 二、后端推理 跨平台部署矩阵

```mermaid
flowchart LR
    ONNX[.onnx模型文件]

    subgraph 部署场景一: Python后端服务
        ONNX --> ORT_CPU["ORT CPU EP<br/>pip install onnxruntime<br/>8核Intel 延迟18ms"]
        ONNX --> ORT_CUDA["ORT CUDA EP 英伟达GPU<br/>pip install onnxruntime-gpu<br/>RTX3090 1.5ms ×50%并行吞吐"]
        ONNX --> ORT_TRT["ORT TensorRT EP<br/>自动子图转TensorRT加速 INT8量化<br/>比CUDA再快×2-3倍"]
    end

    subgraph 部署场景二: C++生产高性能
        ONNX --> ORT_CPP["onnxruntime C++ API<br/>静态库链接<br/>延迟和Python一致 省Python开销"]
        ONNX --> TVM["Apache TVM 自动调优算子<br/>极致性能比ORT再快×1.5-2倍"]
        ONNX --> OpenVINO["Intel OpenVINO EP<br/>Intel CPU/集显优化 ×3加速CPU推理"]
    end

    subgraph 部署场景三: 浏览器/端侧
        ONNX --> ORT_WEB["ONNX Runtime Web<br/>WebAssembly/WASM + WebGL<br/>浏览器Chrome直接跑ResNet50 70ms"]
        ONNX --> ORT_JS["onnxruntime-node 服务端Node.js"]
        ONNX --> MNN["MNN NCNN 手机端侧 量化后ARM 20ms"]
    end
```

---

## 三、量化 INT8/FP16 详细流程

```mermaid
flowchart TD
    FP32[FP32原始ONNX模型 100MB]
    FP32 --> CALIB["校准数据集: 500-1000张代表性图片<br/>⚠️必须是真实业务数据 不是随机噪声"]

    CALIB --> METHOD{选量化方法}

    METHOD -->|"✅ 最准 动态量化 (NLP首选Transformer)"| DynQ["动态量化 Dynamic Quantization<br/>运行时实时计算权重的min/max<br/>激活值FP32算, 权重量化到INT8<br/>代码: quantize_dynamic(model_dtype=QInt8)"]
    DynQ --> S1["模型大小: ×4压缩 100MB→25MB<br/>CPU速度×1.5-2.5加速<br/>精度损失 <0.5%几乎无感"]

    METHOD -->|"✅ 最快 静态量化 (CV首选)"| StaticQ["静态量化 Static Quantization<br/>离线校准跑500张图<br/>记录每个激活值的min/max分布<br/>权重+激活全INT8<br/>代码: quantize_static(calib_reader)"]
    StaticQ --> S2["模型大小: ×4 100MB→25MB<br/>CPU×2.5-4倍加速<br/>GPU TensorRT×5-8加速<br/>精度损失: 调校准集可控制在<1%"]

    METHOD -->|"✅ FP16半精度 GPU首选"| FP16["FP16量化<br/>权重从32位float→16位float<br/>代码: float16.convert_float_to_float16()"]
    FP16 --> S3["模型大小: ×2 100MB→50MB<br/>Ampere架构GPU TensorCore原生加速×2<br/>精度几乎无损 <0.1%"]

    S1 --> Q_MODEL["✅ 量化后模型 生产部署"]
    S2 --> Q_MODEL
    S3 --> Q_MODEL

    Q_MODEL --> EVAL["验证量化后精度<br/>验证集Top1准确率降>1%<br/>→ 加校准数据/换量化方法/回退FP16"]
```

---

## 四、ONNX Runtime 推理Session创建全流程

```mermaid
sequenceDiagram
    participant APP as 业务代码
    participant ORT as ONNX Runtime
    participant GPU as GPU CUDA Driver

    APP->>ORT: 1. new InferenceSession("model.onnx")
    Note over ORT: 读文件+解析IR图结构

    APP->>ORT: 2. SessionOptions设置优化选项
    Note over ORT: 🚀 GraphOptimizationLevel=ORT_ENABLE_ALL<br/> 常量折叠/算子融合/DeadCodeElim

    APP->>ORT: 3. 配置Execution Provider
    alt 选CUDA
        APP->>ORT: append_execution_provider(CUDA)
        ORT->>GPU: 初始化CUDA Context加载GPU kernel代码
        GPU-->>ORT: CUDA Ready
    else 选TensorRT
        APP->>ORT: append_execution_provider(TensorRT)
        ORT->>ORT: 首次启动 子图编译生成TensorRT engine
        Note over ORT: 🏭 这一步会慢 30s-2min 但启动后就快了
    else CPU默认
        ORT->>ORT: MLAS/OpenMP 多线程设置 intra_op_num_threads=8
    end

    APP->>ORT: 4. session.get_inputs() 获取输入名称/shape/type
    Note over APP: ☑️ 校验输入数据类型/shape完全匹配 常见报错根源

    loop 每一张图片
        APP->>ORT: 5. session.run(inputs={input_name: ndarray})
        Note over ORT: CPU<->GPU Memcpy + Kernel Launch 推理
        ORT-->>APP: 6. outputs numpy数组 [Batch,1000] logits
    end
```

---

## 五、模型转换失败常见节点 + 修复路径

```mermaid
flowchart TD
    FAIL[导出失败 / 数值不一致 / 跑不起来]

    FAIL --> C1["类型1: ExportError 导出阶段就报错<br/>'Exporting the operator prim::XXX to ONNX opset version 17 is not supported'"]
    C1 --> F1[修复: opset_version升级17→19/20 新版本支持新算子<br/>或 register_custom_op_symbolic 自己实现符号函数]

    FAIL --> C2["类型2: Checker报错 模型格式非法<br/>'N node X is not zero dimensional but has no shape info'"]
    C2 --> F2[修复: 开dynamic_axes参数 动态轴shape推理<br/>或torch.onnx.export(shape_inference=True)]

    FAIL --> C3["类型3: Shape不匹配 RuntimeError<br/>'Got invalid dimensions for arguments'"]
    C3 --> F3[修复: DummyInput形状和代码真输入完全一致<br/>特别是Batch维度/NLP seq_len维度]

    FAIL --> C4["类型4: 数值差异大 assert_allclose FAIL<br/>PyTorch输出和ORT输出差1%+"]
    C4 --> F4["4步排查法:<br/>1. model.eval()是不是漏了<br/>2. 有没有用torch.no_grad()关闭autograd<br/>3. 关PostTrainingQuantization回到FP32试试<br/>4. opset降级 最新版本可能有bug"]

    FAIL --> C5["类型5: 某些算子不支持CustomOp"]
    C5 --> F5["三选一: a) torch里替换算子成标准算子<br/>b) ORT自定义op实现CPU/CUDA kernel<br/>c) 保持PyTorch原生子图 混合执行 fallback到PyTorch"]
```