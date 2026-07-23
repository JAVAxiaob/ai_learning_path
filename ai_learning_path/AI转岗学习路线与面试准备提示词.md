# AI转岗学习路线与面试准备提示词（按技术分类版）

> 版本 v2.0 · 按技术方向分类组织 · 无需检查运行环境，只需下载解压到目录

## 一、目标用户背景

- **现有技能**: 2年开发经验，熟练掌握 Android（Java/Kotlin）、Vue.js、Java后端，了解 Python 基础
- **转岗目标**: AI相关岗位，**核心策略：不丢弃原有技术栈，以AI赋能Android/前端/Java开发**
- **岗位定位**: 端侧AI开发工程师 / AI应用开发工程师 / 大模型应用开发工程师 / ML工程/AI工程化工程师
- **学习原则**: 由浅入深，循序渐进，**始终以Android/Java/Vue为核心载体融合AI知识**
- **运行环境要求**: **不检查IDE/编译器版本，只要能git clone下载、解压到指定目录即可，不强制立刻运行**

---

## 二、通用要求（所有技术方向都遵守）

### 任务0：GitHub项目搜索与代码下载（**必须首先执行，优先于其他任务**）

**
⚠️ 硬性要求：你必须在生成任何学习路线或知识内容之前，先主动搜索并推荐GitHub开源项目。
每个推荐项目必须给出完整的可执行命令，让ai可以直接 `git clone` 下载到本地且网址要可访问。
**

**操作步骤**：
1. 使用 WebSearch 搜索各技术方向的优秀开源项目（每个方向至少推荐 5 个项目，总计不少于 25 个）
2. 对每个项目按以下格式输出：

```
================================================================
【项目名称】owner/repo-name
【GitHub链接】https://github.com/owner/repo-name
【下载命令】git clone --depth 1 https://github.com/owner/repo-name.git
【解压到目录】D:\ai_learning\android\tflite-example\   （或用户自定义路径）
【Star数】xx.xk  【最后更新】YYYY-MM
【技术栈】语言/框架/核心库
【项目简介】3-5句话概述
【本地运行步骤】（简化版，不检查环境，给出核心命令即可）
  1. 依赖安装：pip install -r requirements.txt 或 ./gradlew build
  2. 配置修改：需要修改的配置文件/环境变量
  3. 启动命令：java -jar xxx.jar 或 python app.py 或 Android Studio Run
【代码阅读路线】（按建议阅读顺序列出关键文件路径）
  entry → 入口类/入口Activity → 核心模块1 → 核心模块2 → ...
【架构分析】模块划分、数据流、关键设计模式
【核心业务流程图】（用Mermaid语法绘制）
【技术迁移点】Android/Java/Vue经验如何迁移到本项目
【扩展/二次开发建议】基于该项目可以做的3-5个功能扩展方向
【学习价值评估】能学到什么具体技术/工程能力
================================================================
```

**项目清单汇总表格（必须输出）**：

| 项目名 | 技术栈 | 难度 | 预计学习时长 | 核心学习点 | git clone 命令 |
|--------|--------|------|-------------|-----------|---------------|
| ...    | ...    | ...  | ...         | ...       | ...           |

---

### 任务1：系统化学习路线 & 代码示例

每个技术方向必须包含以下内容：
- 知识点分级（入门/进阶/高级）
- 每个知识点展开讲解（公式+直觉理解+代码场景）
- 配套 GitHub 项目（含 clone 命令）
- 完整可运行代码示例（Python / Java / Kotlin / JavaScript / Vue）
- 与传统开发经验的结合点

---

### 任务2：面试题库（题量≥250题）

**硬性要求**：
- Java/Android/前端与AI结合的题目占比 ≥65%
- 每道代码题必须**完整可运行**（含 import、类声明、main方法，可直接复制到IDE运行）
- 每题标注：难度（简单/中等/困难）、所属分类、面试出现频率
- 系统设计题必须附带 Mermaid 架构图

---

### 任务3：简历优化 & 项目包装

- 简历关键词策略（传统技术栈 + AI技术栈）
- 传统项目的AI赋能改写方法
- 面试话术模板（为什么转AI / 如何展示学习路径 / 如何体现差异化）

---

### 任务4：时间规划

- 周学习计划表（表格形式）
- 月度里程碑（可量化指标）
- 自检清单（Yes/No 问答）
- 资源推荐（课程/书籍/GitHub仓库）

---

## 三、按技术方向分类的详细要求

---

### 🔷 技术方向1：数学与Python基础

**目标**：补齐AI所需的数学直觉 + Python工程能力

#### 1.1 知识点展开（必须逐项讲解）

| 知识点 | 展开内容 |
|--------|---------|
| 线性代数 | 矩阵乘法 `AB≠BA`、转置 `Aᵀ`、逆矩阵、特征值/特征向量、向量点积/叉积、Embedding直觉 |
| 概率统计 | 条件概率 `P(A|B)`、贝叶斯定理、高斯分布、softmax 直觉、交叉熵损失理解 |
| 梯度下降 | `θ = θ - η·∇L(θ)`、SGD vs Adam、学习率衰减、梯度消失/爆炸 |
| NumPy | `np.dot`/`np.matmul` 区别、广播机制、`np.einsum`、张量形状操作 |
| Pandas | `DataFrame` 特征工程、`groupby`+`transform`、缺失值处理、One-Hot Encoding |

#### 1.2 必须推荐的GitHub项目（至少5个，含 clone 命令）

- NumPy/Pandas 实战项目
- 机器学习基础算法从零实现
- 线性代数可视化教学项目
- Python 数据处理 Pipeline 项目
- 其他相关项目

#### 1.3 必须包含的代码示例（每个知识点至少1个完整可运行的 Python 示例）

例如：纯Python实现矩阵乘法、NumPy实现梯度下降、Pandas完成一个特征工程Pipeline

#### 1.4 面试题（约20-30题，含代码题）

---

### 🔷 技术方向2：机器学习基础

**目标**：掌握传统ML算法原理 + 特征工程 + 评估方法

#### 2.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| 监督学习 | 线性回归（最小二乘）、逻辑回归（sigmoid+交叉熵）、决策树（ID3/C4.5/CART）、SVM、XGBoost/LightGBM |
| 无监督学习 | K-Means（肘部法则）、DBSCAN、PCA（协方差矩阵特征分解） |
| 特征工程 | One-Hot/Label Encoding、归一化vs标准化、特征交叉 |
| 评估指标 | 准确率/精确率/召回率/F1、混淆矩阵、ROC-AUC、MAPE/RMSE |
| 与Java结合 | DJL Translator 实现特征处理、Spring Boot 暴露 ML 推理服务 |
| 与Android结合 | 端侧展示ML预测结果（RecyclerView + MPAndroidChart） |

#### 2.2 必须推荐的GitHub项目（至少5个）

- scikit-learn 实战项目
- XGBoost 应用案例
- 特征工程最佳实践
- 从零实现机器学习算法
- 其他相关项目

#### 2.3 必须包含的代码示例

- Python：使用 sklearn 完成完整的训练+评估 Pipeline
- Java：使用 DJL 加载 sklearn 模型并推理

#### 2.4 面试题（约20-30题）

---

### 🔷 技术方向3：深度学习与PyTorch

**目标**：掌握神经网络、CNN、Transformer、PyTorch工程实践

#### 3.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| 神经网络基础 | MLP、激活函数（ReLU/sigmoid/tanh/GELU）、反向传播直觉、BatchNorm/Residual |
| CNN | 卷积公式 `(H-F+2P)/S+1`、Pooling、LeNet→ResNet→EfficientNet→ViT、Grad-CAM |
| RNN/LSTM | seq2seq、门控机制（输入门/遗忘门/输出门+cell state）、Bidirectional RNN |
| **Transformer（重点）** | Self-Attention公式 `softmax(QKᵀ/√d)·V`、√d缩放的意义、Multi-Head Attention、Positional Encoding（正弦编码公式）、LayerNorm vs BatchNorm、Encoder-Decoder/Decoder-only/Encoder-only |
| PyTorch实战 | `nn.Module`、`DataLoader`、`torch.no_grad()`、`state_dict()`、`torch.onnx.export()` |
| 与Android结合 | PyTorch训练 → 导出 ONNX/TFLite → Android端部署 |
| 与Vue结合 | 使用 ECharts 可视化 loss/accuracy 曲线 |

#### 3.2 必须推荐的GitHub项目（至少5个）

- PyTorch 官方教程精选项目
- The Annotated Transformer（Transformer详细注解实现）
- ResNet/ViT 图像分类实战
- 模型 ONNX 导出与部署实战
- 其他相关项目

#### 3.3 必须包含的代码示例

- Python：用 PyTorch 从零实现一个简化版 Transformer（不少于150行，可直接运行）
- Python：CNN图像分类的完整训练代码
- Java：使用 DJL 加载 ONNX 模型进行推理

#### 3.4 面试题（约30-40题）

---

### 🔷 技术方向4：Android端侧AI开发

**目标**：将AI模型部署到Android设备，实现端侧推理

#### 4.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| TensorFlow Lite | `Interpreter.run(input, output)`、Model Maker迁移学习、模型文件放置到assets |
| 模型量化 | PTQ（训练后量化）vs QAT（训练感知量化）、FP32/FP16/INT8 模型大小对比 |
| 加速代理 | GPU Delegate（OpenGL ES）、NNAPI Delegate（Android系统神经网络API）、Hexagon DSP |
| Google ML Kit | `TextRecognition`/`FaceDetection`/`BarcodeScanning` 开箱即用API与自定义模型混合策略 |
| NCNN/MNN | 腾讯/阿里的端侧推理框架，与TFLite的选型对比 |
| CameraX集成 | 摄像头预览 → 帧回调 → 预处理 → TFLite推理 → 结果展示的完整链路 |
| Android性能优化 | 推理耗时测量、内存占用分析、模型热更新（下载新.tflite替换assets） |
| 端云协同 | 端侧轻量推理 + 云端大模型推理的分工策略、隐私/延迟权衡 |

#### 4.2 必须推荐的GitHub项目（至少5-8个，含 clone 命令）

- tensorflow/examples（TFLite Android示例）
- Google ML Kit Android示例
- NCNN Android Demo
- MNN Android Demo
- 端侧OCR项目（PaddleOCR Android）
- 端侧LLM项目（llama.cpp Android）
- 其他相关项目

#### 4.3 必须包含的代码示例（每个都要**完整可运行**，含import/类声明）

- Kotlin：TFLite 图像分类完整示例（含 CameraX 采集 + Interpreter 推理 + Bitmap预处理）
- Kotlin：ML Kit 文本识别完整示例
- Kotlin：GPU/NNAPI Delegate 加速配置对比代码
- Kotlin：端云协同推理架构示例（OkHttp请求云端API + TFLite端侧推理的切换逻辑）

#### 4.4 面试题（约30-40题，Kotlin/Java代码题占主要）

---

### 🔷 技术方向5：Java后端AI服务

**目标**：用Java技术栈构建AI推理服务，与现有Spring Boot微服务融合

#### 5.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| Deep Java Library (DJL) | `Criteria.builder()` 模型加载策略、`Translator` 接口 `processInput`/`processOutput`、Model Zoo、多引擎支持（PyTorch/TensorFlow/ONNX/MXNet） |
| ONNX Runtime Java | `OrtEnvironment` + `OrtSession`、ONNX模型加载、输入输出Tensor映射 |
| Spring Boot + AI | `@RestController` 暴露 `POST /predict`、`@Async` + `ThreadPoolTaskExecutor` 并发推理、Resilience4j `CircuitBreaker` 容错 |
| 微服务协同 | AI模型服务与业务服务的解耦、Feign/gRPC通信、批量推理 batching |
| 特征工程与数据 | Java端预处理必须与Python训练端完全对齐（推理偏差最大来源）、HuggingFace Tokenizer Java版对齐 |
| JVM性能优化 | 堆内存配置（大模型推理需要更大堆）、G1 GC vs ZGC、线程池参数调优、JNI调用开销 |
| Java调用Python | `ProcessBuilder` 子进程方式、gRPC跨语言服务、REST API调用FastAPI后端 |
| LangChain4j | `@AiService` 接口定义、`ChatLanguageModel` 适配多厂商、Tool/Function Calling |

#### 5.2 必须推荐的GitHub项目（至少5-8个）

- deepjavalibrary/djl-demo（DJL官方示例，含Spring Boot集成）
- deepjavalibrary/djl（DJL核心库源码学习）
- LangChain4j 官方示例
- Spring Boot + ONNX Runtime 推理服务
- mlflow java client 项目
- 其他相关项目

#### 5.3 必须包含的代码示例（每个都要**完整可运行**，含import/package/main方法）

- Java：DJL 加载 PyTorch 预训练模型完成图像分类（含 Translator 实现 + main 方法）
- Java：ONNX Runtime Java 加载 BERT 模型完成文本分类
- Java：Spring Boot 完整推理服务（RestController + DJL模型加载 + 异步推理）
- Java：Java通过 gRPC/REST 调用Python后端AI服务
- Java：LangChain4j 构建企业级 AI 助手服务

#### 5.4 面试题（约30-40题，Java代码题占主要）

---

### 🔷 技术方向6：Vue前端AI应用

**目标**：用前端技术实现AI功能界面与浏览器端推理

#### 6.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| TensorFlow.js | `@tensorflow/tfjs` 加载 `model.json` + shard 文件、Tensor创建与操作、`tf.tidy()` 避免WebGL显存泄漏、`tfjs-backend-wasm`（无GPU场景回退）、`tfjs-backend-webgl` GPU加速 |
| Vue3集成AI | 组合式 API 封装 `useAIModel()`、响应式状态管理推理结果、`onMounted` 加载模型 |
| AI交互界面 | 流式输出（SSE/ReadableStream）、打字机效果、Markdown-it渲染AI回复、代码高亮（highlight.js/prism） |
| WebWorker推理 | 把模型推理放到WebWorker线程，避免阻塞UI主线程 |
| 前端可视化 | ECharts展示推理置信度柱状图、Three.js展示3D点云/Embedding可视化 |
| 前端API调用 | 流式调用OpenAI/企业大模型API（`fetch` + `ReadableStream.getReader`）、请求取消/重试机制 |
| AI组件封装 | `<AIChatBox />`、`<ImageClassifier />`、`<SpeechRecognition />` 等可复用组件设计 |

#### 6.2 必须推荐的GitHub项目（至少5-8个）

- tensorflow/tfjs-examples（TF.js官方示例）
- Vue3 + OpenAI 聊天界面项目
- Next.js / Nuxt + RAG 前端项目
- TensorFlow.js 实时姿态检测/图像识别
- 前端Agent应用
- 其他相关项目

#### 6.3 必须包含的代码示例（完整可运行）

- Vue3 + TF.js：在浏览器中部署图像分类模型的完整单文件组件（`.vue`）
- Vue3：流式输出的 AI 聊天界面组件（使用 fetch + ReadableStream 逐字渲染）
- JavaScript：TF.js WebWorker 后台推理示例
- Vue3：Embedding 向量可视化组件（ECharts 3D散点图）

#### 6.4 面试题（约20-30题，Vue/JavaScript代码题占主要）

---

### 🔷 技术方向7：大模型应用开发（LLM / RAG / Agent）

**目标**：掌握大模型应用的完整工程链路

#### 7.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| Prompt Engineering | Zero-shot / Few-shot / CoT（`Let's think step by step`）/ Self-Consistency / Tree of Thoughts |
| RAG完整链路 | DocumentLoader → RecursiveCharacterTextSplitter（chunk_size/chunk_overlap）→ Embeddings → VectorStore.addDocuments → similar_search_with_score → CrossEncoder rerank → LLM.generate |
| 向量数据库 | Chroma / FAISS / Milvus / Pinecone 的选型对比、相似度计算（cosine/L2/inner product） |
| LangChain核心 | Model / Prompt / Chain / Agent / Memory / Retriever 六大抽象 |
| Agent框架 | ReAct（思考-行动-观察循环）、Tool定义、AgentExecutor、Function Calling / Tool Calling（ChatML格式） |
| 微调技术 | LoRA低秩适配 `BAᵀ`、QLoRA（4-bit量化+LoRA）、Full Fine-tuning vs PEFT显存对比表 |
| 推理优化 | KV Cache、Speculative Decoding、vLLM PagedAttention、批量推理 |
| 与Java结合 | LangChain4j `@AiService`、Spring Boot + RAG、Java向量数据库客户端 |
| 与Android结合 | OkHttp + SSE 流式调用LLM API、语音STT→LLM→TTS语音输出完整链路 |
| 与Vue结合 | ChatGPT风格UI、Markdown渲染+代码高亮+流式打字效果、RAG检索来源展示 |

#### 7.2 必须推荐的GitHub项目（至少5-8个）

- langchain-ai/langchain 官方仓库精选示例
- langchain4j/langchain4j 官方示例（Java版）
- RAG实战项目（如 LlamaIndex 应用）
- LangGraph 多Agent项目
- vLLM 推理服务
- 开源前端Chat项目（如Chatbot UI / LibreChat）
- 其他相关项目

#### 7.3 必须包含的代码示例

- Python：完整的RAG实现（文档加载→切分→向量化→检索→重排→LLM生成，可直接运行）
- Python：ReAct Agent 自定义 Tool 的完整示例
- Java：LangChain4j 实现 RAG 服务（Spring Boot 集成）
- Vue：流式输出的完整Chat组件
- Kotlin：Android端调用LLM流式API的完整示例（OkHttp + SSE + RecyclerView）

#### 7.4 面试题（约30-40题，含系统设计题）

---

### 🔷 技术方向8：AI工程化与MLOps

**目标**：掌握模型从训练到部署的完整工程流程

#### 8.1 知识点展开

| 知识点 | 展开内容 |
|--------|---------|
| 训练Pipeline | 数据→预处理→数据增强（`torchvision.transforms`/`albumentations`）→训练→验证→导出→部署 |
| 超参数调优 | 网格搜索/随机搜索/Bayesian Optimization（Optuna）、学习率调度（CosineAnnealing/LinearWarmup） |
| 模型部署 | 模型导出格式（ONNX/TFLite/TorchScript/GGUF）、服务框架（FastAPI/Tornado/vLLM/TGI） |
| 模型压缩与量化 | PTQ vs QAT、蒸馏（Student-Teacher+温度τ）、剪枝 |
| MLflow | `mlflow.log_model`、Model Registry（Staging/Production/Archived）、MLflow Java Client |
| 模型监控 | 模型漂移（PSI/KS检验）、推理延迟P99、A/B testing |
| 与Java结合 | Spring Boot + MLflow REST API构建模型管理平台、Java实现模型版本切换逻辑 |
| 与Android结合 | 模型热更新（下载新.tflite→替换assets→Interpreter重新加载） |
| CI/CD | GitHub Actions 自动训练+自动部署、Feature Store（Feast） |

#### 8.2 必须推荐的GitHub项目（至少5个）

- mlflow/mlflow 官方示例
- vLLM 推理服务
- 模型量化/蒸馏实战项目
- Feature Store（Feast）
- 其他相关项目

#### 8.3 必须包含的代码示例

- Python：使用 MLflow 跟踪实验并注册模型的完整流程
- Python：使用 Optuna 进行超参数搜索的完整代码
- Java：Spring Boot 集成 MLflow 实现模型热切换
- Python：PTQ 量化 PyTorch 模型并导出 ONNX

#### 8.4 面试题（约15-20题）

---

## 四、输出格式规范

1. **文档格式**: Markdown (.md)
2. **编码格式**: UTF-8
3. **语言**: 中文
4. **按技术分类**: 所有内容（项目推荐、代码示例、面试题）都按上面的8个技术方向分类组织（按照不同的技术方向分类，每个技术方向下包含知识点展开讲解、项目推荐、代码示例、面试题库，最好按照每个技术形成不同的章节，不要全写在一个目录或者一个md文件中）
5. **代码示例**: 提供**完整可运行**代码片段（含import、类声明、main方法、必要的依赖说明）
6. **图表说明**: 使用Mermaid语法绘制流程图和架构图
7. **项目链接**: 每个推荐的 GitHub 项目必须给出完整 URL 和 `git clone` 命令
8. **表格**: 使用 Markdown 表格汇总对比信息（项目清单、时间规划、框架选型等）
9. **运行环境**: **不检查**，只要求给出下载命令和解压到的目录路径即可

---

## 五、立即执行指令

**
⚠️ 当你（AI助手）阅读到这段提示词时，你的**第一个动作**必须是：

1. 列出你将搜索的所有技术方向：
   - 数学与Python基础
   - 机器学习基础
   - 深度学习与PyTorch
   - Android端侧AI开发
   - Java后端AI服务
   - Vue前端AI应用
   - 大模型应用开发（LLM/RAG/Agent）
   - AI工程化与MLOps

2. 使用 WebSearch 搜索上述每个方向的优秀开源项目

3. 按【任务0】的格式输出每个项目卡片（必须包含 GitHub URL + git clone 命令 + 解压到目录 + 代码阅读路径）

4. 输出完所有项目推荐 + 项目清单汇总表格后，再按技术方向依次生成：
   - 知识点展开讲解
   - 代码示例
   - 面试题库
   - 简历优化
   - 时间规划

**不允许跳过项目搜索直接输出理论内容。**
**