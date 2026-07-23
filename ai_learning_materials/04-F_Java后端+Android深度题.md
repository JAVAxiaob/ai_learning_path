# 04-F - 面试题库: Java后端深度题 + Android深度题 (30题)

> 核心: 传统后端和移动开发 + AI工程化, 这是你的差异化竞争力

---

## 第一部分: Java后端深度题 (15题)

### 难度: 简单-中等

**1. Spring Boot集成AI模型推理服务的架构设计**

推荐的分层架构:
1. Controller层: REST/SSE接口, 参数校验, 响应封装
2. Service层: 业务逻辑编排, 多模型路由, 缓存策略
3. Engine层: 模型推理封装 (ONNX/TF/DJL), 与框架解耦
4. Store层: 向量数据库/特征存储的DAO

关键Bean管理:
- 模型实例: 单例 + 懒加载, 避免重复加载
- 线程池: 独立的推理专用线程池, 隔离业务线程
- 缓存: Caffeine本地缓存 + Redis分布式缓存

**2. Java微服务与AI模型协同设计**

微服务拆分原则:
- ModelInferenceService: 纯推理服务, 无状态, 可水平扩展
- FeatureService: 特征提取/存储, 供多个模型共享
- PromptTemplateService: Prompt模板管理 + 变量注入
- ChatHistoryService: 对话历史存储/检索/上下文管理

服务间通信:
- 同步: gRPC/REST (低延迟推理调用)
- 异步: Kafka/RabbitMQ (批量推理/日志收集)

**3. JVM推理性能优化的关键配置**

内存配置:
- Xmx/Xms设置为物理内存70% (留给ONNX Runtime native内存)
- 元空间: MaxMetaspaceSize=256m 足够
- 禁用UseCompressedOops在大堆(>32GB)时反而可能更好

GC选择:
- 低延迟场景: ZGC (-XX:+UseZGC) 或 Shenandoah
- 高吞吐场景: G1GC (-XX:+UseG1GC)
- 推理服务推荐ZGC: 更低的STW停顿, 推理响应更稳定

线程配置:
- 推理引擎内部线程数: 与CPU核心数匹配 (ONNX Runtime的intra_op_num_threads)
- Tomcat线程数: 不要设置太大, 推理是CPU密集型

**4. Java并发编程在AI服务中的应用**

典型应用场景:
- CompletableFuture: 异步推理, 并行调用多个模型后合并结果
- ExecutorService: 批量推理的任务调度
- Semaphore: 限制并发推理数, 避免OOM (GPU显存有限)
- CountDownLatch: 等待多个推理任务完成

关键注意:
- ONNX/DJL模型实例通常是线程安全的, 可以多线程共享
- 但输入Tensor缓冲区要每个线程独立创建 (ThreadLocal优化)

**5. 数据库优化: 特征数据和模型结果存储**

数据分类:
- 特征向量: 用向量数据库(Pinecone/Milvus/Qdrant)或PostgreSQL pgvector
- 模型输出缓存: Redis (key=输入hash, value=推理结果, TTL=可配置)
- 对话历史: PostgreSQL/MongoDB (结构化存储, 支持检索)
- 审计日志: Elasticsearch/ClickHouse (大表分析)

索引设计:
- 向量检索用ANN索引 (HNSW/IVF-FLAT)
- 对话历史按用户ID分片/分区
- 时间范围查询用BRIN索引 (时序数据)

### 难度: 中等-困难

**6. 设计一个高并发的AI推理服务**

三层架构:
- 接入层: Nginx反向代理 + 负载均衡 + 限流
- 服务层: Spring Boot应用 + 推理引擎 + 异步任务队列
- 数据层: Redis缓存 + 向量数据库 + 对象存储

并发控制策略:
- 信号量限制并发推理数 (避免显存/CPU耗尽)
- 请求队列 + 超时熔断 (Resilience4J)
- 降级模式: 大模型不可用时, 回退到小模型/启发式算法

**7. 实现模型版本管理和A/B测试框架**

核心数据结构:
- ModelVersion: versionId, modelPath, metrics, status, trafficRatio
- TrafficAllocation: 按请求特征(用户ID/地域/时间)分配到不同版本
- ExperimentResult: 记录各版本的业务指标(点击率/转化率/延迟)

路由实现:
- request hash % 100 < threshold -> 新版本, 否则默认版本
- 支持白名单/灰度/全量三阶段发布
- 可通过管理API动态调整流量分配 (无需重启)

**8. 使用Spring Boot实现SSE流式响应**

关键技术点:
- 返回类型: SseEmitter (Spring MVC) 或 Flux<ServerSentEvent> (WebFlux)
- MIME类型: text/event-stream
- 消息格式: data: xxx + 空行
- 超时设置: 长连接需要合理的超时和心跳

生产环境注意:
- Nginx需配置 proxy_buffering off 避免缓冲阻塞流式传输
- 浏览器限制每个域名6个SSE连接, 需复用连接
- 异常/网络中断时需要客户端重连机制

**9. Java推理服务的监控和告警**

核心监控指标 (Micrometer + Prometheus):
- inference.request.count: 推理请求数 (按模型/版本分维度)
- inference.latency: 推理延迟分布 (P50/P95/P99)
- inference.error.count: 错误率
- inference.queue.size: 等待队列长度 (积压预警)
- gpu.memory.used: GPU显存占用
- llm.token.cost: API调用成本(按美元/人民币统计)

告警规则示例:
- P99延迟 > 3秒 (持续5分钟) -> 告警 + 自动扩容
- 错误率 > 5% -> 告警 + 切回稳定版本
- GPU显存 > 90% -> 告警

**10. 成本优化: 推理服务的资源利用率提升**

批处理优化:
- 将短时间(如100ms窗口)到达的多个请求打包为一个batch推理
- GPU利用率从 10% 提升到 70%+
- 类似vLLM/Google的动态批处理思想

模型合并部署:
- 小模型共用GPU实例, 通过CUDA MPS共享算力
- 冷热分离: 低频调用的模型动态卸载

弹性伸缩:
- K8s HPA基于QPS/延迟/CPU自定义指标扩缩容
- 预测式扩缩容: 基于历史流量数据预测高峰

### 难度: 困难

**11. 设计一个企业级RAG系统的后端架构**

核心模块:
- DocumentIngestionService: 文档解析(PDF/Word/网页) + 分段(chunking)
- EmbeddingService: 文本向量化 (可缓存, 向量是高价值可复用资产)
- VectorStoreService: 向量CRUD + 相似度检索 + 元数据过滤
- RAGRetrievalService: 检索编排 (混合检索+重排序)
- LLMEngineService: 多LLM供应商适配 (OpenAI/Anthropic/国产模型)
- PromptTemplateService: Prompt模板管理

质量保障:
- 检索评估: Recall@K 指标周期性测试
- 生成评估: LLM-as-judge 自动评分 + 人工抽查
- A/B测试: 不同检索策略对业务指标的影响

**12. 从0到1搭建模型服务化平台 (MLOps 基础)**

核心功能:
- 模型注册中心: 模型元数据+版本+指标管理
- 模型部署流水线: 一键部署到测试/预发/生产环境
- 推理服务运行时: 统一的推理容器模板
- 监控仪表盘: 性能/成本/质量三位一体
- 模型评价体系: 离线指标 + 线上A/B

技术选型:
- 框架: Spring Boot (微服务生态成熟) 或 KServe (K8s原生, 专业MLOps)
- 存储: MinIO/S3 (模型文件) + PostgreSQL (元数据)
- 监控: Prometheus + Grafana (通用) 或 Arize/WhyLabs (AI专用)

**13. 设计一个Agent执行引擎的Java实现**

核心概念:
- Tool: 可被LLM调用的函数 (如Search/Database/Calculator/API调用)
- Planner: LLM解析用户意图, 制定执行计划 (ReAct/Tree-of-Thoughts)
- Memory: 短期(对话历史) + 长期(用户画像/偏好)
- Executor: 执行Planner生成的Tool调用序列

实现要点:
- 使用注解 @Tool 标记工具类方法, 反射调用
- JSON Schema描述Tool接口, 传给LLM
- 支持同步执行和异步流式执行
- 加入最大步数限制, 防止死循环

**14. 实现多租户AI服务的隔离策略**

隔离维度:
- 数据隔离: 按租户ID分片, 向量数据库中区分命名空间
- 模型隔离: 不同租户可使用不同模型/版本
- 配额隔离: 每分钟/每日调用次数限制
- 成本核算: 按租户统计token消耗/推理调用次数

实现要点:
- ThreadLocal 存储 tenantId, 贯穿整个调用链
- 自定义 RateLimiter 注解 + 切面实现配额控制
- 租户级缓存: 共享模型权重, 但缓存按租户前缀隔离

**15. 设计大模型推理服务的稳定性保障体系**

容错策略:
- 故障转移: 主LLM API失败 -> 切到备用供应商 (如GPT失败 -> Claude)
- 降级模式: 复杂模型不可用 -> 使用本地小模型/规则引擎
- 请求熔断: 连续失败N次 -> 熔断窗口内直接返回错误/缓存结果

重试策略:
- 指数退避重试 (Exponential Backoff)
- 只对幂等/可重试错误重试 (超时/网络错误)
- 避免对业务错误(如内容审核不通过)重试

---

## 第二部分: Android深度题 (15题)

### 难度: 简单-中等

**16. Android Jetpack组件与AI功能集成**

典型应用:
- CameraX: 相机帧 -> CV模型推理 (目标检测/人脸识别)
- WorkManager: 后台模型更新/批量推理/数据上报
- Room: 本地存储推理历史/用户偏好/离线缓存
- ViewModel + LiveData: 推理状态管理, 跨生命周期存活
- DataStore: 存储用户选择的模型版本/AI偏好设置

最佳实践:
- 推理放在 ViewModel 中, 不随Activity重建销毁
- 用 WorkManager 执行后台任务 (如批量处理照片)
- 用 Room 缓存离线可用的AI结果 (减少云端API调用)

**17. Android NDK开发与C++模型库调用**

JNI/NDK集成模式:
- Java/Kotlin层: 定义 native 方法声明, 加载.so库
- JNI桥接层: 转换Java对象 <-> C++类型, 异常处理
- C++层: 直接使用 ONNX Runtime C++ API 或 TFLite C API

性能优势:
- 避免Java对象序列化反序列化开销
- 直接使用C++端的内存模型, 减少GC压力
- 可调用针对ARM架构优化的算子实现 (NEON指令集)

注意事项:
- JNIEnv是线程相关, 跨线程调用需AttachCurrentThread
- 注意局部引用溢出 (LocalReference溢出)
- 用Android NDK的arm64-v8a ABI (现代手机主流架构)

**18. Android性能优化: AI推理场景的特殊性**

内存优化:
- Bitmap复用 (inBitmap选项)
- ByteBuffer.allocateDirect 避免堆内存拷贝
- 模型按需加载, 空闲时释放 (参考LRU策略)

延迟优化:
- 模型预热: 启动后跑一次dummy推理, 预热GPU kernel缓存
- 量化模型: float32 -> int8, 推理延迟减少60%+
- 推理线程数: 设为可用核心数, 但避开大核绑定(功耗/发热)

功耗与发热:
- 推理密集型任务与用户交互操作分时执行
- 检测电池温度/电量, 低电量时降低推理频率
- 避免长时间高负载导致CPU降频 (thermal throttling)

**19. MVVM/MVI架构模式与AI数据流设计**

MVI推荐实现:
- UiState: sealed class 定义UI状态 (Loading/Success/Error/Empty)
- UserAction: 用户意图 (输入/上传图片/点击按钮)
- ModelAction: 模型响应 (推理结果/错误/进度)
- ViewModel: 持有State, 接收Action, 调用Model层, 更新State

AI场景的状态流设计:
- 流式推理的中间状态: PartialResult, 可显示在UI上
- 异步推理任务与Activity生命周期解耦 (ViewModel不会销毁)
- 多模型组合的复杂流程: 用Flow/Channel编排数据流

**20. Android端数据采集与隐私合规**

数据类型:
- 训练数据: 用户上传的图片/文本/语音
- 推理日志: 输入/输出/耗时/错误信息 (用于模型优化)
- 用户行为: 点击/滑动/停留, 用于推荐模型特征

合规要点:
- GDPR/个人信息保护法: 数据最小化原则, 明确告知+用户同意
- 数据脱敏: 上传前去除个人信息 (人脸打码/用户名替换)
- 端侧推理优先: 能在本地做的推理不上传
- 数据可删除: 用户有权删除自己的数据 (被遗忘权)

### 难度: 中等-困难

**21. 设计一个端侧推荐系统的Android实现**

架构分层:
- 数据层: 传感器/使用行为采集 -> 本地特征提取
- 模型层: TFLite加载轻量级推荐模型 (DNN/因子分解机)
- 服务层: RecommendationManager (单例, 应用内通用)
- UI层: RecyclerView + Adapter 展示推荐卡片

端云协同:
- 云端: 大规模协同过滤 + 全局候选召回 (候选集下发)
- 端侧: 用户实时兴趣建模 + 个性化重排序 (隐私保护)
- 混合推荐: 基于用户上下文(时间/地点/设备)动态加权

**22. 端云协同推理系统设计**

设计目标: 在不牺牲用户体验前提下, 最大化云端AI能力利用率

核心流程:
1. 用户操作 -> 端侧快速推理 (低延迟, 即时响应)
2. 同时异步上传请求到云端
3. 云端返回更强的结果 -> 增量更新UI
4. 收集用户反馈 -> 用于模型迭代

关键挑战:
- 何时触发云端调用 (置信度阈值/任务复杂度判断)
- 如何避免闪烁/内容抖动 (优雅的UI更新策略)
- 弱网/断网的容错

**23. 端侧模型的动态更新与灰度发布**

更新流程:
1. 服务端检查版本: HEAD请求获取最新模型版本元数据
2. 对比本地版本, 如需更新则下载 (断点续传)
3. 原子替换: 下载完成后校验MD5, 然后重命名替换旧模型
4. 模型热加载: 新建Interpreter实例, 旧实例完成当前请求后释放

灰度策略:
- 按设备IMEI/用户ID哈希分桶
- 先1% -> 5% -> 20% -> 100% 逐步放量
- 监控新版本的推理延迟/错误率/业务指标, 异常则自动回滚

**24. Android端多模型管理与资源优化**

模型库设计:
- ModelLoader: 模型加载器, 支持懒加载
- ModelPool: 对象池模式复用模型实例
- ModelPriority: 优先级管理, 低优先级模型可被内存压力下释放

资源优化:
- 按设备能力分发不同大小的模型 (低端机用tiny模型)
- 仅下载所需模型 (按需分发: App Bundle Dynamic Feature)
- 共享相同基础模型, 仅更新差异部分 (类似Git delta)

**25. 推理失败案例分析与自动上报系统**

失败类型:
- 模型加载失败 (文件损坏/版本不兼容/内存不足)
- 推理失败 (输入shape不匹配/数值溢出/OOM)
- 输出异常 (NaN/Inf, 后处理失败)
- 性能退化 (延迟突然飙升, 可能因系统状态变化)

自动诊断与上报:
- 捕获异常 -> 收集设备信息(机型/系统/内存/CPU)
- 记录输入数据摘要 (hash值/前N字节, 保护隐私)
- 批量压缩 -> WiFi下上传到分析平台 (参考Firebase Crashlytics)
- 建立知识库: 常见错误+解决方案, 用于自动修复或用户提示

### 难度: 困难

**26. 实现Android端联邦学习(Federated Learning)客户端**

基本概念:
- 数据不出端: 训练在设备本地完成
- 只上传模型参数更新 (梯度/权重增量)
- 服务器聚合多方更新 -> 产生新模型 -> 下发到设备

Android端实现:
- TFLite Model Personalization: 在端侧用用户数据微调
- WorkManager约束: 仅在充电+WiFi+空闲时执行训练
- 安全聚合: 用加密技术保护参数更新的隐私

挑战:
- 设备计算能力差异大 (需要动态batch size)
- 训练中断/失败的断点续训
- 通信开销控制 (压缩/量化/只传必要层)

**27. 端侧隐私计算与数据安全**

隐私增强技术:
- 差分隐私: 训练/推理时加入噪声, 防止泄露个体信息
- 联邦学习: 数据不出端, 协同训练
- 同态加密: 加密数据上直接推理 (但目前性能只支持小规模)

Android端实践:
- 本地模型推理优先, 敏感数据不上云
- 上传前做数据脱敏/模糊化处理
- 安全存储推理模型: 加密保护 (防止被窃取后分析)
- 权限最小化: 只申请必要的权限, 动态申请

**28. 实时计算机视觉系统在Android的性能优化**

处理流程优化:
- CameraX ImageAnalysis 设置目标分辨率 (640x480而非4K)
- 图像格式选择: YUV_420_888 -> 直接操作亮度通道做检测
- 避免Bitmap转换: 直接用ByteBuffer喂给TFLite

帧率控制策略:
- 不是每一帧都推理 (如每5帧推理一次, 中间帧复用跟踪结果)
- 目标跟踪算法 (KCF/MOSSE) 比检测快10倍以上, 可插值检测框
- 降采样: 远处/小目标降低推理分辨率

多线程设计:
- Camera线程: 只做图像捕获, 最快速度释放Image
- 预处理线程: resize + 归一化 (可使用Renderscript)
- 推理线程: TFLite Interpreter (独立线程池)
- UI线程: 只做结果绘制, 不做任何计算

**29. Android端大模型 (LLM) 的轻量化部署**

模型压缩路线:
- 模型选择: 用小模型 (TinyLlama 1.1B / Qwen 0.5B) 而非 70B+
- 量化: 4-bit量化 (GGUF格式), 用llama.cpp Android port
- 剪枝: 移除不重要的注意力头/层
- 蒸馏: 用大模型输出教小模型学习

Android端推理:
- mmap方式加载模型 (避免一次载入内存)
- 限制上下文长度 (减少KV Cache内存)
- 目前(2024年)旗舰机可流畅跑 1-3B 4bit模型 (约5-10 token/s)

**30. 端侧AI的产品化经验总结**

技术选型原则:
- 能不用AI就不用: 先探索规则/启发式/传统统计方法
- 简单模型优先: MobileNet/轻量级MLP优先于大模型
- 端云协同: 只在端侧做必须的, 其余放心交给云端

产品化经验:
- 明确的失败策略: AI不可用时的兜底方案
- 用户可控: 用户可以关闭/定制AI功能
- 渐进式增强: 从简单功能开始, 逐步增强AI能力
- 持续迭代: 收集用户反馈 -> 标注数据 -> 优化模型 -> 重新发布

关键成功因素:
- 延迟必须在用户可接受范围内 (<100ms 即时反馈)
- 模型准确率必须够高 (85%+ 是基本门槛, 否则用户会觉得笨)
- 隐私保护必须做到极致 (数据不出端是最强的信任背书)
- 与硬件/系统深度结合, 才能发挥端侧AI的最大价值

---

> 本文件完成面试题库F部分。核心价值: 将Java/Android传统技能与AI深度结合,
> 形成别人难以复制的竞争优势。纯AI算法背景的人不懂后端工程和移动端优化,
> 纯后端/移动端出身的人不懂模型部署和推理优化。你就是两者的交叉点。
