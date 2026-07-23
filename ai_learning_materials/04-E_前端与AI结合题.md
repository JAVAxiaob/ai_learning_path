# 04-E - 面试题库: 前端与AI结合题 (25题)

> 核心: 在Web前端集成AI能力, Vue3 + TensorFlow.js是你的第三主战场

---

## 第一部分: 前端AI基础题 (10题)

### 难度: 简单

**1. 前端AI推理的三种模式对比**

| 模式 | 技术方案 | 适用场景 | 优点 | 缺点 |
|------|---------|---------|-----|-----|
| 纯前端推理 | TensorFlow.js / ONNX Runtime Web | 实时滤镜/简单分类/隐私敏感 | 离线可用,无成本,响应即时 | 模型大小受限,速度比GPU慢10-100x |
| 调用云端API | fetch/axios调用Java后端 | LLM问答/图像生成/OCR | 可运行大模型,精度高 | 依赖网络,有调用成本 |
| 端云混合 | 轻任务本地做+重任务调云端 | 推荐系统/图片识别+搜索 | 兼顾体验与成本 | 架构更复杂 |

**2. 前端调用LLM API的核心代码结构**

关键步骤: 
1. fetch发送POST请求到Java后端 /api/chat
2. 使用 ReadableStream.getReader() 处理流式响应
3. TextDecoder 解码字节流为字符串
4. 解析SSE格式 (data: xxx ... data: [DONE])
5. incrementally更新UI实现打字机效果

关键API: 
- fetch + response.body.getReader()
- TextDecoder utf-8
- async/await 异步控制

**3. 前端流式响应(打字机效果) vs 完整响应**

流式响应优点: 
- 用户感知延迟从10秒降到0 (首个token到达即显示)
- 适合长文本生成场景 (文章/代码/翻译)
- 服务器可边生成边返回, 内存占用更低

技术要点: 
- SSE (Server-Sent Events) 协议
- 后端: Spring Boot 返回 Flux ServerSentEvent
- 前端: EventSource API 或 fetch+ReadableStream

**4. TensorFlow.js WebGL加速原理**

- TF.js把张量运算映射到WebGL shader
- GPU并行计算比CPU快5-20倍
- 数据在WebGL纹理中流转, 避免CPU-GPU数据拷贝
- 内存管理关键: tf.tidy() 自动释放中间张量

**5. 前端AI推理的内存管理最佳实践**

- tf.tidy() 包裹推理代码块 (类似try-with-resources)
- tf.dispose() 手动释放不再需要的张量
- 长运行推理放在 Web Worker 中, 避免阻塞UI线程
- 模型热切换时及时调用 model.dispose()

### 难度: 中等

**6. Vue3中AI推理的架构设计**

推荐架构分层: 
1. composables层: useAI() - 封装模型加载/推理逻辑
2. service层: llmService.ts - API调用封装
3. store层: Pinia 管理对话历史/缓存结果
4. component层: 纯UI组件, 接收props显示结果

**7. AI应用前端性能优化清单**

- 懒加载模型: const model = await import models/mobilenet
- 结果缓存: 相同输入复用缓存结果
- 防抖节流: 输入变化300ms后才调用AI
- Service Worker: 缓存模型文件, 加速二次加载
- Web Worker: 推理在后台线程, 不影响UI响应
- 虚拟列表: 长对话使用virtual scroll渲染

**8. AI聊天界面状态管理 (Pinia示例)**

状态数据结构: 
- messages: Message[] (对话历史)
- isTyping: boolean (AI生成状态)
- currentConversationId: string
- models: string[] (可用模型列表)
- cachedResults: Map (输入哈希->缓存结果)

核心actions: 
- sendMessage(text: string): 发送消息并触发流式接收
- clearConversation(): 清空当前对话
- switchModel(model: string): 切换模型

**9. AI功能的浏览器能力检测与降级方案**

检测项: 
- WebGL/WebGPU 支持 (TF.js需要)
- 摄像头/麦克风权限 (多模态输入需要)
- 网络连接状态 (离线时禁用云端功能)
- Cookie/LocalStorage 可用性 (本地缓存)

降级策略: 
1. 无WebGL: 改用纯云端API调用模式
2. 无摄像头: 隐藏图像输入按钮
3. 离线: 仅启用本地推理功能
4. 弱网: 禁用流式传输, 等待完整响应

**10. 前端Prompt模板管理系统**

模板数据结构定义: 
- id: 唯一标识
- name: 显示名称
- template: 含变量占位符的模板
- variables: string[] 变量名列表
- temperature: 生成多样性参数
- maxTokens: 输出长度限制

---

## 第二部分: 系统设计与工程化题 (15题)

### 难度: 中等-困难

**11. 设计一个AI图片生成Web App**

核心功能模块: 
- Prompt输入框 (关键词建议/历史记录)
- 参数面板 (风格/尺寸/数量/比例)
- 任务队列 (排队/进度条/可取消)
- 结果画廊 (瀑布流/收藏夹/下载)
- 历史记录 (Prompt缓存 + 参数复用)

**12. 设计一个AI辅助写作编辑器**

核心功能: 
- 富文本编辑器 (Quill/Monaco)
- AI补全提示 (灰色幽灵文字 + Tab接受)
- 操作面板 (重写/润色/扩展/翻译)
- 风格切换 (正式/口语/学术/简洁)
- 大纲自动生成

**13. 设计端云结合的推荐系统前端**

前端职责: 
- 用户行为采集 (点击/滚动/停留时间/收藏)
- 本地特征提取 (embedding缓存)
- 轻量级重排序模型 (TFLite)
- 推荐结果展示卡片
- 负反馈收集 (不感兴趣按钮)

**14. AI应用的前端测试策略**

单元测试: 
- Prompt模板填充逻辑
- 流式响应解析器
- 结果缓存命中逻辑

集成测试: 
- 端到端对话流程
- 错误处理和重试机制

E2E测试: 
- Playwright/Cypress 模拟用户交互

**15. AI前端的成本控制**

成本构成: 
- LLM API调用费 (0.002-0.06 USD/1K tokens)
- 图片生成API费 (0.02-0.08 USD/张)
- 云服务器/带宽

优化手段: 
- 结果缓存: 相同问题直接返回缓存答案
- 请求合并: 批量操作减少API调用
- 用户配额: 免费额度/日上限/会员无限
- 模型路由: 简单问题用小模型, 复杂问题用大模型
- Token限制: 控制Prompt和输出长度

---

## 综合设计题 (16-25题)

**16. 多模态LLM前端交互设计**

输入方式: 
- 文本输入 + 图片拖放/粘贴
- 预览图片缩略图 + 删除按钮
- 图片压缩/尺寸检查

**17. AI语音助手前端 (Web版语音对话)**

技术方案: 
- 语音识别: Web Speech API SpeechRecognition
- 语音合成: SpeechSynthesis API
- 唤醒词: Porcupine Web 或 简单关键词检测

**18. 设计一个可扩展的AI能力插件系统**

插件接口定义: 
- id / name / description
- capabilities: string[] 声明支持的能力
- invoke(input) -> Promise output
- render(element) -> 可选自定义渲染

**19. AI前端的用户隐私保护**

隐私设计原则: 
- 隐私模式开关: 仅本地推理/不上传数据
- 透明提示: 明确告知哪些功能会上传数据
- 一键清除: 清除对话历史+清除缓存模型
- GDPR兼容: 数据导出/删除请求

**20. AI推荐系统的前端展示与交互**

展示形式: 
- 卡片列表 (图文混排)
- 推荐理由展示 (为什么推荐这个)
- 交互反馈 (不感兴趣/收藏/立即查看)
- 刷新按钮 (换一批推荐)

**21. AI白板应用设计 (草图到标准图形)**

核心流程: 
1. Canvas绑定手写/绘图事件
2. 检测笔画完成, 提取图像数据
3. 上传到CV模型识别 (如Google Vision/自定义模型)
4. 返回标准图形对象 (矩形/圆形/流程图)
5. 自动替换用户手绘内容为标准图形

**22. WebAssembly在前端AI中的应用**

应用场景: 
- ONNX Runtime Web (WASM版本)
- 自定义C++推理库编译为WASM
- 性能比JS快, 比WebGL更稳定(无GPU依赖)

对比: 
- JS: 最慢, 兼容性最好
- WASM: 中等速度, 兼容性好
- WebGL: 最快, 设备兼容性依赖GPU驱动

**23. 多人协作 + AI共创平台设计**

核心技术: 
- WebSocket/CRDT (Y.js) 实现实时同步
- 光标/选区同步显示
- AI作为虚拟协作者: 第N+1个参与者
- 操作记录/回放/撤销 (包含AI操作)

**24. AI前端组件库设计 (面向AI应用的UI kit)**

核心组件清单: 
- AiChatMessage: AI消息渲染 (Markdown+代码高亮)
- AiSuggest: 带AI建议的输入框
- AiImageUpload: 自动打标签+描述生成
- StreamingText: 流式文本渲染组件
- LoadingDots: AI特色加载动画

**25. 从传统前端到AI前端的技术迁移图谱**

| 传统技能 | 如何迁移到AI场景 | 新知识 |
|---------|----------------|--------|
| Vue3 + TS | 封装AI能力为composables, 类型化API响应 | 流式响应处理/WebGL |
| 状态管理Pinia | 管理对话历史/缓存/AI任务状态 | Vector数据结构设计 |
| REST API调用 | 扩展支持SSE流式协议/异步任务 | Server-Sent Events |
| 表单输入 | 自然语言输入/语音/图片多模态输入 | Web Speech API/媒体处理 |
| 数据展示 | 结果可视化:图表/3D/交互式展示 | ECharts/D3/Three.js |
| 性能优化 | 模型懒加载/推理Web Worker/缓存 | WASM/WebGL原理 |
| 组件设计 | AI组件库设计 (聊天消息/流式文本/AI按钮)| 设计系统扩展 |
| 单元测试 | Prompt逻辑测试/API Mock/集成测试 | Mock AI响应 |

---

> 面试准备重点: 
> 1. Vue3组合式API深入理解 + TypeScript类型系统
> 2. 前端调用AI API的最佳实践 (流式/异步/缓存)
> 3. TF.js/ONNX Runtime Web的使用和优化
> 4. ECharts/D3可视化 + Markdown/代码高亮渲染
> 5. 系统设计能力 (消息队列/状态管理/性能优化)
> 6. 用户体验设计 (动画/响应式/可访问性)
> 7. 成本控制与监控 (API调用成本/错误率统计)
