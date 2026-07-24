# 💻 06 - Vue前端AI应用 章节导览

> **垂直方向推荐度：⭐⭐⭐** (前端+AI的全栈工程师需求在增长，但岗位数量比Java/Android少一些)
> 预计学习周期：1.5周 (10天) | 目标掌握度：⭐⭐⭐ L3实战级
> 配套项目：`06-frontend-ai` 目录下的项目或参考 Vue3 + TF.js 官方示例

---

## 📚 本章节文件索引

| 文件名 | 内容 | 优先级 |
|-------|------|--------|
| **README.md** (本文) | 技术全景 + 前端AI的3种模式 | ⭐⭐⭐ 先读 |
| **TensorFlow.js详解.md** | 浏览器端推理 + 层API/模型API + WebGL加速 | ⭐⭐⭐⭐ 必学 |
| **MediaPipe多媒体AI.md** | Google官方 5行代码: 人脸/手势/姿态/人体分割/人脸468网格 | ⭐⭐⭐⭐ 快速出活 |
| **流式交互UI设计.md** ⭐⭐⭐⭐⭐ | SSE/WebSocket 流式Chat UI + 打字机效果 + Markdown+代码高亮渲染 | ⭐⭐⭐⭐⭐ LLM应用必备 |
| **代码实战.md** | Vue3 + TS + Vite 3个完整项目模板 | ⭐⭐⭐⭐ 必做 |
| **面试题库.md** | 35道前端AI面试题+答案 | ⭐⭐⭐⭐ 面试前看 |
| **GitHub项目推荐.md** | chatbot-ui / Transformers.js 等明星项目 | ⭐⭐⭐ 参考 |

---

## 🧭 前端 + AI 的三种实现模式（按复杂度排序）

```mermaid
graph TD
    subgraph 模式1: 纯API调用 90%场景选这个 ⭐⭐⭐⭐⭐
        VUE[Vue 前端 UI层] -->|HTTP REST / SSE流式| BACKEND[AI后端服务 Spring Boot / Python FastAPI]
        BACKEND -->|API调用| LLM[云端LLM大模型 / 向量数据库]
        %% 特点: 前端只负责展示结果和交互，AI全在后端跑
        %% 优点: 简单+兼容性好+模型安全不泄露+不限模型大小
        %% 缺点: 受网络延迟影响，需要后端服务
    end

    subgraph 模式2: 浏览器端本地推理 TF.js / ONNX Web ⭐⭐⭐
        VUE2[Vue] -->|loadGraphModel| WEBGL[WebGL 2.0 GPU加速 + WASM]
        WEBGL --> INFER[本地推理 MobileNet/BlazePose 等小模型<100MB]
        %% 特点: 完全本地，隐私保护(人脸/医疗不上传)，零网络延迟
        %% 优点: 离线可用 + 隐私合规 + 免服务器成本
        %% 缺点: 模型大小受限(<200MB)，手机浏览器速度较慢
    end

    subgraph 模式3: 混合模式 端云协同 2025高级玩法 ⭐⭐⭐⭐
        VUE3[Vue] -->|小模型本地快速算| LOCAL[本地TF.js 快速分类/检测]
        LOCAL -->|结果不够准/复杂问题| CLOUD[云端大模型兜底]
        %% 例: 图片分类 本地MobileNet先算Top3置信度都<0.7 → 再传云端大模型再算
    end
```

> 💡 **90%的LLM应用选模式1就够了！** 前端只负责：① 好的聊天交互UI(打字机/Markdown渲染/代码块复制) ② 流式SSE/WebSocket接收 ③ 结果可视化(检测框/关键点)。不要把大模型塞到浏览器里！只有当**强隐私**(人脸/医疗不上传)/**完全离线**场景才选模式2。

---

## ⚡ 模式1（LLM前端）核心能力清单（面试必问）

### 1. SSE流式响应 = 打字机效果 (**必做！没有流式用户等5秒全走了**)

Vue3 + Fetch 流式SSE完整模板：
```vue
<script setup lang="ts">
import { ref } from 'vue'
import { marked } from 'marked'          // Markdown渲染
import hljs from 'highlight.js'         // 代码块高亮
import 'highlight.js/styles/github.css'

const answer = ref('')
const isLoading = ref(false)

// ⭐核心：ReadableStream + TextDecoder 逐字处理
async function askStream(question: string) {
  isLoading.value = true
  answer.value = ''
  try {
    const resp = await fetch('/api/ai/chat/stream', {
      method: 'POST',
      body: JSON.stringify({ q: question }),
      headers: { 'Content-Type': 'application/json' }
    })
    // 响应体是 ReadableStream，边到边读边解析
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder('utf-8')
    let done = false
    while (!done) {
      const { value, done: isDone } = await reader.read()
      done = isDone
      if (value) {
        const chunk = decoder.decode(value, { stream: true })
        // SSE格式: data: xxx\n\n  每次去掉前缀
        const lines = chunk.split('\n\n').filter(l => l.startsWith('data: '))
        for (const line of lines) {
          const text = line.slice(6)  // 去掉 "data: "
          if (text === '[DONE]') continue
          answer.value += text       // ⭐逐字追加 = 打字机效果
        }
      }
    }
  } finally {
    isLoading.value = false
  }
}

// Markdown + 代码高亮
const rendered = computed(() =>
  marked.parse(answer.value, { highlight: (c, l) => hljs.highlight(c, { language: l }).value })
)
</script>

<template>
  <div class="chat-ui">
    <div class="answer" v-html="rendered"></div>   <!-- 渲染Markdown -->
    <div v-if="isLoading" class="typing-cursor">▊</div>
  </div>
</template>
```

### 2. 结果可视化：Canvas 目标检测框 / 人脸关键点绘制

TF.js目标检测后，Canvas绘制检测框：
```ts
function drawBoxes(ctx: CanvasRenderingContext2D, boxes: Detection[], imgW: number, imgH: number) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  // 原始图像尺寸 -> Canvas显示尺寸的比例缩放
  const scaleX = ctx.canvas.width / imgW
  const scaleY = ctx.canvas.height / imgH
  for (const b of boxes) {
    if (b.score < 0.5) continue
    // 画矩形框
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    ctx.strokeRect(b.x * scaleX, b.y * scaleY, b.w * scaleX, b.h * scaleY)
    // 标签文字+背景
    ctx.fillStyle = '#00ff00'
    ctx.font = '14px sans-serif'
    const label = `${b.className} ${(b.score * 100).toFixed(0)}%`
    ctx.fillRect(b.x * scaleX, b.y * scaleY - 22, ctx.measureText(label).width + 10, 22)
    ctx.fillStyle = '#000'
    ctx.fillText(label, b.x * scaleX + 5, b.y * scaleY - 6)
  }
}
```

### 3. 生产级Chat UI必备功能清单 (按优先级)

| 优先级 | 功能 | 实现要点 |
|-------|------|---------|
| P0 ⭐⭐⭐⭐⭐ | 多轮对话历史 + Pinia/Vuex管理 | 每条存 {role, content, timestamp} 列表 |
| P0 ⭐⭐⭐⭐⭐ | SSE流式 + 打字机光标 | ReadableStream + 解码追加 |
| P0 ⭐⭐⭐⭐⭐ | Markdown + KaTeX数学公式 + 代码块复制 | marked + katex + highlight.js + Copy按钮 |
| P1 ⭐⭐⭐⭐ | 取消请求 AbortController | `fetch(..., { signal: ctrl.signal })` 点×立刻断 |
| P1 ⭐⭐⭐⭐ | 对话本地持久化 localStorage/IndexedDB | 刷新页面不丢历史 |
| P2 ⭐⭐⭐ | 代码可执行 / 图片展示 / 表格渲染 | render函数里扩展custom renderer |
| P2 ⭐⭐⭐ | 对话导出 Markdown/PDF | 把历史拼MD字符串 + 下载Blob |

---

## 🤖 模式2：TF.js 浏览器端本地推理

5行代码搞定MobileNet图像分类：
```ts
import * as tf from '@tensorflow/tfjs'
// 1. 第一次加载从CDN下JSON模型权重，之后浏览器HTTP缓存
const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/2/default/1', { fromTFHub: true })

// 2. 预处理 HTMLImageElement / Canvas 直接传!
const img = document.getElementById('photo') as HTMLImageElement
const tensor = tf.browser.fromPixels(img)          // 自动 uint8 [0,255]
  .resizeNearestNeighbor([224, 224])               // resize
  .toFloat()
  .expandDims()                                    // 加batch维 [224,224,3]→[1,224,224,3]
  .div(tf.scalar(127.5))                           // [-1, 1] 归一化 MobileNet要求
  .sub(tf.scalar(1))

// 3. 推理
const predictions = (await model.predict(tensor) as tf.Tensor).dataSync() as Float32Array

// 4. ⭐别忘了内存释放！不写的话浏览器跑10次OOM崩溃！
tensor.dispose()
```

> ⚠️ TF.js第一大坑：**每一张图用完的Tensor必须dispose()**，否则浏览器内存疯涨→卡死→崩溃。推荐用 `tf.tidy(() => { ... })` 自动清理函数里所有临时Tensor。

### MediaPipe 5行代码 33点人体姿态 实时视频流：

```ts
import { Pose, Results } from '@mediapipe/pose'
import { Camera } from '@mediapipe/camera_utils'

const pose = new Pose({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
})
pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: true })
pose.onResults((r: Results) => {
  // r.poseLandmarks = 33个 [{x,y,z,visibility}] 直接拿来绘制
  drawPoseKeypoints(ctx, r.poseLandmarks)
})
// 绑定video标签 实时30FPS
const camera = new Camera(videoEl, {
  onFrame: async () => await pose.send({ image: videoEl }),
  width: 640, height: 480
})
camera.start()
```

---

## 🎯 章节结业标准

- [ ] 能手写 Vue3 + SSE 流式打字机 + 取消请求 AbortController
- [ ] 能解释三种前端AI模式各自的适用场景与优缺点
- [ ] 能独立写出Chat UI核心功能：历史/流式/Markdown/代码块复制
- [ ] 知道TF.js内存泄漏大坑：Tensor必须 tf.tidy 或 dispose()
- [ ] 能使用 MediaPipe 完成实时视频的关键点检测
- [ ] 面试题库正确率 ≥ 70%

> 💡 补充：如果你是Vue前端转AI应用，**强烈建议再把第07章(大模型应用)和第05章(Java后端 Spring AI)一起学完**。前端AI岗位少，但**LLM应用全栈工程师**岗位多薪资高，会前后端+AI = 稀缺人才。