# 🔷 技术方向6：Vue前端AI应用

## 6.1 核心技术概述

### 前端AI方案对比

| 方案 | 说明 | 优势 | 适用场景 |
|------|------|------|----------|
| **TensorFlow.js** | Google官方JS推理框架 | 成熟，支持WebGL加速 | 浏览器直接推理CV/NLP模型 |
| **ONNX Runtime Web** | Microsoft Web版推理 | 与PyTorch训练无缝 | ONNX模型 |
| **ML.js社区生态** | Transformers.js等 | 大模型在浏览器中运行 | 轻量级LLM |
| **API调用模式** | Fetch/Axios调用后端AI | 零客户端推理开销 | 复杂模型/大语言模型 |
| **WebWorker + 流式** | WebWorker后台推理 + SSE流式输出 | 不阻塞UI | 实时输出场景 |

### TF.js核心流程

```javascript
// 1. 安装: npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgl

import * as tf from '@tensorflow/tfjs';

// 2. 加载模型（从URL或本地文件）
const model = await tf.loadLayersModel('/models/mobilenet/model.json');
// 或从TF Hub: const model = await tf.loadGraphModel('https://tfhub.dev/...');

// 3. 设置后端
tf.setBackend('webgl');  // GPU加速，性能最佳
// tf.setBackend('wasm'); // 无GPU时回退，需安装 @tensorflow/tfjs-backend-wasm

// 4. 预处理
const tensor = tf.browser.fromPixels(imageElement)  // HTMLImageElement → 张量
    .resizeBilinear([224, 224])                     // 缩放
    .toFloat()
    .sub(tf.tensor1d([123.68, 116.78, 103.94]))     // 减均值
    .div(tf.tensor1d([58.39, 57.12, 57.38]))        // 除标准差
    .expandDims(0);                                  // 增加batch维度: [1,224,224,3]

// 5. 推理
const predictions = model.predict(tensor) as tf.Tensor;
const probs = predictions.dataSync();  // 同步获取结果
// 或异步: const probs = await predictions.data();

// 6. ⚠️ 重要：清理WebGL显存！
tensor.dispose();
predictions.dispose();
tf.disposeVariables();
// 或用 tf.tidy() 自动管理: tf.tidy(() => { 推理逻辑 })
```

### Vue 3 + TF.js组件设计模式

```vue
<template>
  <div class="ai-classifier">
    <input type="file" @change="handleFile" accept="image/*" />
    <img v-if="imgUrl" :src="imgUrl" ref="imgRef" width="224" />
    <div v-if="loading" class="loading">推理中...</div>
    <div v-if="results" class="results">
      <div v-for="(r, i) in results" :key="i">
        {{ i+1 }}. {{ r.label }}: {{ (r.confidence * 100).toFixed(1) }}%
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import * as tf from '@tensorflow/tfjs';

const imgUrl = ref<string>('');
const imgRef = ref<HTMLImageElement | null>(null);
const loading = ref(false);
const results = ref<{label: string, confidence: number}[]>([]);
let model: tf.LayersModel | null = null;
let labels: string[] = [];

onMounted(async () => {
  // 组件挂载时加载模型（只加载一次）
  console.log('正在加载TF.js模型...');
  model = await tf.loadLayersModel('/models/mobilenet/model.json');
  labels = await fetch('/models/mobilenet/labels.txt').then(r => r.text())
                    .then(t => t.trim().split('\n'));
  console.log('✅ TF.js模型加载成功，类别数:', labels.length);
});

async function handleFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !model) return;
  imgUrl.value = URL.createObjectURL(file);
  loading.value = true;
  
  // 图片渲染后再推理
  await new Promise(resolve => setTimeout(resolve, 100));
  if (!imgRef.value) return;
  
  // WebWorker后台推理，避免阻塞UI主线程
  const result = await tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imgRef.value!)
      .resizeBilinear([224, 224]).toFloat()
      .sub(tf.tensor1d([123.68, 116.78, 103.94]))
      .div(tf.tensor1d([58.39, 57.12, 57.38]))
      .expandDims(0);
    const preds = model!.predict(tensor) as tf.Tensor;
    return Array.from(preds.dataSync());
  });
  
  const sorted = result.map((conf, idx) => ({ label: labels[idx], confidence: conf }))
                       .sort((a, b) => b.confidence - a.confidence).slice(0, 3);
  results.value = sorted;
  loading.value = false;
}

onUnmounted(() => {
  // 组件卸载时释放模型
  model?.dispose();
  tf.disposeVariables();
});
</script>
```

### LLM流式输出（Vue 3 + Fetch + ReadableStream）

```vue
<template>
  <div class="chat-ui">
    <div class="messages">
      <div v-for="(msg, i) in messages" :key="i" :class="msg.role">
        {{ msg.content }}
      </div>
    </div>
    <input v-model="input" @keyup.enter="send" :disabled="streaming" />
    <button @click="send" :disabled="streaming">发送</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const messages = ref<{role: string, content: string}[]>([]);
const input = ref('');
const streaming = ref(false);

async function send() {
  if (!input.value.trim()) return;
  const userMsg = input.value.trim();
  messages.value.push({ role: 'user', content: userMsg });
  messages.value.push({ role: 'assistant', content: '' });
  input.value = '';
  streaming.value = true;
  
  // 流式调用: SSE (Server-Sent Events) 或 直接读取响应流
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: userMsg }),
  });
  
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  
  if (reader) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // 解析SSE格式: data: {"content":"xxx"}\n\n
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(l => l.startsWith('data: '));
      for (const line of lines) {
        try {
          const data = JSON.parse(line.replace('data: ', ''));
          messages.value[messages.value.length - 1].content += data.content || '';
        } catch { /* 忽略解析错误 */ }
      }
    }
  }
  streaming.value = false;
}
</script>
```

### 前端AI组件架构

```
┌─────────────────────────────────────────────────────────┐
│                     Vue 3 组件                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ <ImageClass  │  │ <AIChatBox   │  │ <SpeechRecog │ │
│  │  ifier />    │  │    />        │  │    nition /> │ │
│  └───┬──────────┘  └───┬──────────┘  └──────┬───────┘ │
│      │  TF.js推理       │  API流式调用        │ Web Speech │
│      │  WebWorker        │  SSE/ReadableStream│ API        │
└──────┼───────────────────┼────────────────────┼───────────┘
       │                   │                    │
  ┌────▼──────────────┐ ┌──▼─────────────────┐ ┌▼──────────┐
  │ 模型文件 (.json   │ │ 后端LLM API        │ │ 浏览器原生│
  │  + .bin分片)      │ │ (Python/FastAPI)    │ │ API        │
  └───────────────────┘ └──────────────────────┘ └────────────┘
```

---

## 6.2 GitHub项目推荐

| 项目名 | 链接 | 核心学习点 | clone命令 |
|--------|------|-----------|-----------|
| tfjs-examples | github.com/tensorflow/tfjs-examples | **TF.js官方示例** - 图像分类/目标检测/姿态估计 | `git clone --depth 1 https://github.com/tensorflow/tfjs-examples.git` |
| transformers.js | github.com/xenova/transformers.js | **浏览器中的HuggingFace Transformers** | `git clone --depth 1 https://github.com/xenova/transformers.js.git` |
| chatbot-ui | github.com/mckaywrigley/chatbot-ui | **开源ChatGPT风格前端** - Vue/React参考 | `git clone --depth 1 https://github.com/mckaywrigley/chatbot-ui.git` |
| vue-ai-starter | github.com 搜索 vue ai | Vue 3 + TF.js + LLM流式输出模板 | `git clone --depth 1 <your-fork>` |
| onnxruntime-web | github.com/microsoft/onnxruntime | **ONNX Runtime Web版** - WASM/WebGL加速 | `git clone --depth 1 https://github.com/microsoft/onnxruntime.git` |

---

## 6.3 Vue 3完整示例：TF.js图像分类

```vue
<!-- ImageClassifier.vue -->
<template>
  <div class="min-h-screen bg-gray-100 p-6">
    <div class="max-w-2xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
      <div class="p-6">
        <h1 class="text-2xl font-bold mb-4">TF.js 图像分类</h1>
        
        <div class="mb-4">
          <label class="block mb-2 text-sm font-medium">上传图片:</label>
          <input type="file" @change="handleFile" accept="image/*"
                 class="block w-full text-sm border rounded cursor-pointer bg-gray-50" />
        </div>

        <div v-if="!modelLoaded" class="mb-4 p-4 bg-yellow-50 border rounded">
          <p>正在加载模型: {{ Math.round(loadProgress) }}%</p>
          <div class="mt-2 bg-gray-200 rounded-full h-2">
            <div class="bg-yellow-500 h-2 rounded-full" :style="{ width: loadProgress + '%' }"></div>
          </div>
        </div>

        <div v-if="imgUrl" class="mb-4">
          <img :src="imgUrl" ref="imgRef" class="max-w-full rounded shadow" width="224" />
        </div>

        <button @click="predict" :disabled="!modelLoaded || loading"
                class="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white py-2 px-4 rounded">
          {{ loading ? '推理中...' : '开始识别' }}
        </button>

        <div v-if="results.length > 0" class="mt-4">
          <h3 class="font-semibold mb-2">Top-3 识别结果:</h3>
          <div class="space-y-2">
            <div v-for="(r, i) in results" :key="i"
                 class="flex justify-between bg-gray-50 p-3 rounded">
              <span>{{ i+1 }}. {{ r.label }}</span>
              <span class="font-mono text-blue-600">{{ (r.confidence * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <p class="mt-4 text-sm text-gray-500">推理耗时: {{ latency }}ms</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import * as tf from '@tensorflow/tfjs';

// ===== 状态 =====
const imgUrl = ref('');
const imgRef = ref<HTMLImageElement | null>(null);
const loading = ref(false);
const modelLoaded = ref(false);
const loadProgress = ref(0);
const results = ref<{label: string, confidence: number}[]>([]);
const latency = ref(0);

let model: tf.LayersModel | null = null;
let labels: string[] = [];

// ===== 生命周期 =====
onMounted(async () => {
  console.log('[TF.js] 后端可用:', tf.getBackend());
  try {
    model = await tf.loadLayersModel('/models/mobilenet/model.json', {
      onProgress: (fraction: number) => { loadProgress.value = fraction * 100; }
    });
    labels = await fetch('/models/mobilenet/labels.txt')
      .then(r => r.text()).then(t => t.trim().split('\n'));
    modelLoaded.value = true;
    console.log(`✅ 模型加载成功，类别数: ${labels.length}`);
    console.log(`   当前后端: ${tf.getBackend()}, WebGL内存: ${tf.memory().numTensors} tensors`);
  } catch (e) {
    console.error('模型加载失败:', e);
  }
});

onUnmounted(() => {
  model?.dispose();
  tf.disposeVariables();
  console.log('🧹 模型资源已释放');
});

// ===== 事件 =====
function handleFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  imgUrl.value = URL.createObjectURL(file);
  results.value = [];
}

async function predict() {
  if (!model || !imgRef.value) return;
  loading.value = true;
  const t0 = performance.now();
  
  // tf.tidy() 自动清理中间张量，防止WebGL显存泄漏
  const predictions = tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imgRef.value!)
      .resizeBilinear([224, 224]).toFloat()
      .sub(tf.tensor1d([123.68, 116.78, 103.94]))
      .div(tf.tensor1d([58.39, 57.12, 57.38]))
      .expandDims(0);
    return model!.predict(tensor) as tf.Tensor;
  });
  
  const probs = Array.from(predictions.dataSync());
  predictions.dispose();
  
  results.value = probs.map((conf, idx) => ({ label: labels[idx], confidence: conf }))
                       .sort((a, b) => b.confidence - a.confidence).slice(0, 3);
  latency.value = Math.round(performance.now() - t0);
  loading.value = false;
  
  console.log(`🧠 推理完成: ${latency.value}ms, WebGL内存: ${tf.memory().numTensors} tensors`);
}
</script>
```

**package.json依赖**：
```json
{
  "dependencies": {
    "@tensorflow/tfjs": "^4.17.0",
    "@tensorflow/tfjs-backend-webgl": "^4.17.0",
    "vue": "^3.4.0"
  }
}
```

---

## 6.4 面试题库

### 📝 理论题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 6.1 | TF.js的不同后端（CPU/WebGL/WASM）有什么区别？如何选择？ | 中 | ⭐⭐⭐ |
| 6.2 | TF.js的`tf.tidy()`和`dispose()`有什么作用？为什么必须手动管理内存？ | 中 | ⭐⭐⭐⭐ |
| 6.3 | 浏览器流式输出（SSE/ReadableStream）与普通HTTP请求的区别？ | 中 | ⭐⭐⭐ |
| 6.4 | 前端调用LLM API时，如何处理取消请求、重试、超时？ | 中 | ⭐⭐⭐ |
| 6.5 | WebWorker为什么能避免阻塞UI？它与主线程如何通信？ | 中 | ⭐⭐⭐ |
| 6.6 | Vue 3组合式API中，模型加载应该放在哪里？`onMounted`还是组件初始化？ | 简 | ⭐⭐⭐ |
| 6.7 | 图像预处理中，不同模型有不同的归一化方式（如ImageNet均值），如何管理？ | 中 | ⭐⭐⭐ |

### 🖥️ JavaScript/Vue代码题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 6.8 | 用TF.js实现HTML图像→张量的预处理函数（缩放224+ImageNet归一化） | 中 | ⭐⭐⭐⭐ |
| 6.9 | Vue 3 + TF.js实现一个简化版图像分类组件 | 中 | ⭐⭐⭐⭐ |
| 6.10 | 用Fetch + ReadableStream实现流式输出的聊天UI | 中 | ⭐⭐⭐ |
| 6.11 | 将模型推理移到WebWorker，写一个简化的postMessage版本 | 中 | ⭐⭐⭐ |

### 🔗 架构设计题

| 题号 | 题目 | 难度 | 频率 |
|------|------|------|------|
| 6.12 | 设计一个可复用的前端AI组件库：包含`<ImageClassifier/>`/`<AIChatBox/>`/`<SpeechRecognition/>`组件 | 难 | ⭐⭐⭐ |
| 6.13 | 设计一个企业级AI前端应用：流式输出+代码高亮+Markdown渲染+多会话管理 | 难 | ⭐⭐⭐ |

---

> ✅ **方向6（Vue前端AI应用）学习完成自检清单**：
> - [ ] 能用TF.js在浏览器中加载并推理图像分类模型
> - [ ] 理解WebGL显存管理（tf.tidy/dispose）的重要性
> - [ ] 能用Vue 3组合式API写出完整的AI组件
> - [ ] 掌握流式输出的前端实现方式