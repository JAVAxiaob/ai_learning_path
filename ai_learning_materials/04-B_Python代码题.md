# 04-B - 面试题库：Python代码题 (20题)

---

## 第一部分：Python代码题 (20题)

### 难度: 简单

**1. 用NumPy实现简单的线性回归预测**
``python
import numpy as np

def predict_linear_regression(X, weights, bias):
    """线性回归预测: y = X*w + b"""
    return np.dot(X, weights) + bias

# 测试
X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
w = np.array([[0.5], [0.3]])
b = 0.1
print(predict_linear_regression(X_test, w, b))  # [[1.2], [2.8]]
``

**2. 用PyTorch实现一个最简单的全连接神经网络**
``python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 推理示例（你的主要工作）
model = SimpleNN(input_dim=784, hidden_dim=128, output_dim=10)
model.eval()  # 切换到评估模式
with torch.no_grad():  # 推理不需要梯度
    output = model(torch.randn(1, 784))
    print(f"输出形状: {output.shape}")  # [1, 10]
``

**3. 实现图像预处理（和训练时保持一致）**
``python
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """图像预处理: 必须与训练时的预处理完全一致！"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)

    # ImageNet预训练模型的标准归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    return img_array[np.newaxis, ...]  # 增加batch维度
``

**4. 导出PyTorch模型为ONNX格式（供Java/Android使用）**
``python
import torch

model = torch.load('trained_model.pth')
model.eval()

# 创建dummy输入（形状需与实际推理一致）
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=12,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("ONNX导出完成, Java/Android端可加载此文件")
``

**5. 实现简单的分类后处理（softmax+top-k）**
``python
import numpy as np

def postprocess_classification(output, labels, top_k=3):
    """output: 模型原始输出(logits), 形状[batch, num_classes]"""
    # 1. softmax转概率
    exp_output = np.exp(output - np.max(output))  # 减max防溢出
    probs = exp_output / np.sum(exp_output)

    # 2. Top-K
    top_k_indices = np.argsort(probs)[-top_k:][::-1]

    # 3. 格式化结果
    return [(labels[i], float(probs[i])) for i in top_k_indices]
``

### 难度: 中等

**6. 用Python调用LLM API并处理流式响应**
``python
import requests
import json

def chat_with_llm_streaming(prompt, api_key, model="gpt-3.5-turbo"):
    """流式调用LLM API (打字机效果)"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}

    response = requests.post(url, headers=headers, json=data, stream=True)
    full_response = ""
    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            line_data = line.decode('utf-8')[6:]
            if line_data == "[DONE]": break
            try:
                json_data = json.loads(line_data)
                delta = json_data['choices'][0]['delta']
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
                    full_response += delta['content']
            except json.JSONDecodeError: pass
    return full_response
``

**7. 实现一个简单的向量相似度检索**
``python
import numpy as np

class SimpleVectorStore:
    def __init__(self, documents, embeddings):
        self.docs = documents
        # 提前归一化, 加速计算
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(self, query_embedding, top_k=3):
        """余弦相似度检索"""
        q_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, q_norm)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.docs[i], float(similarities[i])) for i in top_k_indices]

# 生产环境: 使用Pinecone/Milvus/FAISS替代
``

**8. 实现数据批处理(Batching)推理**
``python
import torch
import numpy as np

def batch_predict(model, data, batch_size=32):
    """批处理推理: 避免一次性加载到显存"""
    results = []
    n = len(data)
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        batch_result = model(batch)
        results.extend(batch_result)
        torch.cuda.empty_cache()  # 清理GPU内存
    return results
``

**9. 实现一个简单的Embedding缓存**
``python
import hashlib
import json

class EmbeddingCache:
    def __init__(self, model, cache_dict=None):
        self.model = model
        self.cache = cache_dict or {}  # 生产环境用Redis

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text: str):
        key = self._get_cache_key(text)
        if key not in self.cache:
            self.cache[key] = self.model.encode(text)
        return self.cache[key]
``

**10. 实现模型推理的性能监控**
``python
import time
from functools import wraps

def monitor_performance(model_name):
    """推理性能监控装饰器 - Java端可用AOP实现类似逻辑"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            print(f"[{model_name}] Latency: {latency_ms:.2f}ms")
            return result
        return wrapper
    return decorator
``

**11. 实现一个简单的LangChain风格的RAG Chain**
``python
class SimpleRAG:
    """简化版RAG实现 - Java版可使用LangChain4j"""
    def __init__(self, vector_store, llm, prompt_template):
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_template = prompt_template

    def query(self, user_question: str, top_k: int = 3):
        # Step 1: 向量检索Top-K相关文档
        query_embedding = self.vector_store.embed(user_question)
        retrieved_docs = self.vector_store.search(query_embedding, top_k)

        # Step 2: 构建Context
        context = "\n\n".join(f"文档{i+1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs))

        # Step 3: 构建完整Prompt并调用LLM
        full_prompt = self.prompt_template.format(context=context, question=user_question)
        return self.llm.chat(full_prompt), retrieved_docs
``

**12. 实现推理结果的缓存**
``python
import hashlib
import time

class InferenceCache:
    """推理结果缓存 - 生产环境用Redis"""
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def predict_with_cache(self, model, model_version, input_data):
        key = hashlib.sha256(f"{model_version}|{str(input_data)}".encode()).hexdigest()
        # 查缓存
        if key in self.cache:
            result, expire = self.cache[key]
            if time.time() < expire:
                return result, True
            del self.cache[key]
        # 缓存miss: 实际推理
        result = model.predict(input_data)
        self.cache[key] = (result, time.time() + self.ttl)
        return result, False
``

### 难度: 困难

**13. 实现数据漂移检测**
``python
import numpy as np
from scipy import stats

class DataDriftDetector:
    def __init__(self, reference_data: np.ndarray):
        self.ref_mean = reference_data.mean(axis=0)
        self.ref_std = reference_data.std(axis=0)

    def detect_drift(self, current_data: np.ndarray, threshold=0.05):
        drift_results = {}
        for feature_idx in range(current_data.shape[1]):
            # 生成参考分布样本
            ref_feature = np.random.normal(self.ref_mean[feature_idx],
                                          self.ref_std[feature_idx],
                                          size=len(current_data))
            cur_feature = current_data[:, feature_idx]
            # KS检验
            ks_stat, p_value = stats.ks_2samp(ref_feature, cur_feature)
            drift_results[feature_idx] = {"ks_stat": float(ks_stat),
                                         "p_value": float(p_value),
                                         "is_drifted": p_value < threshold}
        return drift_results
``

**14. 实现一个简单的LoRA层**
``python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation): 训练低秩矩阵而非全量参数"""
    def __init__(self, in_features, out_features, rank=4, alpha=8):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # 原始权重(冻结)
        self.w_original = nn.Linear(in_features, out_features, bias=False)
        self.w_original.weight.requires_grad = False

        # LoRA新增参数(可训练)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_out = self.w_original(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return original_out + lora_out

# 推理时可将 A*B 合并回W, 无额外推理开销
``

**15. 实现可扩展的推理服务（支持多种模型格式）**
``python
from abc import ABC, abstractmethod
import numpy as np

class BaseInferenceEngine(ABC):
    @abstractmethod
    def load_model(self, path: str): pass
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray: pass

class OnnxEngine(BaseInferenceEngine):
    def __init__(self):
        import onnxruntime as ort
        self.ort = ort

    def load_model(self, path: str):
        self.session = self.ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: input_data})[0]

class InferenceService:
    def __init__(self):
        self.engines = {'.onnx': OnnxEngine()}
        self.current_engine = None

    def load(self, model_path: str):
        for ext, engine in self.engines.items():
            if model_path.endswith(ext):
                engine.load_model(model_path)
                self.current_engine = engine
                return
        raise ValueError(f"不支持: {model_path}")

    def predict(self, input_data):
        return self.current_engine.predict(input_data)
``

**16. 实现动态批处理推理优化器**
``python
import asyncio
from collections import deque

class DynamicBatchingOptimizer:
    """动态批处理: 把短时间内到达的多个请求合并为一个batch推理"""
    def __init__(self, model, max_batch_size=32, max_wait_ms=5):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.request_queue = deque()
        asyncio.create_task(self._process_loop())

    async def predict(self, input_data):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.request_queue.append((future, input_data))
        return await future

    async def _process_loop(self):
        while True:
            if not self.request_queue:
                await asyncio.sleep(0.001)
                continue
            batch = []
            start = asyncio.get_event_loop().time()
            while (len(batch) < self.max_batch_size and
                   (asyncio.get_event_loop().time() - start) < self.max_wait_ms and
                   self.request_queue):
                batch.append(self.request_queue.popleft())
            # 推理并分发结果
            try:
                results = self.model(torch.cat([b for _, b in batch], dim=0))
                for (f, _), r in zip(batch, results):
                    f.set_result(r)
            except Exception as e:
                for f, _ in batch:
                    f.set_exception(e)
# Java端: 可使用Disruptor + 线程池实现类似机制
``

**17. 实现模型版本管理和热切换**
``python
import threading
from collections import OrderedDict

class ModelVersionManager:
    """多版本共存 + 流量分配 + 热切换"""
    def __init__(self, max_versions=3):
        self.models = OrderedDict()
        self.traffic_allocations = {}
        self.max_versions = max_versions
        self.current_version = None
        self.lock = threading.RLock()

    def load_version(self, version, model_path, traffic_ratio):
        with self.lock:
            if len(self.models) >= self.max_versions:
                oldest = next(iter(self.models))
                if oldest != self.current_version:
                    self.models.popitem(last=False)
            self.models[version] = self._load_model(model_path)
            self.traffic_allocations[version] = traffic_ratio
            if self.current_version is None:
                self.current_version = version

    def predict(self, request_id, input_data):
        import hashlib
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100 / 100
        cumulative = 0
        selected = self.current_version
        for v, ratio in self.traffic_allocations.items():
            cumulative += ratio
            if hash_val < cumulative:
                selected = v
                break
        with self.lock:
            model = self.models[selected]
        return model.predict(input_data), selected
``

**18. 实现RAG检索的重排序优化器**
``python
class RAGWithRerank:
    """两阶段检索: 1)向量粗召回Top-100 2)Cross-Encoder精排Top-10"""
    def __init__(self, vector_store, reranker, llm, recall_k=100, rerank_k=10):
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.recall_k = recall_k
        self.rerank_k = rerank_k

    def query(self, question: str):
        # Phase 1: 向量粗召回
        candidates = self.vector_store.search(question, k=self.recall_k)
        # Phase 2: Cross-Encoder精排
        pairs = [[question, doc] for doc, _ in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip([d for d, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:self.rerank_k]]
        # Phase 3: LLM生成答案
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(top_docs))
        prompt = f"基于以下参考资料回答问题。\n\n参考资料:\n{context}\n\n问题: {question}"
        return self.llm.generate(prompt), top_docs
``

**19. 实现Token成本估算器**
``python
import tiktoken
from collections import defaultdict

class TokenCostEstimator:
    """LLM API调用成本监控 - Java端可用jTokkit库"""
    def __init__(self):
        self.encoders = {}
        self.costs = defaultdict(float)
        self.prices = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015}
        }

    def count_tokens(self, text: str, model: str) -> int:
        if model not in self.encoders:
            self.encoders[model] = tiktoken.encoding_for_model(model)
        return len(self.encoders[model].encode(text))

    def estimate_cost(self, input_text, output_text, model):
        if model not in self.prices: return 0.0
        in_tokens = self.count_tokens(input_text, model)
        out_tokens = self.count_tokens(output_text, model)
        price = self.prices[model]
        cost = (in_tokens / 1000 * price["input"]) + (out_tokens / 1000 * price["output"])
        self.costs[model] += cost
        return cost
``

**20. 实现模型服务Health Check**
``python
import time
import threading
from datetime import datetime

class ModelHealthChecker:
    """模型服务健康检查 - 比传统服务多了模型推理测试"""
    def __init__(self, model, test_input):
        self.model = model
        self.test_input = test_input
        self.history = []
        self._start_monitor()

    def _start_monitor(self):
        def monitor_loop():
            while True:
                try:
                    self.history.append(self.check_full())
                    if len(self.history) > 100:
                        self.history.pop(0)
                except Exception as e:
                    self.history.append({"status": "ERROR", "error": str(e)})
                time.sleep(60)
        threading.Thread(target=monitor_loop, daemon=True).start()

    def check_full(self):
        start = time.time()
        try:
            output = self.model(self.test_input)
            latency = (time.time() - start) * 1000
            return {"status": "HEALTHY", "latency_ms": latency, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"status": "UNHEALTHY", "error": str(e)}

# Java端: Actuator health endpoint + 自定义HealthIndicator
