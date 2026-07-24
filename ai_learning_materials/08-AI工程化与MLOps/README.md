# ⚙️ 08 - AI工程化与MLOps 章节导览

> **进阶岗位必备：中高级/大厂AI平台工程师、MLOps、推理优化工程师的核心考点。**
> 如果你目标是大厂高薪 / AI平台架构方向，必须学。中小公司应用岗位快速浏览概念即可。
> 预计学习周期：2周 (14天) | 目标掌握度：⭐⭐⭐⭐ L4熟练级
> 配套项目路径：`../../08-mlops/mlflow/` / `vllm/`

---

## 📚 本章节文件索引

| 文件名 | 内容 | 优先级 |
|-------|------|--------|
| **README.md** (本文) | MLOps生命周期 + 两大核心工具 | ⭐⭐⭐ 先读 |
| **MLflow全生命周期.md** | Tracking实验追踪/Registry模型注册/PyFunc万能格式 + Spring集成 | ⭐⭐⭐⭐ 必学 |
| **vLLM大模型推理优化.md** ⭐⭐⭐⭐⭐ | PagedAttention核心 + 连续批处理 + 7种推理优化技术 | ⭐⭐⭐⭐⭐ 面试必考 |
| **模型优化技术.md** | 量化三板斧(PTQ/QAT/AWQ/GPTQ) + 剪枝 + 蒸馏 + 算子融合 | ⭐⭐⭐⭐ 推理优化 |
| **容器化与K8s部署.md** | DockerFile最佳实践 + GPU调度 + KEDA弹性伸缩 | ⭐⭐⭐⭐ 平台必做 |
| **数据漂移监控与重训练.md** | KS检验 / PSI / 概念漂移检测 + Airflow自动重训练Pipeline | ⭐⭐⭐⭐ 生产稳定 |
| **代码实战.md** | MLflow完整XGBoost追踪注册 + vLLM Llama3部署 | ⭐⭐⭐⭐ 必做 |
| **面试题库.md** | 80道MLOps面试题 + 答案 (推理显存估算公式必背) | ⭐⭐⭐⭐ 大厂面必出 |
| **GitHub项目推荐.md** | MLflow/vLLM/Kubeflow源码阅读路线 | ⭐⭐⭐ 参考 |

---

## 🔄 MLOps 全生命周期图 (CI/CD + CT = 持续训练)

```mermaid
graph TD
    subgraph 业务侧: 数据采集与标注
        DB1[业务数据库] --> DP[数据处理 Pipeline]
        LOG[用户行为日志] --> DP
        DP --> LAB[标注平台 Label Studio]
    end

    subgraph ML 实验与开发 ⭐ MLflow Tracking
        LAB --> DS[数据科学家 Notebook 快速实验]
        DS --> TR{MLflow Tracking Server}
        TR -->|log_param/log_metric/log_model| PG[(Postgres 元数据)]
        TR -->|log_artifact| S3[(MinIO/S3 模型文件/图)]
        PG --> UI[MLflow Web UI 对比100次实验]
    end

    subgraph 模型注册与审批 ⭐ MLflow Registry
        UI --> MR[模型注册中心 Model Registry]
        MR -->|Version 1/2/3 递增| MR
        MR --> Stage1[Stage Staging 准生产验证]
        Stage1 -->|自动化评估通过| Stage2[Production 生产切流量]
        Stage1 -->|指标不达标| REJ[❌ 打回Archived]
        Stage2 -->|新版本上线| OLD[旧版本 Archived归档保留]
    end

    subgraph CI/CD 自动化部署 ⭐ Jenkins/ArgoCD + K8s
        MR -->|新版本Webhook触发| JENKINS[Jenkins CI: 评估+镜像构建]
        JENKINS -->|打包镜像: 模型+推理服务| REG[镜像Harbor仓库: v1.2.3]
        REG --> K8s[K8s 集群 + ArgoCD GitOps部署]
        K8s --> GPU[NVIDIA GPU 节点池 + nvidia-device-plugin]
        K8s --> HPA[KEDA HPA 按QPS/P99延迟弹性扩缩Pod]
    end

    subgraph 生产监控 + 自动化闭环 ⭐ Prometheus/Grafana
        K8s --> MET[指标采集 推理P99/QPS/错误率]
        K8s --> DRIFT[数据漂移 PSI/KS检验 特征分布监控]
        MET --> ALERT[AlertManager 超阈值告警: 钉钉/企微/邮件]
        DRIFT -->|漂移分数>0.3触发| RETRAIN[Airflow 自动重训练Pipeline]
        RETRAIN --> LAB[重新增量标注+训练]  %% 闭环循环
    end

    %% 线上验证模式
    K8s --> AB[A/B测试 新旧模型 各50%流量]
    K8s --> SHADOW[影子模式 Shadow: 新模型后台跑不生效 只记录预测对比]
```

---

## 🧮 核心1：vLLM - LLM高吞吐推理引擎 ⭐⭐⭐⭐⭐

> **面试第一题必考：vLLM为什么比HuggingFace Transformers快10-20倍？** → 答PagedAttention + Continuous Batching + Chunked Prefill

### 💡 PagedAttention = 操作系统虚拟内存 搬进GPU

| 传统KV Cache的痛点 (HF Transformers) | vLLM PagedAttention的解法 ⭐ |
|-----------------------------------|---------------------------|
| ❌ 每个请求序列长度不同，必须**按最大长度分配连续显存** → 大量Padding碎片，显存利用率只有**30%~40%** | ✅ 把KV按固定大小Block切分(如每Block=16个Token) → 物理上**不需要连续**，用逻辑BlockTable映射 → 显存利用率**90%~95%** |
| ❌ 静态批处理：攒够N个请求一起跑，慢的拖死快的 | ✅ **Continuous Batching 连续批处理**：迭代级别调度，哪个请求生成完立刻用新请求填空位，不用等别人，GPU一直忙 |
| ❌ 长Prefill占住GPU，Decode请求排队 → 延迟抖动大 | ✅ **Chunked Prefill 分块Prefill**：长输入切成小块，插在Decode间隙里算，P99延迟抖动↓80% |

### 🧠 KV Cache 显存估算公式 ⭐⭐⭐⭐⭐ (面试让你现场算！)

```
公式必须背：
KV Cache显存 = 2 × n_layers × n_seq_len × d_model × bytes_per_value × batch_size
         ↑↑↑           ↑          ↑            ↑                ↑
         K+V两份      层数     上下文长度    每层隐层维度       FP16=2字节/FP8=1字节
```

**例题 ( Llama3-8B, 32层, d=4096, 单用户16K上下文, FP16 )：**
> 2 × 32 × 16384 × 4096 × 2字节 = 8,589,934,592 字节 = **8 GB / 单用户！**
> 
> 那8个并发用户呢？64GB！A100 80GB塞不下，怎么办？
> 
> → FP8 KV Cache (×2 减半) = 4GB/人 → 8人只要32GB → A100塞20人没问题。
> → 再加AWQ INT4权重(原FP16 8B模型16GB → 4GB)，总显存：4GB模型 + 32GB KV = 36GB，还剩44GB给更多用户！

### ⚡ vLLM 性能优化7招 背下来 = 高薪

| 技术 | 效果 | 部署命令加参数 |
|-----|------|--------------|
| 1. **AWQ/GPTQ INT4量化** | 显存÷2.2, 速度×1.5 | `--quantization awq` |
| 2. **Chunked Prefill** | 延迟抖动↓80% | `--enable-chunked-prefill` |
| 3. **Continuous Batching** | 吞吐×2~5 默认开启 | (vLLM默认自带) |
| 4. **Speculative Decoding 投机解码** | 解码速度×2~3,分布严格等价 | `--speculative-model meta-llama/Llama-3-8B-Instruct --num-speculative-tokens 5` |
| 5. **Prefix Caching前缀缓存** | 共享System Prompt KV复用×4 | `--enable-prefix-caching` |
| 6. **FP8 KV Cache** | KV显存再÷2, 上下文加倍 | (CUDA 8.9+ Ampere+自动) |
| 7. **LoRA Adapter热切换** | 多任务1份权重+N个LoRA,显存↓N倍 | `--enable-lora --max-loras 8` |

> 💰 **简历黄金句式**: `「vLLM部署Llama3.1-70B：双卡A100张量并行TP=2 + AWQ 4bit + Chunked Prefill + Prefix Caching，支持32K上下文，吞吐从HF 64→218 req/s (3.4×)，P99延迟2.1s→650ms，月GPU成本从$86K→$36K 降低58%」`

---

## 📈 核心2：MLflow - 把实验/模型/部署管起来

### 四大组件一句话记忆

| 组件 | 解决什么问题 | 类比传统开发 |
|-----|------------|------------|
| 🧪 **Tracking 实验追踪** | 100次实验参数/指标/模型 谁好谁坏？不然混了记不清 | 像 Jenkins + JUnit 结果记录 |
| 📦 **Projects 可复现** | 同事/服务器复现训练环境一致？ | Dockerfile + requirements.txt |
| 🗃️ **Models 标准模型格式** | sklearn/xgb/pytorch/tf 格式不一，部署统一接口？ | Maven/Jar包标准化 |
| 🏛️ **Registry 模型注册** | v1/v2/v3 版本管理 Staging/Prod流转？ | Docker镜像仓库 Harbor |

### 🔑 PyFunc = 万能格式 (面试重点)

```python
# 不管底层是sklearn/xgb/pytorch/onnx/手写规则
# 只要包装成pyfunc接口，生产调用只有一行！
import mlflow.pyfunc

# 加载Production最新版
model = mlflow.pyfunc.load_model("models:/fraud-detection-xgb/Production")

# ⭐ 不管什么模型 统一接口 predict(pandas DataFrame) = 输出
predictions = model.predict(X_production_input_df)

# 好处：部署代码一模一样，换模型不改一行代码！
# 坏处：如果输入列变了，那必须新版+输入校验Schema
```

### 🔄 Model Registry Stage 流转 (面试Webhook/CICD)

```
实验阶段 → 注册Version 1 → Staging阶段 → Production → Archived归档
                  ↑             |              ↑
                  |            自动化Staging评估
                  |             通过才转 否则打回
                  | 新版本迭代递增Version
```

> 典型自动化Webhook流程：`新版本自动注册 → Jenkins Staging离线评估1000条Golden集 → AUC>0.93通过 → 自动切10%灰度流量 → 线上A/B 7天指标不劣化 → 自动切100%全量 + 旧版Archive + 发通知企微`

---

## 🚨 核心3：数据漂移 + 概念漂移 = 线上模型效果悄悄下滑的元凶

| 漂移类型 | 定义 | 检测方法 | 触发阈值 |
|---------|------|---------|---------|
| 📊 **数据漂移 Data Drift (协变量偏移)** | 输入特征X分布变了 (如原来客群25-35，突然涌入大量50+新用户) | **PSI (Population Stability Index)**：<br>`Σ (实际占比-预期占比) × ln(实际占比/预期占比)` <br> 或 KS检验/Wasserstein距离 | PSI < 0.1 没问题<br>0.1~0.25 监控告警<br>>0.25 触发重训练 |
| 🧩 **概念漂移 Concept Drift** | X没变，但X→Y的映射关系变了 (如消费者习惯因疫情经济环境改变) | AUC/准确率/F1线上线下指标对比 + 标注小样本回测 | AUC下降>3% 触发重训练 |
| 🔄 **周期性漂移** | 双11/618大促/节假日/季节性 (夏天羽绒服销量低) | 日历特征 + 专门的大促模型 | 提前7天切专用模型 |

> 🚨 一个真实踩坑案例：某信贷模型2023年上线AUC 0.92，2024年4月偷偷降到0.83没人发现！为什么？→ 没上漂移监控，等到坏账率涨了30%才事后发现。上线漂移监控 = 模型界的APM。

---

## 🎯 章节结业标准

- [ ] 能背诵KV Cache显存估算公式 + 现场算一道Llama参数题
- [ ] 能解释PagedAttention vs 传统KV Cache的3个改进点
- [ ] 能说出Continuous Batching vs 静态Batching区别和为什么吞吐差几倍
- [ ] 能说出MLflow四大组件 + PyFunc万能格式的好处
- [ ] 能解释MLflow Staging/Production/Archived三阶段流转机制
- [ ] 能说出PSI的含义 + 数据漂移和概念漂移的区别和检测方法
- [ ] 能独立部署vLLM + Llama3 + 验证OpenAI格式API可用
- [ ] 面试题库正确率 ≥ 75%