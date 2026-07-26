# MLflow 实验追踪与模型管理流程图详解

> 位置: 07-AI项目工程化/mlflow/doc/
> 配套文档: MLflow实验追踪与模型管理.md | MLflow性能优化重难点.md | MLflow面试题汇总.md

---

## 一、MLflow四大组件总览

```mermaid
flowchart TD
    subgraph 数据科学家日常开发
        DS["Jupyter Notebook / Python训练脚本"]
        DS --> TR["MLflow Tracking<br/>🔍 实验/参数/指标/Artifacts追踪"]
    end

    subgraph MLflow 4大组件
        TR --> COMP1["组件1: MLflow Tracking<br/>记录实验: params/metrics/artifacts/code<br/>本地/数据库/服务端存储"]
        TR --> COMP2["组件2: MLflow Projects<br/>📦 可复现打包: 代码+环境+依赖Docker/conda.yaml<br/>别人mlflow run一键复现你的实验"]
        TR --> COMP3["组件3: MLflow Models<br/>🚢 标准化模型格式<br/>5种打包风格: python_function/sklearn/pytorch/onnx/java<br/>部署到: Batch/SageMaker/Docker/TorchServe"]
        COMP3 --> COMP4["组件4: MLflow Model Registry<br/>📚 模型注册中心/版本管理/Stage生命周期<br/>Staging → Production → Archived"]
    end

    subgraph 生产部署
        COMP4 --> DEPLOY1["Seldon Core / KFServing K8s部署"]
        COMP4 --> DEPLOY2["REST API / gRPC 在线服务"]
        COMP4 --> DEPLOY3["Spark Batch / Airflow 离线批处理"]
        COMP4 --> DEPLOY4["Shadow Deployment 影子流量A/B测试"]
    end
```

---

## 二、MLflow Tracking 实验记录完整生命周期

```mermaid
sequenceDiagram
    participant User as 训练代码 train.py
    participant MLflow as MLflow Tracking Client
    participant Store as BackendStore (PostgreSQL+S3)
    participant UI as MLflow Web UI

    Note over User: 代码开头
    User->>MLflow: 1. mlflow.set_tracking_uri("http://mlflow.internal:5000")
    User->>MLflow: 2. mlflow.set_experiment("resnet50_imagenet_exp5")
    Note over MLflow: 如果实验不存在→自动创建 / experiment_id=123

    User->>MLflow: 3. with mlflow.start_run(run_name="bs128_lr4e4_adamw"):
    Note over MLflow: 创建run_id=7f9a3d, 状态=RUNNING, 记录开始时间/用户/源码Git Commit

    loop 训练循环Epoch 1~100
        User->>User: train 1 epoch
        User->>User: val_loss=0.234 val_acc=0.945
        User->>MLflow: 4. mlflow.log_param("batch_size", 128)
        User->>MLflow: 5. mlflow.log_param("optimizer", "AdamW")
        User->>MLflow: 6. mlflow.log_param("lr_scheduler", "CosineAnnealing")
        User->>MLflow: 7. mlflow.log_metric("val_loss", 0.234, step=epoch)
        User->>MLflow: 8. mlflow.log_metric("val_acc", 0.945, step=epoch)
        MLflow->>Store: 写入PostgreSQL params/metrics表
    end

    Note over User: 保存训练产出
    User->>MLflow: 9. mlflow.log_artifact("best_model.pth")  # 权重
    User->>MLflow: 10. mlflow.log_artifact("train_log.csv")  # 日志
    User->>MLflow: 11. mlflow.log_figure(plt.gcf(), "confusion_matrix.png")  # 混淆矩阵图
    User->>MLflow: 12. mlflow.pytorch.log_model(model, "resnet50_model")  # 标准模型格式
    MLflow->>Store: 上传文件到S3: s3://mlflow/123/7f9a3d/artifacts/...

    User->>MLflow: 13. with块自动结束 run_status=FINISHED
    MLflow->>Store: 更新run状态为FINISHED + 结束时间戳

    UI->>Store: 14. 查询实验123 所有Run对比
    Store-->>UI: 返回所有Run表格
    Note over UI: 可筛选/排序/对比2-5个Run的Params差异/Metric曲线
    UI->>Store: 下载best_model.pth或直接注册到Registry
```

---

## 三、Model Registry 模型注册生命周期状态机

```mermaid
stateDiagram-v2
    [*] --> 开发训练
    开发训练 --> 实验产出Run: log_model()
    实验产出Run --> Unregistered: 模型文件在Artifacts但没注册
    Unregistered --> Staging: 【操作1】UI点击Register Model注册<br/>命名"cv/resnet50_detector" + 创建Version 1

    Staging --> QualityCheck: 自动化测试/质检Pipeline触发
    QualityCheck --> Staging_Approved: ✅ 评估指标达标(val_acc>93%,延迟<5ms)
    QualityCheck --> Archived: ❌ 指标不合格 直接归档淘汰

    Staging_Approved --> A_B_TEST: 【操作2】切10%流量A/B Shadow影子模式
    A_B_TEST --> Staging_Approved: 线上P99延迟不达标 → 继续调优
    A_B_TEST --> Production: 【操作3】Version1审核通过<br/>Promote升级到生产✅<br/>流量100%切给Version 1

    Production --> Production: 持续线上服务几个月
    Production --> Staging2: Version2出来了 Staging挑战老版本
    Staging2 --> A_B_TEST2: 10%流量切给V2做A/B
    A_B_TEST2 --> Production2: V2更好, V1→Archived归档
    A_B_TEST2 --> Production: V2更差, V2归档V1继续生产

    Production2 --> Archived: 半年后V3上线→V2归档
    Staging_Approved --> Archived: 被新版本替代淘汰
    Production --> Archived: 淘汰退役

    Archived --> [*]: 最终状态 永久保存历史可回溯
```

---

## 四、MLflow Projects 可复现实验环境打包流程

```mermaid
flowchart TD
    subgraph 本地开发 你机器上
        CODE["train.py 训练代码"]
        YAML["conda.yaml / requirements.txt<br/>name: pytorch_env<br/>dependencies:<br/>- pytorch=2.1<br/>- torchvision=0.16<br/>- mlflow=2.9"]
        META["MLproject 元文件<br/>name: ResNet50-ImageNet<br/>entry_points:<br/>  main:<br/>    parameters:<br/>      batch_size: {type:int, default:64}<br/>      lr: {type: float, default:0.001}<br/>    command: 'python train.py --bs={batch_size} --lr={lr}'"]
    end

    CODE --> GIT
    YAML --> GIT
    META --> GIT["Git Commit 提交到GitHub: yourname/resnet-exp"]

    subgraph 同事机器 / 服务器
        REPRODUCE["一键命令 同事复现实验:"]
        REPRODUCE --> CMD["$ mlflow run https://github.com/yourname/resnet-exp<br/>                 -P batch_size=128 -P lr=4e-4<br/>                 --experiment-name resnet50_repro"]
        CMD --> STEP1["✅ 自动步骤1: Git Clone代码到临时目录"]
        STEP1 --> STEP2["✅ 自动步骤2: conda创建新环境<br/>安装conda.yaml中所有指定版本依赖<br/>(避免污染全局!)"]
        STEP2 --> STEP3["✅ 自动步骤3: 参数注入到命令行占位符"]
        STEP3 --> STEP4["✅ 自动步骤4: python train.py 运行训练<br/>自动把所有log/artifacts记录到Tracking Server"]
    end

    STEP4 --> ENSURE["🤝 可复现: 同代码+同依赖+同参数=相同结果<br/>(消除'我机器上跑得好好的'扯皮)"]
```

---

## 五、部署架构 企业级 Tracking Server + 存储后端

```mermaid
flowchart TD
    subgraph 用户侧 数据科学家工作站
        DS["100个数据科学家<br/>每人本地Python代码<br/>mlflow.set_tracking_uri"]
    end

    subgraph MLflow服务端集群 Nginx+2节点高可用
        DS --> LB["Nginx 负载均衡<br/>HTTPS终止+OIDC认证鉴权"]
        LB --> TR1["MLflow Tracking Server #1<br/>Gunicorn workers=8"]
        LB --> TR2["MLflow Tracking Server #2<br/>Gunicorn workers=8"]
    end

    subgraph 后端存储
        TR1 --> PG["主数据库 PostgreSQL 15<br/>存储: experiments/runs/params/metrics/tags<br/>主从复制 每日备份"]
        TR2 --> PG
        TR1 --> S3["对象存储 MinIO/S3<br/>Artifacts: .pth模型/CSV日志/PNG图/onnx<br/>分桶 生命周期归档到冷存储"]
        TR2 --> S3
    end

    subgraph 模型服务
        S3 --> REG["MLflow Model Registry<br/>生产模型版本审批流"]
        REG --> K8S["Seldon Core on K8s 部署<br/>A/B Staging/Production 2组Deployment<br/>Istio按比例切流量"]
    end

    subgraph 监控运维
        PG --> MONITOR["Grafana监控面板<br/>活跃实验数/日均Runs数/存储增长趋势<br/>S3磁盘告警/PostgreSQL慢查询"]
        S3 --> MONITOR
    end
```

---

## 六、模型从训练到生产的全流水线 CI/CD

```mermaid
flowchart LR
    subgraph 阶段1 开发
        A[数据科学家本地Notebook] --> B[MLflow Tracking记录N次实验Run]
    end

    subgraph 阶段2 注册
        B --> C[选Top3最优Run → 注册到Registry → Staging版本]
    end

    subgraph 阶段3 CI评估
        C --> D[GitHub Actions 触发评估Pipeline<br/>离线测试集评估精度+延迟+吞吐量]
        D --> E{评估达标?<br/>Precision>95%, P95<5ms}
        E -->|No| F[驳回 通知数据科学家继续调]
        E -->|Yes| G[Approved 通过 打标签]
    end

    subgraph 阶段4 CD部署
        G --> H[ArgoCD部署SeldonManifest到Staging K8s命名空间]
        H --> I[Istio 5%影子流量]
        I --> J{影子P99延迟+错误率监控7天没问题?}
        J -->|No| K[回滚 继续调优]
        J -->|Yes| L[全量切换 Production 100%流量]
    end

    subgraph 阶段5 运行
        L --> M[Prometheus指标采集+Grafana告警]
        M --> N{模型漂移检测? 数据分布变了精度掉3%+}
        N -->|Yes 触发重训| A
        N -->|No| M
    end
```