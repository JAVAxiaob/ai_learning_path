# MLflow实验追踪与模型管理面试题汇总 (30题)

> 位置: 07-AI项目工程化/mlflow/doc/
> 配套文档: MLflow实验追踪与模型管理.md | MLflow流程图详解.md | MLflow性能优化重难点.md

---

## 一、基础架构（8题）

### Q1. MLflow四大组件: Tracking/Projects/Models/Registry 各负责什么职责边界

### Q2. 为什么不能用本地Excel/Notebook+Git 代替MLflow Tracking? 实验多实验对比表格/指标曲线/Artifacts 为什么手动画表格对比

### Q3. Params vs Metrics vs Artifacts vs Tags 四类存什么区别? 超参数/指标/文件/标签 举例各存什么

### Q4. MLflow标准模型格式5种flavor: python_function/sklearn/pytorch/onnx/java 跨框架加载predict

### Q5. Experiment vs Run的层级关系: Experiment包含N个Run, 命名规范team/project/任务名

### Q6. mlflow.start_run() 嵌套父子Run嵌套场景: 5折交叉验证5个子Run挂在同一个父Experiment下

### Q7. set_tracking_uri三种存储后端: 本地文件/SQLite/PostgreSQL+S3 存储对比场景

### Q8. Projects 为什么能一键复现实验? MLproject元文件+conda.yaml+代码版本依赖 Git Pull

---

## 二、Tracking 深度（7题）

### Q9. 数据爆炸问题Metrics千万级metrics行数: 每天1亿行metrics怎么解? 4层防护采样/批量提交/数据库索引/降采样查询

### Q10. log_metric vs log_metrics批量写法: 10万HTTP请求 vs 1次批量请求 性能差10倍

### Q11. Artifacts大文件爆炸: 每个epoch存1GB模型100epoch=100GB/实验: save_top_k=3只留最好3个

### Q12. PostgreSQL 3大表索引优化: params/metrics/runs 联合索引如何加 慢查询从10s→100ms

### Q13. Nested Run嵌套场景: AutoML 100次超参数搜索如何组织层级化树形展示

### Q14. Tags vs Parameters: 实验乱码乱传参自动记录系统信息/Git commit/diff源码补丁自动记录 复现

### Q15. Search Runs UI 对比5个Run: 选2-5个Runs对比Params差异高亮不同超参Metrics曲线

---

## 三、Registry & 生命周期（7题）

### Q16. 模型注册5阶段状态机: Unregistered/Staging/Production/Archived流转条件审批

### Q17. Staging→Production质量门禁: 离线精度阈值+延迟<5ms+A/B测试P值<0.05显著性检验

### Q18. 影子部署Shadow Mode A/B测试: Istio按比例切流量+2版本并行, 线上数据双写指标看新旧误差

### Q19. 模型版本语义化Version: v1.2.3含义Major/Minor/Patch 什么时候升级策略兼容性回滚

### Q20. 模型漂移触发重训练: 数据分布漂移KS检验PSI>0.2 自动触发训练pipeline重训

### Q21. Registry vs 多团队命名规范: team/department/project/modelname 避免冲突覆盖

### Q22. 审批工作流代码: transition_model_version_stage API自动化CI流水线不能手动点按钮

---

## 四、企业级部署（8题）

### Q23. 高可用部署架构: 2 Tracking Server双节点+Nginx+PostgreSQL主从+MinIO分布式

### Q24. Tracking vs Gunicorn多进程workers=8 threads=4 替换mlflow server启动

### Q25. 权限认证: OIDC/SAML单点登录+Nginx Basic中间件RBAC角色权限

### Q26. 生命周期分层存储: SSD热30天→HDD温180天→Glacier冷归档7年合规

### Q27. CI/CD集成: GitHub Actions自动注册→评估→A/B→Production ArgoCD部署Seldon

### Q28. 可观测性: Prometheus+Grafana监控活跃实验Run/存储增长率/慢查询

### Q29. 数据库连接池: SQLAlchemy pool_size参数 防止TooManyConnections

### Q30. 多租户隔离: 实验名称前缀/独立命名空间Bucket权限 避免A团队看B团队模型