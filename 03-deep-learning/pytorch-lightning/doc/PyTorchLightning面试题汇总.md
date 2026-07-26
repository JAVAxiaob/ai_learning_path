# PyTorch Lightning面试题汇总 (35题)

> 位置: 03-deep-learning/pytorch-lightning/doc/
> 配套文档: PyTorchLightning模块化训练框架.md | PyTorchLightning流程图详解.md | PyTorchLightning性能优化重难点.md

---

## 一、基础架构题（8题）

### Q1. LightningModule vs PyTorch原生nn.Module区别是什么？7个钩子方法对比forward/training_step/validation_step/configure_optimizers

### Q2. DataModule 4个钩子：prepare_data(单卡只运行一次 vs setup(每卡运行  作用区别？为什么不能prepare_data不能调self.model.to(device)

### Q3. Trainer 接管了PyTorch哪些工程细节？zero_grad/backward/step/eval模式/no_grad/DDP/AMP/保存ckpt

### Q4. training_step 返回值三种写法return有何区别？return loss 对比return {'loss':loss, 'log':{'acc':acc}对比self.log()

### Q5. self.log() 的 on_step / on_epoch / prog_bar / logger 4个参数详解？为什么不要手动写TensorBoard writer

### Q6. Callbacks vs Hooks区别？非侵入式Callbacks插件机制对比重写Module钩子的解耦思想

### Q7. strategy参数的值: ddp vs ddp_spawn vs ddp_fork 三种分布式启动方式区别？Windows/MacSpawn原因

### Q8. auto_lr_find自动学习率查找原理：LR Finder怎么实现的？loss斜率最小点取lr

---

## 二、性能优化题（10题）

### Q9. precision参数: 16-mixed vs bf16-mixed vs 32-true vs 64-true 选型对比表 新显卡3090/4090首选bf16

### Q10. FP16 GradScaler动态缩放原理? loss×65536倍放大后再backward再除，为何解决梯度下溢？

### Q11. gradient_clip_val梯度裁剪norm原理/training_step范数裁剪方式/value裁剪对权重梯度超过阈值直接clip掉怎么选

### Q12. accumulate_grad_batches梯度累积：等效大batch效果？为什么梯度累积×4不等于真实Batch×4收敛速度略差于真Batch

### Q13. activation_checkpointing梯度检查点：速度换显存Tradeoff原理/怎么选开启哪些层？第一层和最后层别开浪费

### Q14. DDP AllReduce梯度同步流程：NCCL Ring算法如何保证每张卡相同权重？为什么DDP比DataParallel快很多

### Q15. FSDP分片策略对比SHARD_GRAD_OP/FULL_SHARD/HYBRID_SHARD ZeRO2/ZeRO3显存省多少

### Q16. Flash Attention2为什么O(N)显存比传统O(N²)快2-4倍？Tiling+Recomputation原理

### Q17. torch.compile三种模式 default/max-autotune/reduce-overhead效果对比表 PyTorch2.0 JIT

### Q18. num_workers/pin_memory/prefetch_factor/persistent_workers DataLoader四件套配置的建议值/常见错误

---

## 三、分布式训练题（7题）

### Q19. prepare_data 只在Rank0执行一次，setup在每张卡都执行的设计原理？多卡同时写数据集下载冲突解决

### Q20. DistributedSampler自动切分数据集1/N不重叠：多卡shuffle=True每卡为什么epoch不同随机种子

### Q21. 梯度同步AllReduce vs ReduceScatter+AllGather(FSDP ZeRO3) 带宽O(N²)/O(N) 区别

### Q22. SyncBatchNorm 跨卡同步BN统计均值方差：单卡batch_size=2导致BatchNorm效果差怎么办

### Q23. 多机多卡 node=2 machines × devices=8 MASTER_PORT/MASTER_ADDR/NCCL_SOCKET_IFNAME环境变量

### Q24. 断点续训 resume_from_checkpoint：恢复optimizer状态/scheduler步数/RNG随机种子 训练无缝接着跑

### Q25. DeepSpeed ZeRO Stage1/2/3/offload区别：70B模型怎么调参数策略

---

## 四、工程实践题（10题）

### Q26. EarlyStopping回调参数monitor/delta/patience/restore_best_weights 连续多少轮没涨 恢复最优权重原理

### Q27. ModelCheckpoint save_top_k/save_last/every_n_train_steps/monitor=val_acc保存最好3个best模型

### Q28. Trainer(fast_dev_run=True)快速Debug模式：跑1批2batch验证有没有bug怎么加速开发循环没报错

### Q29. Trainer(limit_train_batches=0.1)小数据10%过拟合Sanity Check验证pipeline通不通

### Q30. 梯度监控Callback梯度范数分布/权重范数: 0.00001=梯度消失/1000+=梯度爆炸

### Q31. trainer.validate() vs trainer.test()区别：为什么test必须训练做完最后一次性跑test集

### Q32. profiler='simple'/'advanced'/'pytorch' 找训练瓶颈Top10耗时算子/Kernel 10%时间花在哪层

### Q33. swa学习率计划/StochasticWeightAveraging最后15%训练步平均权重提升1-2%点原理

### Q34. 日志记录多logger同时记录TB+W&B+MLflow多协作：多次实验怎么横向对比

### Q35. 生产ONNX导出PL模型：LightningModule→torch.onnx.export(model.to_onnx()流程注意事项