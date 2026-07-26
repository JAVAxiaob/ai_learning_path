# PyTorch Lightning 流程图详解

> 位置: 03-deep-learning/pytorch-lightning/doc/
> 配套文档: PyTorchLightning模块化训练框架.md | PyTorchLightning性能优化重难点.md | PyTorchLightning面试题汇总.md

---

## 一、PL核心架构7模块流程图

```mermaid
flowchart TD
    subgraph DataModule数据
        A1["prepare_data() ⭐只在1卡/GPU执行一次<br/>→ 下载数据集/解压/预处理Tokenize"]
        A1 --> A2["setup(stage) 每张卡执行<br/>→ train_dataset/val_dataset/test_dataset实例化"]
        A2 --> A3["train_dataloader() DataLoader批量<br/>→ num_workers/pin_memory/shuffle"]
        A2 --> A4["val_dataloader() 验证集Dataloader<br/>→ 分布式DistributedSampler"]
    end

    subgraph LightningModule模型
        B1["__init__ 定义层: Transformer/ViT层"]
        B1 --> B2["forward(x) 推理路径<br/>⚠️ forward只写前向推理"]
        B2 --> B3["training_step(batch, idx) 训练单步<br/>→ 算loss return loss/log_dict"]
        B2 --> B4["validation_step(batch, idx) 验证单步<br/>→ 算val_acc/val_loss记录"]
        B2 --> B5["configure_optimizers()<br/>→ AdamW + CosineAnnealingLR Scheduler"]
    end

    subgraph Trainer引擎⭐核心
        C1["Trainer(max_epochs=100, devices=4, strategy='ddp')<br/>→ 接管所有工程细节"]
        C1 --> C2["Callbacks回调: EarlyStopping/ModelCheckpoint/LearningRateMonitor<br/>→ 插件机制 非侵入加功能"]
        C1 --> C3["Logger日志: TensorBoard/W&B/MLflow<br/>→ 自动log_metrics"]
        C1 --> C4["Precision精度: 16-mixed bf16-mixed 32<br/>→ 自动混合精度AMP"]
        C1 --> C4_2["Strategy分布式: ddp/fsdp/deepspeed_stage3<br/>→ 8行代码切分布式不用改模型"]
    end

    A3 --> C1
    A4 --> C1
    B3 --> C1
    B4 --> C1
    C1 --> FIT[trainer.fit(model, datamodule)]
```

---

## 二、完整训练生命周期钩子执行顺序

```mermaid
flowchart TD
    START["trainer.fit() 开始训练"]
    START --> HOOK0[on_fit_start 训练开始前]

    HOOK0 --> EPOCH_LOOP{for epoch 1..N}
    EPOCH_LOOP --> HOOK1["🟢 on_train_epoch_start<br/>每个epoch开始前"]

    HOOK1 --> BATCH_LOOP{for batch in train_loader}

    BATCH_LOOP --> HOOK2["on_train_batch_start<br/>每批前: 可改学习率/数据增广"]
    HOOK2 --> STEP["🔵 training_step(batch, batch_idx)<br/>⭐业务代码核心: loss = model(x,y)<br/>⚠️ PL自动: backward() + optimizer.step() + zero_grad()"]

    STEP --> HOOK3["on_before_backward(loss)<br/>backward之前: 可加梯度惩罚项"]
    HOOK3 --> BW["⚙️ 自动loss.backward()"]
    BW --> HOOK4["on_after_backward<br/>backward之后: 梯度裁剪clip_gradients自动"]
    HOOK4 --> OPT["⚙️ 自动optimizer.step() + zero_grad()"]

    OPT --> HOOK5["on_train_batch_end<br/>批结束: log batch_loss"]
    HOOK5 --> BATCH_LOOP

    BATCH_LOOP -->|Epoch训练完| HOOK6["🟡 on_validation_epoch_start"]
    HOOK6 --> VAL_LOOP{for batch in val_loader}
    VAL_LOOP --> VSTEP["🔵 validation_step(batch, idx)<br/>用户写: val_loss/acc 不用手动汇总"]
    VSTEP --> HOOK7["on_validation_batch_end"]
    HOOK7 --> VAL_LOOP

    VAL_LOOP --> HOOK8["🟡 on_validation_epoch_end<br/>⭐ 这里做: self.log('val_acc_epoch', epoch_avg)<br/>→ EarlyStopping/ModelCheckpoint回调读这个值"]
    HOOK8 --> HOOK9["on_train_epoch_end 训练+验证都结束"]

    HOOK9 --> CKPT{"🔧 ModelCheckpoint回调触发<br/>if 当前val_acc是历史最优<br/>→ 自动保存best.ckpt权重"}
    HOOK9 --> ES{"🔧 EarlyStopping回调<br/>val_acc连续10epoch没涨<br/>→ 自动break训练循环 防过拟合"}
    HOOK9 --> SCHED["🔧 scheduler.step()<br/>CosineAnnealingLR每个epoch降学习率"]

    ES -->|未终止| EPOCH_LOOP
    ES -->|patience耗尽| END[on_fit_end 训练结束]
    CKPT --> EPOCH_LOOP
    SCHED --> EPOCH_LOOP
    EPOCH_LOOP -->|max_epochs跑完| END
```

---

## 三、分布式DDP 4卡并行流程图

```mermaid
sequenceDiagram
    participant HOST as 主进程 spawn启动
    participant GPU0 as GPU Rank0
    participant GPU1 as GPU Rank1
    participant GPU2 as GPU Rank2
    participant GPU3 as GPU Rank3

    HOST->>GPU0: spawn子进程1
    HOST->>GPU1: spawn子进程2
    HOST->>GPU2: spawn子进程3
    HOST->>GPU3: spawn子进程4
    Note over GPU0,GPU3: 4个子进程完全独立！各跑一份完整代码

    GPU0->>GPU0: DataModule.setup(stage='fit')
    GPU1->>GPU1: 相同代码各自setup
    GPU2->>GPU2: ⚠️ prepare_data()只Rank0执行一次<br/>其他Rank跳过防重复下载
    GPU3->>GPU3:

    Note over GPU0,GPU3: 🔄 DistributedSampler自动切数据集<br/>每张卡拿到数据的1/4不重叠！

    loop 每个batch并行
        GPU0->>GPU0: forward(batch_0) = loss_0
        GPU1->>GPU1: forward(batch_1) = loss_1
        GPU2->>GPU2: forward(batch_2) = loss_2
        GPU3->>GPU3: forward(batch_3) = loss_3

        GPU0->>GPU0: backward → grad_0
        GPU1->>GPU1: backward → grad_1
        GPU2->>GPU2: backward → grad_2
        GPU3->>GPU3: backward → grad_3

        Note over GPU0,GPU3: 📡 AllReduce同步梯度<br/>每张卡的梯度平均: (g0+g1+g2+g3)/4<br/>NCCL通信 4卡间高速互联
        GPU0-->GPU1: NCCL Send
        GPU1-->GPU2: NCCL Send
        GPU2-->GPU3: NCCL Ring

        GPU0->>GPU0: optimizer.step() 更新权重
        GPU1->>GPU1: optimizer.step() 相同梯度→相同权重<br/>4卡模型始终保持一致！
        GPU2->>GPU2: optimizer.step()
        GPU3->>GPU3: optimizer.step()
    end

    Note over GPU0: ✅ 只Rank0做: 保存ckpt/Log/TensorBoard<br/>其他Rank不做避免重复写文件！
    GPU0->>HOST: 保存 best_model.ckpt
```

---

## 四、16-bit混合精度AMP执行流程

```mermaid
flowchart TD
    INPUT["FP32输入数据 [B,3,224,224] float32"]

    INPUT --> CAST1["🔽 自动cast权重和输入 → FP16/BF16<br/>节省显存×2 + TensorCore加速×2"]
    CAST1 --> FWD["⭐ forward 计算用FP16<br/>Linear/Conv/Attention TensorCore原生加速"]
    FWD --> LOSS["算loss FP16 → 可能出现Grad Underflow!<br/>小梯度值溢出成0!"]

    LOSS --> SCALE["📈 自动GradScaler 动态放大loss<br/>loss × scale_factor(通常=65536)<br/>→ 放大后梯度也×65536 不会变成0"]
    SCALE --> BW["loss_scaled.backward()"]
    BW --> GRAD["得到梯度: FP16 存放大后的梯度"]
    GRAD --> UN_SCALE["📉 unscale_: 梯度 / scale_factor 还原真实梯度"]
    GRAD --> CHECK{"🚨 检查梯度中是否有Inf/NaN溢出?"}

    CHECK -->|Yes溢出了| SKIP["❌ 跳过本次step 不更新权重<br/>scale_factor自动 /2 下次降低"]
    CHECK -->|No没溢出| OK["✅ clip_grad 梯度裁剪 防爆炸"]
    OK --> STEP["optimizer.step()<br/>权重更新用FP32 master副本算<br/>避免累积精度损失"]
    STEP --> GROW["scale_factor自动 ×1.0001 缓慢涨回去<br/>动态寻找最大值"]

    SKIP --> NEXT["下一批继续"]
    GROW --> NEXT
```

---

## 五、Callbacks非侵入插件机制

```mermaid
flowchart LR
    subgraph 业务代码
        M[LightningModule<br/>只写算法forward/step 纯算法代码<br/>✅ 不包含任何工程细节]
    end

    subgraph Trainer回调链 按顺序触发
        M --> C1["📦 ModelCheckpoint<br/>top-k best模型自动保存<br/>monitor=val_acc mode=max save_top_k=3"]
        C1 --> C2["🛑 EarlyStopping<br/>patience=10 10轮没涨就停<br/>restore_best_weights自动回滚最优"]
        C2 --> C3["📊 LearningRateMonitor<br/>自动把lr记录到TensorBoard曲线"]
        C3 --> C4["📝 RichProgressBar<br/>精美进度条ETA/剩余时间/速度"]
        C4 --> C5["🧊 StochasticWeightAveraging<br/>最后15%训练步SWA平均权重 提1-2%点"]
        C5 --> C6["🔧 自定义: GradientMonitorCallback<br/>每100步统计梯度范数/权重范数<br/>→ 检测梯度消失/爆炸"]
    end

    C6 --> Logger["📈 多Logger同时记录<br/>→ TensorBoard本地看<br/>→ Weights&Biases云端协作看<br/>→ MLflow实验追踪对比"]
```

---

## 六、从PyTorch原生迁移PL对比流程

```mermaid
flowchart TD
    subgraph PyTorch原生 100行样板代码
        PT1[手动: model.train() / eval()]
        PT2[手动: optimizer.zero_grad()]
        PT3[手动: loss.backward()]
        PT4[手动: torch.nn.utils.clip_grad_norm_()]
        PT5[手动: optimizer.step() / scheduler.step()]
        PT6[手动: with torch.no_grad() + torch.cuda.amp.autocast()]
        PT7[手动: GradScaler 混合精度调scale_factor]
        PT8[手动: DistributeDataParallel DDP封装]
        PT9[手动: torch.save(state_dict) + 比较best_acc]
        PT10[手动: writer.add_scalar TensorBoard写日志]
        PT11[手动:  EarlyStopping计数器逻辑]
    end

    subgraph LightningModule 20行只写业务
        PL1["🔵 training_step(batch, idx) return loss<br/>→ PT的1,2,3,4,5,6,7自动"]
        PL2["🔵 validation_step 写验证逻辑<br/>→ 自动no_grad/eval模式"]
        PL3["🔵 configure_optimizers return optimizer,scheduler<br/>→ step顺序自动"]
        PL4["Trainer(devices=4, strategy='ddp', precision='16-mixed')<br/>→ PT的8自动分布式"]
        PL5["Callbacks=[ModelCheckpoint, EarlyStopping]<br/>→ PT的9,11自动"]
        PL6["Trainer(logger=TensorBoardLogger)<br/>→ PT的10自动log"]
    end

    PT1 --> PL1
    PT2 --> PL1
    PT3 --> PL1
    PT4 --> PL1
    PT5 --> PL1
    PT6 --> PL1
    PT7 --> PL1
    PT8 --> PL4
    PT9 --> PL5
    PT10 --> PL6
    PT11 --> PL5
```