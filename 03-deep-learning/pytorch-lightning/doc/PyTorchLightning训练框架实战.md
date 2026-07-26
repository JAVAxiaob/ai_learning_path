# PyTorch Lightning 工程化解析

> 位置: 03-deep-learning/pytorch-lightning/
> 简历推荐: 4星 | 岗位: 深度学习工程师 (生产训练必备)

---

## 一、Lightning核心理念

```
Research代码三分离: Engineering (Lightning替你做) vs Science (你写LightningModule)
Lightning自动帮你做(不用再手写):
├── model.to(device) / .cuda()  自动切设备
├── torch.no_grad() / .train()/.eval()  模式切换
├── AMP混合精度 autocast + GradScaler  precision=16-mixed
├── DDP/FSDP多卡分布式 accelerator='gpu', devices=N, strategy='ddp'
├── Checkpoint + EarlyStop Callback 自动保存/早停
├── TensorBoard / WandB Logger  self.log()自动记录
├── Gradient Clipping  gradient_clip_val=1.0
├── Gradient Accumulation  accumulate_grad_batches=K → batch_size×K
├── Autolog  MLflow/Neptune等实验追踪自动对接
└── Auto LR Finder / Auto Batch Size Finder  自动调参
```

## 二、标准代码模板

```python
class LitModel(pl.LightningModule):
    # ===== 你的工作: 模型结构 =====
    def __init__(self):
        super().__init__()
        self.net = MyFancyModel()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):        # inference用
        return self.net(x)

    # ===== 你的工作: 训练/验证/测试 逻辑 =====
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss   # Lightning自动: backward + optimizer.step + zero_grad!

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def test_step(self, batch, _):
        self.validation_step(batch, _)

    # ===== 你的工作: 优化器配置 =====
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return {"optimizer": opt, "lr_scheduler": sch}

# ===== 工程部分 Lightning帮你做 (一行配置!) =====
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu", devices=2, strategy="ddp_find_unused_parameters_true",  # 双卡DDP
    precision="16-mixed",                    # AMP混合精度
    callbacks=[
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=3),
        EarlyStopping(monitor="val_acc", patience=5, mode="max"),
        LearningRateMonitor(logging_interval="step"),
    ],
    gradient_clip_val=1.0,                   # 梯度裁剪
    accumulate_grad_batches=4,               # 梯度累积=4倍batch
    logger=pl.loggers.TensorBoardLogger("logs/"),
)
trainer.fit(LitModel(), train_loader, val_loader)  # 一键训练,工程代码0行手写
trainer.test(ckpt_path="best")  # 用最佳checkpoint测试
```

## 三、功能对照表 (对比纯PyTorch)

| 功能 | 纯PyTorch手写行数 | Lightning配置 |
|-----|------------------|-------------|
| DDP双卡分布式 | ~50行样板 | `devices=2, strategy='ddp'` |
| AMP混合精度 | 4处改autocast/scaler | `precision='16-mixed'` |
| Checkpoint+EarlyStop | ~30行判断 | 2个Callback |
| Logger日志 | 手动wandb.log | `self.log()`自动同步 |
| Gradient Accumulation | for循环计数判断 | `accumulate_grad_batches=4` |
| 梯度裁剪 | 手动clip_grad_norm_ | `gradient_clip_val=1.0` |

## 四、简历黄金句式

| 写法 |
|-----|
| 「基于PyTorch Lightning搭建图像分类训练框架：双卡DDP+AMP混合精度+Gradient Accumulation 4×，单epoch训练时间从2h→35min (3.4×加速)」 |
| 「Checkpoint+EarlyStop+AutoLRFinder完整pipeline：CIFAR100 EfficientNet-B3 Top1从78%→84.1%，实验周转速度↑5倍」 |
| 「FSDP+Lightning训练2B参数语言模型：8卡A10，ZeRO Stage3分片+Activation Checkpointing，单卡峰值显存42GB→16GB，吞吐1.8k tok/s/gpu」 |

## 五、面试题

**Q Lightning DataModule vs 裸DataLoader 好处？**
> A: DataModule 把 setup(train/val/test/predict) 4种数据集、DataLoader、transform预处理全部封装到一个类，代码模块化+可复现+不同项目直接复用DataModule不用改训练代码。

**Q self.log()的on_step/on_epoch/prog_bar/logger参数？**
> A: on_step=True：记录每个step即时值；on_epoch=True：自动累积整个epoch算平均(验证必用这个)；prog_bar=True：显示在进度条上(不写文件)；logger=True：写到TensorBoard/WandB。