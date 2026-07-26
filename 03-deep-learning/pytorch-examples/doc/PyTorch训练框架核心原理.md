# PyTorch Examples 训练框架解析

> 位置: 03-deep-learning/pytorch-examples/
> 简历推荐: 4星 | 岗位: 所有深度学习岗位

---

## 一、项目模块结构

```
pytorch-examples/
├── mnist/main.py              MNIST手写数字 (标准训练循环模板!)
├── cifar10/                   CIFAR10图像分类 (数据增强)
├── imagenet/                  ImageNet多卡DDP分布式训练
├── dcgan/                     生成对抗网络 GAN (MNIST数字生成)
├── word_language_model/       RNN/LSTM语言模型 (PTB数据集)
├── cpp/                       C++ libtorch前端 (C++推理生产部署)
│   ├── mnist/mnist.cpp        C++ MNIST训练
│   ├── custom-dataset/        C++ Dataset自定义
│   ├── transfer-learning/     C++迁移学习 (图像分类)
│   └── distributed/dist-mnist.cpp  C++ DDP多卡
└── distributed/
    ├── FSDP/                  Fully Sharded Data Parallel 大模型训练
    └── FSDP2/                 最新FSDP2改进版
```

## 二、Production级训练循环模板 (必背!)

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)

    def train_one_epoch(self, epoch):
        self.model.train()  # 训练模式 启用BN/Dropout
        for data, target in self.train_loader:
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad(set_to_none=True)  # 清空梯度(更快)
            loss = self.criterion(self.model(data), target)  # 正向
            loss.backward()                    # 反向 自动求导
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪防爆
            self.optimizer.step()             # 更新参数
        self.scheduler.step()                 # 每个epoch调一次LR

    @torch.no_grad()  # 推理禁用梯度 省显存+快
    def validate(self):
        self.model.eval()  # 评估模式 冻结BN/Dropout
        loss, correct = 0, 0
        for data, target in self.val_loader:
            data, target = data.cuda(), target.cuda()
            out = self.model(data)
            loss += self.criterion(out, target).item()
            correct += (out.argmax(1) == target).sum().item()
        acc = 100.*correct/len(self.val_loader.dataset)
        # 保存最佳Checkpoint
        if acc > self.best_acc:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_acc': acc,
            }, f'best.pth')
```

## 三、PyTorch 8大常见坑排查表

| 症状 | 根因 | 修复 |
|-----|------|-----|
| Loss不下降梯度0 | 忘了zero_grad()或LR过大/数据bug | 每次backward前zero_grad, LR调小10倍 |
| Loss=NaN | 梯度爆炸/除0/log负数 | clip_grad_norm_, 检查输入标签, 加1e-8 eps |
| Train好Val差 | 没切换model.eval()或过拟合 | 手动切换train/eval, 加正则/增强 |
| CrossEntropy Loss大 Acc低 | 最后层重复加Softmax | PyTorch CE内置Softmax, Linear直接喂不用Softmax |
| DDP多卡不收敛 | SyncBatchNorm没加/LR没线性缩放 | 单机多卡加convert_sync_batchnorm, LR×卡数 |
| 显存爆OOM | batch太大/中间变量没释放 | batch_size减半 + 梯度检查点ActivationCheckpoint |
| 数据Loader卡死 | Windows num_workers>0死锁 | Windows设num_workers=0, Linux设CPU核心数 |
| RuntimeError设备不一致 | model/data不在同一device | 统一 .to(device) 显式指定 |

## 四、简历黄金句式

| 写法 |
|-----|
| 「搭建Production级PyTorch训练框架：AMP混合精度+CosineLR+梯度裁剪+EarlyStop，CIFAR10 ResNet18从baseline 92%→95.8%」 |
| 「Fully Sharded Data Parallel (FSDP) 训练1.3B参数GPT：4卡A10训练吞吐538 tok/s/gpu，比原生DDP显存↓62%」 |
| 「libtorch C++推理落地：TorchScript导出ResNet50，C++单线程推理37ms，对比Python服务P99延迟↓71%」 |

## 五、面试题

**Q DataLoader 中 pin_memory / num_workers / persistent_workers 作用？**
> A: pin_memory=True: CPU tensor锁页内存→GPU拷贝更快(快30%)；num_workers=N: N个子进程预加载数据(非阻塞)；persistent_workers=True: epoch结束不销毁worker进程，避免反复fork开销(快20%)。

**Q model.train()/eval() 影响哪些层？**
> A: Dropout: train时随机丢弃/eval时全部保留；BatchNorm: train时用当前batch统计mean/var更新running_mean/var, eval时固定用训练阶段running统计；LayerNorm不受影响。

**Q torch.no_grad() vs torch.inference_mode()？**
> A: 都禁用梯度计算省显存，inference_mode()更激进(连Autograd视图追踪都关)，比no_grad再快5-10%。