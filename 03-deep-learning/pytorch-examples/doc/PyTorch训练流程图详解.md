# PyTorch 训练流程图详解

> 位置: 03-deep-learning/pytorch-examples/

---

## 一、标准训练循环流程

```mermaid
graph TD
    A[初始化阶段] --> B[模型/优化器/损失函数]
    B --> C[训练循环]
    C --> D[数据加载]
    D --> E[前向传播]
    E --> F[计算损失]
    F --> G[反向传播]
    G --> H[梯度裁剪]
    H --> I[参数更新]
    I --> J{是否结束?}
    J -->|否| D
    J -->|是| K[验证评估]
    K --> L[保存模型]
```

## 二、训练循环时序图

```mermaid
sequenceDiagram
    participant Trainer as Trainer
    participant Model as Model
    participant Optimizer as Optimizer
    participant Loader as DataLoader
    
    Trainer->>Model: model.train()
    loop 每个epoch
        loop 每个batch
            Loader->>Trainer: (data, target)
            Trainer->>Optimizer: zero_grad()
            Trainer->>Model: forward(data)
            Model-->>Trainer: output
            Trainer->>Trainer: loss = criterion(output, target)
            Trainer->>Trainer: loss.backward()
            Trainer->>Trainer: clip_grad_norm_(params, 1.0)
            Trainer->>Optimizer: step()
        end
        Trainer->>Trainer: scheduler.step()
        Trainer->>Trainer: validate()
    end
```

## 三、数据加载流程

```mermaid
flowchart LR
    A[Dataset] --> B[__getitem__]
    B --> C[数据读取]
    C --> D[数据预处理]
    D --> E[返回样本]
    E --> F[DataLoader]
    F --> G[batch采样]
    G --> H[多线程预加载]
    H --> I[返回batch]
```

```python
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# DataLoader配置
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## 四、前向传播流程

```mermaid
graph LR
    A[输入Tensor] --> B[Layer1 Conv]
    B --> C[BN]
    C --> D[ReLU]
    D --> E[Layer2 Conv]
    E --> F[BN]
    F --> G[ReLU]
    G --> H[Pooling]
    H --> I[Flatten]
    I --> J[Linear]
    J --> K[输出Logits]
```

## 五、反向传播流程

```mermaid
graph TD
    A[损失值] --> B[backward()]
    B --> C[梯度计算]
    C --> D[链式法则]
    D --> E[梯度累积]
    E --> F[梯度裁剪]
    F --> G[参数更新]
    G --> H[优化器step]
```

```python
# 完整训练循环
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = validate()
    
    scheduler.step(val_acc)
```