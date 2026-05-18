# 任务 4：模型改进与准确率对比

本文档记录第 4 个实验任务：在保持训练超参数相同的前提下，对模型进行改进，并比较改进前后的准确率。

本次新增/更新代码：

```text
scripts\task4_model_improvement.py
```

运行日志：

```text
logs\task4_improvement.log
```

结果汇总：

```text
runs\task4_improvement\summary.json
```

## 1. 实验目标

PDF 中第 (4) 点要求：

```text
对模型进行改进，以获得更高的准确率。需要保证改进模型前后的超参数设置相同，且需要对前后准确率进行对比。
```

本实验继续使用任务 2 已提取好的 MobileNetV2 backbone 特征：

```text
runs\task2_custom\train_features.npz
runs\task2_custom\test_features.npz
```

这样可以保证改进前后模型的输入数据完全一致，只改变分类头结构。

## 2. 公平对比设置

改进前后保持完全相同的训练超参数：

| 超参数 | 设置 |
|---|---:|
| epoch | 30 |
| batch_size | 64 |
| learning_rate | 0.03 |
| momentum | 0.9 |
| weight_decay | 0.00004 |
| optimizer | Momentum |
| loss | SoftmaxCrossEntropyWithLogits |
| seed | 11 |
| 输入特征 | 同一份 MobileNetV2 1280 维特征 |

唯一改变的是模型分类头结构：

```text
改进前：LinearHead
改进后：MultiBranchMLPHead
```

## 3. 运行命令

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\task4_model_improvement.py `
  --output_dir runs\task4_improvement `
  --epochs 30 `
  --batch_size 64 `
  --lr 0.03 `
  --momentum 0.9 `
  --weight_decay 0.00004
```

输出模型：

```text
runs\task4_improvement\baseline_linear\ckpt\baseline_linear-30_41.ckpt
runs\task4_improvement\improved_multibranch\ckpt\improved_multibranch-30_41.ckpt
```

## 4. 改进前模型

改进前使用单层线性分类头：

```python
class LinearHead(nn.Cell):
    """Baseline classifier: one linear layer from feature to class logits."""

    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)
```

结构为：

```text
Dense(1280, 26)
```

优点：

```text
结构简单
参数少
训练快
在 MobileNetV2 特征已经较好时表现稳定
```

不足：

```text
只能学习单一线性分类边界；
当类别之间存在更复杂的非线性关系时，表达能力有限。
```

## 5. 改进后模型

改进后使用多分支 MLP 分类头：

```python
class MultiBranchMLPHead(nn.Cell):
    """Improved head: average logits from several MLP classifiers."""

    def __init__(self, in_channels=1280, hidden_channels=512, num_classes=26, branches=3, dropout=0.1):
        super().__init__()
        self.branches = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(hidden_channels, num_classes),
                )
                for _ in range(branches)
            ]
        )

    def construct(self, x):
        logits = self.branches[0](x)
        for i in range(1, len(self.branches)):
            logits = logits + self.branches[i](x)
        return logits / len(self.branches)
```

结构为：

```text
3 个并行分支，每个分支：
Dense(1280, 512)
ReLU
Dropout(0.1)
Dense(512, 26)

最终输出：
3 个分支 logits 求平均
```

改进点解释：

```text
1. 每个 MLP 分支比单层线性头更有表达能力；
2. ReLU 引入非线性，可以学习更复杂的类别边界；
3. Dropout(0.1) 在训练时随机屏蔽部分隐藏单元，降低过拟合；
4. 多分支平均 logits 类似轻量集成，可以降低单个分类器的偶然误差。
```

这个改进的代价是参数量和训练时间增加，但目标是换取更高准确率。

## 6. 训练代码说明

### 6.1 读取同一份特征

代码：

```python
def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32),
        train["labels"].astype(np.int32),
        test["features"].astype(np.float32),
        test["labels"].astype(np.int32),
    )
```

改进前后都使用同一份 `train_features` 和 `test_features`，保证输入一致。

### 6.2 共用同一个训练函数

代码：

```python
def train_one_model(name, net, train_features, train_labels, test_features, test_labels, args, output_dir):
    train_dataset = create_dataset(train_features, train_labels, args.batch_size, shuffle=True)
    test_dataset = create_dataset(test_features, test_labels, args.batch_size, shuffle=False)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(
        net.trainable_params(),
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"})
```

这个函数同时用于：

```text
baseline_linear
improved_multibranch
```

所以两者训练流程一致。

### 6.3 记录共同超参数

代码：

```python
common_hyperparams = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "seed": args.seed,
}
```

这部分写入 `summary.json`，用于证明改进前后超参数一致。

## 7. 实验结果

结果来自：

```text
runs\task4_improvement\summary.json
```

| 模型 | 结构 | epoch | batch_size | lr | 参数量 | 训练时间 | 测试准确率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 改进前 | Dense(1280, 26) | 30 | 64 | 0.03 | 33,306 | 2.81 s | 87.69% |
| 改进后 | 3 分支 MLP logits 平均 | 30 | 64 | 0.03 | 2,007,630 | 12.18 s | 88.85% |

准确率提升：

```text
88.85% - 87.69% = 1.15 个百分点
```

换算到 260 张测试图片上：

```text
改进前约预测正确 228 张
改进后约预测正确 231 张
```

## 8. 训练 loss 对比

改进前最后几轮 loss：

```text
epoch 26 avg_loss=0.036607
epoch 27 avg_loss=0.035954
epoch 28 avg_loss=0.037266
epoch 29 avg_loss=0.037093
epoch 30 avg_loss=0.031623
```

改进后最后几轮 loss：

```text
epoch 26 avg_loss=0.054910
epoch 27 avg_loss=0.050154
epoch 28 avg_loss=0.051263
epoch 29 avg_loss=0.043497
epoch 30 avg_loss=0.040420
```

虽然改进后训练 loss 不一定比线性头更低，但测试准确率更高，说明多分支结构起到了更好的泛化作用。

## 9. 结果分析

改进后准确率提升的原因：

```text
线性头只学习一个分类边界；
多分支 MLP 学习多个不同的非线性分类器；
Dropout 让每个分支训练时看到略有差异的隐藏表示；
最终平均 logits，可以降低单个分支误判带来的影响。
```

代价：

```text
参数量从 33,306 增加到 2,007,630；
训练时间从 2.81 秒增加到 12.18 秒；
准确率提升 1.15 个百分点。
```

所以该改进是有效的，但属于“用更多模型容量和训练时间换取准确率提升”。

## 10. 本任务完成情况

已经完成：

```text
设计改进前模型 LinearHead
设计改进后模型 MultiBranchMLPHead
保证改进前后超参数完全相同
运行训练和测试
保存 checkpoint
保存 summary.json
对比改进前后准确率
```

对 PDF 要求的对应：

```text
对模型进行改进：单层线性分类头改为多分支 MLP 分类头
获得更高准确率：87.69% 提升到 88.85%
保证超参数相同：epoch、batch_size、lr、momentum、weight_decay、seed 相同
进行准确率对比：已给出表格和提升幅度
```

## 11. 可写入正式报告的结论

本实验在保持训练超参数一致的前提下，对分类头结构进行了改进。改进前模型为单层线性分类头 `Dense(1280, 26)`，测试准确率为 87.69%；改进后模型为 3 分支 MLP 分类头，每个分支为 `Dense(1280,512)+ReLU+Dropout(0.1)+Dense(512,26)`，最终对三个分支的 logits 求平均，测试准确率为 88.85%，提升 1.15 个百分点。结果说明，在 MobileNetV2 预训练特征已经较好的情况下，使用多分支非线性分类头可以进一步提升垃圾分类准确率，但也会带来参数量和训练时间的增加。
