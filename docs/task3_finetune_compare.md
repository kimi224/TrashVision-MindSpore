# 任务 3：冻结微调、全量微调、LoRA 微调对比

本文档记录第 3 个实验任务：实现并比较冻结微调、全量微调、LoRA 微调三种方式，说明原理、冻结层或 LoRA 层位置，并对内存占用、准确率、训练时间做对比。

本次新增代码：

```text
scripts\task3_finetune_compare.py
```

运行日志：

```text
logs\task3_compare.log
```

结果汇总：

```text
runs\task3_compare\summary.json
```

## 1. 实验说明

本机当前实验环境使用 CPU 运行 MindSpore，因此这里统计的是：

```text
进程峰值内存 peak_rss_mb
```

不是 GPU 显存。如果后续换到 GPU 环境，可以把这里的内存统计替换成 GPU 显存统计。

为了让实验能在 CPU 上完成，本次设置如下：

```text
冻结微调：使用任务 2 已提取的 MobileNetV2 特征，训练分类头 20 轮
LoRA 微调：使用任务 2 已提取的 MobileNetV2 特征，在分类头上加入 LoRA，训练 20 轮
全量微调：直接使用图片训练完整 MobileNetV2，训练 1 轮
```

为什么全量微调只跑 1 轮？

```text
全量微调需要对完整 MobileNetV2 做前向和反向传播；
CPU 上非常慢，1 轮已经耗时约 213 秒；
峰值内存约 4.6 GB；
如果跑 20 轮，预计耗时会很长。
```

所以本任务重点是比较三种方式的资源差异和工作机制，而不是给全量微调充分训练后的最终上限。

## 2. 运行命令

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\task3_finetune_compare.py `
  --output_dir runs\task3_compare `
  --freeze_epochs 20 `
  --lora_epochs 20 `
  --full_epochs 1 `
  --image_batch_size 32 `
  --feature_batch_size 64
```

输出文件：

```text
runs\task3_compare\freeze\ckpt\freeze-20_41.ckpt
runs\task3_compare\lora\ckpt\lora-20_41.ckpt
runs\task3_compare\full\ckpt\full-1_82.ckpt
runs\task3_compare\summary.json
```

## 3. 三种微调方式原理

### 3.1 冻结微调

冻结微调的思想：

```text
保留预训练模型的 backbone；
不更新 backbone 参数；
只训练最后的分类头；
适合数据量较小、算力较弱的场景。
```

本实验中冻结了：

```text
MobileNetV2Backbone 的全部参数
```

训练了：

```text
LinearHead.classifier，也就是 Dense(1280, 26)
```

代码：

```python
class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)
```

这里输入是任务 2 已经缓存好的 1280 维图片特征，输出是 26 类垃圾分类分数。

### 3.2 全量微调

全量微调的思想：

```text
加载预训练 backbone；
新建 26 类分类头；
backbone 和 head 全部参与训练；
模型可以调整更多参数，理论上上限更高；
但计算量、内存占用和过拟合风险也更高。
```

本实验中训练了：

```text
MobileNetV2Backbone 全部参数
MobileNetV2Head 全部参数
```

代码：

```python
def run_full_strategy(data_dir, pretrain_ckpt, epochs, lr, batch_size, output_dir):
    backbone = load_backbone(pretrain_ckpt, trainable=True)
    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=26)
    net = mobilenet_v2(backbone, head)
```

其中：

```python
backbone = load_backbone(pretrain_ckpt, trainable=True)
```

表示 backbone 参数是可训练的。

全量微调直接读取原始图片：

```python
train_dataset = create_image_dataset(data_dir / "train", batch_size, train=True)
test_dataset = create_image_dataset(data_dir / "test", batch_size, train=False)
```

所以它比冻结微调和 LoRA 微调更耗时、更占内存。

### 3.3 LoRA 微调

LoRA 的全称是 Low-Rank Adaptation，低秩适配。

它的核心思想：

```text
原始大权重 W 不更新；
额外增加两个小矩阵 A 和 B；
训练时只更新 A 和 B；
实际效果相当于让模型学习一个低秩增量 ΔW = A × B。
```

本实验中 LoRA 没有加在 MobileNetV2 的所有卷积层上，而是加在分类头线性层上：

```text
LoRALinearHead.classifier
```

具体冻结了：

```text
MobileNetV2Backbone 全部参数
LoRALinearHead.base.weight
LoRALinearHead.base.bias
```

训练了：

```text
LoRALinearHead.lora_a
LoRALinearHead.lora_b
LoRALinearHead.lora_bias
```

LoRA 设置：

```text
rank = 8
alpha = 16
scaling = alpha / rank = 2
```

代码：

```python
class LoRALinearHead(nn.Cell):
    """Dense layer with frozen base weight and trainable LoRA adapters."""

    def __init__(self, in_channels=1280, num_classes=26, rank=8, alpha=16.0):
        super().__init__()
        self.base = nn.Dense(in_channels, num_classes)
        self.base.weight.set_data(initializer(Zero(), self.base.weight.shape, ms.float32))
        self.base.bias.set_data(initializer(Zero(), self.base.bias.shape, ms.float32))
        self.lora_a = Parameter(initializer(HeUniform(), (in_channels, rank), ms.float32), name="lora_a")
        self.lora_b = Parameter(initializer(Zero(), (rank, num_classes), ms.float32), name="lora_b")
        self.lora_bias = Parameter(initializer(Zero(), (num_classes,), ms.float32), name="lora_bias")
        self.scaling = alpha / rank
        set_trainable(self.base, False)

    def construct(self, x):
        return self.base(x) + ops.matmul(ops.matmul(x, self.lora_a), self.lora_b) * self.scaling + self.lora_bias
```

这一版 LoRA 的作用是：

```text
用更少的可训练参数学习分类头的低秩变化；
相比普通 Dense 分类头，参数更少；
但表达能力也更受限制，所以准确率低于冻结微调。
```

## 4. 关键辅助代码

### 4.1 统计可训练参数

代码：

```python
def count_params(params):
    return int(sum(np.prod(tuple(param.shape)) for param in params))
```

用这个函数统计：

```text
trainable_params：参与训练的参数量
total_params：当前策略包含的总参数量
```

### 4.2 设置参数是否训练

代码：

```python
def set_trainable(cell, trainable):
    for param in cell.get_parameters():
        param.requires_grad = trainable
```

冻结 backbone 或 base dense 时，调用：

```python
set_trainable(backbone, False)
set_trainable(self.base, False)
```

全量微调时，调用：

```python
set_trainable(backbone, True)
```

### 4.3 统计训练时间和内存

代码：

```python
class MemoryAndLoss(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
```

训练过程中每个 step 更新一次峰值内存：

```python
def _update_peak(self):
    self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
```

最后记录：

```text
train_seconds：训练耗时
peak_rss_mb：训练过程中的进程峰值内存
```

## 5. 实验结果

结果来自：

```text
runs\task3_compare\summary.json
```

| 微调方式 | epoch | 可训练参数量 | 总参数量 | 训练时间 | 峰值内存 | 测试准确率 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结微调 | 20 | 33,306 | 33,306 | 3.45 s | 306.89 MB | 87.69% |
| LoRA 微调 | 20 | 10,474 | 43,780 | 3.68 s | 319.32 MB | 43.08% |
| 全量微调 | 1 | 2,291,290 | 2,291,290 | 213.52 s | 4644.19 MB | 49.62% |

## 6. 结果分析

### 6.1 参数量对比

冻结微调：

```text
只训练 Dense(1280, 26)
参数量 = 1280 × 26 + 26 = 33,306
```

LoRA 微调：

```text
训练 lora_a: 1280 × 8 = 10,240
训练 lora_b: 8 × 26 = 208
训练 lora_bias: 26
可训练参数总量 = 10,474
```

全量微调：

```text
训练 MobileNetV2Backbone + MobileNetV2Head
可训练参数量 = 2,291,290
```

所以参数规模关系是：

```text
LoRA < 冻结微调 << 全量微调
```

### 6.2 内存占用对比

冻结微调和 LoRA 微调都使用任务 2 缓存的 1280 维特征，不需要对完整图片反复跑 MobileNetV2，因此内存较低：

```text
冻结微调：306.89 MB
LoRA 微调：319.32 MB
```

全量微调需要保存完整网络的前向中间结果，用于反向传播，所以内存明显更高：

```text
全量微调：4644.19 MB
```

全量微调约为冻结微调的：

```text
4644.19 / 306.89 ≈ 15.13 倍
```

### 6.3 训练时间对比

冻结微调和 LoRA 微调训练的是小型分类器，所以非常快：

```text
冻结微调 20 轮：3.45 秒
LoRA 微调 20 轮：3.68 秒
```

全量微调虽然只跑 1 轮，但已经花了：

```text
213.52 秒
```

如果粗略按线性估算，全量微调 20 轮可能需要一个多小时。因此在 CPU 环境下，全量微调成本很高。

### 6.4 准确率对比

冻结微调最高：

```text
87.69%
```

原因是它直接训练完整的 `Dense(1280, 26)` 分类头，表达能力足够，而且训练稳定。

LoRA 微调较低：

```text
43.08%
```

主要原因：

```text
本实验只在分类头上加 LoRA，没有把 LoRA 加到 backbone 的卷积层；
rank=8 的低秩增量表达能力有限；
训练的是低秩适配矩阵，不如完整 Dense 分类头灵活。
```

全量微调只有 1 轮，准确率为：

```text
49.62%
```

不能说明全量微调最终效果差，只能说明：

```text
在当前 CPU 资源限制下，全量微调 1 轮还没有充分收敛；
它消耗的时间和内存明显更高。
```

## 7. 三种方法适用场景

冻结微调适合：

```text
数据集较小
硬件资源有限
希望快速得到可用结果
预训练模型和目标任务差距不大
```

全量微调适合：

```text
数据量较大
GPU/昇腾等算力充足
目标任务和预训练任务差异较大
希望模型整体适应新任务
```

LoRA 微调适合：

```text
大模型参数很多
不想保存完整模型副本
希望只训练少量适配参数
需要多任务切换或低成本微调
```

在本实验这种小型 MobileNetV2 + 小数据集 + CPU 环境下，冻结微调是最实用的选择。

## 8. 本任务完成情况

已经完成：

```text
实现冻结微调
实现全量微调
实现 LoRA 微调
指出冻结层和 LoRA 层
统计可训练参数量
统计训练耗时
统计 CPU 进程峰值内存
统计测试准确率
保存 checkpoint、日志和 summary
```

任务 3 对 PDF 要求的对应关系：

```text
使用其他微调方式：已实现冻结、全量、LoRA
说明工作原理：已在第 3 节说明
指出冻结层/LoRA 层：已在第 3 节说明
显存/内存对比：CPU 环境下记录 peak_rss_mb
准确率区别：已给出表格
训练时间差异：已给出表格
```

## 9. 可写入正式报告的结论

本实验在 MindSpore CPU 环境下比较了三种迁移学习策略。冻结微调只训练 `Dense(1280, 26)` 分类头，训练参数 33,306 个，20 轮耗时 3.45 秒，峰值内存 306.89 MB，测试准确率 87.69%，综合表现最好。LoRA 微调只训练 `lora_a`、`lora_b` 和 `lora_bias`，可训练参数减少到 10,474 个，但由于只在分类头上做低秩适配，准确率为 43.08%。全量微调训练 MobileNetV2 全部参数，参数量 2,291,290 个，1 轮耗时 213.52 秒，峰值内存 4644.19 MB，准确率 49.62%；该结果主要受 CPU 训练时间限制影响，不能代表充分训练后的最终上限。总体来看，在小数据集和普通 CPU 环境下，冻结微调最适合本实验。
