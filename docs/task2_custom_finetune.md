# 任务 2：参考开源项目编写自己的微调代码

本文档记录第 2 个实验任务：在任务 1 成功复现 MindSpore 官方 MobileNetV2 垃圾分类项目的基础上，参考开源项目思路，编写我们自己的类似代码，完成模型加载和迁移微调。

本次没有直接运行官方的 `train.py` 或 `eval.py`，而是新增了自己的脚本：

```text
scripts\task2_custom_finetune.py
```

## 1. 任务目标

PDF 中第 (2) 点要求：

```text
在(1)的基础上，通过学习开源项目，写了类似的代码（可以参考源代码、并非直接运行已有代码），进行模型加载、不同任务上微调等任务。
```

我的理解是：

```text
任务 1：能跑通别人写好的官方项目
任务 2：看懂官方项目后，自己写一个类似流程
```

所以本任务做了这些事：

```text
1. 继续使用 MobileNetV2 预训练模型
2. 自己写数据读取、特征提取、分类头训练、评估流程
3. 把预训练 backbone 迁移到 26 类垃圾分类任务
4. 冻结 backbone，只训练自己写的分类头
5. 保存训练日志、模型 checkpoint 和结果 summary
```

## 2. 与任务 1 的区别

任务 1 直接运行官方入口：

```text
third_party\...\mobilenetv2\train.py
third_party\...\mobilenetv2\eval.py
```

任务 2 新增自己的入口：

```text
scripts\task2_custom_finetune.py
```

本次复用的内容：

```text
MobileNetV2Backbone 网络结构
预训练权重 mobilenetv2_cpu_gpu.ckpt
垃圾分类数据集 data\data_en
```

本次自己实现的内容：

```text
命令行参数解析
数据集读取和预处理
预训练 backbone 加载
backbone 冻结
特征提取与缓存
新的 26 类分类头
分类头训练
测试集评估
结果 summary 保存
```

这样既不是简单复制官方 `train.py`，也不是从零乱写，而是“学习官方代码后重写一个更清晰的小版本”。

## 3. 文件和运行结果

代码文件：

```text
scripts\task2_custom_finetune.py
```

运行日志：

```text
logs\task2_custom.log
```

输出目录：

```text
runs\task2_custom
```

输出文件包括：

```text
runs\task2_custom\train_features.npz
runs\task2_custom\test_features.npz
runs\task2_custom\summary.json
runs\task2_custom\ckpt\task2_head-30_41.ckpt
```

最终测试准确率：

```text
Task2 eval result: {'acc': 0.8923076923076924}
```

也就是：

```text
89.23%
```

作为对比，任务 1 官方复现结果是：

```text
86.54%
```

两次训练流程不同，所以这个对比只能说明“本次自写流程在当前设置下跑通且效果正常”，不能作为严格模型改进结论。严格改进对比会放到后面的任务 4 做。

## 4. 运行命令

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\task2_custom_finetune.py `
  --epochs 30 `
  --output_dir runs\task2_custom `
  --image_batch_size 64 `
  --head_batch_size 64 `
  --force_extract
```

解释：

```text
--epochs 30：分类头训练 30 轮
--output_dir：输出日志、特征缓存、checkpoint 的目录
--image_batch_size 64：提取图片特征时每批 64 张
--head_batch_size 64：训练分类头时每批 64 条特征
--force_extract：强制重新提取 backbone 特征
```

注意：因为当前项目路径里有中文，MindSpore 的 C++ 数据集读取层对中文绝对路径支持不好，所以脚本内部传给 MindSpore 的是相对路径。

## 5. 核心代码讲解

### 5.1 引入官方 backbone，但不直接跑官方训练入口

代码位置：

```python
OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT
    / "third_party"
    / "mindspore_r1.3_sparse"
    / "model_zoo"
    / "official"
    / "cv"
    / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))

from src.mobilenetV2 import MobileNetV2Backbone
```

这里我们只复用 `MobileNetV2Backbone` 这个网络结构。官方的 `train.py`、`eval.py` 没有被调用。

为什么可以复用 backbone？

```text
任务要求允许参考源代码；
MobileNetV2 网络结构本身是开源项目的核心模型定义；
我们的重点是自己写迁移学习流程，而不是重新发明 MobileNetV2。
```

### 5.2 自己定义新的垃圾分类头

代码：

```python
class LinearGarbageHead(nn.Cell):
    """A small classification head for cached MobileNetV2 features."""

    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)
```

解释：

```text
backbone 输出 1280 维图像特征；
垃圾分类任务有 26 类；
所以我们用 Dense(1280, 26) 得到 26 个类别分数。
```

这就是“把预训练模型迁移到新任务”的关键：前面的特征提取器保留，最后的分类器换成适合垃圾分类的新分类器。

### 5.3 自己写数据读取和预处理

代码：

```python
def create_image_dataset(split_dir, batch_size):
    transform_img = [
        vision.Decode(),
        vision.Resize((256, 256)),
        vision.CenterCrop(224),
        vision.Normalize(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        ),
        vision.HWC2CHW(),
    ]
    transform_label = transforms.TypeCast(ms.int32)

    dataset = ds.ImageFolderDataset(
        path_for_mindspore(split_dir),
        shuffle=False,
        num_parallel_workers=1,
    )
    dataset = dataset.map(transform_img, input_columns="image", num_parallel_workers=1)
    dataset = dataset.map(transform_label, input_columns="label", num_parallel_workers=1)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
```

这里的流程是：

```text
Decode：读取图片内容
Resize：统一缩放到 256 x 256
CenterCrop：中心裁剪到 224 x 224
Normalize：按 ImageNet 均值方差标准化
HWC2CHW：把图片格式改成 MindSpore 模型需要的通道优先格式
```

为什么用 ImageNet 的均值和方差？

```text
预训练 MobileNetV2 通常是在大规模自然图片数据上训练的；
用相同的归一化方式，输入分布更接近预训练时的输入；
这样迁移学习更稳定。
```

### 5.4 加载预训练模型并冻结 backbone

代码：

```python
def load_frozen_backbone(pretrain_ckpt):
    backbone = MobileNetV2Backbone()
    params = load_checkpoint(str(pretrain_ckpt))
    not_loaded = load_param_into_net(backbone, params)
    for param in backbone.get_parameters():
        param.requires_grad = False
    backbone.set_train(False)
    print(f"Backbone trainable params after freezing: {sum(p.requires_grad for p in backbone.get_parameters())}")
    return backbone
```

运行日志里显示：

```text
Backbone trainable params after freezing: 0
Unloaded params info: ([], [])
```

这说明：

```text
1. 预训练权重成功加载进 backbone；
2. 没有缺失参数；
3. backbone 中可训练参数数量为 0；
4. 后面训练时不会更新 backbone。
```

为什么冻结？

```text
垃圾分类数据集只有 2593 张训练图片，比较小；
如果把整个 MobileNetV2 都训练，容易过拟合；
冻结 backbone 后，只训练最后的分类头，训练更快，也更稳。
```

### 5.5 先提取特征，再训练分类头

代码：

```python
feature_map = model.predict(image).asnumpy()
pooled = feature_map.mean(axis=(2, 3)).astype(np.float32)
features.append(pooled)
labels.append(item["label"].astype(np.int32))
```

backbone 对每张图片输出的是一个特征图，形状大致可以理解为：

```text
[batch_size, 1280, height, width]
```

这里手动做：

```text
mean(axis=(2, 3))
```

意思是对高和宽两个维度求平均，把特征图变成 1280 维向量：

```text
[batch_size, 1280]
```

然后保存成：

```text
train_features.npz
test_features.npz
```

这样做的好处：

```text
backbone 已经冻结，同一张图片每次提取到的特征固定；
先把特征缓存下来，后面训练分类头时不需要反复跑整张 MobileNetV2；
训练速度会快很多。
```

### 5.6 用 NumpySlicesDataset 训练分类头

代码：

```python
def create_feature_dataset(features, labels, batch_size, shuffle):
    dataset = ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)
```

这一步把缓存好的特征和标签重新包装成 MindSpore 数据集。训练分类头时，输入不再是图片，而是 1280 维特征。

### 5.7 损失函数、优化器和评估

代码：

```python
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.Momentum(
    params=head.trainable_params(),
    learning_rate=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
model = Model(head, loss_fn=loss, optimizer=optimizer, metrics={"acc"})
```

解释：

```text
SoftmaxCrossEntropyWithLogits：多分类常用损失函数
Momentum：带动量的梯度下降优化器
metrics={"acc"}：评估时计算准确率
```

注意这里传给优化器的是：

```python
head.trainable_params()
```

也就是说优化器只更新我们自己写的分类头，不更新 backbone。

## 6. 训练过程

训练日志中的 loss：

```text
epoch 1/30,  avg_loss=1.899261
epoch 5/30,  avg_loss=0.337563
epoch 10/30, avg_loss=0.149374
epoch 15/30, avg_loss=0.091943
epoch 20/30, avg_loss=0.058737
epoch 25/30, avg_loss=0.040037
epoch 30/30, avg_loss=0.031397
```

可以看到 loss 明显下降，说明新的分类头在学习垃圾分类任务。

最终评估：

```text
Task2 eval result: {'acc': 0.8923076923076924}
```

测试集一共有 260 张图片，所以 0.8923 大约表示：

```text
260 张测试图中约 232 张预测正确
```

## 7. 本任务完成情况

已经完成：

```text
学习并参考任务 1 的开源项目
编写自己的微调脚本 scripts\task2_custom_finetune.py
加载 MobileNetV2 预训练 backbone
冻结 backbone
为垃圾分类任务新建 Dense(1280, 26) 分类头
提取并缓存 train/test 特征
训练新的分类头
保存分类头 checkpoint
在测试集上评估准确率
```

最终结果：

```text
测试准确率：89.23%
```

## 8. 与 PDF 要求的对应总结

本任务满足 PDF 第 (2) 点：

```text
通过学习开源项目，写了类似的代码；
不是直接运行已有 train.py；
进行了模型加载；
把预训练模型迁移到垃圾分类这个不同任务；
完成微调训练和评估。
```

后续任务可以继续在这个脚本基础上扩展：

```text
任务 3：实现并比较冻结微调、全量微调、LoRA 微调
任务 4：改进模型结构或训练策略，并做准确率对比
```
