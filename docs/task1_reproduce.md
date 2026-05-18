# 任务 1：复现 MindSpore MobileNetV2 垃圾分类开源项目

本文档记录当前项目中已经完成的第 1 个实验任务：成功复现并运行开源项目，说明代码原理、操作步骤和算法流程。所有命令都在项目根目录 `D:\人工智能实验\AIhomework3` 下执行，并且 Python 解释器统一使用当前文件夹的虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

## 1. PDF 要求对应关系

PDF 中的实验主题是“手写数字识别与垃圾分类”，核心要求是加载预训练模型，并在 26 类垃圾分类数据集上做微调。PDF 给出的参考资料指向 MindSpore 官方资料，数据集下载地址为：

```text
https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/MindStudio-pc/data_en.zip
```

本步骤完成的是要求细则中的第 (1) 点：

```text
成功复现实现、运行开源项目，详细说明代码原理、各种操作、算法流程。
```

## 2. 本次复现使用的内容

开源项目：MindSpore 官方 r1.3 仓库中的 MobileNetV2 示例。

本地代码位置：

```text
third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2
```

数据集位置：

```text
data\data_en
```

预训练权重位置：

```text
pretrain_checkpoint\mobilenetv2_cpu_gpu.ckpt(.venv) D:\人工智能实验\AIhomework3>
 *  还原的历史记录 
                                    where python                        
D:\人工智能实验\AIhomework3\.venv\Scripts\python.exe                    
C:\Users\shilf\AppData\Local\Programs\Python\Python312\python.exe       
C:\Users\shilf\AppData\Local\Programs\Python\Python310\python.exe       
C:\Users\shilf\AppData\Local\Programs\Python\Python313\python.exe              
C:\Users\shilf\AppData\Local\Microsoft\WindowsApps\python.exe  
```

训练输出位置：

```text
runs\task1_reproduce\ckpt_0
```

日志位置：

```text
logs\task1_train.log
logs\task1_eval.log
```

## 3. 环境说明

当前虚拟环境中的主要依赖：

```text
Python 3.12.10
mindspore 2.9.0
numpy 1.26.4
pillow 12.2.0
PyYAML 6.0.3
```

注意：我没有修改系统 Python 或全局环境。`PyYAML` 是安装到当前项目的 `.venv` 里的，因为官方示例需要用它读取 yaml 配置文件。

## 4. 数据准备

数据集解压后结构如下：

```text
data\data_en
├── train
│   ├── Banana Peel
│   ├── Basketball
│   └── ...
└── test
    ├── Banana Peel
    ├── Basketball
    └── ...
```

统计结果：

```text
训练集：2593 张图片
测试集：260 张图片
类别数：26 类
```

MindSpore 的 `ImageFolderDataset` 会根据文件夹名自动生成类别标签。例如 `train\Battery` 文件夹里的图片会被当成 Battery 类。

官方评估脚本默认读取 `validation_preprocess` 文件夹，而 PDF 数据集里叫 `test`，所以我在项目内复制了一份：

```powershell
Copy-Item -Recurse -Force 'data\data_en\test' 'data\data_en\validation_preprocess'
```

这一步只是为了适配官方脚本的目录名，不改变图片内容。

## 5. 兼容性处理

官方 r1.3 示例使用旧接口：

```python
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
```

当前环境是 MindSpore 2.9.0，新版本已经改成：

```python
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
```

所以我只在本地开源代码副本中做了一个兼容补丁：

```python
try:
    import mindspore.dataset.vision.c_transforms as C
    import mindspore.dataset.transforms.c_transforms as C2
except ModuleNotFoundError:
    import mindspore.dataset.vision as C
    import mindspore.dataset.transforms as C2
```

另外，MindSpore 在 Windows 下读取包含中文的绝对路径时会把路径编码弄乱，所以运行时使用相对路径，例如 `data\data_en`，不要传 `D:\人工智能实验\AIhomework3\data\data_en`。

## 6. 复现命令

在项目根目录执行训练：

```powershell
$env:PYTHONPATH='third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2'
.\.venv\Scripts\python.exe third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\train.py `
  --config_path third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\default_config_cpu.yaml `
  --platform CPU `
  --dataset_path data\data_en `
  --pretrain_ckpt pretrain_checkpoint\mobilenetv2_cpu_gpu.ckpt `
  --freeze_layer backbone `
  --epoch_size 15 `
  --batch_size 150 `
  --save_checkpoint True `
  --save_checkpoint_path runs\task1_reproduce\
```

执行评估：

```powershell
$env:PYTHONPATH='third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2'
.\.venv\Scripts\python.exe third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\eval.py `
  --config_path third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\default_config_cpu.yaml `
  --platform CPU `
  --dataset_path data\data_en `
  --pretrain_ckpt runs\task1_reproduce\ckpt_0\mobilenetv2_15.ckpt `
  --batch_size 10
```

如果 PowerShell 显示 MindSpore warning，不一定代表失败。关键看日志中是否有训练 loss 或评估结果。

## 7. 本次运行结果

训练设置：

```text
模型：MobileNetV2
平台：CPU
类别数：26
微调方式：冻结 backbone，只训练分类头
epoch：15
batch_size：150
学习率：余弦衰减，lr_max=0.03，lr_end=0.03
优化器：Momentum
损失函数：带 label smoothing 的交叉熵
```

训练 loss 变化：

```text
epoch 1  avg loss: 2.256
epoch 5  avg loss: 1.100
epoch 10 avg loss: 0.968
epoch 15 avg loss: 0.911
```

最终评估结果：

```text
result:{'acc': 0.8653846153846154}
```

也就是测试集准确率约为：

```text
86.54%
```

## 8. 代码原理说明

### 8.1 数据读取与预处理

文件：

```text
src\dataset.py
```

训练时使用的主要处理流程：

```text
ImageFolderDataset 读取图片
RandomCropDecodeResize 随机裁剪并缩放到 224 x 224
RandomHorizontalFlip 随机水平翻转
RandomColorAdjust 随机调整亮度、对比度、饱和度
Normalize 做标准化
HWC2CHW 把图片格式从 高宽通道 改为 通道高宽
batch 打包成一批数据
```

这些操作的作用是让模型看到更多变化后的图片，降低过拟合。

评估时不做随机增强，主要是：

```text
Decode
Resize
CenterCrop
Normalize
HWC2CHW
```

这样测试结果更稳定。

### 8.2 MobileNetV2 网络结构

文件：

```text
src\mobilenetV2.py
```

MobileNetV2 分成两部分：

```text
backbone：负责提取图像特征
head：负责把特征映射成 26 类垃圾分类结果
```

backbone 的核心模块是 `InvertedResidual`。可以简单理解为：

```text
先用 1x1 卷积扩展通道
再用 depthwise 卷积提取空间特征
最后用 1x1 卷积压回目标通道
如果输入输出尺寸一致，就加残差连接
```

这种结构计算量小，适合轻量级图像分类。

head 的结构很简单：

```text
GlobalAvgPooling
Dense(1280, 26)
```

`GlobalAvgPooling` 会把每个通道的特征图压缩成一个数，`Dense` 层输出 26 个分数，每个分数对应一个垃圾类别。

### 8.3 预训练模型加载与冻结微调

文件：

```text
src\models.py
```

本次使用了：

```powershell
--freeze_layer backbone
```

训练脚本中的逻辑是：

```python
load_ckpt(backbone_net, config.pretrain_ckpt, trainable=False)
```

意思是：加载预训练的 backbone 参数，并把 backbone 中的参数设置为不参与训练。这样训练时只更新分类头 `head_net`。

为什么这样做？

```text
预训练 backbone 已经学会了边缘、纹理、形状等通用视觉特征；
垃圾分类数据集比较小，从头训练容易过拟合；
只训练分类头更快，也更适合先做开源项目复现。
```

### 8.4 损失函数与优化器

损失函数：

```text
CrossEntropyWithLabelSmooth
```

交叉熵用来衡量模型预测和真实标签的差距。Label smoothing 会让标签不要过于绝对，比如不是把正确类设为 1、其他类设为 0，而是稍微“柔和”一点，这通常能提升泛化能力。

优化器：

```text
Momentum
```

Momentum 可以理解为带“惯性”的梯度下降，更新参数时会参考前几步的方向，让训练更稳定。

学习率：

```text
src\lr_generator.py
```

代码使用余弦形式生成学习率。学习率决定每一步参数更新幅度，合适的学习率能让模型更快收敛。

## 9. 算法流程总结

完整流程可以概括为：

```text
1. 下载垃圾分类数据集和 MobileNetV2 预训练权重
2. 按 train/test 文件夹组织图片
3. 创建 ImageFolderDataset
4. 对训练图片做随机裁剪、翻转、颜色增强和标准化
5. 构建 MobileNetV2 backbone 和分类 head
6. 加载预训练 backbone 权重
7. 冻结 backbone，只训练新的 26 类分类 head
8. 使用交叉熵损失计算预测误差
9. 使用 Momentum 优化器更新 head 参数
10. 保存每个 epoch 的 checkpoint
11. 用最后一个 checkpoint 在测试集上评估准确率
```

## 10. 当前完成情况

已经完成：

```text
PDF 阅读
官方开源代码下载
数据集下载与解压
预训练权重下载
MindSpore 2.9 兼容补丁
15 epoch 冻结微调复现
测试集评估
复现日志和 checkpoint 保存
```

后续任务可以继续在此基础上做：

```text
任务 2：参考开源项目写自己的类似代码
任务 3：比较冻结微调、全量微调、LoRA 微调
任务 4：改进模型并对比准确率
```
