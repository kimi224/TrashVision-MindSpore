# AIhomework3 源码说明

本项目完成 PDF 中 4 个任务：

1. 复现 MindSpore 官方 MobileNetV2 垃圾分类微调项目。
2. 参考开源项目编写自己的微调代码。
3. 对比冻结微调、全量微调、LoRA 微调。
4. 在相同超参数下改进分类头并比较准确率。

## 我们自己写的主要源码

```text
scripts/task2_custom_finetune.py
scripts/task3_finetune_compare.py
scripts/task4_model_improvement.py
```

## 报告文档

```text
docs/task1_reproduce.md
docs/task2_custom_finetune.md
docs/task3_finetune_compare.md
docs/task4_model_improvement.md
```

## 重要结果

```text
任务 1：官方复现冻结微调，准确率 86.54%
任务 2：自写微调流程，准确率 89.23%
任务 3：冻结/LoRA/全量微调对比
任务 4：分类头改进，准确率从 87.69% 提升到 88.46%
```

## 环境说明

项目运行时使用当前工程下的虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

主要依赖：

```text
mindspore 2.9.0
numpy 1.26.4
pillow 12.2.0
PyYAML 6.0.3
psutil 7.2.2
```

## 不包含的内容

提交源码整理目录中不包含：

```text
.venv 虚拟环境
data 数据集
pretrain_checkpoint 预训练权重
third_party 克隆的开源项目源码
大型 checkpoint 训练结果
```

这些内容体积较大，且不是我们自己编写的源码。报告中已经记录下载地址、运行命令和结果。
