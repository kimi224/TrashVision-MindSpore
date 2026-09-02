# TrashVision-MindSpore

基于 **MindSpore + MobileNetV2** 的 26 类垃圾分类迁移学习实验：从开源项目复现出发，自研微调流程，系统对比三种微调策略，并在严格同超参的前提下改进分类头。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MindSpore](https://img.shields.io/badge/MindSpore-2.9.0-orange)
![Platform](https://img.shields.io/badge/Platform-CPU-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> 一句话概括：**把预训练 MobileNetV2 当特征提取器冻结起来，把全部实验变成"在 1280 维特征上训练小分类器"的问题**——于是训练从百秒级降到秒级，结构搜索、多种子、集成这些需要几十次训练的对比才做得动。

---

## 目录

- [项目背景与任务](#项目背景与任务)
- [亮点](#亮点)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [实现思想](#实现思想)
- [实验结果](#实验结果)
- [注意事项与踩坑清单](#注意事项与踩坑清单)
- [后续可改进方向](#后续可改进方向)
- [参考与致谢](#参考与致谢)
- [许可证](#许可证)

---

## 项目背景与任务

数据集为 26 类生活垃圾图片（训练集 2593 张、测试集 260 张），基座模型为 ImageNet 预训练的 MobileNetV2。项目按课程要求拆成 4 个递进任务：

| 编号 | 任务                                                | 产出脚本                                             | 结果                                    |
| :--: | --------------------------------------------------- | ---------------------------------------------------- | --------------------------------------- |
|  1   | 复现 MindSpore 官方 MobileNetV2 微调项目            | 直接运行 `third_party` 内官方 `train.py` / `eval.py` | 86.54%                                  |
|  2   | 参考开源项目**自写**微调流程（模型加载 + 迁移微调） | `scripts/task2_custom_finetune.py`                   | 89.23%                                  |
|  3   | 冻结微调 / 全量微调 / LoRA 微调三者对比             | `scripts/task3_finetune_compare.py`                  | 见[策略对比表](#任务-3三种微调策略对比) |
|  4   | 在超参完全相同的前提下改进模型并对比准确率          | `scripts/task4_model_improvement.py`                 | 87.69% → **90.77%**                     |

每个任务的详细原理、代码讲解与结论写在 [`docs/`](docs) 下对应的文档中，本 README 负责"整体架构 + 复现路径 + 思想与坑"。

---

## 亮点

- **一条可复现的流水线**：任务 2 产出的特征缓存是任务 3、4 的共同输入，保证对比公平。
- **特征缓存加速**：backbone 冻结后只跑一次前向，把特征存成 `.npz`，后续所有训练都在 1280 维向量上做，单次训练从分钟级降到秒级。
- **手写 LoRA 适配层**：`lora_a` / `lora_b` / `lora_bias` 三个可训练张量实现低秩增量，并统计可训练参数量。
- **可横向对比的资源指标**：CPU 环境下统一记录训练耗时与进程峰值内存（`peak_rss_mb`），不是只看准确率。
- **完整的搜索轨迹**：从线性头 → 多分支 MLP → SE 注意力 / 特征门控 → 多种子集成，每一步的准确率都留在 `runs/**/summary*.json` 里。

---

## 目录结构

```text
trashvision-mindspore/
├── README.md                       # 本文件：架构说明、实现思想、复现步骤
├── LICENSE                         # MIT
├── requirements.txt                # 依赖清单（已验证版本）
│
├── docs/                           # 四个任务的实验报告（原理 + 代码讲解 + 结论）
│   ├── task1_reproduce.md
│   ├── task2_custom_finetune.md
│   ├── task3_finetune_compare.md
│   └── task4_model_improvement.md
│
├── scripts/                        # 我们自己写的源码
│   ├── task2_custom_finetune.py    # 任务 2：自写迁移微调流程（含特征提取与缓存）
│   ├── task3_finetune_compare.py   # 任务 3：冻结 / 全量 / LoRA 三种策略对比
│   ├── task4_model_improvement.py  # 任务 4：基线 vs 改进分类头（严格同超参）
│   └── exploration/                # 任务 4 的扩展探索（非报告主线，可复现搜索过程）
│       ├── task4_head_search.py    #   小头结构快速搜索（18 组候选）
│       ├── task4_phase2_search.py  #   更宽/更深/不同激活 + 集成
│       ├── task4_phase3_ensemble.py#   多模型 logits 集成 + 翻转 TTA
│       ├── task4_phase4_search.py  #   SE 注意力 + 超宽分支
│       ├── task4_best_head.py      #   SE + 特征门控 + 超宽多分支的单模型最强组合
│       └── task4_final_ensemble.py #   最终多种子、多结构集成（本项目最高准确率）
│
├── third_party/                    # 官方开源代码（vendored，非本人作品）
│   └── mindspore_r1.3_sparse/      # MindSpore r1.3 Model Zoo 的 MobileNetV2 稀疏检出
│       └── model_zoo/official/cv/mobilenetv2/
│
├── runs/                           # 实验结果（仅 summary*.json 入库，其余已 gitignore）
│   ├── task4_head_search.json      #   18 组候选头的准确率/耗时/参数量
│   ├── task2_custom/               #   summary.json + train/test 特征缓存
│   ├── task3_compare/              #   summary.json（三种策略的资源与准确率）
│   └── task4_improvement/          #   summary*.json（基线、各阶段搜索、最终集成）
│
├── logs/                           # 训练/评估日志（*.log，不入库）
├── data/                           # 数据集（不入库，见「数据准备」）
└── pretrain_checkpoint/            # 预训练权重（不入库，见「数据准备」）
```

各顶层目录职责：

| 目录                            | 是否入库 | 说明                                                                          |
| ------------------------------- | :------: | ----------------------------------------------------------------------------- |
| `docs/`                         |    ✅    | 课程报告正文，每个任务一份，含原理与结论，可直接作为写作素材                  |
| `scripts/`                      |    ✅    | 自研源码；`exploration/` 是与报告主线无关的扩展搜索，删掉不影响四个任务       |
| `third_party/`                  |    ✅    | 官方 MobileNetV2 代码副本（约 4.5 MB），已移除嵌套 `.git` 并打了 2.x 兼容补丁 |
| `runs/`                         |   部分   | 只提交结果摘要 JSON，便于对照复现；`*.ckpt` / `*.meta` / `*.npz` 均忽略       |
| `logs/`                         |    ❌    | 运行日志，重新跑脚本即可生成                                                  |
| `data/`、`pretrain_checkpoint/` |    ❌    | 体积大且非原创，按 README 指引自行下载                                        |

---

## 快速开始

### 1. 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

已验证环境：Windows 11 / Python 3.12.10 / mindspore 2.9.0（CPU）/ numpy 1.26.4。

### 2. 数据准备

1. 下载垃圾分类数据集（课程 PDF 提供）：

   ```text
   https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/MindStudio-pc/data_en.zip
   ```

2. 解压到 `data/data_en`，结构如下（`ImageFolderDataset` 按文件夹名自动编号 0–25）：

   ```text
   data/data_en/
   ├── train/{Banana Peel, Battery, ..., Vegetable Leaf}/
   └── test/{Banana Peel, Battery, ..., Vegetable Leaf}/
   ```

3. 官方 `eval.py` 只读 `validation_preprocess`，而数据集里叫 `test`，复制一份目录适配官方脚本：

   ```powershell
   Copy-Item -Recurse -Force 'data\data_en\test' 'data\data_en\validation_preprocess'
   ```

4. 下载官方预训练权重 `mobilenetv2_cpu_gpu.ckpt`（链接见 `third_party/mindspore_r1.3_sparse/model_zoo/official/cv/mobilenetv2/README_CN.md` 与官方 Model Zoo 页面），放到 `pretrain_checkpoint/`。

### 3. 一键复现（按顺序执行）

> 所有命令都在仓库根目录执行；下文用 `python` 表示已激活虚拟环境的解释器（Windows 亦可写 `.venv\Scripts\python.exe`）。

**任务 1 — 复现官方项目**（冻结 backbone，15 epoch）

```powershell
$env:PYTHONPATH='third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2'
python third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\train.py `
  --config_path third_party\mindspore_r1.3_sparse\model_zoo\official\cv\mobilenetv2\default_config_cpu.yaml `
  --platform CPU --dataset_path data\data_en `
  --pretrain_ckpt pretrain_checkpoint\mobilenetv2_cpu_gpu.ckpt `
  --freeze_layer backbone --epoch_size 15 --batch_size 150 `
  --save_checkpoint True --save_checkpoint_path runs\task1_reproduce\
```

**任务 2 — 自研微调流程**（特征提取 + 缓存 + 训练分类头）

```powershell
python scripts\task2_custom_finetune.py --epochs 30 --output_dir runs\task2_custom --force_extract
```

> 这一步会生成 `runs/task2_custom/{train,test}_features.npz`，**任务 3 和任务 4 都依赖它**，必须先跑。
> 已存在缓存时脚本会直接复用；更换 backbone、预处理或数据集后请务必加 `--force_extract` 重新提取。

**任务 3 — 三种微调策略对比**

```powershell
python scripts\task3_finetune_compare.py --output_dir runs\task3_compare `
  --freeze_epochs 20 --lora_epochs 20 --full_epochs 1
```

**任务 4 — 分类头改进（严格同超参对比）**

```powershell
python scripts\task4_model_improvement.py --output_dir runs\task4_improvement `
  --epochs 30 --batch_size 64 --lr 0.03 --momentum 0.9 --weight_decay 0.00004
```

**可选：任务 4 的扩展搜索 / 集成**

```powershell
python scripts\exploration\task4_head_search.py
python scripts\exploration\task4_phase2_search.py
python scripts\exploration\task4_final_ensemble.py
```

耗时参考（CPU）：任务 2 特征提取数分钟、分类头训练秒级；任务 3 全量微调 1 轮约 3.5 分钟；任务 4 单次搜索 3–70 秒。

---

## 实现思想

### 1. 总原则：把"特征提取"和"分类决策"解耦

26 类垃圾数据只有 2593 张训练图，直接在 CPU 上微调整个 MobileNetV2 既不现实也容易过拟合（实测 1 轮就要 213 秒、4.6 GB 内存）。因此整个项目围绕一个拆分展开：

```text
backbone（MobileNetV2，ImageNet 预训练，冻结，只做特征提取）
        │  全局平均池化 → 1280 维向量，缓存成 .npz
        ▼
head（小型分类器，可训练，全部实验的发生地）
```

**为什么要缓存特征**——这是本项目的关键工程决策，收益有三：

1. **速度**：backbone 冻结后同一张图每次特征完全相同，没必要每个 epoch 重跑整网；缓存后单次训练从百秒级降到秒级。
2. **可比**：缓存让"结构搜索 / 多种子 / 集成"这些需要跑几十次训练的对比变得可行，也让所有候选头看到**逐字节相同**的输入。
3. **可复现**：超参、loss 曲线、参数量、耗时全部写进 `summary*.json`，任何人重跑都能对上。

代价也很明确：backbone 冻住后，模型的上限被预训练特征的质量锁死。所以任务 4 的改进全部集中在 head 上——这是"在给定特征下如何把决策边界做得更好"的问题。

### 2. 任务 1：复现不是"跑通"，而是"能解释"

复现官方 r1.3 项目时重点做了三件适配工作，它们都记录在 `docs/task1_reproduce.md`：

- **版本兼容补丁**：MindSpore 2.x 把 `mindspore.dataset.vision.c_transforms` 合并进了 `mindspore.dataset.vision`，用 `try/except ModuleNotFoundError` 做双版本兼容，不改动官方逻辑。
- **目录名适配**：官方评估脚本只读 `validation_preprocess`，因此复制一份 `test` 目录（只改目录名，不动任何图片）。
- **路径编码**：MindSpore 的 C++ 数据读取层对含中文的绝对路径处理有问题，一律使用相对路径。

### 3. 任务 2：学官方，但不抄官方

只复用官方的 `MobileNetV2Backbone` 网络结构定义，**不调用**官方 `train.py` / `eval.py`。命令行解析、数据管线、预训练权重加载、backbone 冻结、特征池化与缓存、分类头训练、评估与 summary 落盘全部自己实现。

```python
class LinearGarbageHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)
```

这就是迁移学习的最小形态：**保留通用特征提取器，把最后的分类器换成适配新任务的新头**。

### 4. 任务 3：三种微调策略，比的不只是准确率

| 策略      | 做法                                                                                                   | 训练参数  |
| --------- | ------------------------------------------------------------------------------------------------------ | --------- |
| 冻结微调  | `backbone` 全部 `requires_grad=False`，优化器只接收 `head.trainable_params()`                          | 33,306    |
| 全量微调  | backbone + head 全部可训练，直接读原始图片                                                             | 2,291,290 |
| LoRA 微调 | 冻结 `base` 权重，额外训练 `lora_a (1280×8)`、`lora_b (8×26)`、`lora_bias`，`scaling = alpha/rank = 2` | 10,474    |

LoRA 的核心是把更新量限制成低秩增量 `ΔW = A × B`：

```python
def construct(self, x):
    return self.base(x) + ops.matmul(ops.matmul(x, self.lora_a), self.lora_b) * self.scaling + self.lora_bias
```

由于 CPU 上给卷积层加 LoRA 就必须跑整网（成本高到失去对比意义），本实验把 LoRA 加在分类头线性层上——**重点是比较"低秩适配"这一机制在参数量、耗时、内存上的特征，而不是追求它的绝对精度**。这一点在解读结果时非常关键。

### 5. 任务 4：先立规矩，再谈改进

改进类实验最容易犯的错是"顺手调了学习率又换了结构，最后不知道提升来自哪里"。本任务的公平对比协议是：

- 同一份缓存特征（`runs/task2_custom/*.npz`），数据划分完全一致；
- 同一套超参：`epoch=30, batch_size=64, lr=0.03, momentum=0.9, weight_decay=4e-5, seed=11`；
- 同一个 `train_one_model()` 训练函数与同一个评估流程；
- 超参与指标一并写入 `summary.json`，可事后核验。

结构演进路线与背后的动机：

```text
LinearHead (1280→26)                     线性边界，参数少、训练快，是干净的基线
   ↓ 加非线性与正则
MultiBranchMLPHead (3×(1280→512→26) 平均)  分支多样性 + Dropout，logits 平均相当于轻量集成
   ↓ 探索结构变体
输入 BN / 残差分支 / 多尺度分支 / bottleneck 分支
   ↓ 引入注意力与门控
SE 特征重加权 + 逐样本特征门控 + 超宽多分支 (5×1024)
   ↓ 集成降方差
多种子 + 多结构 logits 平均（ensemble_4arch）
```

可以提炼出的三条经验：

1. **非线性 + Dropout 是主要收益来源**，多分支平均进一步降低单分支偶然误差；
2. **容量不是越大越好**：超宽分支（926 万参数、62 秒）与较省的结构（295 万参数、30 秒）在本数据集上准确率相同（90.38%）；
3. **集成的收益很快撞上天花板**：多个 90.38% 的模型平均后仍是 90.38%–90.77%，说明剩余误差主要来自冻结特征本身的判别性，而不是分类头的表达能力。

---

## 实验结果

> 数据来源：`runs/**/summary*.json`；测试集固定 260 张图片（1 张 ≈ 0.385 个百分点）。

### 总览

| 任务 | 方法                                             |          测试准确率 |
| ---- | ------------------------------------------------ | ------------------: |
| 1    | 官方项目复现（冻结 backbone，15 epoch）          |              86.54% |
| 2    | 自研流程（冻结 backbone + 自写分类头，30 epoch） |              89.23% |
| 3    | 三种微调策略对比（见下表）                       |                   — |
| 4    | 分类头改进：基线 → 改进 → 搜索 → 集成            | 87.69% → **90.77%** |

### 任务 3：三种微调策略对比

| 策略      | epoch | 可训练参数 | 训练时间 |   峰值内存 | 测试准确率 |
| --------- | ----: | ---------: | -------: | ---------: | ---------: |
| 冻结微调  |    20 |     33,306 |   3.45 s |  306.89 MB | **87.69%** |
| LoRA 微调 |    20 |     10,474 |   3.68 s |  319.32 MB |     43.08% |
| 全量微调  |     1 |  2,291,290 | 213.52 s | 4644.19 MB |     49.62% |

结论：小数据集 + CPU 场景下**冻结微调性价比最高**（约为全量微调内存的 1/15）。LoRA 与全量微调的准确率偏低是实验设置的必然结果——LoRA 只作用于分类头且 rank=8，全量微调只跑了 1 轮尚未收敛，**不代表两种方法本身的上限**。

### 任务 4：分类头改进

| 模型         | 结构                                     |    参数量 | 训练时间 |                测试准确率 |
| ------------ | ---------------------------------------- | --------: | -------: | ------------------------: |
| 基线         | `Dense(1280, 26)`                        |    33,306 |     ~3 s | 87.69%（同批重跑 88.08%） |
| 初版改进     | 3 分支 MLP(512) logits 平均              | 2,007,630 |   12.2 s |                    89.62% |
| 结构搜索     | `multi3_512_d01`                         | 2,007,630 |   17.4 s |                    90.38% |
| 单模型最优   | `vwide5_1024_s11`（5 分支 ×1024 + 门控） | 9,267,330 |   62.3 s |                    90.38% |
| 轻量同类     | `diverse_s11`                            | 2,955,624 |   30.0 s |                    90.38% |
| **最终集成** | `ensemble_4arch`（多结构 + 多种子平均）  |         — |        — |                **90.77%** |

提升幅度：基线 87.69% → 90.77%，**+3.08 个百分点**（约多分对 8 张测试图）。

---

## 注意事项

1. **中文路径**：MindSpore 的 C++ 数据集层处理含中文的绝对路径会出错。请始终在项目根目录运行，并传入 `data\data_en` 这类**相对路径**（脚本内部已用 `path_for_mindspore()` 做转换）。
2. **MindSpore 2.x 与 r1.3 官方代码的 import 差异**：`c_transforms` 已被合并，官方代码直接跑会 `ModuleNotFoundError`。`third_party` 副本中已加兼容补丁，请勿用原始仓库覆盖该目录。
3. **官方评估脚本的目录约定**：必须存在 `data/data_en/validation_preprocess`，否则评估读不到数据。
4. **特征缓存的依赖关系**：任务 3、4 依赖任务 2 生成的 `runs/task2_custom/*.npz`。改了 backbone、图像预处理或数据集后，**必须 `--force_extract` 重新提取**，否则会用旧特征得出错误结论。
5. **测试集只有 260 张**：1 张图 = 0.385 个百分点。完全相同的 `LinearHead` 在不同批次里也出现过 87.69% / 88.08% / 89.62%；因此**不要过度解读 0.4 个百分点的差异**，要看趋势和多次平均。
6. **CPU 上全量微调极慢**：1 轮约 213 秒、峰值内存 4.6 GB。本仓库对它的结论只针对"1 轮未收敛"这一状态，不能外推到充分训练后的效果。
7. **内存指标口径**：统计的是 CPU 进程峰值内存（`peak_rss_mb`），**不是 GPU 显存**。换到 GPU / 昇腾环境请替换统计方式。
8. **大文件不入库**：`data/`、`pretrain_checkpoint/`、`*.ckpt`、`*.meta`、`*.npz`、`logs/` 均已加入 `.gitignore`，clone 后需按上文自行准备数据与权重；`runs/` 只保留摘要 JSON。
9. **随机性无法完全消除**：脚本固定了 `seed`（默认 11），但 MindSpore 数据加载与初始化仍有波动，做严谨结论建议跑多个 seed 取平均。
10. **文档与最新搜索的口径差异**：`docs/task4_model_improvement.md` 记录的是报告版本（88.85%），而 `runs/task4_improvement/summary_final.json` 记录的是后续扩展搜索的最终结果（90.77%）。两者不矛盾，前者是提交报告时的快照，后者是仓库当前代码能达到的结果。
11. **`third_party` 不是本人作品**：它是 MindSpore r1.3 官方 Model Zoo 的稀疏检出副本（已删除嵌套 `.git`，避免被误认为 submodule），仅供任务 1 复现与 backbone 结构复用。

---

## 后续可改进方向

- 在 GPU / 昇腾上把全量微调跑满，补齐任务 3 中"未收敛"的那一格；
- 把 LoRA 插到 backbone 的卷积层（而非仅分类头），验证低秩适配在特征提取阶段的真实表现；
- 用更强的增广（多尺度裁剪、颜色抖动）重新提取特征，或引入 TTA 提升推理稳定性；
- 测试集太小，可改为多次随机划分 / 交叉验证来降低评估方差；
- 用 `export.py` 导出 MindIR，做一个最小推理 Demo。

---

## 参考与致谢

- 数据集：课程提供的 26 类垃圾分类数据集 `data_en.zip`（见[数据准备](#2-数据准备)）。
- MindSpore r1.3 Model Zoo — MobileNetV2：官方开源代码副本位于 `third_party/mindspore_r1.3_sparse/model_zoo/official/cv/mobilenetv2`，上游见 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2>。
- MobileNetV2: Sandler et al., _MobileNetV2: Inverted Residuals and Linear Bottlenecks_, CVPR 2018（<https://arxiv.org/pdf/1801.04381.pdf>）。
- LoRA: Hu et al., _LoRA: Low-Rank Adaptation of Large Language Models_, ICLR 2022（<https://arxiv.org/abs/2106.09685>）。

---

## 许可证

本项目自研代码（`scripts/`、`docs/`、本仓库新增文件）采用 [MIT](LICENSE) 许可证；`third_party/` 下的代码版权归 MindSpore 官方所有，遵循其原始许可证。
