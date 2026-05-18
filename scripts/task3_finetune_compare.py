"""Task 3: compare freeze, full, and LoRA fine-tuning.

The script measures trainable parameters, elapsed training time, peak process
memory, and test accuracy for three transfer-learning strategies.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import HeUniform, Zero, initializer
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT / "third_party" / "mindspore_r1.3_sparse" / "model_zoo" / "official" / "cv" / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))

from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 fine-tuning comparison")
    parser.add_argument("--data_dir", default="data/data_en")
    parser.add_argument("--pretrain_ckpt", default="pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt")
    parser.add_argument("--output_dir", default="runs/task3_compare")
    parser.add_argument("--feature_source", default="runs/task2_custom")
    parser.add_argument("--freeze_epochs", type=int, default=20)
    parser.add_argument("--lora_epochs", type=int, default=20)
    parser.add_argument("--full_epochs", type=int, default=2)
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lora_lr", type=float, default=0.001)
    parser.add_argument("--full_lr", type=float, default=0.001)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    return parser.parse_args()


def project_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_for_mindspore(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def count_params(params):
    return int(sum(np.prod(tuple(param.shape)) for param in params))


def set_trainable(cell, trainable):
    for param in cell.get_parameters():
        param.requires_grad = trainable


def create_image_dataset(split_dir, batch_size, train):
    if train:
        image_ops = [
            vision.Decode(),
            vision.Resize((256, 256)),
            vision.RandomCrop(224),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
            vision.HWC2CHW(),
        ]
    else:
        image_ops = [
            vision.Decode(),
            vision.Resize((256, 256)),
            vision.CenterCrop(224),
            vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
            vision.HWC2CHW(),
        ]
    dataset = ds.ImageFolderDataset(path_for_mindspore(split_dir), shuffle=train, num_parallel_workers=1)
    dataset = dataset.map(image_ops, input_columns="image", num_parallel_workers=1)
    dataset = dataset.map(transforms.TypeCast(ms.int32), input_columns="label", num_parallel_workers=1)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def create_feature_dataset(features, labels, batch_size, shuffle):
    dataset = ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)


class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


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


class MemoryAndLoss(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.epoch_losses = []
        self._losses = []
        self._epoch_start = 0.0

    def epoch_begin(self, run_context):
        self._losses = []
        self._epoch_start = time.time()
        self._update_peak()

    def step_end(self, run_context):
        self._update_peak()
        loss = run_context.original_args().net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, Tensor):
            loss = float(np.mean(loss.asnumpy()))
        self._losses.append(float(loss))

    def epoch_end(self, run_context):
        self._update_peak()
        epoch = run_context.original_args().cur_epoch_num
        avg_loss = float(np.mean(self._losses)) if self._losses else float("nan")
        self.epoch_losses.append(avg_loss)
        print(f"{self.name}: epoch {epoch}, avg_loss={avg_loss:.6f}, time={time.time() - self._epoch_start:.3f}s")

    def _update_peak(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32),
        train["labels"].astype(np.int32),
        test["features"].astype(np.float32),
        test["labels"].astype(np.int32),
    )


def load_backbone(pretrain_ckpt, trainable):
    backbone = MobileNetV2Backbone()
    params = load_checkpoint(str(pretrain_ckpt))
    load_param_into_net(backbone, params)
    set_trainable(backbone, trainable)
    return backbone


def run_feature_strategy(name, model_cell, train_data, test_data, epochs, lr, batch_size, output_dir):
    train_features, train_labels, test_features, test_labels = train_data + test_data
    train_dataset = create_feature_dataset(train_features, train_labels, batch_size, shuffle=True)
    test_dataset = create_feature_dataset(test_features, test_labels, batch_size, shuffle=False)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(model_cell.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=4e-5)
    model = Model(model_cell, loss_fn=loss, optimizer=optimizer, metrics={"acc"})
    monitor = MemoryAndLoss(name)
    ckpt_dir = output_dir / name / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        prefix=name,
        directory=str(ckpt_dir),
        config=CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=1),
    )
    start = time.time()
    model.train(epochs, train_dataset, callbacks=[monitor, ckpt], dataset_sink_mode=False)
    train_seconds = time.time() - start
    metrics = model.eval(test_dataset, dataset_sink_mode=False)
    return {
        "strategy": name,
        "epochs": epochs,
        "train_seconds": train_seconds,
        "peak_rss_mb": monitor.peak_rss / 1024 / 1024,
        "accuracy": float(metrics["acc"]),
        "trainable_params": count_params(model_cell.trainable_params()),
        "total_params": count_params(model_cell.get_parameters()),
        "epoch_losses": monitor.epoch_losses,
    }


def run_full_strategy(data_dir, pretrain_ckpt, epochs, lr, batch_size, output_dir):
    backbone = load_backbone(pretrain_ckpt, trainable=True)
    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=26)
    net = mobilenet_v2(backbone, head)
    train_dataset = create_image_dataset(data_dir / "train", batch_size, train=True)
    test_dataset = create_image_dataset(data_dir / "test", batch_size, train=False)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=4e-5)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"})
    monitor = MemoryAndLoss("full")
    ckpt_dir = output_dir / "full" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        prefix="full",
        directory=str(ckpt_dir),
        config=CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=1),
    )
    start = time.time()
    model.train(epochs, train_dataset, callbacks=[monitor, ckpt], dataset_sink_mode=False)
    train_seconds = time.time() - start
    metrics = model.eval(test_dataset, dataset_sink_mode=False)
    return {
        "strategy": "full",
        "epochs": epochs,
        "train_seconds": train_seconds,
        "peak_rss_mb": monitor.peak_rss / 1024 / 1024,
        "accuracy": float(metrics["acc"]),
        "trainable_params": count_params(net.trainable_params()),
        "total_params": count_params(net.get_parameters()),
        "epoch_losses": monitor.epoch_losses,
    }


def main():
    args = parse_args()
    ms.set_seed(7)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_path(args.data_dir)
    pretrain_ckpt = project_path(args.pretrain_ckpt)
    feature_dir = project_path(args.feature_source)

    train_features, train_labels, test_features, test_labels = load_features(feature_dir)
    num_classes = len(np.unique(train_labels))

    results = []
    freeze_head = LinearHead(train_features.shape[1], num_classes)
    results.append(
        run_feature_strategy(
            "freeze",
            freeze_head,
            (train_features, train_labels),
            (test_features, test_labels),
            args.freeze_epochs,
            args.lr,
            args.feature_batch_size,
            output_dir,
        )
    )

    lora_head = LoRALinearHead(train_features.shape[1], num_classes, args.lora_rank, args.lora_alpha)
    results.append(
        run_feature_strategy(
            "lora",
            lora_head,
            (train_features, train_labels),
            (test_features, test_labels),
            args.lora_epochs,
            args.lora_lr,
            args.feature_batch_size,
            output_dir,
        )
    )

    results.append(run_full_strategy(data_dir, pretrain_ckpt, args.full_epochs, args.full_lr, args.image_batch_size, output_dir))

    summary = {
        "task": "task3_finetune_compare",
        "data": {
            "train_samples": int(train_features.shape[0]),
            "test_samples": int(test_features.shape[0]),
            "feature_dim": int(train_features.shape[1]),
            "num_classes": int(num_classes),
        },
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "layers": ["LoRALinearHead.classifier"]},
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
