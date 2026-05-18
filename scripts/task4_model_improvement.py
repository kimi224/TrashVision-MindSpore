"""Task 4: improve the classifier head under identical hyperparameters."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT / "third_party" / "mindspore_r1.3_sparse" / "model_zoo" / "official" / "cv" / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))

from src.mobilenetV2 import MobileNetV2Backbone  # noqa: E402


class LinearHead(nn.Cell):
    """Baseline classifier: one linear layer from feature to class logits."""

    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class TTALinearHead(nn.Cell):
    """Improved inference head: average logits from original and flipped features."""

    def __init__(self, trained_linear_head):
        super().__init__()
        self.classifier = trained_linear_head.classifier

    def construct(self, x_pair):
        original = x_pair[:, 0, :]
        flipped = x_pair[:, 1, :]
        return (self.classifier(original) + self.classifier(flipped)) / 2.0


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


class ImprovedMLPHead(nn.Cell):
    """Improved classifier head with hidden representation and regularization."""

    def __init__(self, in_channels=1280, hidden_channels=256, num_classes=26, dropout=0.2):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_channels, num_classes),
        )

    def construct(self, x):
        return self.net(x)


class EpochLogger(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.losses = []
        self.epoch_losses = []
        self.epoch_start = 0.0

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_start = time.time()

    def step_end(self, run_context):
        loss = run_context.original_args().net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, Tensor):
            loss = float(np.mean(loss.asnumpy()))
        self.losses.append(float(loss))

    def epoch_end(self, run_context):
        epoch = run_context.original_args().cur_epoch_num
        avg_loss = float(np.mean(self.losses)) if self.losses else float("nan")
        self.epoch_losses.append(avg_loss)
        print(f"{self.name}: epoch {epoch}, avg_loss={avg_loss:.6f}, time={time.time() - self.epoch_start:.3f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4 model improvement")
    parser.add_argument("--feature_dir", default="runs/task2_custom")
    parser.add_argument("--data_dir", default="data/data_en")
    parser.add_argument("--pretrain_ckpt", default="pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt")
    parser.add_argument("--output_dir", default="runs/task4_improvement")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def project_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32),
        train["labels"].astype(np.int32),
        test["features"].astype(np.float32),
        test["labels"].astype(np.int32),
    )


def path_for_mindspore(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def create_flipped_image_dataset(split_dir, batch_size):
    image_ops = [
        vision.Decode(),
        vision.Resize((256, 256)),
        vision.CenterCrop(224),
        vision.HorizontalFlip(),
        vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW(),
    ]
    dataset = ds.ImageFolderDataset(path_for_mindspore(split_dir), shuffle=False, num_parallel_workers=1)
    dataset = dataset.map(image_ops, input_columns="image", num_parallel_workers=1)
    dataset = dataset.map(transforms.TypeCast(ms.int32), input_columns="label", num_parallel_workers=1)
    return dataset.batch(batch_size, drop_remainder=False)


def create_resized_image_dataset(split_dir, batch_size):
    image_ops = [
        vision.Decode(),
        vision.Resize((224, 224)),
        vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW(),
    ]
    dataset = ds.ImageFolderDataset(path_for_mindspore(split_dir), shuffle=False, num_parallel_workers=1)
    dataset = dataset.map(image_ops, input_columns="image", num_parallel_workers=1)
    dataset = dataset.map(transforms.TypeCast(ms.int32), input_columns="label", num_parallel_workers=1)
    return dataset.batch(batch_size, drop_remainder=False)


def load_frozen_backbone(pretrain_ckpt):
    backbone = MobileNetV2Backbone()
    params = load_checkpoint(str(pretrain_ckpt))
    load_param_into_net(backbone, params)
    for param in backbone.get_parameters():
        param.requires_grad = False
    backbone.set_train(False)
    return backbone


def extract_flipped_features(split_name, data_dir, pretrain_ckpt, output_dir, batch_size):
    feature_file = output_dir / f"{split_name}_flip_features.npz"
    if feature_file.exists():
        data = np.load(feature_file)
        return data["features"].astype(np.float32), data["labels"].astype(np.int32)

    backbone = load_frozen_backbone(pretrain_ckpt)
    model = Model(backbone)
    dataset = create_flipped_image_dataset(data_dir / split_name, batch_size)
    features, labels = [], []
    total = dataset.get_dataset_size()
    for index, item in enumerate(dataset.create_dict_iterator(output_numpy=True), start=1):
        feature_map = model.predict(Tensor(item["image"], ms.float32)).asnumpy()
        pooled = feature_map.mean(axis=(2, 3)).astype(np.float32)
        features.append(pooled)
        labels.append(item["label"].astype(np.int32))
        print(f"Extract flipped {split_name} features: batch {index}/{total}", flush=True)
    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    np.savez_compressed(feature_file, features=feature_array, labels=label_array)
    return feature_array, label_array


def extract_resized_features(split_name, data_dir, pretrain_ckpt, output_dir, batch_size):
    feature_file = output_dir / f"{split_name}_resize_features.npz"
    if feature_file.exists():
        data = np.load(feature_file)
        return data["features"].astype(np.float32), data["labels"].astype(np.int32)

    backbone = load_frozen_backbone(pretrain_ckpt)
    model = Model(backbone)
    dataset = create_resized_image_dataset(data_dir / split_name, batch_size)
    features, labels = [], []
    total = dataset.get_dataset_size()
    for index, item in enumerate(dataset.create_dict_iterator(output_numpy=True), start=1):
        feature_map = model.predict(Tensor(item["image"], ms.float32)).asnumpy()
        pooled = feature_map.mean(axis=(2, 3)).astype(np.float32)
        features.append(pooled)
        labels.append(item["label"].astype(np.int32))
        print(f"Extract resized {split_name} features: batch {index}/{total}", flush=True)
    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    np.savez_compressed(feature_file, features=feature_array, labels=label_array)
    return feature_array, label_array


def create_dataset(features, labels, batch_size, shuffle):
    dataset = ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)


def create_pair_dataset(original_features, flipped_features, labels, batch_size):
    pair_features = np.stack([original_features, flipped_features], axis=1).astype(np.float32)
    dataset = ds.NumpySlicesDataset(
        {"data": pair_features, "label": labels.astype(np.int32)},
        shuffle=False,
    )
    return dataset.batch(batch_size, drop_remainder=False)


def count_params(params):
    return int(sum(np.prod(tuple(param.shape)) for param in params))


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
    logger = EpochLogger(name)
    ckpt_dir = output_dir / name / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        prefix=name,
        directory=str(ckpt_dir),
        config=CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=1),
    )
    start = time.time()
    model.train(args.epochs, train_dataset, callbacks=[logger, ckpt], dataset_sink_mode=False)
    train_seconds = time.time() - start
    metrics = model.eval(test_dataset, dataset_sink_mode=False)
    result = {
        "name": name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "train_seconds": train_seconds,
        "accuracy": float(metrics["acc"]),
        "trainable_params": count_params(net.trainable_params()),
        "total_params": count_params(net.get_parameters()),
        "epoch_losses": logger.epoch_losses,
    }
    print(f"{name} eval result: {metrics}")
    return result


def eval_tta_model(base_head, test_features, test_flip_features, test_labels, args):
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    tta_model = Model(TTALinearHead(base_head), loss_fn=loss, metrics={"acc"})
    tta_dataset = create_pair_dataset(test_features, test_flip_features, test_labels, args.batch_size)
    metrics = tta_model.eval(tta_dataset, dataset_sink_mode=False)
    result = {
        "name": "improved_tta",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "train_seconds": 0.0,
        "accuracy": float(metrics["acc"]),
        "trainable_params": count_params(base_head.trainable_params()),
        "total_params": count_params(base_head.get_parameters()),
        "epoch_losses": [],
    }
    print(f"improved_tta eval result: {metrics}")
    return result


def main():
    args = parse_args()
    ms.set_seed(args.seed)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    feature_dir = project_path(args.feature_dir)
    data_dir = project_path(args.data_dir)
    pretrain_ckpt = project_path(args.pretrain_ckpt)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_labels, test_features, test_labels = load_features(feature_dir)
    in_channels = int(train_features.shape[1])
    num_classes = int(len(np.unique(train_labels)))

    common_hyperparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    baseline_head = LinearHead(in_channels, num_classes)
    baseline = train_one_model(
        "baseline_linear",
        baseline_head,
        train_features,
        train_labels,
        test_features,
        test_labels,
        args,
        output_dir,
    )
    improved = train_one_model(
        "improved_multibranch",
        MultiBranchMLPHead(in_channels, hidden_channels=512, num_classes=num_classes, branches=3, dropout=0.1),
        train_features,
        train_labels,
        test_features,
        test_labels,
        args,
        output_dir,
    )

    summary = {
        "task": "task4_model_improvement",
        "data": {
            "train_samples": int(train_features.shape[0]),
            "test_samples": int(test_features.shape[0]),
            "feature_dim": in_channels,
            "num_classes": num_classes,
        },
        "common_hyperparams": common_hyperparams,
        "baseline_model": "LinearHead: Dense(1280, 26)",
        "improved_model": "MultiBranchMLPHead: mean of 3 branches, each Dense(1280,512)+ReLU+Dropout(0.1)+Dense(512,26)",
        "results": [baseline, improved],
        "accuracy_delta": improved["accuracy"] - baseline["accuracy"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
