"""Task 2 custom MobileNetV2 transfer-learning script.

This script intentionally does not call the official train.py/eval.py entry
points. It reuses the MobileNetV2 network definition from the downloaded
MindSpore example, then implements our own data pipeline, checkpoint loading,
feature extraction, classifier training, and evaluation flow.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

from src.mobilenetV2 import MobileNetV2Backbone  # noqa: E402


class LinearGarbageHead(nn.Cell):
    """A small classification head for cached MobileNetV2 features."""

    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class EpochSummary(Callback):
    """Print compact epoch summaries for the report log."""

    def __init__(self):
        super().__init__()
        self.losses = []
        self.epoch_start = 0.0

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_start = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, Tensor):
            loss = float(np.mean(loss.asnumpy()))
        self.losses.append(float(loss))

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        elapsed = time.time() - self.epoch_start
        avg_loss = float(np.mean(self.losses)) if self.losses else float("nan")
        print(
            f"epoch {cb_params.cur_epoch_num}/{cb_params.epoch_num}, "
            f"avg_loss={avg_loss:.6f}, time={elapsed:.3f}s",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 custom fine-tuning")
    parser.add_argument("--data_dir", default="data/data_en")
    parser.add_argument("--pretrain_ckpt", default="pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt")
    parser.add_argument("--output_dir", default="runs/task2_custom")
    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--head_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--force_extract", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_classes(split_dir):
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])


def path_for_mindspore(path):
    """Use relative paths to avoid Windows non-ASCII absolute-path issues."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def create_image_dataset(split_dir, batch_size):
    transform_img = [
        vision.Decode(),
        vision.Resize((256, 256)),
        vision.CenterCrop(224),
        vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW(),
    ]
    transform_label = transforms.TypeCast(ms.int32)

    dataset = ds.ImageFolderDataset(path_for_mindspore(split_dir), shuffle=False, num_parallel_workers=1)
    dataset = dataset.map(transform_img, input_columns="image", num_parallel_workers=1)
    dataset = dataset.map(transform_label, input_columns="label", num_parallel_workers=1)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def load_frozen_backbone(pretrain_ckpt):
    backbone = MobileNetV2Backbone()
    params = load_checkpoint(str(pretrain_ckpt))
    not_loaded = load_param_into_net(backbone, params)
    for param in backbone.get_parameters():
        param.requires_grad = False
    backbone.set_train(False)
    print(f"Loaded pretrained backbone from: {pretrain_ckpt}")
    print(f"Backbone trainable params after freezing: {sum(p.requires_grad for p in backbone.get_parameters())}")
    print(f"Unloaded params info: {not_loaded}")
    return backbone


def extract_or_load_features(backbone, split_name, split_dir, output_dir, batch_size, force_extract):
    feature_file = output_dir / f"{split_name}_features.npz"
    if feature_file.exists() and not force_extract:
        data = np.load(feature_file)
        print(f"Loaded cached {split_name} features: {feature_file}")
        return data["features"].astype(np.float32), data["labels"].astype(np.int32)

    dataset = create_image_dataset(split_dir, batch_size)
    model = Model(backbone)
    features, labels = [], []
    total_batches = dataset.get_dataset_size()
    for index, item in enumerate(dataset.create_dict_iterator(output_numpy=True), start=1):
        image = Tensor(item["image"], ms.float32)
        feature_map = model.predict(image).asnumpy()
        pooled = feature_map.mean(axis=(2, 3)).astype(np.float32)
        features.append(pooled)
        labels.append(item["label"].astype(np.int32))
        print(f"Extract {split_name} features: batch {index}/{total_batches}", flush=True)

    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    np.savez_compressed(feature_file, features=feature_array, labels=label_array)
    print(f"Saved {split_name} features to: {feature_file}")
    return feature_array, label_array


def create_feature_dataset(features, labels, batch_size, shuffle):
    dataset = ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)


def train_and_eval_head(train_features, train_labels, test_features, test_labels, args, output_dir):
    head = LinearGarbageHead(in_channels=train_features.shape[1], num_classes=len(np.unique(train_labels)))
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(
        params=head.trainable_params(),
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model = Model(head, loss_fn=loss, optimizer=optimizer, metrics={"acc"})

    train_dataset = create_feature_dataset(train_features, train_labels, args.head_batch_size, shuffle=True)
    test_dataset = create_feature_dataset(test_features, test_labels, args.head_batch_size, shuffle=False)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=5)
    callbacks = [
        EpochSummary(),
        ModelCheckpoint(prefix="task2_head", directory=str(ckpt_dir), config=ckpt_config),
    ]

    print("Start training custom classification head")
    model.train(args.epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
    metrics = model.eval(test_dataset, dataset_sink_mode=False)
    print(f"Task2 eval result: {metrics}")
    return metrics


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    data_dir = resolve_project_path(args.data_dir)
    pretrain_ckpt = resolve_project_path(args.pretrain_ckpt)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    class_names = list_classes(train_dir)

    backbone = load_frozen_backbone(pretrain_ckpt)
    train_features, train_labels = extract_or_load_features(
        backbone, "train", train_dir, output_dir, args.image_batch_size, args.force_extract
    )
    test_features, test_labels = extract_or_load_features(
        backbone, "test", test_dir, output_dir, args.image_batch_size, args.force_extract
    )
    metrics = train_and_eval_head(train_features, train_labels, test_features, test_labels, args, output_dir)

    summary = {
        "task": "task2_custom_finetune",
        "num_classes": len(class_names),
        "classes": class_names,
        "train_samples": int(train_features.shape[0]),
        "test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()
