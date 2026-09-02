"""Task 4: improve the classifier head under identical hyperparameters.

This version adds stronger head architectures beyond the original MultiBranchMLPHead:
  - InputBNMultiBranchHead: input BatchNorm + deeper 3-layer MLP branches
  - ResidualMultiBranchHead: residual connections within each branch
  - MultiScaleBranchHead: branches with different widths for scale diversity
  - DeepBottleneckBranchHead: bottleneck-structured deep branches
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT / "third_party" / "mindspore_r1.3_sparse" / "model_zoo" / "official" / "cv" / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))


# ─── baseline head ───────────────────────────────────────────────────────────

class LinearHead(nn.Cell):
    """Baseline classifier: one linear layer from feature to class logits."""

    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


# ─── original improved head (from task4 doc) ─────────────────────────────────

class MultiBranchMLPHead(nn.Cell):
    """Original improved head: average logits from several 2-layer MLP classifiers."""

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


# ─── new improved heads ──────────────────────────────────────────────────────

class InputBNMultiBranchHead(nn.Cell):
    """Input BatchNorm + deeper 3-layer MLP branches with bottleneck structure.

    Architecture:
      Input → BatchNorm(1280)
      → 4 branches, each:
        Dense(1280, 512) → BN → ReLU → Dropout
        Dense(512, 256) → BN → ReLU → Dropout
        Dense(256, 26)
      → Average logits
    """

    def __init__(self, in_channels=1280, hidden1=512, hidden2=256, num_classes=26, branches=4, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branches = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(in_channels, hidden1),
                    nn.BatchNorm1d(hidden1),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(hidden1, hidden2),
                    nn.BatchNorm1d(hidden2),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(hidden2, num_classes),
                )
                for _ in range(branches)
            ]
        )

    def construct(self, x):
        x = self.input_bn(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class ResidualMultiBranchHead(nn.Cell):
    """Multi-branch MLP with residual skip-connections in each branch.

    Each branch has a residual block: h = ReLU(BN(fc2(ReLU(BN(fc1(x)))))) + fc1(x)

    Architecture:
      Input → BatchNorm(1280)
      → 3 branches, each (residual block):
        Dense(1280, 512) → BN → ReLU → Dropout
        Dense(512, 512) → BN → ReLU (residual add) → Dropout
        Dense(512, 26)
      → Average logits
    """

    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=3, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branch_fc1 = nn.CellList([nn.Dense(in_channels, hidden) for _ in range(branches)])
        self.branch_bn1 = nn.CellList([nn.BatchNorm1d(hidden) for _ in range(branches)])
        self.branch_fc2 = nn.CellList([nn.Dense(hidden, hidden) for _ in range(branches)])
        self.branch_bn2 = nn.CellList([nn.BatchNorm1d(hidden) for _ in range(branches)])
        self.branch_out = nn.CellList([nn.Dense(hidden, num_classes) for _ in range(branches)])
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.input_bn(x)
        out = None
        for i in range(len(self.branch_fc1)):
            h = self.relu(self.branch_bn1[i](self.branch_fc1[i](x)))
            h = self.drop(h)
            r = self.branch_bn2[i](self.branch_fc2[i](h))
            h = self.relu(h + r)
            h = self.drop(h)
            b = self.branch_out[i](h)
            out = b if out is None else out + b
        return out / len(self.branch_fc1)


class MultiScaleBranchHead(nn.Cell):
    """Multi-branch head where each branch has a different hidden width.

    Different widths allow each branch to capture patterns at different scales
    (wider = more capacity, narrower = more regularization).

    Architecture:
      Input → BatchNorm(1280)
      → Branch 0: Dense(1280, 256) → BN → ReLU → Dropout → Dense(256, 26)
      → Branch 1: Dense(1280, 384) → BN → ReLU → Dropout → Dense(384, 26)
      → Branch 2: Dense(1280, 512) → BN → ReLU → Dropout → Dense(512, 26)
      → Branch 3: Dense(1280, 768) → BN → ReLU → Dropout → Dense(768, 26)
      → Average logits
    """

    def __init__(self, in_channels=1280, branch_widths=(256, 384, 512, 768), num_classes=26, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branches = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(in_channels, w),
                    nn.BatchNorm1d(w),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(w, num_classes),
                )
                for w in branch_widths
            ]
        )

    def construct(self, x):
        x = self.input_bn(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class DeepBottleneckBranchHead(nn.Cell):
    """Deep bottleneck branches that first compress, then expand features.

    Architecture:
      Input → BatchNorm(1280)
      → 4 branches, each:
        Dense(1280, 256) → BN → ReLU
        Dense(256, 512) → BN → ReLU → Dropout
        Dense(512, 256) → BN → ReLU → Dropout
        Dense(256, 26)
      → Average logits
    """

    def __init__(self, in_channels=1280, bottleneck=256, expanded=512, num_classes=26, branches=4, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branches = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(in_channels, bottleneck),
                    nn.BatchNorm1d(bottleneck),
                    nn.ReLU(),
                    nn.Dense(bottleneck, expanded),
                    nn.BatchNorm1d(expanded),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(expanded, bottleneck),
                    nn.BatchNorm1d(bottleneck),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(bottleneck, num_classes),
                )
                for _ in range(branches)
            ]
        )

    def construct(self, x):
        x = self.input_bn(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


# ─── training utilities ──────────────────────────────────────────────────────

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
    parser.add_argument("--output_dir", default="runs/task4_improvement")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
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


def create_dataset(features, labels, batch_size, shuffle):
    dataset = ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
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


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ms.set_seed(args.seed)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    feature_dir = project_path(args.feature_dir)
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

    # 1. Baseline
    baseline = train_one_model(
        "baseline_linear",
        LinearHead(in_channels, num_classes),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    # 2. Original improved head (multi-branch MLP)
    original_improved = train_one_model(
        "improved_multibranch",
        MultiBranchMLPHead(in_channels, hidden_channels=512, num_classes=num_classes, branches=3, dropout=0.1),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    # 3. InputBN + deeper branches (4 branches, 1280→512→256→26)
    input_bn_deep = train_one_model(
        "improved_inputbn_deep",
        InputBNMultiBranchHead(in_channels, hidden1=512, hidden2=256, num_classes=num_classes, branches=4, dropout=0.15),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    # 4. Residual multi-branch (3 branches with skip connections)
    residual_multi = train_one_model(
        "improved_residual_multi",
        ResidualMultiBranchHead(in_channels, hidden=512, num_classes=num_classes, branches=3, dropout=0.1),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    # 5. Multi-scale branches (4 branches: 256, 384, 512, 768)
    multiscale = train_one_model(
        "improved_multiscale",
        MultiScaleBranchHead(in_channels, branch_widths=(256, 384, 512, 768), num_classes=num_classes, dropout=0.1),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    # 6. Deep bottleneck branches (4 branches, 1280→256→512→256→26)
    deep_bottleneck = train_one_model(
        "improved_deep_bottleneck",
        DeepBottleneckBranchHead(in_channels, bottleneck=256, expanded=512, num_classes=num_classes, branches=4, dropout=0.15),
        train_features, train_labels, test_features, test_labels, args, output_dir,
    )

    all_results = [baseline, original_improved, input_bn_deep, residual_multi, multiscale, deep_bottleneck]
    best = max(all_results, key=lambda r: r["accuracy"])

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
        "baseline_accuracy": baseline["accuracy"],
        "best_model": best["name"],
        "best_accuracy": best["accuracy"],
        "accuracy_delta_vs_baseline": best["accuracy"] - baseline["accuracy"],
        "results": all_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nBest model: {best['name']} with accuracy {best['accuracy']:.4f}")
    print(f"Delta vs baseline ({baseline['accuracy']:.4f}): {best['accuracy'] - baseline['accuracy']:.4f}")


if __name__ == "__main__":
    main()
