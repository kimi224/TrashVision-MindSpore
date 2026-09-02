"""Task 4 Phase 4: SE attention + extreme ensemble architectures.

Key ideas:
  - SE (Squeeze-and-Excitation) feature reweighting before multi-branch classification
  - 7-branch very wide ensemble
  - Combined SE + very wide
"""

import json
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import Callback


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── heads ──────────────────────────────────────────────────────────────────

class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class SEMultiBranchHead(nn.Cell):
    """Squeeze-and-Excitation feature reweighting + multi-branch MLP ensemble.

    SE learns a per-dimension importance weight, allowing the model to focus
    on the most discriminative features before branching.
    """
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=4, dropout=0.1, se_reduction=16):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # SE: feature-wise attention
        se_hidden = in_channels // se_reduction
        self.se_squeeze = nn.Dense(in_channels, se_hidden)
        self.se_excite = nn.Dense(se_hidden, in_channels)
        self.se_relu = nn.ReLU()
        self.se_sigmoid = nn.Sigmoid()
        # Branches
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        # SE reweighting
        w = self.se_sigmoid(self.se_excite(self.se_relu(self.se_squeeze(x))))
        x = x * w
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class SEVeryWideMultiBranchHead(nn.Cell):
    """SE feature attention + deep 3-layer MLP branches in a wide ensemble."""
    def __init__(self, in_channels=1280, hidden=1024, num_classes=26, branches=5, dropout=0.12):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # SE module
        se_hidden = in_channels // 16
        self.se_squeeze = nn.Dense(in_channels, se_hidden)
        self.se_excite = nn.Dense(se_hidden, in_channels)
        self.se_relu = nn.ReLU()
        self.se_sigmoid = nn.Sigmoid()
        # Deep branches
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 2, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        w = self.se_sigmoid(self.se_excite(self.se_relu(self.se_squeeze(x))))
        x = x * w
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class MegaEnsembleHead(nn.Cell):
    """7-branch ensemble with SE attention and deep branches."""
    def __init__(self, in_channels=1280, hidden=768, num_classes=26, branches=7, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # SE module
        se_hidden = in_channels // 16
        self.se_squeeze = nn.Dense(in_channels, se_hidden)
        self.se_excite = nn.Dense(se_hidden, in_channels)
        self.se_relu = nn.ReLU()
        self.se_sigmoid = nn.Sigmoid()
        # 7 branches
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 2, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        w = self.se_sigmoid(self.se_excite(self.se_relu(self.se_squeeze(x))))
        x = x * w
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class FeatureGatedMultiBranchHead(nn.Cell):
    """Learn a per-sample feature gate, then branch.

    Each sample gets a different feature emphasis, making the ensemble
    more adaptive to per-sample characteristics.
    """
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=5, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # Feature gate: learns which features to emphasize per sample
        self.gate = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // 8),
            nn.ReLU(),
            nn.Dense(in_channels // 8, in_channels),
            nn.Sigmoid(),
        )
        # Branches operate on gated features
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 2, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        g = self.gate(x)
        x = x * g
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class DeepSEHead(nn.Cell):
    """Deep SE with richer feature interaction: SE at both input and bottleneck."""
    def __init__(self, in_channels=1280, bottleneck=256, num_classes=26, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # Input SE
        se_hidden = in_channels // 16
        self.se1_squeeze = nn.Dense(in_channels, se_hidden)
        self.se1_excite = nn.Dense(se_hidden, in_channels)
        # Shared feature transformer (like a "neck")
        self.shared = nn.SequentialCell(
            nn.Dense(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # Multiple classifiers on the shared representation
        self.classifiers = nn.CellList([
            nn.SequentialCell(
                nn.Dropout(p=dropout),
                nn.Dense(256, num_classes),
            ) for _ in range(5)
        ])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.input_bn(x)
        w = self.sigmoid(self.se1_excite(self.relu(self.se1_squeeze(x))))
        x = x * w
        shared = self.shared(x)
        out = self.classifiers[0](shared)
        for i in range(1, len(self.classifiers)):
            out = out + self.classifiers[i](shared)
        return out / len(self.classifiers)


# ─── training ───────────────────────────────────────────────────────────────

class EpochLogger(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.losses = []
        self.epoch_losses = []

    def epoch_begin(self, run_context):
        self.losses = []

    def step_end(self, run_context):
        loss = run_context.original_args().net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, Tensor):
            loss = float(np.mean(loss.asnumpy()))
        self.losses.append(float(loss))

    def epoch_end(self, run_context):
        avg = float(np.mean(self.losses)) if self.losses else float("nan")
        self.epoch_losses.append(avg)
        print(f"{self.name}: epoch {run_context.original_args().cur_epoch_num}, loss={avg:.6f}")


def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (train["features"].astype(np.float32), train["labels"].astype(np.int32),
            test["features"].astype(np.float32), test["labels"].astype(np.int32))


def make_dataset(features, labels, batch_size, shuffle):
    return ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    ).batch(batch_size, drop_remainder=False)


def count_params(params):
    return int(sum(np.prod(tuple(p.shape)) for p in params))


def train_one(name, net, train_f, train_l, test_f, test_l, seed):
    ms.set_seed(seed)
    train_ds = make_dataset(train_f, train_l, 64, True)
    test_ds = make_dataset(test_f, test_l, 64, False)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"acc"})
    logger = EpochLogger(name)
    start = time.time()
    model.train(30, train_ds, callbacks=[logger], dataset_sink_mode=False)
    seconds = time.time() - start
    acc = float(model.eval(test_ds, dataset_sink_mode=False)["acc"])
    result = {"name": name, "accuracy": acc, "train_seconds": seconds,
              "params": count_params(net.trainable_params()),
              "epoch_losses": logger.epoch_losses}
    print(f"{name}: acc={acc:.4f}, time={seconds:.1f}s")
    return result, net


def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    feature_dir = PROJECT_ROOT / "runs/task2_custom"
    output_dir = PROJECT_ROOT / "runs/task4_improvement"

    train_f, train_l, test_f, test_l = load_features(feature_dir)
    in_channels = int(train_f.shape[1])
    num_classes = int(len(np.unique(train_l)))

    results = {}

    # Baseline
    r, _ = train_one("baseline_linear", LinearHead(in_channels, num_classes),
                     train_f, train_l, test_f, test_l, 11)
    results["baseline"] = r

    # SE + MultiBranch (4 branches, hidden=512)
    r, se_net = train_one("se_multi4_512", SEMultiBranchHead(in_channels, 512, num_classes, 4, 0.1),
                          train_f, train_l, test_f, test_l, 11)
    results["se_multi4_512"] = r

    # SE + Very Wide (5 branches, hidden=1024, deep)
    r, se_vwide = train_one("se_vwide5_1024", SEVeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12),
                            train_f, train_l, test_f, test_l, 11)
    results["se_vwide5_1024"] = r

    # Mega Ensemble (7 branches, hidden=768)
    r, mega = train_one("mega7_768", MegaEnsembleHead(in_channels, 768, num_classes, 7, 0.15),
                        train_f, train_l, test_f, test_l, 11)
    results["mega7_768"] = r

    # Feature Gated (learned per-sample gate)
    r, fg = train_one("feature_gated5_512", FeatureGatedMultiBranchHead(in_channels, 512, num_classes, 5, 0.1),
                      train_f, train_l, test_f, test_l, 11)
    results["feature_gated5_512"] = r

    # Deep SE (shared representation + multiple classifiers)
    r, dse = train_one("deep_se_shared", DeepSEHead(in_channels, 256, num_classes, 0.1),
                       train_f, train_l, test_f, test_l, 11)
    results["deep_se_shared"] = r

    best = max(results.values(), key=lambda r: r["accuracy"])
    print("\n" + "=" * 60)
    print(f"Best: {best['name']} = {best['accuracy']:.4f}")
    print(f"Baseline: {results['baseline']['accuracy']:.4f}")
    print(f"Delta: {best['accuracy'] - results['baseline']['accuracy']:.4f}")

    for r in sorted(results.values(), key=lambda r: r["accuracy"], reverse=True):
        print(f"  {r['name']}: {r['accuracy']:.4f}")

    # Save
    out = output_dir / "summary_phase4.json"
    out.write_text(json.dumps({k: v["accuracy"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
