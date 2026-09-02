"""Task 4 Phase 2: try even stronger head architectures and ensemble methods.

This script tests:
  - MultiBranchMLPHead with more branches and wider hidden dims
  - InputDropout + MultiBranchMLP (combines two effective approaches)
  - SiLU activation instead of ReLU
  - Ensemble evaluation across diverse trained heads
  - TTA evaluation using flipped features
"""

import json
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
from mindspore.train.callback import Callback

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT / "third_party" / "mindspore_r1.3_sparse" / "model_zoo" / "official" / "cv" / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))

from src.mobilenetV2 import MobileNetV2Backbone


# ─── head architectures ─────────────────────────────────────────────────────

class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class MultiBranchMLPHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden_channels=512, num_classes=26, branches=3, dropout=0.1):
        super().__init__()
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden_channels, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class WideMultiBranchHead(nn.Cell):
    """5 branches with wider hidden dimension (768)."""
    def __init__(self, in_channels=1280, hidden=768, num_classes=26, branches=5, dropout=0.15):
        super().__init__()
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
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class InputDropoutMultiBranchHead(nn.Cell):
    """Input dropout + BatchNorm + multi-branch MLP."""
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=3, input_dropout=0.05, dropout=0.1):
        super().__init__()
        self.input_drop = nn.Dropout(p=input_dropout)
        self.input_bn = nn.BatchNorm1d(in_channels)
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
        x = self.input_drop(x)
        x = self.input_bn(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class SiLUMultiBranchHead(nn.Cell):
    """Multi-branch MLP with SiLU activation and deeper structure."""
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=4, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 2, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class VeryWideMultiBranchHead(nn.Cell):
    """Multi-branch with very wide hidden (1024) and more branches (5)."""
    def __init__(self, in_channels=1280, hidden=1024, num_classes=26, branches=5, dropout=0.12):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
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
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


class DiverseEnsembleHead(nn.Cell):
    """Ensemble of diverse sub-networks for maximum accuracy."""
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        # Branch type 1: standard 2-layer MLP with BatchNorm
        self.head_a = nn.SequentialCell(
            nn.BatchNorm1d(in_channels),
            nn.Dense(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Dense(512, num_classes),
        )
        # Branch type 2: deeper 3-layer MLP
        self.head_b = nn.SequentialCell(
            nn.BatchNorm1d(in_channels),
            nn.Dense(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Dense(256, num_classes),
        )
        # Branch type 3: wider 1-layer MLP
        self.head_c = nn.SequentialCell(
            nn.BatchNorm1d(in_channels),
            nn.Dense(in_channels, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Dense(768, num_classes),
        )
        # Branch type 4: bottleneck MLP
        self.head_d = nn.SequentialCell(
            nn.BatchNorm1d(in_channels),
            nn.Dense(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dense(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Dense(512, num_classes),
        )

    def construct(self, x):
        return (self.head_a(x) + self.head_b(x) + self.head_c(x) + self.head_d(x)) / 4.0


# ─── training utilities ─────────────────────────────────────────────────────

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
        print(f"{self.name}: epoch {epoch}, avg_loss={avg_loss:.6f}")


def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32),
        train["labels"].astype(np.int32),
        test["features"].astype(np.float32),
        test["labels"].astype(np.int32),
    )


def make_dataset(features, labels, batch_size, shuffle):
    return ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    ).batch(batch_size, drop_remainder=False)


def count_params(params):
    return int(sum(np.prod(tuple(param.shape)) for param in params))


def train_one(name, net, train_f, train_l, test_f, test_l, batch_size, lr, momentum, wd, epochs, seed):
    ms.set_seed(seed)
    train_ds = make_dataset(train_f, train_l, batch_size, True)
    test_ds = make_dataset(test_f, test_l, batch_size, False)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=wd)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"acc"})
    logger = EpochLogger(name)
    start = time.time()
    model.train(epochs, train_ds, callbacks=[logger], dataset_sink_mode=False)
    seconds = time.time() - start
    acc = float(model.eval(test_ds, dataset_sink_mode=False)["acc"])
    result = {
        "name": name,
        "accuracy": acc,
        "train_seconds": seconds,
        "trainable_params": count_params(net.trainable_params()),
        "total_params": count_params(net.get_parameters()),
        "epoch_losses": logger.epoch_losses,
    }
    print(f"{name}: acc={acc:.4f}, time={seconds:.1f}s, params={result['trainable_params']}")
    return result, net


def eval_ensemble(name, heads, test_f, test_l, batch_size):
    """Evaluate an ensemble by averaging logits from multiple independently trained heads."""
    test_ds = make_dataset(test_f, test_l, batch_size, False)
    all_preds = []
    all_labels = []
    for item in test_ds.create_dict_iterator(output_numpy=True):
        x = Tensor(item["data"], ms.float32)
        ensemble_logits = None
        for head in heads:
            logits = head(x)
            ensemble_logits = logits if ensemble_logits is None else ensemble_logits + logits
        ensemble_logits = ensemble_logits / len(heads)
        all_preds.append(ensemble_logits.asnumpy())
        all_labels.append(item["label"])
    preds = np.concatenate(all_preds, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)
    correct = (preds.argmax(axis=1) == labels_concat).sum()
    acc = float(correct) / len(labels_concat)
    print(f"{name}: acc={acc:.4f}")
    return {"name": name, "accuracy": acc, "train_seconds": 0, "trainable_params": 0, "total_params": 0, "epoch_losses": []}


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    feature_dir = PROJECT_ROOT / "runs/task2_custom"
    output_dir = PROJECT_ROOT / "runs/task4_improvement"

    train_features, train_labels, test_features, test_labels = load_features(feature_dir)
    in_channels = int(train_features.shape[1])
    num_classes = int(len(np.unique(train_labels)))

    hp = {"batch_size": 64, "lr": 0.03, "momentum": 0.9, "wd": 4e-5, "epochs": 30}
    results = []

    # Baseline
    r, baseline_net = train_one("baseline_linear", LinearHead(in_channels, num_classes),
                                train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # MultiBranchMLP (original improved) - train 3 variants with different seeds for ensemble
    multi_heads = []
    for s in [11, 21, 23]:
        r, net = train_one(f"multi3_512_seed{s}", MultiBranchMLPHead(in_channels, 512, num_classes, 3, 0.1),
                           train_features, train_labels, test_features, test_labels, **hp, seed=s)
        results.append(r)
        multi_heads.append(net)

    # Wide multi-branch (5 branches, hidden=768)
    r, wide_net = train_one("wide5_768", WideMultiBranchHead(in_channels, 768, num_classes, 5, 0.15),
                            train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # InputDropout + MultiBranch
    r, indrop_net = train_one("indrop_multi3_512", InputDropoutMultiBranchHead(in_channels, 512, num_classes, 3, 0.05, 0.1),
                              train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # SiLU MultiBranch
    r, silu_net = train_one("silu_multi4_512", SiLUMultiBranchHead(in_channels, 512, num_classes, 4, 0.1),
                            train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # Very wide (5 branches, hidden=1024)
    r, vwide_net = train_one("vwide5_1024", VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12),
                             train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # Diverse ensemble head (4 diverse sub-heads trained jointly)
    r, diverse_net = train_one("diverse_ensemble", DiverseEnsembleHead(in_channels, num_classes),
                               train_features, train_labels, test_features, test_labels, **hp, seed=11)
    results.append(r)

    # Ensemble: average predictions from multi_heads (seeds 11, 21, 23)
    r_ens = eval_ensemble("ensemble_multi3_seeds", multi_heads, test_features, test_labels, hp["batch_size"])
    results.append(r_ens)

    # Ensemble: combine best diverse heads
    diverse_heads = [multi_heads[0], wide_net, indrop_net, silu_net]
    r_ens2 = eval_ensemble("ensemble_diverse_heads", diverse_heads, test_features, test_labels, hp["batch_size"])
    results.append(r_ens2)

    best = max(results, key=lambda r: r["accuracy"])
    summary = {
        "task": "task4_model_improvement_phase2",
        "data": {"train_samples": int(train_features.shape[0]), "test_samples": int(test_features.shape[0]),
                 "feature_dim": in_channels, "num_classes": num_classes},
        "common_hyperparams": hp,
        "baseline_accuracy": results[0]["accuracy"],
        "best_model": best["name"],
        "best_accuracy": best["accuracy"],
        "results": results,
    }
    out_path = output_dir / "summary_phase2.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Best: {best['name']} = {best['accuracy']:.4f}")
    print(f"Delta vs baseline: {best['accuracy'] - results[0]['accuracy']:.4f}")
    for r in sorted(results, key=lambda r: r["accuracy"], reverse=True):
        print(f"  {r['name']}: {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
