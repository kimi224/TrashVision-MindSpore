"""Task 4: Test the absolute best single-model head architecture.

Combines all winning ideas:
  - SE feature attention (learns which features matter)
  - Feature gating (per-sample adaptive feature emphasis)
  - Deep multi-branch MLP (ensemble effect)
  - Many wide branches (5-7)
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


class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)
    def construct(self, x):
        return self.classifier(x)


class UltimateHead(nn.Cell):
    """Combines SE attention, feature gating, and deep multi-branch ensemble.

    Architecture:
      Input(1280) → BatchNorm → SE → Gate → 7 deep branches → Average logits

    Each branch: Dense(1280,896) → BN → ReLU → Dropout → Dense(896,448) → BN → ReLU → Dropout → Dense(448,26)
    """
    def __init__(self, in_channels=1280, hidden=896, num_classes=26, branches=7, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        # SE: channel attention
        se_r = 16
        self.se_sq = nn.Dense(in_channels, in_channels // se_r)
        self.se_ex = nn.Dense(in_channels // se_r, in_channels)
        self.se_act = nn.ReLU()
        self.se_sig = nn.Sigmoid()
        # Feature gate
        self.gate = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // 8),
            nn.ReLU(),
            nn.Dense(in_channels // 8, in_channels),
            nn.Sigmoid(),
        )
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
        self.n_branches = float(branches)

    def construct(self, x):
        x = self.input_bn(x)
        # SE attention
        se_w = self.se_sig(self.se_ex(self.se_act(self.se_sq(x))))
        x = x * se_w
        # Feature gate
        g = self.gate(x)
        x = x * g
        # Ensemble branches
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / self.n_branches


class UltimateHeadV2(nn.Cell):
    """Alternative: wider with fewer but deeper branches."""
    def __init__(self, in_channels=1280, num_classes=26, dropout=0.12):
        super().__init__()
        hidden = 1024
        branches = 6
        self.input_bn = nn.BatchNorm1d(in_channels)
        se_r = 16
        self.se_sq = nn.Dense(in_channels, in_channels // se_r)
        self.se_ex = nn.Dense(in_channels // se_r, in_channels)
        self.se_act = nn.ReLU()
        self.se_sig = nn.Sigmoid()
        self.gate = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // 8), nn.ReLU(),
            nn.Dense(in_channels // 8, in_channels), nn.Sigmoid(),
        )
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 2, hidden // 4), nn.BatchNorm1d(hidden // 4), nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden // 4, num_classes),
            ) for _ in range(branches)
        ])
        self.n_branches = float(branches)

    def construct(self, x):
        x = self.input_bn(x)
        x = x * self.se_sig(self.se_ex(self.se_act(self.se_sq(x))))
        x = x * self.gate(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / self.n_branches


# ─── training ───────────────────────────────────────────────────────────────

class SimpleLogger(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.epoch_losses = []
        self.step_losses = []

    def epoch_begin(self, run_context):
        self.step_losses = []

    def step_end(self, run_context):
        loss = run_context.original_args().net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, Tensor):
            loss = float(np.mean(loss.asnumpy()))
        self.step_losses.append(float(loss))

    def epoch_end(self, run_context):
        avg = float(np.mean(self.step_losses)) if self.step_losses else float("nan")
        self.epoch_losses.append(avg)
        ep = run_context.original_args().cur_epoch_num
        if ep % 10 == 0 or ep <= 3:
            print(f"  {self.name}: epoch {ep}, loss={avg:.6f}")


def load_features():
    d = PROJECT_ROOT / "runs/task2_custom"
    train = np.load(d / "train_features.npz")
    test = np.load(d / "test_features.npz")
    return (train["features"].astype(np.float32), train["labels"].astype(np.int32),
            test["features"].astype(np.float32), test["labels"].astype(np.int32))


def make_ds(f, l, shuffle):
    return ds.NumpySlicesDataset(
        {"data": f.astype(np.float32), "label": l.astype(np.int32)}, shuffle=shuffle,
    ).batch(64, drop_remainder=False)


def train_one(name, net, train_f, train_l, test_f, test_l, seed):
    ms.set_seed(seed)
    train_ds = make_ds(train_f, train_l, True)
    test_ds = make_ds(test_f, test_l, False)
    model = Model(net,
        loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean"),
        optimizer=nn.Momentum(net.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5),
        metrics={"acc"},
    )
    logger = SimpleLogger(name)
    start = time.time()
    model.train(30, train_ds, callbacks=[logger], dataset_sink_mode=False)
    seconds = time.time() - start
    acc = float(model.eval(test_ds, dataset_sink_mode=False)["acc"])
    params = int(sum(np.prod(tuple(p.shape)) for p in net.trainable_params()))
    print(f"  {name}: acc={acc:.4f}, time={seconds:.1f}s, params={params}")
    return {"name": name, "accuracy": acc, "train_seconds": seconds, "params": params,
            "epoch_losses": logger.epoch_losses}, net


def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    train_f, train_l, test_f, test_l = load_features()
    in_channels, num_classes = int(train_f.shape[1]), int(len(np.unique(train_l)))
    output_dir = PROJECT_ROOT / "runs/task4_improvement"

    results = {}

    # Baseline
    print("=== Baseline ===")
    r, net = train_one("baseline_linear", LinearHead(in_channels, num_classes),
                       train_f, train_l, test_f, test_l, 11)
    results["baseline"] = r

    # Ultimate head V1 (7 branches, hidden=896)
    print("\n=== UltimateHead (7 branches, hidden=896) ===")
    r, net = train_one("ultimate_v1", UltimateHead(in_channels, 896, num_classes, 7, 0.15),
                       train_f, train_l, test_f, test_l, 11)
    results["ultimate_v1"] = r

    # Ultimate head V2 (6 branches, hidden=1024, 4-layer deep)
    print("\n=== UltimateHeadV2 (6 branches, hidden=1024, 4-layer) ===")
    r, net = train_one("ultimate_v2", UltimateHeadV2(in_channels, num_classes, 0.12),
                       train_f, train_l, test_f, test_l, 11)
    results["ultimate_v2"] = r

    # Ultimate V1 with seed=42
    print("\n=== UltimateHead (seed=42) ===")
    r, net = train_one("ultimate_v1_s42", UltimateHead(in_channels, 896, num_classes, 7, 0.15),
                       train_f, train_l, test_f, test_l, 42)
    results["ultimate_v1_s42"] = r

    best = max(results.values(), key=lambda r: r["accuracy"])
    print("\n" + "=" * 60)
    print(f"BEST: {best['name']} = {best['accuracy']:.4f}")
    print(f"Baseline: {results['baseline']['accuracy']:.4f}")
    print(f"Delta: {best['accuracy'] - results['baseline']['accuracy']:.4f}")
    for r in sorted(results.values(), key=lambda r: r["accuracy"], reverse=True):
        print(f"  {r['name']}: {r['accuracy']:.4f}")

    out = output_dir / "summary_best_head.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
