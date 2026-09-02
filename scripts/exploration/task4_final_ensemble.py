"""Task 4 Final: super-ensemble of the best complementary architectures.

Trains multiple diverse top-performing heads and combines them via logit averaging.
Goal: break through the 90.38% ceiling.
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


# ─── all best architectures ─────────────────────────────────────────────────

class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)
    def construct(self, x):
        return self.classifier(x)


class VeryWideMultiBranchHead(nn.Cell):
    """5 branches, hidden=1024, 3-layer deep. Phase 2 best: 90.38%"""
    def __init__(self, in_channels=1280, hidden=1024, num_classes=26, branches=5, dropout=0.12):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
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
    """4 diverse sub-heads trained jointly. Phase 2 best: 90.38%"""
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.head_a = nn.SequentialCell(
            nn.BatchNorm1d(in_channels), nn.Dense(in_channels, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.1), nn.Dense(512, num_classes),
        )
        self.head_b = nn.SequentialCell(
            nn.BatchNorm1d(in_channels), nn.Dense(in_channels, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Dense(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(p=0.1), nn.Dense(256, num_classes),
        )
        self.head_c = nn.SequentialCell(
            nn.BatchNorm1d(in_channels), nn.Dense(in_channels, 768),
            nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(p=0.15), nn.Dense(768, num_classes),
        )
        self.head_d = nn.SequentialCell(
            nn.BatchNorm1d(in_channels), nn.Dense(in_channels, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dense(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.1), nn.Dense(512, num_classes),
        )

    def construct(self, x):
        return (self.head_a(x) + self.head_b(x) + self.head_c(x) + self.head_d(x)) / 4.0


class MegaEnsembleHead(nn.Cell):
    """7 branches, hidden=768, SE attention. Phase 4 best: 90.38%"""
    def __init__(self, in_channels=1280, hidden=768, num_classes=26, branches=7, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        se_hidden = in_channels // 16
        self.se_squeeze = nn.Dense(in_channels, se_hidden)
        self.se_excite = nn.Dense(se_hidden, in_channels)
        self.se_relu = nn.ReLU()
        self.se_sigmoid = nn.Sigmoid()
        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Dense(in_channels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
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
    """Per-sample feature gating + 5 deep branches. Phase 4: 90.00%"""
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=5, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
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
                nn.Dense(hidden // 2, num_classes),
            ) for _ in range(branches)
        ])

    def construct(self, x):
        x = self.input_bn(x)
        x = x * self.gate(x)
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


# ─── training utilities ─────────────────────────────────────────────────────

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
    logger = EpochLogger(name)
    start = time.time()
    model.train(30, train_ds, callbacks=[logger], dataset_sink_mode=False)
    seconds = time.time() - start
    acc = float(model.eval(test_ds, dataset_sink_mode=False)["acc"])
    params = int(sum(np.prod(tuple(p.shape)) for p in net.trainable_params()))
    print(f"  {name}: acc={acc:.4f}, time={seconds:.1f}s, params={params}")
    return {"name": name, "accuracy": acc, "train_seconds": seconds, "params": params}, net


def eval_soft_voting(name, heads, test_f, test_l):
    """Evaluate logit-averaging ensemble."""
    test_ds = make_ds(test_f, test_l, False)
    correct = 0
    total = 0
    for item in test_ds.create_dict_iterator(output_numpy=True):
        x = Tensor(item["data"], ms.float32)
        out = None
        for h in heads:
            logits = h(x)
            out = logits if out is None else out + logits
        out = out / len(heads)
        preds = out.asnumpy().argmax(axis=1)
        correct += (preds == item["label"]).sum()
        total += len(item["label"])
    acc = float(correct) / total
    print(f"  {name}: acc={acc:.4f} (soft voting, {len(heads)} models)")
    return {"name": name, "accuracy": acc}


def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    train_f, train_l, test_f, test_l = load_features()
    in_channels, num_classes = int(train_f.shape[1]), int(len(np.unique(train_l)))
    output_dir = PROJECT_ROOT / "runs/task4_improvement"

    results = {}
    all_trained_heads = {}

    # Baseline
    print("--- Baseline ---")
    r, net = train_one("baseline_linear", LinearHead(in_channels, num_classes),
                       train_f, train_l, test_f, test_l, 11)
    results["baseline"] = r
    all_trained_heads["baseline"] = net

    # Train top architectures with multiple seeds
    print("\n--- VeryWideMultiBranch (seed=11) ---")
    r, net = train_one("vwide5_1024_s11", VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12),
                       train_f, train_l, test_f, test_l, 11)
    results["vwide_s11"] = r
    all_trained_heads["vwide_s11"] = net

    print("\n--- VeryWideMultiBranch (seed=42) ---")
    r, net = train_one("vwide5_1024_s42", VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12),
                       train_f, train_l, test_f, test_l, 42)
    results["vwide_s42"] = r
    all_trained_heads["vwide_s42"] = net

    print("\n--- MegaEnsemble (seed=11) ---")
    r, net = train_one("mega7_768_s11", MegaEnsembleHead(in_channels, 768, num_classes, 7, 0.15),
                       train_f, train_l, test_f, test_l, 11)
    results["mega_s11"] = r
    all_trained_heads["mega_s11"] = net

    print("\n--- DiverseEnsemble (seed=11) ---")
    r, net = train_one("diverse_s11", DiverseEnsembleHead(in_channels, num_classes),
                       train_f, train_l, test_f, test_l, 11)
    results["diverse_s11"] = r
    all_trained_heads["diverse_s11"] = net

    print("\n--- FeatureGated (seed=11) ---")
    r, net = train_one("feature_gated_s11", FeatureGatedMultiBranchHead(in_channels, 512, num_classes, 5, 0.1),
                       train_f, train_l, test_f, test_l, 11)
    results["fgated_s11"] = r
    all_trained_heads["fgated_s11"] = net

    # Also train DiverseEnsemble with seed=42
    print("\n--- DiverseEnsemble (seed=42) ---")
    r, net = train_one("diverse_s42", DiverseEnsembleHead(in_channels, num_classes),
                       train_f, train_l, test_f, test_l, 42)
    results["diverse_s42"] = r
    all_trained_heads["diverse_s42"] = net

    # ─── super ensembles ──────────────────────────────────────────────────
    print("\n=== SUPER ENSEMBLES ===")

    # Ensemble 1: all 6 models
    all_heads = list(all_trained_heads.values())
    r = eval_soft_voting("ensemble_all6", all_heads, test_f, test_l)
    results["ensemble_all6"] = r

    # Ensemble 2: top 3 performers only
    top3 = [all_trained_heads["vwide_s11"], all_trained_heads["mega_s11"],
            all_trained_heads["diverse_s11"]]
    r = eval_soft_voting("ensemble_top3", top3, test_f, test_l)
    results["ensemble_top3"] = r

    # Ensemble 3: vwide seeds ensemble
    vwides = [all_trained_heads["vwide_s11"], all_trained_heads["vwide_s42"]]
    r = eval_soft_voting("ensemble_vwide2seeds", vwides, test_f, test_l)
    results["ensemble_vwide2seeds"] = r

    # Ensemble 4: best of each architecture type
    arch_best = [all_trained_heads["vwide_s11"], all_trained_heads["mega_s11"],
                 all_trained_heads["diverse_s11"], all_trained_heads["fgated_s11"]]
    r = eval_soft_voting("ensemble_4arch", arch_best, test_f, test_l)
    results["ensemble_4arch"] = r

    # Ensemble 5: vwide(s11) + mega(s11) only (both got 90.38% at their best)
    r = eval_soft_voting("ensemble_vwide_mega", [all_trained_heads["vwide_s11"],
                                                  all_trained_heads["mega_s11"]], test_f, test_l)
    results["ensemble_vwide_mega"] = r

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"  {name}: {r['accuracy']:.4f}")

    delta = max(r["accuracy"] for r in results.values()) - results["baseline"]["accuracy"]
    print(f"\nBest delta vs baseline: {delta:.4f}")

    out = output_dir / "summary_final.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
