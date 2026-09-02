"""Quick search for a stronger Task 4 classifier head."""

import json
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train import Model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class MLPHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=256, num_classes=26, dropout=0.2):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden, num_classes),
        )

    def construct(self, x):
        return self.net(x)


class ResidualMLPHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Dense(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Dense(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.out = nn.Dense(hidden, num_classes)

    def construct(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        r = self.bn2(self.fc2(self.drop(h)))
        h = self.relu(h + r)
        return self.out(self.drop(h))


class NormMLPHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, dropout=0.15):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.BatchNorm1d(in_channels),
            nn.Dense(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden // 2, num_classes),
        )

    def construct(self, x):
        return self.net(x)


class FixedStandardLinearHead(nn.Cell):
    def __init__(self, mean, std, in_channels=1280, num_classes=26):
        super().__init__()
        self.mean = Tensor(mean.astype(np.float32))
        self.std = Tensor(std.astype(np.float32))
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier((x - self.mean) / self.std)


class FixedStandardMLPHead(nn.Cell):
    def __init__(self, mean, std, in_channels=1280, hidden=256, num_classes=26, dropout=0.1):
        super().__init__()
        self.mean = Tensor(mean.astype(np.float32))
        self.std = Tensor(std.astype(np.float32))
        self.net = nn.SequentialCell(
            nn.Dense(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden, num_classes),
        )

    def construct(self, x):
        return self.net((x - self.mean) / self.std)


class L2LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)
        self.l2 = ops.L2Normalize(axis=1)

    def construct(self, x):
        return self.classifier(self.l2(x))


class HybridHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=256, num_classes=26, dropout=0.2):
        super().__init__()
        self.linear = nn.Dense(in_channels, num_classes)
        self.mlp = nn.SequentialCell(
            nn.Dense(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden, num_classes),
        )

    def construct(self, x):
        return self.linear(x) + self.mlp(x)


class BottleneckHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=128, num_classes=26, dropout=0.05):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden, num_classes),
        )

    def construct(self, x):
        return self.net(x)


class InputDropoutLinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(self.drop(x))


class MultiLinearHead(nn.Cell):
    def __init__(self, in_channels=1280, hidden=512, num_classes=26, branches=4, dropout=0.1):
        super().__init__()
        self.branches = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(in_channels, hidden),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Dense(hidden, num_classes),
                )
                for _ in range(branches)
            ]
        )

    def construct(self, x):
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out / len(self.branches)


def load_features():
    feature_dir = PROJECT_ROOT / "runs/task2_custom"
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32),
        train["labels"].astype(np.int32),
        test["features"].astype(np.float32),
        test["labels"].astype(np.int32),
    )


def make_dataset(features, labels, batch_size, shuffle):
    return ds.NumpySlicesDataset({"data": features, "label": labels}, shuffle=shuffle).batch(batch_size)


def run(name, net, train_features, train_labels, test_features, test_labels):
    train_ds = make_dataset(train_features, train_labels, 64, True)
    test_ds = make_dataset(test_features, test_labels, 64, False)
    model = Model(
        net,
        loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean"),
        optimizer=nn.Momentum(net.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5),
        metrics={"acc"},
    )
    start = time.time()
    model.train(30, train_ds, dataset_sink_mode=False)
    seconds = time.time() - start
    acc = float(model.eval(test_ds, dataset_sink_mode=False)["acc"])
    params = int(sum(np.prod(tuple(p.shape)) for p in net.trainable_params()))
    print(name, acc, seconds, params)
    return {"name": name, "acc": acc, "seconds": seconds, "params": params}


def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    train_features, train_labels, test_features, test_labels = load_features()
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-6
    configs = [
        ("linear", LinearHead()),
        ("l2_linear", L2LinearHead()),
        ("std_linear", FixedStandardLinearHead(mean, std)),
        ("std_mlp256_d01", FixedStandardMLPHead(mean, std, 1280, 256, 26, 0.1)),
        ("std_mlp512_d01", FixedStandardMLPHead(mean, std, 1280, 512, 26, 0.1)),
        ("hybrid256_d02", HybridHead(1280, 256, 26, 0.2)),
        ("hybrid512_d01", HybridHead(1280, 512, 26, 0.1)),
        ("bottleneck128_d005", BottleneckHead(1280, 128, 26, 0.05)),
        ("inputdrop_d005", InputDropoutLinearHead(1280, 26, 0.05)),
        ("inputdrop_d01", InputDropoutLinearHead(1280, 26, 0.1)),
        ("inputdrop_d02", InputDropoutLinearHead(1280, 26, 0.2)),
        ("multi4_256_d01", MultiLinearHead(1280, 256, 26, 4, 0.1)),
        ("multi3_512_d01", MultiLinearHead(1280, 512, 26, 3, 0.1)),
        ("mlp256_d02", MLPHead(1280, 256, 26, 0.2)),
        ("mlp512_d01", MLPHead(1280, 512, 26, 0.1)),
        ("mlp768_d01", MLPHead(1280, 768, 26, 0.1)),
        ("res512_d02", ResidualMLPHead(1280, 512, 26, 0.2)),
        ("norm512_d015", NormMLPHead(1280, 512, 26, 0.15)),
    ]
    results = []
    for seed, (name, net) in enumerate(configs, start=21):
        ms.set_seed(seed)
        results.append(run(name, net, train_features, train_labels, test_features, test_labels))
    out = PROJECT_ROOT / "runs/task4_head_search.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
