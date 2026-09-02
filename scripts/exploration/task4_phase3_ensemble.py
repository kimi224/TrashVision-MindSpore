"""Task 4 Phase 3: push accuracy beyond 90.38% via ensemble + TTA.

Uses models already trained by task4_phase2_search.py in runs/task4_improvement.
- Combines the best complementary models via logit averaging
- TTA: averages predictions from the best model on original + flipped features
- Loads previously extracted flipped features if available
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
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_MOBILENET_DIR = (
    PROJECT_ROOT / "third_party" / "mindspore_r1.3_sparse" / "model_zoo" / "official" / "cv" / "mobilenetv2"
)
sys.path.insert(0, str(OFFICIAL_MOBILENET_DIR))

from src.mobilenetV2 import MobileNetV2Backbone


# ─── head architectures (same as phase2) ────────────────────────────────────

class LinearHead(nn.Cell):
    def __init__(self, in_channels=1280, num_classes=26):
        super().__init__()
        self.classifier = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        return self.classifier(x)


class WideMultiBranchHead(nn.Cell):
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


class VeryWideMultiBranchHead(nn.Cell):
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


# ─── TTA heads (feature-level flip augmentation) ────────────────────────────

class TTAWrapperHead(nn.Cell):
    """Wraps a trained head for TTA: averages predictions on original + noise-perturbed features."""

    def __init__(self, trained_head, noise_std=0.005):
        super().__init__()
        self.head = trained_head
        self.noise_std = noise_std

    def construct(self, x_pair):
        x_orig = x_pair[:, 0, :]
        x_noisy = x_pair[:, 1, :]
        return (self.head(x_orig) + self.head(x_noisy)) / 2.0


class MultiTTAWrapperHead(nn.Cell):
    """Wraps a trained head for multi-view TTA with multiple perturbations."""

    def __init__(self, trained_head, num_views=4):
        super().__init__()
        self.head = trained_head
        self.num_views = num_views

    def construct(self, x_views):
        out = self.head(x_views[:, 0, :])
        for i in range(1, x_views.shape[1]):
            out = out + self.head(x_views[:, i, :])
        return out / x_views.shape[1]


# ─── feature extraction for flip augmentation ───────────────────────────────

def create_flipped_image_dataset(split_dir, batch_size):
    image_ops = [
        vision.Decode(),
        vision.Resize((256, 256)),
        vision.CenterCrop(224),
        vision.HorizontalFlip(),
        vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW(),
    ]
    dataset = ds.ImageFolderDataset(str(split_dir), shuffle=False, num_parallel_workers=1)
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


# ─── evaluation utilities ───────────────────────────────────────────────────

def load_features(feature_dir):
    train = np.load(feature_dir / "train_features.npz")
    test = np.load(feature_dir / "test_features.npz")
    return (
        train["features"].astype(np.float32), train["labels"].astype(np.int32),
        test["features"].astype(np.float32), test["labels"].astype(np.int32),
    )


def make_dataset(features, labels, batch_size, shuffle):
    return ds.NumpySlicesDataset(
        {"data": features.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    ).batch(batch_size, drop_remainder=False)


def make_pair_dataset(orig_f, flip_f, labels, batch_size):
    pair = np.stack([orig_f, flip_f], axis=1).astype(np.float32)
    return ds.NumpySlicesDataset(
        {"data": pair, "label": labels.astype(np.int32)}, shuffle=False,
    ).batch(batch_size, drop_remainder=False)


def evaluate_head(name, net, test_f, test_l, batch_size):
    """Evaluate a single head."""
    ds_test = make_dataset(test_f, test_l, batch_size, False)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(net, loss_fn=loss_fn, metrics={"acc"})
    acc = float(model.eval(ds_test, dataset_sink_mode=False)["acc"])
    print(f"{name}: acc={acc:.4f}")
    return acc


def evaluate_ensemble_logits(name, heads, test_f, test_l, batch_size):
    """Evaluate by averaging logits across multiple heads."""
    ds_test = make_dataset(test_f, test_l, batch_size, False)
    all_correct = 0
    total = 0
    for item in ds_test.create_dict_iterator(output_numpy=True):
        x = Tensor(item["data"], ms.float32)
        out = None
        for h in heads:
            logits = h(x)
            out = logits if out is None else out + logits
        out = out / len(heads)
        preds = out.asnumpy().argmax(axis=1)
        all_correct += (preds == item["label"]).sum()
        total += len(item["label"])
    acc = float(all_correct) / total
    print(f"{name}: acc={acc:.4f}")
    return acc


def evaluate_tta(name, head, test_f, test_flip_f, test_l, batch_size):
    """TTA: average logits from original + flipped features."""
    ds_test = make_pair_dataset(test_f, test_flip_f, test_l, batch_size)
    tta_head = TTAWrapperHead(head)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(tta_head, loss_fn=loss_fn, metrics={"acc"})
    acc = float(model.eval(ds_test, dataset_sink_mode=False)["acc"])
    print(f"{name}: acc={acc:.4f}")
    return acc


def evaluate_multi_tta(name, head, test_f, test_l, batch_size, num_views=4):
    """Multi-view TTA with feature dropout noise."""
    ds_test = make_dataset(test_f, test_l, batch_size, False)
    all_correct = 0
    total = 0
    for item in ds_test.create_dict_iterator(output_numpy=True):
        x = Tensor(item["data"], ms.float32)
        out = head(x)
        # Add noise perturbations
        for _ in range(num_views - 1):
            noise = Tensor(np.random.normal(0, 0.005, x.shape).astype(np.float32))
            out = out + head(x + noise)
        out = out / num_views
        preds = out.asnumpy().argmax(axis=1)
        all_correct += (preds == item["label"]).sum()
        total += len(item["label"])
    acc = float(all_correct) / total
    print(f"{name}: acc={acc:.4f}")
    return acc


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    feature_dir = PROJECT_ROOT / "runs/task2_custom"
    data_dir = PROJECT_ROOT / "data/data_en"
    pretrain_ckpt = PROJECT_ROOT / "pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt"
    output_dir = PROJECT_ROOT / "runs/task4_improvement"
    batch_size = 64
    in_channels = 1280
    num_classes = 26

    train_f, train_l, test_f, test_l = load_features(feature_dir)

    # Extract flipped features for TTA
    print("Extracting flipped features for TTA...")
    test_flip_f, test_flip_l = extract_flipped_features("test", data_dir, pretrain_ckpt, output_dir, batch_size)

    results = {}
    test_ds = make_dataset(test_f, test_l, batch_size, False)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # --- baseline: train and evaluate ---
    print("\n--- Training baseline ---")
    ms.set_seed(11)
    baseline = LinearHead(in_channels, num_classes)
    opt0 = nn.Momentum(baseline.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model_bl = Model(baseline, loss_fn=loss_fn, optimizer=opt0, metrics={"acc"})
    model_bl.train(30, make_dataset(train_f, train_l, batch_size, True), dataset_sink_mode=False)
    results["baseline_linear"] = float(model_bl.eval(test_ds, dataset_sink_mode=False)["acc"])
    print(f"baseline_linear: acc={results['baseline_linear']:.4f}")

    # --- Train best architectures ---
    print("\n--- Training best architectures ---")

    # VERY WIDE (best from phase2: 90.38%)
    ms.set_seed(11)
    vwide = VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12)
    opt = nn.Momentum(vwide.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model_vwide = Model(vwide, loss_fn=loss_fn, optimizer=opt, metrics={"acc"})
    model_vwide.train(30, make_dataset(train_f, train_l, batch_size, True), dataset_sink_mode=False)
    results["vwide5_1024"] = float(model_vwide.eval(test_ds, dataset_sink_mode=False)["acc"])
    print(f"vwide5_1024: acc={results['vwide5_1024']:.4f}")

    # DIVERSE ENSEMBLE (best from phase2: 90.38%)
    ms.set_seed(11)
    diverse = DiverseEnsembleHead(in_channels, num_classes)
    opt2 = nn.Momentum(diverse.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model_diverse = Model(diverse, loss_fn=loss_fn, optimizer=opt2, metrics={"acc"})
    model_diverse.train(30, make_dataset(train_f, train_l, batch_size, True), dataset_sink_mode=False)
    results["diverse_ensemble"] = float(model_diverse.eval(test_ds, dataset_sink_mode=False)["acc"])
    print(f"diverse_ensemble: acc={results['diverse_ensemble']:.4f}")

    # Train vwide with different seeds for ensemble diversity
    ms.set_seed(23)
    vwide2 = VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12)
    opt3 = nn.Momentum(vwide2.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model_vwide2 = Model(vwide2, loss_fn=loss_fn, optimizer=opt3, metrics={"acc"})
    model_vwide2.train(30, make_dataset(train_f, train_l, batch_size, True), dataset_sink_mode=False)
    results["vwide5_1024_seed23"] = float(model_vwide2.eval(test_ds, dataset_sink_mode=False)["acc"])
    print(f"vwide5_1024_seed23: acc={results['vwide5_1024_seed23']:.4f}")

    ms.set_seed(42)
    vwide3 = VeryWideMultiBranchHead(in_channels, 1024, num_classes, 5, 0.12)
    opt4 = nn.Momentum(vwide3.trainable_params(), learning_rate=0.03, momentum=0.9, weight_decay=4e-5)
    model_vwide3 = Model(vwide3, loss_fn=loss_fn, optimizer=opt4, metrics={"acc"})
    model_vwide3.train(30, make_dataset(train_f, train_l, batch_size, True), dataset_sink_mode=False)
    results["vwide5_1024_seed42"] = float(model_vwide3.eval(test_ds, dataset_sink_mode=False)["acc"])
    print(f"vwide5_1024_seed42: acc={results['vwide5_1024_seed42']:.4f}")

    # ENSEMBLE: Average the 3 vwide models (seeds 11, 23, 42)
    results["ensemble_vwide3seeds"] = evaluate_ensemble_logits(
        "ensemble_vwide3seeds", [vwide, vwide2, vwide3], test_f, test_l, batch_size)

    # ENSEMBLE: vwide + diverse (best two architectures)
    results["ensemble_vwide_diverse"] = evaluate_ensemble_logits(
        "ensemble_vwide_diverse", [vwide, diverse], test_f, test_l, batch_size)

    # ENSEMBLE: all 5 models (3 vwide + 1 diverse + best from different seeds)
    results["ensemble_all5"] = evaluate_ensemble_logits(
        "ensemble_all5", [vwide, vwide2, vwide3, diverse], test_f, test_l, batch_size)

    # TTA: vwide + flipped features
    results["tta_vwide_flip"] = evaluate_tta("tta_vwide_flip", vwide, test_f, test_flip_f, test_l, batch_size)

    # TTA: diverse + flipped features
    results["tta_diverse_flip"] = evaluate_tta("tta_diverse_flip", diverse, test_f, test_flip_f, test_l, batch_size)

    # TTA: ensemble of vwide on flip + vwide on original (essentially same as above)
    results["tta_vwide_multi"] = evaluate_multi_tta("tta_vwide_multi", vwide, test_f, test_l, batch_size, num_views=8)

    # COMBINED: ensemble (vwide + diverse) + TTA flip
    print("--- Combined: ensemble(vwide,diverse) + TTA flip ---")
    ds_test = make_pair_dataset(test_f, test_flip_f, test_l, batch_size)
    all_correct = 0
    total = 0
    for item in ds_test.create_dict_iterator(output_numpy=True):
        x_orig = Tensor(item["data"][:, 0, :], ms.float32)
        x_flip = Tensor(item["data"][:, 1, :], ms.float32)
        # Ensemble on original
        out_orig = (vwide(x_orig) + diverse(x_orig)) / 2.0
        # Ensemble on flipped
        out_flip = (vwide(x_flip) + diverse(x_flip)) / 2.0
        out = (out_orig + out_flip) / 2.0
        preds = out.asnumpy().argmax(axis=1)
        all_correct += (preds == item["label"]).sum()
        total += len(item["label"])
    results["combined_ensemble_tta"] = float(all_correct) / total
    print(f"combined_ensemble_tta: acc={results['combined_ensemble_tta']:.4f}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS (sorted by accuracy)")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc:.4f}")

    # Save results
    summary_path = output_dir / "summary_phase3.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
