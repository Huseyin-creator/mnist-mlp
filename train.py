import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_mnist, batches, normalize
from model import MLP

EXPERIMENTS = {
    "deney1_temel": dict(hidden=[256, 128], dropout=0.0, optimizer="adam", lr=1e-3, augment=False),
    "deney2_genis": dict(hidden=[512, 256], dropout=0.0, optimizer="adam", lr=1e-3, augment=False),
    "deney3_derin": dict(hidden=[512, 256, 128], dropout=0.0, optimizer="adam", lr=1e-3, augment=False),
    "deney4_sgd": dict(hidden=[256, 128], dropout=0.0, optimizer="sgd", lr=1e-2, augment=False),
    "deney5_dropout": dict(hidden=[512, 256], dropout=0.2, optimizer="adam", lr=1e-3, augment=False),
    "deney6_veri_artirma": dict(hidden=[512, 256], dropout=0.2, optimizer="adam", lr=1e-3, augment=True),
}

OUT = "sonuclar"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def augment(x):
    n = x.size(0)
    angle = (torch.rand(n, device=x.device) - 0.5) * 2 * math.radians(12)
    scale = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.2
    tx = (torch.rand(n, device=x.device) - 0.5) * 0.2
    ty = (torch.rand(n, device=x.device) - 0.5) * 0.2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, tx], 1),
        torch.stack([sin, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    bg = normalize(torch.tensor(0.0, device=x.device))
    return F.grid_sample(x - bg, grid, align_corners=False) + bg


def evaluate(model, x, y, criterion):
    model.eval()
    loss_sum, correct = 0.0, 0
    preds = []
    with torch.no_grad():
        for xb, yb in batches(x, y, 2000):
            out = model(xb)
            loss_sum += criterion(out, yb).item() * len(yb)
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            preds.append(p)
    return loss_sum / len(y), correct / len(y), torch.cat(preds)


def train_one(name, cfg, data, device, epochs, batch_size, seed):
    torch.manual_seed(seed)
    model = MLP(cfg["hidden"], cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    x_tr, y_tr = data["train"]
    x_val, y_val = data["val"]
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch = 0.0, 0
    os.makedirs(f"{OUT}/{name}", exist_ok=True)

    print(f"\n=== {name} === {cfg}")
    sync(device)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in batches(x_tr, y_tr, batch_size, shuffle=True):
            if cfg["augment"]:
                xb = augment(xb)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()

        tr_loss, tr_acc = loss_sum / len(y_tr), correct / len(y_tr)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, criterion)
        for k, v in zip(hist, (tr_loss, tr_acc, val_loss, val_acc)):
            hist[k].append(v)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), f"{OUT}/{name}/model.pt")
    sync(device)
    train_time = time.time() - t0

    plot_history(hist, name, best_epoch)
    result = {
        "config": cfg,
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "train_loss": hist["train_loss"][best_epoch - 1],
        "train_acc": hist["train_acc"][best_epoch - 1],
        "val_loss": hist["val_loss"][best_epoch - 1],
        "val_acc": best_val,
        "train_time_sec": round(train_time, 1),
        "history": hist,
    }
    print(f"en iyi epoch {best_epoch}, val acc {best_val:.4f}, sure {train_time:.1f} sn")
    return result


def plot_history(hist, name, best_epoch):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["train_loss"], label="train")
    a1.plot(ep, hist["val_loss"], label="validation")
    a1.set_title(f"{name} - loss")
    a2.plot(ep, hist["train_acc"], label="train")
    a2.plot(ep, hist["val_acc"], label="validation")
    a2.set_title(f"{name} - accuracy")
    for a in (a1, a2):
        a.axvline(best_epoch, color="gray", ls="--", lw=0.8)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}/grafik.png", dpi=120)
    plt.close(fig)


def plot_comparison(results):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        ep = range(1, len(r["history"]["val_loss"]) + 1)
        a1.plot(ep, r["history"]["val_loss"], label=name)
        a2.plot(ep, r["history"]["val_acc"], label=name)
    a1.set_title("validation loss")
    a2.set_title("validation accuracy")
    a2.set_ylim(0.96, 1.0)
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/karsilastirma.png", dpi=120)
    plt.close(fig)


def plot_confusion(preds, labels, name):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("tahmin")
    ax.set_ylabel("gercek")
    ax.set_title(f"confusion matrix (test) - {name}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=120)
    plt.close(fig)


def write_table(results, best):
    lines = [
        "| Deney | Gizli katmanlar | Dropout | Optimizer | LR | Veri artırma | Parametre | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Test acc | Süre (sn) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        c = r["config"]
        n = f"**{name}**" if name == best else name
        lines.append(
            f"| {n} | {'-'.join(map(str, c['hidden']))} | {c['dropout']} | {c['optimizer']} | {c['lr']} | "
            f"{'var' if c['augment'] else 'yok'} | {r['params']:,} | {r['best_epoch']} | {r['train_loss']:.4f} | "
            f"{r['val_loss']:.4f} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['train_time_sec']} |"
        )
    with open(f"{OUT}/sonuclar.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deney", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    data = load_mnist(device=device, seed=args.seed)
    for k, (x, _) in data.items():
        print(f"{k}: {len(x)}")

    os.makedirs(OUT, exist_ok=True)
    results = {}
    for name in args.deney:
        results[name] = train_one(name, EXPERIMENTS[name], data, device, args.epochs, args.batch_size, args.seed)

    best = max(results, key=lambda n: results[n]["val_acc"])
    print(f"\nvalidation'a gore en iyi model: {best}")

    criterion = nn.CrossEntropyLoss()
    x_te, y_te = data["test"]
    for name, r in results.items():
        model = MLP(r["config"]["hidden"], r["config"]["dropout"]).to(device)
        model.load_state_dict(torch.load(f"{OUT}/{name}/model.pt", map_location=device))
        r["test_loss"], r["test_acc"], preds = evaluate(model, x_te, y_te, criterion)
        if name == best:
            plot_confusion(preds.cpu().numpy(), y_te.cpu().numpy(), name)

    plot_comparison(results)
    write_table(results, best)

    summary = {
        "best": best,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                            "loss": "CrossEntropyLoss", "split": "50000 train / 10000 val / 10000 test"},
        "results": {n: {k: v for k, v in r.items() if k != "history"} for n, r in results.items()},
    }
    with open(f"{OUT}/ozet.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for name, r in results.items():
        with open(f"{OUT}/{name}/gecmis.json", "w") as f:
            json.dump(r["history"], f, indent=2)


if __name__ == "__main__":
    main()
