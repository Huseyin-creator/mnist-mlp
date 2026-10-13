import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import normalize
from model import MLP

OUT = "sonuclar"


def load_model(name, cfg):
    model = MLP(cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(f"{OUT}/{name}/model.pt", map_location="cpu"))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rakamlar", default="el_yazisi/islenmis/rakamlar.npy")
    ap.add_argument("--gercek", required=True)
    args = ap.parse_args()

    with open(f"{OUT}/ozet.json") as f:
        summary = json.load(f)
    best = summary["best"]

    digits = np.load(args.rakamlar)
    x = normalize(torch.tensor(digits, dtype=torch.float32).unsqueeze(1))
    truth = [int(c) for c in args.gercek]
    assert len(truth) == len(digits), f"{len(digits)} rakam bulundu ama gercek numara {len(truth)} haneli"

    rows = []
    best_probs = None
    for name, r in summary["results"].items():
        model = load_model(name, r["config"])
        with torch.no_grad():
            probs = torch.softmax(model(x), 1).numpy()
        pred = probs.argmax(1)
        correct = int((pred == np.array(truth)).sum())
        rows.append((name, "".join(map(str, pred)), correct))
        if name == best:
            best_probs = probs

    lines = [f"Gercek numara: {args.gercek}", "",
             "| Deney | Tahmin | Doğru rakam | Doğruluk |", "|---|---|---|---|"]
    for name, pred, correct in rows:
        n = f"**{name}**" if name == best else name
        lines.append(f"| {n} | {pred} | {correct}/{len(truth)} | %{100 * correct / len(truth):.1f} |")
    lines += ["", f"En iyi model ({best}) icin rakam bazinda olasiliklar:", "",
              "| # | Gerçek | Tahmin | Güven | 2. tahmin |", "|---|---|---|---|---|"]
    for i, p in enumerate(best_probs):
        top = p.argsort()[::-1]
        lines.append(f"| {i + 1} | {truth[i]} | {top[0]} | %{100 * p[top[0]]:.1f} | {top[1]} (%{100 * p[top[1]]:.1f}) |")
    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/el_yazisi_sonuc.md", "w") as f:
        f.write(text + "\n")

    n = len(digits)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.8))
    for i in range(n):
        p = best_probs[i]
        ok = p.argmax() == truth[i]
        axes[0, i].imshow(digits[i], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{p.argmax()} (%{100 * p.max():.0f})", color="green" if ok else "red", fontsize=10)
        axes[1, i].bar(range(10), p, color=["green" if k == truth[i] else "gray" for k in range(10)])
        axes[1, i].set_xticks(range(10))
        axes[1, i].tick_params(labelsize=6)
        axes[1, i].set_ylim(0, 1)
        if i:
            axes[1, i].set_yticks([])
    pred_str = "".join(str(p.argmax()) for p in best_probs)
    fig.suptitle(f"{best} | gercek: {args.gercek}  tahmin: {pred_str}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/el_yazisi_tahmin.png", dpi=120)


if __name__ == "__main__":
    main()
