import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ink_map(gray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return cv2.subtract(bg, gray)


def find_digits(ink, min_thresh=25):
    t, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (ink > max(t, min_thresh)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(stats[i][:4]) for i in range(1, n) if stats[i][4] > 30]
    if not boxes:
        return mask, []
    heights = np.array([b[3] for b in boxes])
    boxes = [b for b in boxes if b[3] > 0.4 * np.median(heights)]
    boxes.sort(key=lambda b: b[0])

    merged = []
    for b in boxes:
        if merged:
            x, y, w, h = merged[-1]
            overlap = min(x + w, b[0] + b[2]) - max(x, b[0])
            if overlap > 0.5 * min(w, b[2]):
                nx, ny = min(x, b[0]), min(y, b[1])
                merged[-1] = (nx, ny, max(x + w, b[0] + b[2]) - nx, max(y + h, b[1] + b[3]) - ny)
                continue
        merged.append(b)
    return mask, merged


def to_mnist(ink, mask, box, pad=4):
    x, y, w, h = box
    y0, x0 = max(0, y - pad), max(0, x - pad)
    crop = ink[y0:y + h + pad, x0:x + w + pad].astype(np.float32)
    crop *= cv2.dilate(mask[y0:y + h + pad, x0:x + w + pad], np.ones((3, 3), np.uint8))
    crop = np.clip(crop / np.percentile(crop[crop > 0], 95), 0, 1)

    side = max(h, w)
    k = max(1, round(side / 28))
    crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    ys, xs = np.nonzero(crop > 0.05)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    s = 20.0 / max(ch, cw)
    nh, nw = max(1, round(ch * s)), max(1, round(cw * s))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small

    m = cv2.moments(canvas)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    canvas = cv2.warpAffine(canvas, M, (28, 28))
    return np.clip(canvas / canvas.max(), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="el_yazisi/ham/ogrenci_no.jpg")
    ap.add_argument("--out", default="el_yazisi/islenmis")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = ink_map(gray)
    mask, boxes = find_digits(ink)
    print(f"{len(boxes)} rakam bulundu")

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.endswith(".png"):
            os.remove(os.path.join(args.out, f))

    digits = []
    for i, box in enumerate(boxes):
        d = to_mnist(ink, mask, box)
        digits.append(d)
        cv2.imwrite(os.path.join(args.out, f"rakam_{i + 1:02d}.png"), (d * 255).astype(np.uint8))
    np.save(os.path.join(args.out, "rakamlar.npy"), np.stack(digits))

    vis = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)
        cv2.putText(vis, str(i + 1), (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    m = 80
    roi = (slice(max(0, min(ys) - m), max(ys) + m), slice(max(0, min(xs) - m), max(xs) + m))

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(cv2.cvtColor(vis[roi], cv2.COLOR_BGR2RGB))
    ax.set_title("1) orijinal fotograf + bulunan rakamlar")
    ax.axis("off")
    ax = fig.add_subplot(3, 1, 2)
    ax.imshow(ink[roi], cmap="gray")
    ax.set_title("2) gri ton, arka plan cikarilmis (siyah zemin / beyaz rakam)")
    ax.axis("off")
    n = len(digits)
    for i, d in enumerate(digits):
        ax = fig.add_subplot(3, n, 2 * n + i + 1)
        ax.imshow(d, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("3) 28x28", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig("el_yazisi/on_isleme.png", dpi=110)
    print(f"kaydedildi: {args.out}/, el_yazisi/on_isleme.png")


if __name__ == "__main__":
    main()
