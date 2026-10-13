import torch
from torchvision import datasets

MEAN = 0.1307
STD = 0.3081


def normalize(x):
    return (x - MEAN) / STD


def load_mnist(root="data", val_size=10000, seed=42, device="cpu"):
    train_full = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    x = train_full.data.float().div(255).unsqueeze(1)
    y = train_full.targets
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    x_test = test.data.float().div(255).unsqueeze(1)
    y_test = test.targets

    splits = {
        "train": (normalize(x[train_idx]), y[train_idx]),
        "val": (normalize(x[val_idx]), y[val_idx]),
        "test": (normalize(x_test), y_test),
    }
    return {k: (a.to(device), b.to(device)) for k, (a, b) in splits.items()}


def batches(x, y, batch_size, shuffle=False):
    n = len(x)
    order = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]
