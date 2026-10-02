import numpy as np
import torch


def _synthetic(n_train=6000, n_test=1000, num_classes=10, noise=2.0, seed=0):
    """Tiny fake image dataset for smoke tests (no download needed)."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(num_classes, 1, 28, 28, generator=g)

    def make(n):
        y = torch.randint(0, num_classes, (n,), generator=g)
        x = protos[y] + noise * torch.randn(n, 1, 28, 28, generator=g)
        return x, y

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte, num_classes


def load_dataset(name: str, root: str = "./data"):
    """Returns (x_train, y_train, x_test, y_test, num_classes) as tensors,
    images shaped N x C x H x W and normalised per channel."""
    if name == "synthetic":
        return _synthetic()

    from torchvision import datasets

    cls = {
        "mnist": datasets.MNIST,
        "fmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }[name]
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)

    def to_tensors(ds):
        x = torch.as_tensor(np.asarray(ds.data)).float() / 255.0
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        y = torch.as_tensor(np.asarray(ds.targets)).long()
        return x, y

    xtr, ytr = to_tensors(train)
    xte, yte = to_tensors(test)
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True)
    return (xtr - mean) / std, ytr, (xte - mean) / std, yte, 10


def train_val_split(labels, val_frac: float, seed: int):
    """Hold out val_frac of the training samples (stratified by class) for
    hyperparameter tuning, so the test set is only used for final results.
    Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    tr, va = [], []
    for c in np.unique(labels):
        ids = rng.permutation(np.where(labels == c)[0])
        k = int(round(val_frac * len(ids)))
        va.append(ids[:k])
        tr.append(ids[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def dirichlet_partition(labels, n_clients: int, alpha: float, seed: int,
                        min_size: int = 10, max_tries: int = 100):
    """Split sample indices across clients with class proportions drawn from
    Dirichlet(alpha). Small alpha (0.1) = very non-IID; large (100) ~ IID."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = labels.max() + 1
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in range(num_classes):
            ids = np.where(labels == c)[0]
            rng.shuffle(ids)
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(ids, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    return [np.array(p) for p in parts]
