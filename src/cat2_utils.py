"""
cat2_utils.py
=============
Shared helpers for DSA 4050-XA CAT 2: image classification with PyTorch
(10 classes, 4,000 labelled training images, 46,000 test images to predict).

Every notebook imports this module, so data loading, augmentation and the way
results are saved are identical everywhere.

Contents
  1. Paths and folders (Google Drive project + fast local copy of the images)
  2. Reproducibility (one seed for Python, NumPy and PyTorch)
  3. Saving outputs for the write-up (figures, tables, key results, experiments)
  4. Data: labels, the custom Dataset class (subclasses torch.utils.data.Dataset)
  5. Data augmentation / preprocessing transforms

Author : Selmah Tzindori (669602)
Course : DSA 4050-XA Deep Learning, USIU-Africa, Fall 2026 (Dr. Alex Wambua)
"""

import os
import json
import glob
import random
import shutil
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# 1. PATHS
# =============================================================================
ON_COLAB = os.path.isdir("/content")
DRIVE = "/content/drive/MyDrive"

# Project folder on Google Drive (models, outputs, processed files persist here)
PROJECT_ROOT = os.environ.get(
    "CAT2_ROOT",
    f"{DRIVE}/DSA4050/cat2-image-classification" if ON_COLAB
    else os.path.join(os.path.expanduser("~"), "DSA4050", "cat2-image-classification"))

PATHS = {
    "root":        PROJECT_ROOT,
    "processed":   os.path.join(PROJECT_ROOT, "data", "processed"),   # folds, image stats
    "models":      os.path.join(PROJECT_ROOT, "models"),              # trained weights (.pt)
    "optuna":      os.path.join(PROJECT_ROOT, "models", "optuna"),    # hyper-parameter search database
    "figures":     os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "tables":      os.path.join(PROJECT_ROOT, "outputs", "tables"),
    "logs":        os.path.join(PROJECT_ROOT, "outputs", "logs"),
    "submission":  os.path.join(PROJECT_ROOT, "outputs", "submission"),
    "src":         os.path.join(PROJECT_ROOT, "src"),
}
RESULTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "results.json")
EXPERIMENTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "experiments.csv")

# Fast local copy of the images (Colab's disk is wiped every session; reading
# 50,000 small PNGs straight from Drive would be very slow)
LOCAL_DATA = "/content/cat2_data" if ON_COLAB else os.path.join(PROJECT_ROOT, "_local_data")

SEED = 42


def make_folders(verbose: bool = True) -> None:
    """Create every project folder (safe to re-run)."""
    for name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"  ✔ {name:<11} {path}")


def find_source_folder(search_root: str = DRIVE, max_depth: int = 3) -> str:
    """
    Locate the lecturer's shared folder (added to My Drive as a shortcut) by
    looking for a folder that contains BOTH train.zip and Train.csv.
    Searching a few levels deep means the exact folder name does not matter.
    """
    env = os.environ.get("CAT2_SOURCE")
    if env:
        return env
    # Most common case: the shortcut sits directly in My Drive (e.g. "Advanced ML")
    for entry in sorted(os.listdir(search_root)):
        p = os.path.join(search_root, entry)
        if os.path.isfile(os.path.join(p, "train.zip")) and os.path.isfile(os.path.join(p, "Train.csv")):
            return p
    root_depth = search_root.rstrip("/").count("/")
    # Drive shortcuts appear as links in Colab, so links must be followed
    for dirpath, dirnames, filenames in os.walk(search_root, followlinks=True):
        if dirpath.count("/") - root_depth >= max_depth:
            dirnames[:] = []                       # do not go deeper
        if "train.zip" in filenames and "Train.csv" in filenames:
            return dirpath
    raise FileNotFoundError(
        "Could not find a folder with train.zip and Train.csv in My Drive. Open the CAT's "
        "Google Drive link, click 'Add shortcut to Drive' -> My Drive, then re-run.")


def restore_data(source: str = None, local_dir: str = LOCAL_DATA) -> Path:
    """
    Copy train.zip + the two CSV files from the shared Drive folder to Colab's
    local disk and unzip the images (once per session, ~30 s).
    Returns the local folder, which contains train/<id>.png, Train.csv, Test.csv.
    """
    local = Path(local_dir)
    if (local / "train").is_dir() and len(os.listdir(local / "train")) >= 50_000:
        print(f"Data already available at {local}")
        return local
    source = source or find_source_folder()
    print("Source folder:", source)
    local.mkdir(parents=True, exist_ok=True)
    for f in ["Train.csv", "Test.csv"]:
        shutil.copy(os.path.join(source, f), local / f)
    tmp_zip = local / "train.zip"
    shutil.copy(os.path.join(source, "train.zip"), tmp_zip)     # one big file: fast
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(local)
    tmp_zip.unlink()
    n = len(os.listdir(local / "train"))
    print(f"Unzipped {n:,} images to {local / 'train'}")
    return local


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    """Same seed for Python, NumPy and PyTorch (CPU + GPU) -> repeatable results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"Random seed set to {seed}")


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 3. SAVING OUTPUTS FOR THE WRITE-UP
# =============================================================================
def _index(kind, name, path, notebook, caption):
    f = os.path.join(PATHS["logs"], "outputs_index.csv")
    row = pd.DataFrame([{"kind": kind, "name": name, "path": path, "notebook": notebook,
                         "caption": caption, "saved_at": _now()}])
    if os.path.exists(f):
        old = pd.read_csv(f)
        row = pd.concat([old[~((old["kind"] == kind) & (old["name"] == name))], row], ignore_index=True)
    row.to_csv(f, index=False)


def save_figure(fig, name: str, notebook: str, caption: str = "", dpi: int = 200) -> str:
    """Save a matplotlib figure to outputs/figures/<name>.png and record its caption."""
    path = os.path.join(PATHS["figures"], f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _index("figure", name, path, notebook, caption)
    print(f"Figure saved -> {path}")
    return path


def save_table(df: pd.DataFrame, name: str, notebook: str, caption: str = "", index: bool = False) -> str:
    """Save a DataFrame to outputs/tables/<name>.csv."""
    path = os.path.join(PATHS["tables"], f"{name}.csv")
    df.to_csv(path, index=index)
    _index("table", name, path, notebook, caption)
    print(f"Table saved  -> {path}")
    return path


def load_results() -> dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def log_result(key: str, value, notebook: str) -> None:
    """Store one key number for the write-up in outputs/results.json."""
    res = load_results()
    res[key] = {"value": value, "notebook": notebook, "saved_at": _now()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"Result logged: {key} = {value}")


def log_experiment(name: str, metrics: dict, notebook: str, params: dict = None, notes: str = "") -> pd.DataFrame:
    """One row per trained model in outputs/experiments.csv (the comparison table)."""
    row = {"model": name, **metrics, "params": json.dumps(params or {}, default=str),
           "notes": notes, "notebook": notebook, "saved_at": _now()}
    if os.path.exists(EXPERIMENTS_FILE):
        df = pd.read_csv(EXPERIMENTS_FILE)
        df = pd.concat([df[df["model"] != name], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EXPERIMENTS_FILE, index=False)
    print(f"Experiment logged: {name} -> {metrics}")
    return df


def project_summary() -> None:
    def count(folder, ext):
        return len(glob.glob(os.path.join(folder, f"*{ext}"))) if os.path.isdir(folder) else 0
    print("Project root :", PROJECT_ROOT)
    print("Figures      :", count(PATHS["figures"], ".png"))
    print("Tables       :", count(PATHS["tables"], ".csv"))
    print("Models       :", count(PATHS["models"], ".pt"))
    print("Results      :", len(load_results()))
    print("Experiments  :", len(pd.read_csv(EXPERIMENTS_FILE)) if os.path.exists(EXPERIMENTS_FILE) else 0)


# =============================================================================
# 4. DATA
# =============================================================================
# The ten classes in alphabetical order; a class's position is its label index.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_labels(data_dir) -> tuple:
    """Return (train_df, test_df). train_df gets a numeric 'target' column (0-9)."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "Train.csv")
    test = pd.read_csv(data_dir / "Test.csv")
    train["target"] = train["label"].map(CLASS_TO_IDX)
    return train, test


def load_images(ids, image_dir) -> np.ndarray:
    """Read PNGs <image_dir>/<id>.png into one uint8 array of shape (N, H, W, 3)."""
    image_dir = Path(image_dir)
    out = []
    for i in ids:
        with Image.open(image_dir / f"{i}.png") as im:
            out.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return np.stack(out)


try:
    import torch
    from torch.utils.data import Dataset

    class CIFARCSVDataset(Dataset):
        """
        Custom PyTorch Dataset (subclasses torch.utils.data.Dataset).

        Each item is (image_tensor, label). Images are looked up by their id in a
        DataFrame (Train.csv or Test.csv); test images have no label, so -1 is
        returned instead.

        Parameters
        ----------
        df        : DataFrame with an 'id' column (and 'target' for labelled data)
        image_dir : folder holding <id>.png files
        transform : torchvision transform applied to each PIL image (augmentation
                    for training, plain preprocessing for validation/test)
        images    : optional pre-loaded uint8 array (N, H, W, 3) in the same order
                    as df. The whole dataset is only ~150 MB in memory, so
                    pre-loading avoids re-reading thousands of PNGs every epoch.
        """

        def __init__(self, df: pd.DataFrame, image_dir=None, transform=None, images: np.ndarray = None):
            self.ids = df["id"].to_numpy()
            self.targets = df["target"].to_numpy() if "target" in df else np.full(len(df), -1)
            self.image_dir = Path(image_dir) if image_dir is not None else None
            self.transform = transform
            self.images = images
            if images is not None and len(images) != len(df):
                raise ValueError("images and df must have the same length")

        def __len__(self) -> int:
            return len(self.ids)

        def _load(self, idx: int) -> Image.Image:
            if self.images is not None:
                return Image.fromarray(self.images[idx])
            with Image.open(self.image_dir / f"{self.ids[idx]}.png") as im:
                return im.convert("RGB")

        def __getitem__(self, idx: int):
            img = self._load(idx)
            if self.transform is not None:
                img = self.transform(img)
            return img, int(self.targets[idx])

except ImportError:          # lets the module import on a machine without PyTorch
    pass


# =============================================================================
# 5. TRANSFORMS (preprocessing + data augmentation)
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)    # pretrained networks expect these statistics
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, img_size: int = 32, randaug_magnitude: int = 9,
                     erase_prob: float = 0.25, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Training (train=True): data augmentation, applied randomly every epoch
        - random crop with 4-px padding (small shifts of the object)
        - horizontal flip (a left-facing horse is still a horse)
        - RandAugment: 2 random photometric/geometric operations of strength
          `randaug_magnitude` (0 = off ... 30 = extreme)
        - resize to img_size (pretrained CNNs work best at larger sizes)
        - random erasing of a patch (forces use of the whole object)
    Evaluation (train=False): only resize + normalise, so results are repeatable.
    """
    from torchvision import transforms as T
    resize = [T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)] if img_size != 32 else []
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose(resize + norm)
    aug = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if randaug_magnitude > 0:
        aug.append(T.RandAugment(num_ops=2, magnitude=int(randaug_magnitude)))
    tail = [T.RandomErasing(p=erase_prob, scale=(0.02, 0.2))] if erase_prob > 0 else []
    return T.Compose(aug + resize + norm + tail)


def denormalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Undo Normalize for display: tensor (3,H,W) -> uint8 array (H,W,3)."""
    arr = t.detach().cpu().numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


# =============================================================================
# 6. TRAINING AND PREDICTION (shared by every model)
# =============================================================================
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2):
    """DataLoader with sensible defaults (workers + pinned memory on GPU)."""
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=g,
                      persistent_workers=num_workers > 0)


def train_model(model, train_loader, val_loader, epochs: int, lr: float, weight_decay: float = 0.05,
                label_smoothing: float = 0.1, optimizer: str = "adamw", warmup_epochs: int = 1,
                epoch_callback=None, verbose: bool = True):
    """
    Train `model` and track validation accuracy every epoch.

    - Optimiser: AdamW (default) or SGD with Nesterov momentum
    - Learning rate: linear warm-up, then cosine decay to ~0 (per batch)
    - Loss: cross-entropy with label smoothing (softer targets, better calibration)
    - Mixed precision (float16) on GPU for ~2x speed
    - epoch_callback(epoch, val_acc) is called after each epoch; Optuna uses it to
      prune (stop early) unpromising trials. It may raise to stop training.

    Returns a dict: history (per-epoch metrics), best_val_acc, best_epoch, best_state.
    """
    import math, copy, time
    import torch
    import torch.nn as nn

    device = get_device()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total, warm = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_lambda(step):                        # warm-up then cosine
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], {"best_val_acc": -1.0, "best_epoch": -1, "best_state": None}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            run_correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        row = {"epoch": epoch, "train_loss": run_loss / n, "train_acc": run_correct / n,
               "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        history.append(row)
        if val_acc > best["best_val_acc"]:
            best.update(best_val_acc=val_acc, best_epoch=epoch,
                        best_state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
        if verbose:
            print(f"epoch {epoch:>3}/{epochs}  train_loss {row['train_loss']:.3f}  train_acc {row['train_acc']:.3f}  "
                  f"val_loss {val_loss:.3f}  val_acc {val_acc:.4f}  ({row['seconds']:.0f}s)")
        if epoch_callback is not None:
            epoch_callback(epoch, val_acc)
    return {"history": pd.DataFrame(history), **best}


def evaluate(model, loader, loss_fn=None):
    """Average loss and accuracy on a labelled loader (no augmentation, no gradients)."""
    import torch
    import torch.nn as nn
    device = get_device()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            tot_loss += loss_fn(out.float(), y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def predict_proba(model, loader, tta: bool = False) -> np.ndarray:
    """
    Class probabilities for every image in `loader` (in loader order).
    tta=True adds test-time augmentation: the prediction is averaged with the
    prediction for the horizontally flipped image.
    """
    import torch
    device = get_device()
    model.to(device).eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def plot_history(hist: pd.DataFrame, title: str):
    """Training vs validation accuracy and loss per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["acc", "loss"]):
        ax.plot(hist["epoch"], hist[f"train_{m}"], label="train", color="#9CC9BD")
        ax.plot(hist["epoch"], hist[f"val_{m}"], label="validation", color="#2E7D6B")
        ax.set_title(f"{title}: {'accuracy' if m == 'acc' else 'loss'}")
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig
