import os
import copy
import random
import platform
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import timm
import shap
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             classification_report)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
SHORT = ["Early Blight", "Late Blight", "Healthy"]

N_PER_CLASS = 100
IMG_SIZE = 264
DENOISE = True
USE_AUGMENTATION = True
EPOCHS = 30
BATCH = 16
GOA_POP = 8
GOA_ITERS = 10
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_W = None


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close()


def dhash(path, s=8):
    im = Image.open(path).convert("L").resize((s + 1, s), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).tobytes()


def collect_images():
    rng = random.Random(SEED)
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cname)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        seen, unique = set(), []
        for f in files:
            p = os.path.join(folder, f)
            h = dhash(p)
            if h not in seen:
                seen.add(h)
                unique.append(p)
        n = min(N_PER_CLASS, len(unique))
        if n < N_PER_CLASS:
            print(f"[warn] {cname}: only {len(unique)} unique images, using {n}")
        chosen = rng.sample(unique, n)
        print(f"{cname}: {len(files)} files, {len(files) - len(unique)} duplicates removed, "
              f"using {n}")
        paths += chosen
        labels += [ci] * n
    return paths, labels


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    if DENOISE:
        img = img.filter(ImageFilter.MedianFilter(3))
    return img


train_tf = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([transforms.ToTensor()])


class LeafDS(Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.labels[i]


def fig_samples(raw_paths, labels):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for ci in range(3):
        idx = labels.index(ci)
        ax[ci].imshow(Image.open(raw_paths[idx]).convert("RGB"))
        ax[ci].set_title(SHORT[ci]); ax[ci].axis("off")
    savefig("01_sample_per_class.png")


def fig_preprocessing_steps(raw_paths, labels):
    steps = ["Original", f"Resize {IMG_SIZE}x{IMG_SIZE}", "Noise removal",
             "Normalize (0-1)", "Rotation", "Horizontal flip", "Vertical flip",
             "Zoom 1.2x", "Brightness"]
    fig, ax = plt.subplots(3, len(steps), figsize=(2.2 * len(steps), 7))
    for ci in range(3):
        p = raw_paths[labels.index(ci)]
        orig = Image.open(p).convert("RGB")
        res = orig.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        den = res.filter(ImageFilter.MedianFilter(3))
        norm = np.asarray(den, dtype=np.float32) / 255.0
        c = int(IMG_SIZE / 1.2)
        zoom = TF.resized_crop(den, (IMG_SIZE - c) // 2, (IMG_SIZE - c) // 2, c, c,
                               [IMG_SIZE, IMG_SIZE])
        panels = [orig, res, den, norm, TF.rotate(den, 30), TF.hflip(den),
                  TF.vflip(den), zoom, TF.adjust_brightness(den, 1.4)]
        for j, im in enumerate(panels):
            ax[ci, j].imshow(im); ax[ci, j].axis("off")
            if ci == 0:
                ax[ci, j].set_title(steps[j], fontsize=9)
    savefig("02_preprocessing_steps.png")


def fig_before_after(raw_paths, labels):
    fig, ax = plt.subplots(3, 2, figsize=(6, 9))
    for ci in range(3):
        p = raw_paths[labels.index(ci)]
        ax[ci, 0].imshow(Image.open(p).convert("RGB"))
        ax[ci, 1].imshow(np.asarray(preprocess(p), dtype=np.float32) / 255.0)
        ax[ci, 0].set_ylabel(SHORT[ci]); ax[ci, 0].set_xticks([]); ax[ci, 0].set_yticks([])
        ax[ci, 1].axis("off")
        if ci == 0:
            ax[ci, 0].set_title("Before"); ax[ci, 1].set_title("After")
    savefig("03_before_after_per_class.png")


def fig_history(hist, name, prefix):
    ep = range(1, len(hist["tr_acc"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, hist["tr_acc"], label="Training"); plt.plot(ep, hist["va_acc"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{name} Training and Validation Accuracy"); plt.legend()
    plt.grid(alpha=.3); savefig(f"{prefix}_accuracy.png")
    plt.figure(figsize=(6, 4))
    plt.plot(ep, hist["tr_loss"], label="Training"); plt.plot(ep, hist["va_loss"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{name} Training and Validation Loss"); plt.legend()
    plt.grid(alpha=.3); savefig(f"{prefix}_loss.png")


class SimpleCNN(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.5), nn.Linear(256, nc))

    def forward(self, x):
        return self.head(self.f(x))


class ViTWrap(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=nc)

    def forward(self, x):
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.m(x)


def build_effnet():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 3))
    return m


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def train_model(model, name, tr_loader, va_loader, lr=1e-3, epochs=EPOCHS):
    model.to(device)
    crit = nn.CrossEntropyLoss(weight=CLASS_W)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}
    best_acc, best_state = -1, None
    for ep in range(epochs):
        tl, ta = run_epoch(model, tr_loader, crit, opt)
        vl, va = run_epoch(model, va_loader, crit)
        sched.step()
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{name}] epoch {ep+1}/{epochs}  train acc {ta:.3f}  val acc {va:.3f}")
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def metrics(y, p):
    return {"Accuracy": accuracy_score(y, p),
            "Precision": precision_score(y, p, average="macro", zero_division=0),
            "Recall": recall_score(y, p, average="macro", zero_division=0),
            "F1-Score": f1_score(y, p, average="macro", zero_division=0)}


class WeightedSVMEnsemble:
    def __init__(self, params):
        self.p = params
        w = np.array(params[4:7], dtype=float)
        self.w = w / w.sum()
        self.models = [
            SVC(kernel="rbf", C=10 ** params[0], gamma=10 ** params[1],
                probability=True, random_state=SEED),
            SVC(kernel="poly", degree=3, C=10 ** params[2], gamma="scale",
                probability=True, random_state=SEED),
            SVC(kernel="linear", C=10 ** params[3], probability=True, random_state=SEED),
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.w, self.models))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


BOUNDS = np.array([[-1, 3],
                   [-5, -1],
                   [-1, 3],
                   [-3, 1],
                   [0.05, 1], [0.05, 1], [0.05, 1]])


def goa_optimize(fitness, bounds, pop=GOA_POP, iters=GOA_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    X = lo + rng.random((pop, len(lo))) * span
    fit = np.array([fitness(x) for x in X])
    history = [fit.max()]
    for it in range(iters):
        best = X[fit.argmax()].copy()
        for i in range(pop):
            j = rng.integers(pop)
            sign = 1.0 if fit[j] > fit[i] else -1.0
            step_best = rng.random(len(lo)) * (best - X[i])
            step_social = sign * rng.random(len(lo)) * (X[j] - X[i]) * 0.5
            noise = (1 - it / iters) * rng.normal(0, 0.1, len(lo)) * span
            new = np.clip(X[i] + step_best + step_social + noise, lo, hi)
            f_new = fitness(new)
            if f_new >= fit[i]:
                X[i], fit[i] = new, f_new
        history.append(fit.max())
        print(f"[GOA] iteration {it+1}/{iters}  best fitness {fit.max():.4f}")
    return X[fit.argmax()], fit.max(), history


def lib_versions():
    rows = [("Python", platform.python_version()),
            ("NumPy", np.__version__),
            ("Pandas", pd.__version__),
            ("Matplotlib", matplotlib.__version__),
            ("Seaborn", sns.__version__),
            ("Pillow", PIL.__version__),
            ("PyTorch", torch.__version__),
            ("Torchvision", torchvision.__version__),
            ("timm", timm.__version__),
            ("SHAP", shap.__version__),
            ("scikit-learn", sklearn.__version__)]
    return pd.DataFrame(rows, columns=["Library", "Version"])


def system_config():
    rows = [("Operating System", platform.platform()),
            ("Processor", platform.processor() or platform.machine()),
            ("CPU Cores", os.cpu_count()),
            ("Python Version", platform.python_version()),
            ("PyTorch Version", torch.__version__),
            ("Compute Device", device.type.upper())]
    try:
        import psutil
        rows.insert(3, ("RAM (GB)", round(psutil.virtual_memory().total / 1024 ** 3, 1)))
    except Exception:
        pass
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        rows.append(("GPU", props.name))
        rows.append(("GPU Memory (GB)", round(props.total_memory / 1024 ** 3, 1)))
        rows.append(("CUDA Version", torch.version.cuda))
    return pd.DataFrame(rows, columns=["Component", "Specification"])


def main():
    set_seed()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    torch.backends.cudnn.benchmark = True

    paths, labels = collect_images()
    counts = np.bincount(labels)
    print("Class counts:", dict(zip(SHORT, counts)))
    if counts.max() != counts.min():
        print("Classes are imbalanced -> augmentation is needed.")
    else:
        print("Classes are balanced.")

    fig_samples(paths, labels)
    fig_preprocessing_steps(paths, labels)
    fig_before_after(paths, labels)

    imgs = [preprocess(p) for p in paths]

    idx = np.arange(len(imgs))
    tr_i, tmp_i = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.5,
                                  stratify=[labels[i] for i in tmp_i], random_state=SEED)
    sub = lambda ii: ([imgs[i] for i in ii], [labels[i] for i in ii])
    (Xtr_img, ytr), (Xva_img, yva), (Xte_img, yte) = sub(tr_i), sub(va_i), sub(te_i)
    print(f"Split -> train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}")
    global CLASS_W
    CLASS_W = torch.tensor(len(ytr) / (3 * np.bincount(ytr)),
                           dtype=torch.float32).to(device)

    tr_ds = LeafDS(Xtr_img, ytr, train_tf if USE_AUGMENTATION else eval_tf)
    tr_ds_plain = LeafDS(Xtr_img, ytr, eval_tf)
    va_ds, te_ds = LeafDS(Xva_img, yva, eval_tf), LeafDS(Xte_img, yte, eval_tf)
    tr_loader = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True, num_workers=0)
    trp_loader = DataLoader(tr_ds_plain, 32, num_workers=0)
    va_loader = DataLoader(va_ds, 32, num_workers=0)
    te_loader = DataLoader(te_ds, 32, num_workers=0)

    print("\n=== Training EfficientNet-B0 (from scratch) ===")
    eff, eff_hist = train_model(build_effnet(), "EfficientNet-B0", tr_loader, va_loader)
    fig_history(eff_hist, "EfficientNet-B0", "04_05_effnet")
    torch.save(eff.state_dict(), os.path.join(OUT_DIR, "efficientnet_b0_best.pth"))

    y_eff_true, y_eff_pred = predict_loader(eff, te_loader)
    eff_only = metrics(y_eff_true, y_eff_pred)

    eff.classifier = nn.Identity()
    Ftr, _ = extract_features(eff, trp_loader)
    Fva, _ = extract_features(eff, va_loader)
    Fte, _ = extract_features(eff, te_loader)
    scaler = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = scaler.transform(Ftr), scaler.transform(Fva), scaler.transform(Fte)
    ytr, yva, yte = np.array(ytr), np.array(yva), np.array(yte)
    print("Feature shape:", Ftr.shape)

    print("\n=== Giraffe Optimization Algorithm ===")

    def fitness(params):
        try:
            ens = WeightedSVMEnsemble(params).fit(Ftr, ytr)
            proba = ens.predict_proba(Fva)
            acc = (proba.argmax(1) == yva).mean()
            true_p = proba[np.arange(len(yva)), yva].mean()
            return acc + 0.01 * true_p
        except Exception:
            return 0.0

    best_params, best_fit, goa_hist = goa_optimize(fitness, BOUNDS)
    print("Best parameters:", np.round(best_params, 3))
    plt.figure(figsize=(5.5, 4))
    plt.plot(range(0, len(goa_hist)), goa_hist, marker="o")
    plt.xlabel("GOA iteration"); plt.ylabel("Best fitness (val)")
    plt.title("GOA Convergence Curve"); plt.grid(alpha=.3); savefig("goa_convergence.png")

    ens = WeightedSVMEnsemble(best_params).fit(Ftr, ytr)
    proba = ens.predict_proba(Fte)
    pred = proba.argmax(1)
    prop = metrics(yte, pred)
    print("\nProposed method (test):", {k: round(v, 4) for k, v in prop.items()})
    print(classification_report(yte, pred, target_names=SHORT, zero_division=0))

    svm_single = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED).fit(Ftr, ytr)
    svm_only = metrics(yte, svm_single.predict(Fte))
    ens_default = WeightedSVMEnsemble([1, -3, 1, -1, 1, 1, 1]).fit(Ftr, ytr)
    ens_no_goa = metrics(yte, ens_default.predict(Fte))

    ablation = pd.DataFrame({
        "EfficientNet-B0 only": eff_only,
        "EfficientNet-B0 + SVM": svm_only,
        "EfficientNet-B0 + SVM Ensemble (no GOA)": ens_no_goa,
        "EfficientNet-B0 + SVM Ensemble + GOA (Proposed)": prop,
    }).T.round(4)
    ablation.index.name = "Configuration"

    plt.figure(figsize=(5.5, 4.5))
    plt.bar(list(prop.keys()), list(prop.values()),
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for i, v in enumerate(prop.values()):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.1); plt.ylabel("Score")
    plt.title("Performance Metrics of Proposed Method")
    savefig("11_performance_metrics.png")

    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SHORT, yticklabels=SHORT)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix of Proposed Method")
    savefig("06_confusion_matrix.png")

    proba_plot = proba

    yb = label_binarize(yte, classes=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    for k in range(3):
        fpr, tpr, _ = roc_curve(yb[:, k], proba_plot[:, k])
        plt.plot(fpr, tpr, label=f"{SHORT[k]} (AUC={auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(yb.ravel(), proba_plot.ravel())
    plt.plot(fpr, tpr, "k--", label=f"Micro-average (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "grey", lw=.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of Proposed Method"); plt.legend(loc="lower right"); savefig("08_roc_curve.png")

    plt.figure(figsize=(6, 5))
    for k in range(3):
        pr, rc, _ = precision_recall_curve(yb[:, k], proba_plot[:, k])
        plt.plot(rc, pr, label=f"{SHORT[k]} (AP={average_precision_score(yb[:, k], proba_plot[:, k]):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision Recall Curve of Proposed Method")
    plt.legend(loc="lower left"); savefig("09_precision_recall_curve.png")

    print("\n=== SHAP (this can take a few minutes) ===")
    bg = shap.sample(Ftr, 10, random_state=SEED)
    explainer = shap.KernelExplainer(ens.predict_proba, bg)
    sv = explainer.shap_values(Fte, nsamples=300)
    if isinstance(sv, list):
        sv_list = sv
    else:
        sv = np.array(sv)
        sv_list = [sv[:, :, k] for k in range(sv.shape[2])] if sv.ndim == 3 else [sv]
    fnames = [f"feat_{i}" for i in range(Fte.shape[1])]
    shap.summary_plot(sv_list, Fte, feature_names=fnames, class_names=SHORT,
                      plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Summary Plot for All Classes")
    savefig("07_shap_summary_plot.png")

    print("\n=== Baselines (from scratch) ===")
    results = {}
    baselines = {"CNN": (SimpleCNN(), 1e-3),
                 "ResNet50": (models.resnet50(weights=None, num_classes=3), 1e-3),
                 "MobileNetV2": (models.mobilenet_v2(weights=None, num_classes=3), 1e-3),
                 "Vision Transformer": (ViTWrap(), 3e-4)}
    for name, (net, lr) in baselines.items():
        net, h = train_model(net, name, tr_loader, va_loader, lr=lr)
        y_true, y_pred = predict_loader(net, te_loader)
        results[name] = metrics(y_true, y_pred)
        del net
        torch.cuda.empty_cache()
    results["Proposed Method"] = prop

    df = pd.DataFrame(results).T.round(4)
    df.index.name = "Model"
    print("\n", df)
    df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1.05))
    plt.xticks(rotation=20); plt.ylabel("Score"); plt.title("Comparison of All Models on Test Set")
    plt.legend(loc="lower right"); savefig("10_baseline_comparison.png")

    df.to_csv(os.path.join(OUT_DIR, "table1_baseline_comparison.csv"))
    ablation.to_csv(os.path.join(OUT_DIR, "table2_ablation_study.csv"))
    lib_versions().to_csv(os.path.join(OUT_DIR, "table3_libraries.csv"), index=False)
    system_config().to_csv(os.path.join(OUT_DIR, "table4_system_config.csv"), index=False)

    print(f"\nDone. All results are saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()