import os
import random

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

SEED = 42
FS = 173.61
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}
DATA = "D:/Work/Test_Data/archive/Epileptic_Seizure_Recognition.csv"


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def load_and_preprocess():
    df = pd.read_csv(DATA)
    n_raw = len(df)
    df = df.drop(columns=["Unnamed"])
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    X = df.drop(columns="y").values.astype("float32")
    y5 = df["y"].values
    y = (y5 == 1).astype(int)
    idx = np.arange(len(X))

    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    mu, sd = X[tr].mean(), X[tr].std()
    Xz = (X - mu) / sd

    return dict(X=Xz[..., None], Xflat=Xz, y=y, y5=y5, tr=tr, va=va, te=te, n_raw=n_raw)


def extract_features(Xflat):
    N, L = Xflat.shape
    spec = np.abs(np.fft.rfft(Xflat, axis=1)) ** 2
    freqs = np.fft.rfftfreq(L, d=1 / FS)
    total = spec[:, 1:].sum(1) + 1e-12

    feats = [
        spec[:, (freqs >= lo) & (freqs < hi)].sum(1) / total
        for lo, hi in BANDS.values()
    ]

    p = spec[:, 1:] / total[:, None]
    feats.append(-(p * np.log2(p + 1e-12)).sum(1))

    ent = []
    for row in Xflat:
        h, _ = np.histogram(row, bins=16)
        h = h / h.sum()
        ent.append(-(h[h > 0] * np.log2(h[h > 0])).sum())
    feats.append(np.array(ent))

    return np.stack(feats, 1).astype("float32")


FEAT_NAMES = [f"{b}_power" for b in BANDS] + ["spectral_entropy", "amplitude_entropy"]


def build_model(hp, use_feat=True, use_attn=True, use_bilstm=True, two_branch=True,
                n_feat=7, L=178):
    sig = layers.Input((L, 1))
    fin = layers.Input((n_feat,))

    def branch(k):
        x = layers.Conv1D(hp["filters"], k, padding="same")(sig)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    x = branch(3)
    if two_branch:
        x = layers.Concatenate()([x, branch(1)])
    x = layers.MaxPooling1D(2)(x)

    if use_bilstm:
        x = layers.Bidirectional(layers.LSTM(hp["units"], return_sequences=True))(x)
    else:
        x = layers.Conv1D(2 * hp["units"], 3, padding="same", activation="relu")(x)

    if use_attn:
        score = layers.Dense(1)(layers.Dense(hp["attn_dim"], activation="tanh")(x))
        att = layers.Softmax(axis=1, name="attention_weights")(score)
        ctx = layers.Flatten()(layers.Dot(axes=1)([att, x]))
    else:
        ctx, att = layers.GlobalAveragePooling1D()(x), None

    if use_feat:
        f = layers.Dense(16, activation="relu")(fin)
        ctx = layers.Concatenate()([ctx, f])

    ctx = layers.Dropout(hp["dropout"])(ctx)
    ctx = layers.Dense(32, activation="relu")(ctx)
    out = layers.Dense(2, activation="softmax")(ctx)

    m = Model([sig, fin], out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


DEFAULT_HP = dict(filters=16, units=32, attn_dim=32, lr=1e-3, dropout=0.3)


def get_attention_model(m):
    try:
        return Model(m.inputs, m.get_layer("attention_weights").output)
    except ValueError:
        return None


import json, time, platform, os, joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
FIG = f"{OUT}/figures"
TAB = f"{OUT}/tables"
MOD = "models"
for p in (FIG, TAB, MOD):
    os.makedirs(p, exist_ok=True)

CACHE = f"{MOD}/cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def save_cache():
    json.dump(cache, open(CACHE, "w"))


def safe(n):
    return n.replace("/", "-")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

set_seed()
d = load_and_preprocess()
X, y, y5, tr, va, te = d["X"], d["y"], d["y5"], d["tr"], d["va"], d["te"]

F = extract_features(d["Xflat"])
fm, fs_ = F[tr].mean(0), F[tr].std(0) + 1e-8
Fz = (F - fm) / fs_
log("data", X.shape, "features", F.shape, "seizure ratio", y.mean().round(3))

rows = []
for nm, ix in [
    ("Train (70%)", tr),
    ("Validation (15%)", va),
    ("Test (15%)", te),
    ("Total", np.arange(len(y))),
]:
    rows.append({
        "Split": nm,
        "Samples": len(ix),
        "Seizure (class 1)": int(y[ix].sum()),
        "Non-seizure (classes 2-5)": int((1 - y[ix]).sum()),
        **{f"Original class {c}": int((y5[ix] == c).sum()) for c in range(1, 6)},
    })
pd.DataFrame(rows).to_csv(f"{TAB}/table1_dataset_distribution.csv", index=False)

SPACE = {
    "filters": (8, 32, "int"),
    "units": (16, 64, "int"),
    "attn_dim": (16, 64, "int"),
    "lr": (1e-4, 1e-2, "log"),
    "dropout": (0.1, 0.5, "float"),
}


def decode(p):
    hp = {}
    for v, (k, (lo, hi, kind)) in zip(p, SPACE.items()):
        if kind == "int":
            hp[k] = int(round(lo + v * (hi - lo)))
        elif kind == "log":
            hp[k] = float(10 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo))))
        else:
            hp[k] = float(lo + v * (hi - lo))
    return hp


rng = np.random.default_rng(SEED)
sub_tr = rng.choice(tr, 3000, replace=False)
sub_va = rng.choice(va, 800, replace=False)


def fitness(p):
    hp = decode(p)
    tf.keras.backend.clear_session()
    set_seed()
    m = build_model(hp)
    m.fit([X[sub_tr], Fz[sub_tr]], y[sub_tr], epochs=3, batch_size=64, verbose=0)
    return float(m.evaluate([X[sub_va], Fz[sub_va]], y[sub_va], verbose=0)[0])


def fox_optimize(pop=5, iters=4):
    P = rng.random((pop, len(SPACE)))
    fit = np.array([fitness(p) for p in P])
    curve = [fit.min()]
    log("FO init best", fit.min())

    for it in range(1, iters + 1):
        a = 2 * (1 - it / iters)
        best = P[fit.argmin()].copy()

        for i in range(pop):
            if rng.random() >= 0.18:
                t = rng.random(len(SPACE))
                jump = 0.5 * 9.81 * t ** 2 / 9.81 * 2
                new = (
                    P[i]
                    + (a / 2) * rng.random(len(SPACE)) * (best - P[i]) * jump
                    + 0.05 * a * rng.normal(size=len(SPACE))
                )
            else:
                new = best + 0.15 * a * rng.normal(size=len(SPACE))

            new = np.clip(new, 0, 1)
            f = fitness(new)
            if f < fit[i]:
                P[i], fit[i] = new, f

        curve.append(fit.min())
        log(f"FO iter {it}/{iters} best val-loss {fit.min():.4f}", decode(P[fit.argmin()]))

    return decode(P[fit.argmin()]), curve


if "fo" not in cache:
    t0 = time.time()
    hp, curve = fox_optimize()
    cache["fo"] = {"hp": hp, "curve": curve, "time": time.time() - t0}
    save_cache()

BEST_HP, FO_CURVE = cache["fo"]["hp"], cache["fo"]["curve"]
log("best hp", BEST_HP)

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
CW = {0: cw[0], 1: cw[1]}


def metrics(yt, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    return {
        "Accuracy": accuracy_score(yt, pred),
        "Precision": precision_score(yt, pred),
        "Recall": recall_score(yt, pred),
        "F1": f1_score(yt, pred),
        "AUC": roc_auc_score(yt, prob),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def run_keras(name, builder, use_feat=True, epochs=20):
    if name in cache:
        return

    log("training", name)
    tf.keras.backend.clear_session()
    set_seed()
    m = builder()

    inp = lambda ix: [X[ix], Fz[ix]]

    t0 = time.time()
    h = m.fit(
        inp(tr), y[tr],
        validation_data=(inp(va), y[va]),
        epochs=epochs,
        batch_size=64,
        class_weight=CW,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
    )
    ttime = time.time() - t0

    t0 = time.time()
    prob = m.predict(inp(te), verbose=0)[:, 1]
    itime = (time.time() - t0) / len(te)

    m.save(f"{MOD}/{safe(name)}.keras")
    cache[name] = {
        "hist": {k: [float(v) for v in vs] for k, vs in h.history.items()},
        "train_time": ttime,
        "infer_ms": itime * 1000,
        "prob": prob.tolist(),
        "metrics": metrics(y[te], prob),
        "params": int(m.count_params()),
    }
    save_cache()
    log(
        name,
        {k: round(v, 4) for k, v in cache[name]["metrics"].items() if k in ("Accuracy", "F1", "AUC")},
        f"{ttime:.0f}s",
    )


def run_sklearn(name, model):
    if name in cache:
        return

    log("training", name)
    t0 = time.time()
    model.fit(d["Xflat"][tr], y[tr])
    ttime = time.time() - t0

    t0 = time.time()
    prob = model.predict_proba(d["Xflat"][te])[:, 1]
    itime = (time.time() - t0) / len(te)

    joblib.dump(model, f"{MOD}/{safe(name)}.joblib")
    cache[name] = {
        "train_time": ttime,
        "infer_ms": itime * 1000,
        "prob": prob.tolist(),
        "metrics": metrics(y[te], prob),
    }
    save_cache()
    log(name, round(cache[name]["metrics"]["Accuracy"], 4))


def simple_cnn():
    i = layers.Input((178, 1))
    f = layers.Input((7,))
    x = layers.Conv1D(32, 3, activation="relu")(i)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(layers.Dropout(.3)(x))
    o = layers.Dense(2, activation="softmax")(x)
    m = Model([i, f], o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def simple_bilstm():
    i = layers.Input((178, 1))
    f = layers.Input((7,))
    x = layers.MaxPooling1D(2)(i)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation="relu")(layers.Dropout(.3)(x))
    o = layers.Dense(2, activation="softmax")(x)
    m = Model([i, f], o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


run_keras("Proposed (FO-CNN-BiLSTM-Attn)", lambda: build_model(BEST_HP))

run_keras("w/o Fox Optimization", lambda: build_model(DEFAULT_HP))
run_keras("w/o FFT/entropy features", lambda: build_model(BEST_HP, use_feat=False))
run_keras("w/o Attention", lambda: build_model(BEST_HP, use_attn=False))
run_keras("w/o Bi-LSTM", lambda: build_model(BEST_HP, use_bilstm=False))
run_keras("w/o k=1 CNN branch", lambda: build_model(BEST_HP, two_branch=False))

run_sklearn("SVM (RBF)", SVC(probability=True, random_state=SEED))
run_sklearn("Random Forest", RandomForestClassifier(200, random_state=SEED, n_jobs=-1))
run_keras("1D-CNN", simple_cnn)
run_keras("Bi-LSTM", simple_bilstm)

log("all models trained")
json.dump(cache, open(CACHE, "w"))


import json, os, platform, joblib, sklearn
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix,
    accuracy_score, f1_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
FIG = f"{OUT}/figures"
TAB = f"{OUT}/tables"
MOD = "models"
cache = json.load(open(f"{MOD}/cache.json"))
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})


def savefig(n):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{n}.png")
    plt.close()


d = load_and_preprocess()
X, y, y5, tr, va, te = d["X"], d["y"], d["y5"], d["tr"], d["va"], d["te"]
F = extract_features(d["Xflat"])
fm, fs_ = F[tr].mean(0), F[tr].std(0) + 1e-8
Fz = (F - fm) / fs_

MAIN = "Proposed (FO-CNN-BiLSTM-Attn)"
M = cache[MAIN]
prob = np.array(M["prob"])
yt = y[te]
pred = (prob >= .5).astype(int)
BEST_HP = cache["fo"]["hp"]

h = M["hist"]
ep = range(1, len(h["loss"]) + 1)

plt.figure(figsize=(6, 4))
plt.plot(ep, h["loss"], label="Training")
plt.plot(ep, h["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
savefig("fig1_loss_curve")

plt.figure(figsize=(6, 4))
plt.plot(ep, h["accuracy"], label="Training")
plt.plot(ep, h["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")
plt.legend()
savefig("fig2_accuracy_curve")

c = cache["fo"]["curve"]
plt.figure(figsize=(6, 4))
plt.plot(range(len(c)), c, "o-")
plt.xlabel("Fox Optimization iteration")
plt.ylabel("Best validation loss")
plt.title("Fox Optimization convergence")
savefig("fig3_fox_convergence")

att_m = get_attention_model(tf.keras.models.load_model(f"{MOD}/{MAIN}.keras"))
A = att_m.predict([X[te], Fz[te]], verbose=0)[..., 0]
A178 = np.repeat(A, 2, axis=1)[:, :178]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t = np.arange(178) / FS

for lab, nm in [(1, "Seizure"), (0, "Non-seizure")]:
    ax[0].plot(t, A178[yt == lab].mean(0), label=nm)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Mean attention weight")
ax[0].set_title("Average attention over test samples")
ax[0].legend()

i = np.where((yt == 1) & (pred == 1))[0][0]
ax[1].plot(t, X[te][i, :, 0], "k", lw=.8, label="EEG (z-score)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")
a2 = ax[1].twinx()
a2.fill_between(t, A178[i], color="r", alpha=.3, label="attention")
a2.set_ylabel("Attention")
a2.grid(False)
ax[1].set_title("Attention on one correctly detected seizure")
savefig("fig4_attention_weights")

fpr, tpr, _ = roc_curve(yt, prob)
plt.figure(figsize=(5, 4.5))
plt.plot(fpr, tpr, label=f"AUC = {M['metrics']['AUC']:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend()
savefig("fig5_roc_curve")

p, r, _ = precision_recall_curve(yt, prob)
plt.figure(figsize=(5, 4.5))
plt.plot(r, p, label=f"AP = {average_precision_score(yt, prob):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend()
savefig("fig6_precision_recall_curve")

cm = confusion_matrix(yt, pred)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.grid(False)
for (a, b), v in np.ndenumerate(cm):
    plt.text(b, a, v, ha="center", va="center", fontsize=14,
             color="white" if v > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Non-seizure", "Seizure"])
plt.yticks([0, 1], ["Non-seizure", "Seizure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (test)")
savefig("fig7_confusion_matrix")


def bar(names, title, fn, cols=("Accuracy", "Precision", "Recall", "F1", "AUC")):
    plt.figure(figsize=(9, 4.5))
    w = .8 / len(cols)
    for j, cn in enumerate(cols):
        plt.bar(np.arange(len(names)) + j * w,
                [cache[n]["metrics"][cn] for n in names], w, label=cn)
    plt.xticks(np.arange(len(names)) + .4 - w / 2,
               [n.replace(" (", "\n(") for n in names],
               rotation=20, ha="right", fontsize=8)
    lo = min(cache[n]["metrics"][cn] for n in names for cn in cols)
    plt.ylim(max(0, lo - .05), 1.01)
    plt.title(title)
    plt.legend(ncol=5, fontsize=8, loc="lower right")
    savefig(fn)


BASE = ["SVM (RBF)", "Random Forest", "1D-CNN", "Bi-LSTM", MAIN]
ABL = [MAIN, "w/o Fox Optimization", "w/o FFT/entropy features",
       "w/o Attention", "w/o Bi-LSTM", "w/o k=1 CNN branch"]

bar(BASE, "Baseline model comparison (test set)", "fig8_baseline_comparison")
bar(ABL, "Ablation study (test set)", "fig9_ablation_study")

cols = ["Accuracy", "Precision", "Recall", "F1", "AUC", "FPR", "FNR"]
for names, fn in [(BASE, "table4_baseline_comparison"), (ABL, "table5_ablation_study")]:
    df = pd.DataFrame([
        {"Model": n, **{k: round(cache[n]["metrics"][k], 4) for k in cols}}
        for n in names
    ])
    df.to_csv(f"{TAB}/{fn}.csv", index=False)


def add_noise(Xf, snr, rng):
    pw = (Xf ** 2).mean(1, keepdims=True)
    return Xf + rng.normal(size=Xf.shape) * np.sqrt(pw / 10 ** (snr / 10))


NM = {
    MAIN: "Proposed",
    "1D-CNN": "1D-CNN",
    "Bi-LSTM": "Bi-LSTM",
    "SVM (RBF)": "SVM",
    "Random Forest": "Random Forest",
}
loaded = {
    n: (joblib.load(f"{MOD}/{n}.joblib") if n in ("SVM (RBF)", "Random Forest")
        else tf.keras.models.load_model(f"{MOD}/{n}.keras"))
    for n in NM
}

SNRS = [None, 20, 10, 5, 0]
res = {n: [] for n in NM}
rng = np.random.default_rng(0)

for snr in SNRS:
    Xf = d["Xflat"][te] if snr is None else add_noise(d["Xflat"][te], snr, rng)
    Fn = (extract_features(Xf) - fm) / fs_
    for n, m in loaded.items():
        if n in ("SVM (RBF)", "Random Forest"):
            pr = m.predict_proba(Xf)[:, 1]
        else:
            pr = m.predict([Xf[..., None], Fn], verbose=0)[:, 1]
        res[n].append(accuracy_score(yt, (pr >= .5).astype(int)))

plt.figure(figsize=(6.5, 4.5))
for n in NM:
    plt.plot(range(len(SNRS)), res[n], "o-", label=NM[n], lw=2.5 if n == MAIN else 1.2)
plt.xticks(range(len(SNRS)), ["Clean"] + [f"{s} dB" for s in SNRS[1:]])
plt.xlabel("Signal-to-noise ratio")
plt.ylabel("Test accuracy")
plt.title("Noise robustness (Gaussian noise)")
plt.legend()
savefig("fig10_noise_robustness")

pd.DataFrame({
    "SNR": ["Clean"] + [f"{s} dB" for s in SNRS[1:]],
    **{NM[n]: np.round(res[n], 4) for n in NM},
}).to_csv(f"{TAB}/noise_robustness.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
nm = [n.replace(" (", "\n(") for n in BASE]

ax[0].bar(nm, [cache[n]["train_time"] for n in BASE], color="tab:blue")
ax[0].set_ylabel("Training time (s)")
ax[0].set_title("Training time")

ax[1].bar(nm, [cache[n]["infer_ms"] for n in BASE], color="tab:orange")
ax[1].set_ylabel("Inference (ms / sample)")
ax[1].set_title("Inference time")

for a in ax:
    a.tick_params(axis="x", labelsize=7, rotation=20)
savefig("fig11_computational_time")

y5t = y5[te]
cls = pd.DataFrame([
    {
        "Original class": c,
        "Meaning": mn,
        "Samples": int((y5t == c).sum()),
        "Target": "Seizure" if c == 1 else "Non-seizure",
        "Correctly classified": round(float((pred[y5t == c] == (1 if c == 1 else 0)).mean()), 4),
    }
    for c, mn in [
        (1, "Seizure activity"),
        (2, "Tumour area"),
        (3, "Healthy area"),
        (4, "Eyes closed"),
        (5, "Eyes open"),
    ]
])
cls.to_csv(f"{TAB}/class_wise_performance.csv", index=False)

plt.figure(figsize=(6.5, 4.2))
plt.bar(
    [f"Class {c}\n({m})" for c, m in zip(cls["Original class"], cls["Meaning"])],
    cls["Correctly classified"],
    color=["tab:red"] + ["tab:blue"] * 4,
)
plt.ylim(max(0, cls["Correctly classified"].min() - .05), 1.005)
plt.ylabel("Fraction correctly classified")
plt.xticks(fontsize=7)
plt.title("Class-wise performance (original 5 classes)")
savefig("fig12_classwise_performance")

def get_cpu():
    try:
        return next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l)
    except Exception:
        return platform.processor() or platform.machine()


def get_ram_gb():
    try:
        return next(int(l.split()[1]) / 1e6 for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 1e9
    except Exception:
        return 0


cpu = get_cpu()
ram = get_ram_gb()

pd.DataFrame(
    [
        ("Processor", cpu),
        ("CPU cores used", os.cpu_count()),
        ("RAM", f"{ram:.1f} GB"),
        ("GPU", "None (CPU only)"),
        ("Operating system", platform.platform()),
        ("Python", platform.python_version()),
        ("TensorFlow / Keras", f"{tf.__version__} / {tf.keras.__version__}"),
        ("scikit-learn", sklearn.__version__),
        ("NumPy / Pandas", f"{np.__version__} / {pd.__version__}"),
    ],
    columns=["Item", "Configuration"],
).to_csv(f"{TAB}/table2_experimental_setup.csv", index=False)

fo = cache["fo"]
pd.DataFrame(
    [
        ("Input signal shape", "(N, 178, 1)  (178 time samples, 1 channel)"),
        ("Input feature vector shape", "(N, 7)"),
        ("Sampling frequency",
         f"{FS} Hz (4097 points / 23.6 s per Bonn recording; each sample = 178 points ~ 1.03 s)"),
        ("Frequency bands (Hz)", "delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-60"),
        ("FFT features (5)",
         "relative band power = sum of |FFT|^2 in the band / total power (DC excluded), "
         "using rfft of the z-scored 178-point signal"),
        ("Spectral entropy (1)", "H = -sum p_k log2 p_k, p_k = normalised power spectrum"),
        ("Amplitude entropy (1)", "Shannon entropy of a 16-bin amplitude histogram of the signal"),
        ("CNN branches",
         f"Conv1D(kernel 3) and Conv1D(kernel 1), {BEST_HP['filters']} filters each, "
         "BatchNorm, ReLU; concatenated; MaxPool(2)"),
        ("Bi-LSTM units (per direction)", BEST_HP["units"]),
        ("Attention dimension", BEST_HP["attn_dim"]),
        ("Learning rate (Adam)", f"{BEST_HP['lr']:.5f}"),
        ("Dropout rate", f"{BEST_HP['dropout']:.3f}"),
        ("Feature fusion",
         "Attention context vector (2 x units) concatenated with Dense(16) of the 7 FFT/entropy features"),
        ("Classifier", "Dense(32, ReLU) -> Dense(2, SoftMax)"),
        ("Loss / batch size / max epochs",
         "Sparse categorical cross-entropy / 64 / 20 (early stopping, patience 4)"),
        ("Class imbalance", "balanced class weights"),
        ("Fox Optimization",
         "population 5, 4 iterations, fitness = validation loss after 3 epochs on 3000 training samples; "
         "search ranges: filters 8-32, units 16-64, attention dim 16-64, lr 1e-4 to 1e-2 (log), "
         f"dropout 0.1-0.5; runtime {fo['time']:.0f} s"),
        ("Total trainable parameters", cache[MAIN]["params"]),
    ],
    columns=["Parameter", "Value"],
).to_csv(f"{TAB}/table3_hyperparameter_configuration.csv", index=False)

print("report done")
print(pd.read_csv(f"{TAB}/table4_baseline_comparison.csv").to_string())
print(pd.read_csv(f"{TAB}/table5_ablation_study.csv").to_string())