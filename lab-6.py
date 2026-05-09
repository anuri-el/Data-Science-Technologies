import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV_PATH = os.path.join(OUTPUT_DIR, "l5_nbu_exchange_rates_2y.csv")
# IMG_PATH = os.path.join(OUTPUT_DIR, "l5_img.jpg")

C = dict(
    sub="#8B949E", gold="#FFD700",
    true="#26C6DA", pred="#FF9800", anom="#EF5350",
    clean="#42A5F5", train="#4CAF50", val="#AB47BC",
)

N = 5000
A0 = 10.0 
A1 = 0.06 
NOISE_STD = 12.0 
ANOM_RATE = 0.10 
WINDOW = 50 
HORIZON = 1
TEST_FRAC = 0.20  
RNG = np.random.default_rng(42)

def main():
    print(f"\n{SEP}")
    t, y_trend, y_clean, y_noisy, anom_mask = generate_dataset()

    print(f"y = {A1} * t + {A0}")
    print(f"N = {N}")
    print(f"mu=0, sigma={NOISE_STD}")
    print(f"Anomalies: {anom_mask.sum()} points ({ANOM_RATE*100:.0f}%), uniform distribution")
    
    print(f"\nStatistics y_noisy:")
    print(f"min={y_noisy.min():.3f}  max={y_noisy.max():.3f}")
    print(f"mu = {y_noisy.mean():.3f} | sigma = {y_noisy.std():.3f}")
    snr = 20 * np.log10(y_clean.std() / NOISE_STD)
    print(f"SNR = {snr:.2f} dB")


    print(f"\n{SEP}")
    data = build_train_test(y_noisy, y_clean)
    print(f"Window (look-back) : {WINDOW} steps")
    print(f"Horizon : {HORIZON} steps")
    print(f"Train: {data['n_train']}  Val: {data['n_val']}  Test: {data['n_test']}")


    plot_dataset_overview(t, y_trend, y_clean, y_noisy, anom_mask, data["n_train"], data["n_val"], N, "l6_dataset_overview.png")
    plot_distributions(y_clean, y_trend, y_noisy, anom_mask, "l6_distributions.png")


def generate_dataset():
    t = np.arange(N, dtype=float)
    y_trend = A1 * t + A0

    noise = RNG.normal(loc=0.0, scale=NOISE_STD, size=N)
    y_clean = y_trend + noise

    y_noisy = y_clean.copy()
    anom_idx = RNG.choice(N, size=int(N * ANOM_RATE), replace=False)
    anom_mask = np.zeros(N, dtype=bool)
    anom_mask[anom_idx] = True
    y_range = y_clean.max() - y_clean.min()
    y_noisy[anom_idx] = RNG.uniform(
        y_clean.min() - 0.3 * y_range,
        y_clean.max() + 0.3 * y_range,
        size=len(anom_idx)
    )

    return t, y_trend, y_clean, y_noisy, anom_mask


def prepare_sequences(series: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.array(X), np.array(y).squeeze()


def build_train_test(y_noisy: np.ndarray, y_clean: np.ndarray):
    scaler = RobustScaler()
    y_sc = scaler.fit_transform(y_noisy.reshape(-1, 1)).ravel()

    y_sc    = np.clip(y_sc, -5, 5)
    y_cl_sc = scaler.transform(y_clean.reshape(-1, 1)).ravel()
    y_cl_sc = np.clip(y_cl_sc, -5, 5)

    X, y = prepare_sequences(y_sc, WINDOW, HORIZON)

    n_test = int(len(X) * TEST_FRAC)
    n_val = int(len(X) * 0.10)
    n_train = len(X) - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    X_train_3d = X_train[:, :, np.newaxis]
    X_val_3d = X_val[:, :, np.newaxis]
    X_test_3d = X_test[:, :, np.newaxis]

    return dict(
        scaler=scaler, y_sc=y_sc, y_cl_sc=y_cl_sc,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_train_3d=X_train_3d,
        X_val_3d=X_val_3d,
        X_test_3d=X_test_3d,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )



def plot_dataset_overview(t, y_trend, y_clean, y_noisy, anom_mask, n_tr, n_val, N, fname):
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(t, y_trend, color=C["true"], lw=2.2, ls="--", alpha=0.85, label="y = 0.012t + 5", zorder=4)
    ax.plot(t, y_clean, color=C["clean"], lw=0.8, alpha=0.45, label="sigma=20")
    ax.plot(t[~anom_mask], y_noisy[~anom_mask], color=C["sub"], lw=0.7, alpha=0.60, label="Noisy")
    ax.scatter(t[anom_mask], y_noisy[anom_mask], color=C["anom"], s=10, zorder=5, label=f"Anomalies ({anom_mask.sum()}, 10%)", alpha=0.8)
    
    ax.axvspan(0, n_tr, alpha=0.06, color=C["train"])
    ax.axvspan(n_tr, n_tr + n_val, alpha=0.06, color=C["val"])
    ax.axvspan(n_tr + n_val, N, alpha=0.06, color=C["gold"])
    ax.text(n_tr * 0.5, y_noisy.max() * 0.95, "TRAIN", ha="center")
    ax.text(n_tr + n_val * 0.5, y_noisy.max() * 0.95, "VAL", ha="center")
    ax.text(n_tr + n_val + (N - n_tr - n_val) * 0.5, y_noisy.max() * 0.95, "TEST", ha="center")
    
    ax.set_title(f"Dataset: N={N}, linear trend + N(0,{NOISE_STD}) + 10% anomalies")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(ncol=4)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_distributions(y_clean, y_trend, y_noisy, anom_mask, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    residuals = y_clean - y_trend
    axes[0].hist(residuals, bins=60, density=True, color=C["clean"], alpha=0.70, label="Empirical")
    xr = np.linspace(residuals.min(), residuals.max(), 300)
    axes[0].plot(xr, norm.pdf(xr, 0, NOISE_STD), color=C["true"], lw=2.0, label=f"N(0,{NOISE_STD})")
    axes[0].set_title("Noise Distribution (Normal)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    axes[1].hist(y_noisy[~anom_mask], bins=60, density=True, color=C["clean"], alpha=0.65, label="Normal")
    axes[1].hist(y_noisy[anom_mask], bins=30, density=True, color=C["anom"], alpha=0.65, label="Anomalies")
    axes[1].set_title("Normal vs Anomalies")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()