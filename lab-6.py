import os
import numpy as np
from sklearn.preprocessing import RobustScaler


SEP = "=" * 67


OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "l5_nbu_exchange_rates_2y.csv")
IMG_PATH = os.path.join(OUTPUT_DIR, "l5_img.jpg")


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
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        X_train_3d=X_train_3d,
        X_val_3d=X_val_3d,
        X_test_3d=X_test_3d,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )





if __name__ == "__main__":
    main()