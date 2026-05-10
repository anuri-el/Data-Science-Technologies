import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, probplot

SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

EPOCHS = 50
BATCH = 256
LR = 5e-4
PATIENCE = 10


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
    data = build_train_test(y_noisy, y_clean, y_trend)
    print(f"Window (look-back) : {WINDOW} steps")
    print(f"Horizon : {HORIZON} steps")
    print(f"Train: {data['n_train']}  Val: {data['n_val']}  Test: {data['n_test']}")


    print(f"\n{SEP}")
    results = run_all_architectures(data)

    print(f"\n  Epochs max={EPOCHS}  Batch={BATCH}  LR={LR}  EarlyStopping(patience={PATIENCE})")
    print(f"\n  {'Architecture':<14} {'Params':>8} {'Epochs':>7} {'RMSE':>9} {'MAE':>9} {'R2':>8} {'MAPE%':>8} {'Time,s':>7}")
    for result in results:
        print(f"  {result['cfg']['name']:<14} {result['n_params']:>8,} {result['epochs']:>7} {result['rmse']:>9.4f} {result['mae']:>9.4f} {result['r2']:>8.5f} {result['mape']:>8.3f} {result['t_train']:>7.2f}")


    n_tr = data["n_train"]
    n_val = data["n_val"]

    t_test_start = WINDOW + n_tr + n_val
    t_test_arr = t[t_test_start : t_test_start + data["n_test"]]

    best_idx = int(np.argmax([r["r2"] for r in results]))
    best_r = results[best_idx]
    worst_r = results[int(np.argmin([r["r2"] for r in results]))]

    plot_dataset_overview(t, y_trend, y_clean, y_noisy, anom_mask, data["n_train"], data["n_val"], N, "l6_dataset_overview.png")
    plot_distributions(y_clean, y_trend, y_noisy, anom_mask, "l6_distributions.png")
    plot_test_predictions(t_test_arr, best_r, worst_r, "l6_test_predictions.png")
    plot_training_curves(results, "l6_training_curves.png")
    plot_metrics_comparison(results, "l6_metrics_comparison.png")
    plot_residuals_analysis(t_test_arr, best_r, "l6_residuals_analysis.png")


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


def build_train_test(y_noisy: np.ndarray, y_clean: np.ndarray, y_trend: np.ndarray):
    total_raw = N - WINDOW - HORIZON + 1
    n_test = int(total_raw * TEST_FRAC)
    n_val = int(total_raw * 0.10)
    n_train = total_raw - n_test - n_val
    train_end = n_train + WINDOW 
    
    scaler = RobustScaler()
    scaler.fit(y_clean[:train_end].reshape(-1, 1))
 
    X_sc_noisy = scaler.transform(y_noisy.reshape(-1, 1)).ravel()
    y_sc_clean = scaler.transform(y_clean.reshape(-1, 1)).ravel()

    X_list, y_list = [], []
    w_means, w_stds = [], []

    for i in range(total_raw):
        window = X_sc_noisy[i : i + WINDOW]
        target = y_sc_clean[i + WINDOW : i + WINDOW + HORIZON]
 
        w_mean = window.mean()
        w_std = window.std() + 1e-8
 
        X_list.append((window - w_mean) / w_std)
        y_list.append((target - w_mean) / w_std)
        w_means.append(w_mean)
        w_stds.append(w_std)
 
    X_all = np.array(X_list)
    y_all = np.array(y_list).squeeze()
    w_means = np.array(w_means)
    w_stds = np.array(w_stds)
 
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
    X_test, y_test = X_all[n_train+n_val:], y_all[n_train+n_val:]
 
    wm_test = w_means[n_train+n_val:]
    ws_test = w_stds[ n_train+n_val:]
 
    X_train_3d = X_train[:, :, np.newaxis]
    X_val_3d = X_val[:, :, np.newaxis]
    X_test_3d = X_test[:, :, np.newaxis]

    return dict(
        scaler=scaler,
        y_trend=y_trend,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        wm_test=wm_test, ws_test=ws_test,
        X_train_3d=X_train_3d,
        X_val_3d=X_val_3d,
        X_test_3d=X_test_3d,
        n_train=n_train, n_val=n_val, n_test=len(X_test),
    )


ARCHITECTURES = [
    dict(name="MLP_S", label="MLP small (32-16-1)", use_3d=False, params="~1.6K"),
    dict(name="MLP_L", label="MLP large (128-64-32-16-1)", use_3d=False, params="~15K"),
    dict(name="LSTM_S", label="LSTM small (32)-16-1", use_3d=True, params="~9.4K"),
    dict(name="LSTM_M", label="LSTM стек (64)-(32)-16-1", use_3d=True, params="~37K"),
    dict(name="GRU", label="GRU стек (64)-(32)-16-1", use_3d=True, params="~28K"),
    dict(name="CONV1D", label="Conv1D+LSTM (64)-(32)-32-16", use_3d=True, params="~22K"),
]


def build_model(arch: str, input_shape_2d: tuple, input_shape_3d: tuple):
    inp_2d = keras.Input(shape=input_shape_2d, name="input_2d")
    inp_3d = keras.Input(shape=input_shape_3d, name="input_3d")

    reg = regularizers.l2(1e-4)

    if arch == "MLP_S":
        x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(inp_2d)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_2d, outputs=out, name=arch)

    elif arch == "MLP_M":
        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(inp_2d)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_2d, outputs=out, name=arch)

    elif arch == "MLP_L":
        x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(inp_2d)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.20)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_2d, outputs=out, name=arch)

    elif arch == "MLP_DEEP":
        x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(inp_2d)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        for units in [128, 64, 32, 16]:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_2d, outputs=out, name=arch)

    elif arch == "LSTM_S":
        x = layers.LSTM(32, return_sequences=False)(inp_3d)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_3d, outputs=out, name=arch)

    elif arch == "LSTM_M":
        x = layers.LSTM(64, return_sequences=True)(inp_3d)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.10)(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_3d, outputs=out, name=arch)

    elif arch == "GRU":
        x = layers.GRU(64, return_sequences=True)(inp_3d)
        x = layers.GRU(32)(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_3d, outputs=out, name=arch)

    elif arch == "CONV1D":
        x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inp_3d)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.LSTM(32)(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return keras.Model(inputs=inp_3d, outputs=out, name=arch)

    else:
        raise ValueError(f"Невідома архітектура: {arch}")


def train_model(model: keras.Model, data: dict, use_3d: bool):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss="mse", metrics=["mae"])

    Xtr = data["X_train_3d"] if use_3d else data["X_train"]
    Xv = data["X_val_3d"] if use_3d else data["X_val"]

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0),
    ]

    t0 = time.perf_counter()
    hist = model.fit(
        Xtr, data["y_train"],
        validation_data=(Xv, data["y_val"]),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=cb_list, verbose=0,
    )
    elapsed = time.perf_counter() - t0
    return hist, elapsed


def evaluate_model(model: keras.Model, data: dict, use_3d: bool):
    Xte = data["X_test_3d"] if use_3d else data["X_test"]
    y_pred_sc = model.predict(Xte, verbose=0).ravel()
 
    wm = data["wm_test"]
    ws = data["ws_test"]
    y_pred_gsc = y_pred_sc * ws + wm
    y_true_gsc = data["y_test"] * ws + wm
 
    scaler = data["scaler"]
    y_pred = scaler.inverse_transform(y_pred_gsc.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_true_gsc.reshape(-1, 1)).ravel()
 
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, mape=mape,
                y_pred=y_pred, y_true=y_true, y_pred_sc=y_pred_sc)


def run_all_architectures(data: dict):
    inp_2d = (WINDOW,)
    inp_3d = (WINDOW, 1)
    results = []

    for cfg in ARCHITECTURES:
        model = build_model(cfg["name"], inp_2d, inp_3d)
        n_params = model.count_params()

        hist, t_train = train_model(model, data, cfg["use_3d"])
        metrics = evaluate_model(model, data, cfg["use_3d"])
        epochs_done = len(hist.history["loss"])

        results.append(dict(
            cfg=cfg, model=model, hist=hist,
            n_params=n_params, epochs=epochs_done,
            t_train=t_train, **metrics,
        ))
        keras.backend.clear_session()

    return results


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


def plot_test_predictions(t_test_arr, best_r, worst_r, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    n_show = min(400, len(t_test_arr))
    ax1.plot(t_test_arr[:n_show], best_r["y_true"][:n_show], color=C["true"], lw=1.5, alpha=0.85, label="Actual (test)", zorder=4)
    ax1.plot(t_test_arr[:n_show], best_r["y_pred"][:n_show],
             color=C["pred"], lw=1.8, ls="--",
             label=f"Prediction {best_r['cfg']['name']} (R2={best_r['r2']:.4f})", zorder=5)
    ax1.plot(t_test_arr[:n_show], worst_r["y_pred"][:n_show],
             color=C["anom"], lw=1.2, ls=":", alpha=0.7,
             label=f"Prediction {worst_r['cfg']['name']} (R2={worst_r['r2']:.4f})")
    ax1.fill_between(t_test_arr[:n_show],
                     best_r["y_true"][:n_show] - NOISE_STD,
                     best_r["y_true"][:n_show] + NOISE_STD,
                     alpha=0.08, color=C["true"], label="±sigma noise")
    ax1.set_title(f"Test Set Predictions")
    ax1.set_xlabel("Time, t")
    ax1.set_ylabel("Value y")
    ax1.legend()
    
    lim = [min(best_r["y_true"]), max(best_r["y_true"])]
    ax2.scatter(best_r["y_true"], best_r["y_pred"], c=C["pred"], s=4, alpha=0.40)
    ax2.plot(lim, lim, color=C["true"], lw=2, ls="--", label="Perfect prediction")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_title(f"Scatter: Actual vs Prediction ({best_r['cfg']['name']})")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Prediction")
    ax2.legend()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_training_curves(results, fname):
    n_cols = 3
    n_archs = len(results)
    n_rows = (n_archs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_archs > 1 else [axes]
    
    for i, r in enumerate(results):
        ax = axes[i]
        hist = r["hist"]
        ep = range(1, len(hist.history["loss"]) + 1)
        ax.plot(ep, hist.history["loss"], color=C["train"], lw=1.5, label="Train Loss")
        ax.plot(ep, hist.history["val_loss"], color=C["val"], lw=1.5, ls="--", label="Val Loss")
        best_ep = int(np.argmin(hist.history["val_loss"])) + 1
        ax.axvline(best_ep, color=C["gold"], lw=1.2, ls=":", alpha=0.8, label=f"Best ep={best_ep}")
        ax.set_title(f"{r['cfg']['name']}  R2={r['r2']:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7)
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_metrics_comparison(results, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names_a = [r["cfg"]["name"] for r in results]
    
    rmses_a = [r["rmse"] for r in results]
    bars1 = axes[0, 0].bar(names_a, rmses_a, color=C.get("c0", "#2196F3"), alpha=0.85)
    axes[0, 0].axhline(NOISE_STD, color=C["anom"], ls="--", lw=1.5, alpha=0.7, label=f"Noise sigma={NOISE_STD}")
    axes[0, 0].set_title("RMSE by Architecture")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].legend()
    
    r2s = [r["r2"] for r in results]
    bars2 = axes[0, 1].bar(names_a, r2s, color=C.get("c1", "#4CAF50"), alpha=0.85)
    axes[0, 1].axhline(1.0, color=C["true"], ls="--", lw=1.2, alpha=0.5, label="R2=1 (ideal)")
    axes[0, 1].set_title("R2 by Architecture")
    axes[0, 1].set_ylabel("R2")
    axes[0, 1].legend()
    
    maes_a = [r["mae"] for r in results]
    bars3 = axes[1, 0].bar(names_a, maes_a, color=C.get("c2", "#FF9800"), alpha=0.85)
    axes[1, 0].set_title("MAE by Architecture")
    axes[1, 0].set_ylabel("MAE")
    
    times_a = [r["t_train"] for r in results]
    bars4 = axes[1, 1].bar(names_a, times_a, color=C.get("c3", "#9C27B0"), alpha=0.85)
    axes[1, 1].set_title("Training Time by Architecture")
    axes[1, 1].set_ylabel("Seconds")
    
    for bars in [bars1, bars2, bars3, bars4]:
        for b, v in zip(bars, [b.get_height() for b in bars]):
            ax = bars[0].axes
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.3f}" if isinstance(v, float) else f"{v:.1f}s", ha="center", va="bottom")
    
    for ax in axes.flatten():
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_residuals_analysis(t_test_arr, best_r, fname):
    resid = best_r["y_true"] - best_r["y_pred"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(t_test_arr, resid, color=C["pred"], lw=0.9, alpha=0.75)
    axes[0, 0].axhline(0, color=C["true"], lw=1.5, ls="--")
    axes[0, 0].fill_between(t_test_arr, -NOISE_STD, NOISE_STD,
                            alpha=0.10, color=C["true"], label="+-sigma noise")
    axes[0, 0].axhline(NOISE_STD, color=C["anom"], lw=1.0, ls=":", alpha=0.7)
    axes[0, 0].axhline(-NOISE_STD, color=C["anom"], lw=1.0, ls=":", alpha=0.7)
    axes[0, 0].set_title(f"Prediction Residuals: mu={resid.mean():.3f}  sigma={resid.std():.3f}")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("y_true - y_pred")
    axes[0, 0].legend()
    
    axes[0, 1].hist(resid, bins=50, density=True, color=C["pred"], alpha=0.75)
    xr2 = np.linspace(resid.min(), resid.max(), 300)
    axes[0, 1].plot(xr2, norm.pdf(xr2, resid.mean(), resid.std()),
                    color=C["true"], lw=2.0,
                    label=f"N(mu={resid.mean():.2f}, sigma={resid.std():.2f})")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    
    (osm, osr), (slope, intercept, r_qq) = probplot(resid, dist="norm")
    axes[1, 0].scatter(osm, osr, c=C["pred"], s=4, alpha=0.4)
    axes[1, 0].plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept], color=C["true"], lw=2, label=f"r={r_qq:.4f}")
    axes[1, 0].set_title("Residual Q-Q Plot")
    axes[1, 0].set_xlabel("Theoretical quantiles")
    axes[1, 0].set_ylabel("Sample quantiles")
    axes[1, 0].legend()
    
    zoom = 60
    tx = t_test_arr[:zoom]
    axes[1, 1].plot(tx, best_r["y_true"][:zoom], "o-", color=C["true"], lw=1.8, ms=5, label="Actual")
    axes[1, 1].plot(tx, best_r["y_pred"][:zoom], "s--", color=C["pred"], lw=1.8, ms=5, label=f"Prediction ({best_r['cfg']['name']})")
    for i in range(zoom):
        axes[1, 1].plot([tx[i], tx[i]], [best_r["y_true"][i], best_r["y_pred"][i]], color=C["anom"], lw=0.8, alpha=0.5)
    axes[1, 1].set_title(f"Detailed Prediction (first {zoom} points)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].legend()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()