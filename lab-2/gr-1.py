# 1. Отримання вхідних даних із властивостями, заданими в Лр_1;
# 2. Модель вхідних даних із аномальними вимірами;
# 3. Очищення вхідних даних від аномальних вимірів. Спосіб виявлення аномалій та
# очищення обрати самостійно;

import numpy as np
import matplotlib.pyplot as plt




def main():
    N = 365


    print("============= Отримання вхідних даних =============")
    t, trend, y_clean = generate_base_data(N)
    print(f"N={N}")

    y_noisy, anom_mask = inject_anomalies(y_clean)
    print(f"N={N}, anom_count={anom_mask.sum()}")

    y_interp, detected = clean_data(y_noisy, t)



    # orig vs anom plot
    plt.plot(t, y_clean)
    plt.plot(t, y_noisy)
    plt.plot(t, y_interp)
    plt.show()


def generate_base_data(n: int):
    t = np.linspace(1, 365, n)
    a, b, c = 0.000033, -0.006952, 41.751589
    trend = a * t**2 + b * t + c
    df = 5
    noise_raw = np.random.chisquare(df, n)
    noise = 1.2 * (noise_raw - df)
    y_clean = trend + noise

    return t, trend, y_clean


def inject_anomalies(y: np.array, rate: float = 0.07, mult: float = 4):
    n = len(y)
    y_noisy = y.copy()
    sigma = np.std(y)
    n_anom = int(n * rate)
    idx = np.random.choice(n, size=n_anom, replace=False)
    signs = np.random.choice([-1, 1], size=n_anom)
    amps = mult * sigma * (1 + np.random.uniform(0, 1, size=n_anom))
    y_noisy[idx] += signs * amps

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    return y_noisy, mask


def iqr_detector(y: np.ndarray, k: float = 2.2):
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    return (y < q1 - k * iqr) | (y > q3 + k * iqr)


def modified_zscore_detector(y: np.ndarray, threshold: float = 3.5):
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    if mad == 0:
        mad = np.std(y)
    mz = 0.6745 * (y - median) / mad
    return np.abs(mz) > threshold


def clean_data(y_noisy: np.ndarray, t: np.ndarray):
    mask_iqr = iqr_detector(y_noisy)
    mask_mz = modified_zscore_detector(y_noisy)
    detected = mask_iqr & mask_mz

    y_interp = y_noisy.copy()
    if detected.any():
        good_idx = np.where(~detected)[0]
        bad_idx = np.where(detected)[0]
        y_interp[bad_idx] = np.interp(t[bad_idx], t[good_idx], y_noisy[good_idx])
    return y_interp, detected


if __name__=="__main__":
    main()