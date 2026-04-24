# 1. Отримання вхідних даних із властивостями, заданими в Лр_1;
# 2. Модель вхідних даних із аномальними вимірами;
# 3. Очищення вхідних даних від аномальних вимірів. Спосіб виявлення аномалій та
# очищення обрати самостійно;

import numpy as np
import matplotlib.pyplot as plt


RNG = np.random.default_rng()
ANOMALY_PCT = 0.08



def main():
    N = 365


    print("============= Отримання вхідних даних =============")
    t, y_orig = generate_base_data(N)

    y_anom, true_mask = inject_anomalies(y_orig, ANOMALY_PCT)
    print(f"N={N}, anom_count={true_mask.sum()} ({ANOMALY_PCT*100:.0f}%)")
    print(f"mean_orig={y_orig.mean():.3f}, std_orig={y_orig.std():.3f}")



    # orig vs anom plot
    plt.plot(t, y_orig)
    plt.plot(t, y_anom)
    plt.show()


def generate_base_data(n: int):
    t = np.linspace(1, 365, n)
    a, b, c = 0.000033, -0.006952, 41.751589
    trend = a * t**2 + b * t + c
    df = 4
    noise = RNG.chisquare(df, n) - df
    noise *= 1.8
    y = trend + noise

    return t, y


def inject_anomalies(y: np.array, pct: float):
    n = len(y)
    y_anom = y.copy()
    n_anom = int(n * pct)
    idx = RNG.choice(n, n_anom, replace=False)
    sigma_glob = np.std(y)

    for i, ix in enumerate(idx):
        if i < n_anom // 2:
            sign = RNG.choice([-1, 1])
            y_anom[ix] += sign * RNG.uniform(4, 8) * sigma_glob
        else:
            seg = slice(ix, min(ix + 3, n))
            y_anom[seg] += RNG.choice([-1, 1]) * RNG.uniform(3, 6) * sigma_glob

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    return y_anom, mask



if __name__=="__main__":
    main()