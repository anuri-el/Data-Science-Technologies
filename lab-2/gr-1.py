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
    tp = int((detected & anom_mask).sum())
    fp = int((detected & ~anom_mask).sum())
    fn = int((~detected & anom_mask).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    print("Cleaning (Ensemble IQR + Modified Z-score):")
    print(f"Detected anomalies: {detected.sum()}")
    print(f"True positives = {tp}, FP={fp}, FN={fn}, precision={prec:.3f}, recall={rec:.3f}")

    # dirty data
    best_d, results_d = setect_best_degree(t, y_noisy)
    print_quality_table(results_d, best_d["deg"])
    print(f"Best degree: {best_d["deg"]}, r2={best_d["r2"]:.5f}, RMSE={best_d["rmse"]:.4f}")

    # clean data
    best_c, results_c = setect_best_degree(t, y_interp)
    print_quality_table(results_c, best_c["deg"])
    print(f"Best degree: {best_c["deg"]}, r2={best_c["r2"]:.5f}, RMSE={best_c["rmse"]:.4f}")

    theta_d, y_lsm_d = lsm_fit(t, y_noisy, best_d["deg"])
    theta_c, y_lsm_c = lsm_fit(t, y_interp, best_c["deg"])
    print(f"theta_d (deg={best_d['deg']})" + ", ".join(f"{c:.4f}" for c in theta_d))
    print(f"theta_c (deg={best_c['deg']})" + ", ".join(f"{c:.4f}" for c in theta_c))

    p_check = np.polyfit(t, y_interp, best_c["deg"])
    diff = np.max(np.abs(p_check - theta_c))
    print(f"max|theta_lsm - theta_polyfit| = {diff:.2e} {'OK' if diff < 1e-6 else 'high diff'}")

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


def compute_aic_bic(y: np.ndarray, y_hat: np.ndarray, k: int):
    n = len(y)
    rss = np.sum((y - y_hat) ** 2)
    if rss <= 0:
        rss = 1e-12
    ll = -n / 2 * np.log(rss /n) - n / 2 * (1 + np.log(2 * np.pi))
    aic = 2 * (k + 1) - 2 * ll
    bic = np.log(n) * (k + 1) - 2 * ll
    return aic, bic


def r_squared(y: np.ndarray, y_hat: np.ndarray):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) **2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def rmse(y: np.ndarray, y_hat: np.ndarray):
    return np.sqrt(np.mean((y - y_hat) ** 2))


def setect_best_degree(t: np.ndarray, y: np.ndarray, max_deg: int = 8):
    results = []
    for deg in range(1, max_deg + 1):
        coeffs = np.polyfit(t, y, deg)
        y_hat = np.polyval(coeffs, t)
        aic, bic = compute_aic_bic(y, y_hat, deg)
        r2 = r_squared(y, y_hat)
        rms = rmse(y, y_hat)
        results.append(dict(deg=deg, aic=aic, bic=bic, r2=r2, rmse=rms, coeffs=coeffs))
    
    best = min(results, key=lambda x: x["bic"])
    return best, results


def lsm_fit(t: np.ndarray, y: np.ndarray, deg: int):
    n = len(t)
    PHI = np.vander(t, deg + 1, increasing=False)
    A = PHI.T @ PHI
    b = PHI.T @ y
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(PHI, y, rcond=None)[0]
    y_hat = PHI @ theta
    return theta, y_hat


def print_quality_table(results: list[dict], best_deg: int):
    print(f"{'Deg':>4} {'AIC':>10} {'BIC':>10} {'R2':>8} {'RMSE':>8}")
    for r in results:
        star = "BEST" if r["deg"] == best_deg else ""
        print(f"{r['deg']:>4} {r['aic']:>10.2f} {r['bic']:>10.2f} {r['r2']:>8.5f} {r['rmse']:>8.4f} {star}")


if __name__=="__main__":
    main()