import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

N = 3650
a, b, c = 0.000033, -0.006952, 41.751589

def main():
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

    t_ext, y_ext_d = extrapolate(theta_d, t)
    _, y_ext_c = extrapolate(theta_c, t)
    n_ext = len(t_ext)
    print(f"прогноз на {n_ext} точок:")
    print(f"lsm_d: [{y_ext_d[0]:.4f} .. {y_ext_d[-1]:.4f}]")
    print(f"lsm_c: [{y_ext_c[0]:.4f} .. {y_ext_c[-1]:.4f}]")

    trend_ext = a * t_ext**2 + b *t_ext + c
    print(f"ideal: [{trend_ext[0]:.4f} .. {trend_ext[-1]:.4f}]")

    print(f"RMSE_d forecast: { rmse(trend_ext, y_ext_d):.4f}")
    print(f"RMSE_c forecast: { rmse(trend_ext, y_ext_c):.4f}")
    
    print("GROUP 2")

    y_noisy2, anom_mask2 = inject_anomalies(y_clean, rate=0.08)
    y_interp2, _ = clean_data(y_noisy2, t)

    ftype_c, finfo_c = choose_filter(y_clean, t)
    ftype_n, finfo_n = choose_filter(y_interp2, t)
    
    for tag, fi in [("clean", finfo_c), ("noisy/cleaned", finfo_n)]:
        print(f"[{tag}] speed: {fi['speed']:.6f}")
        print(f"accel: {fi['accel']:.6f} (threshold={fi['threshold']})")
        print(f"filter: {fi['type']}")
        print(f"alpha={fi['alpha']:.4f} beta={fi['beta']:.5f}" + (f"gamma={fi['gamma']:.6f}" if fi["gamma"] else ""))



    # orig vs anom plot
    plt.plot(t, y_clean)
    plt.plot(t, y_noisy)
    plt.plot(t, y_interp)
    plt.show()


def generate_base_data(n: int):
    t = np.linspace(1, 365, n)
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


def extrapolate(theta: np.ndarray, t_obs: np.ndarray, frac: float = 0.5):
    n_ext = int(len(t_obs) * frac)
    dt = t_obs[1] - t_obs[0]
    t_ext = t_obs[-1] + dt * np.arange(1, n_ext + 1)
    deg = len(theta) - 1
    PHI_ext = np.vander(t_ext, deg + 1, increasing=False)
    y_ext = PHI_ext @ theta
    return t_ext, y_ext


def estimate_trend_acceleration(y: np.ndarray, t: np.ndarray):
    p = np.polyfit(t, y, 2)
    speed = abs(p[1])
    accel = abs(p[0]) * 2
    return speed, accel


def alpha_beta_filter(z: np.ndarray, alpha: float, beta: float):
    n = len(z)
    x_hat = np.zeros(n)
    v_hat = np.zeros(n)
    dt = 1

    x_hat[0] = z[0]
    v_hat[0] = 0

    sigma_inn = np.std(np.diff(z)) if len(z) > 1 else 1
    clip_lim = 5 * sigma_inn
    for k in range(1, n):
        x_pred = x_hat[k - 1] + v_hat[k - 1] * dt
        innov = z[k] - x_pred
        innov = np.clip(innov, -clip_lim, clip_lim)
        x_hat[k] = x_pred + alpha * innov
        v_hat[k] = v_hat[k - 1] + (beta / dt) * innov
    
    return x_hat, v_hat


def alpha_beta_gamma_filter(z: np.ndarray, alpha: float, beta: float, gamma: float):
    n = len(z)
    x_hat = np.zeros(n)
    v_hat = np.zeros(n)
    a_hat = np.zeros(n)
    dt = 1

    x_hat[0] = z[0]
    v_hat[0] = 0
    a_hat[0] = 0

    sigma_inn = np.std(np.diff(z)) if len(z) > 1 else 1
    clip_lim = 5 * sigma_inn
    accel_lim = 3 * np.std(z) / max(n ** 2, 1)

    for k in range(1, n):
        dt2 = dt * dt
        x_pred = x_hat[k - 1] + v_hat[k - 1] * dt + 0.5 * a_hat[k - 1] * dt2
        v_pred = v_hat[k - 1] + a_hat[k - 1] * dt
        a_pred = a_hat[k - 1]

        innov = z[k] - x_pred
        innov = np.clip(innov, -clip_lim, clip_lim)
        
        x_hat[k] = x_pred + alpha * innov
        v_hat[k] = v_pred + (beta / dt) * innov

        raw_a = a_pred + (gamma / (0.5 * dt2)) * innov
        a_hat[k] = np.clip(raw_a, -accel_lim * 1e6, accel_lim * 1e6)
    
    return x_hat, v_hat, a_hat


def optimize_abg_params(z: np.ndarray, use_gamma: bool = True):
    n_val = len(z) // 5
    z_tr = z[:-n_val]
    z_val = z[-n_val:]
    
    if use_gamma:
        def cost_abg(alpha):
            if alpha <= 0 or alpha >= 1:
                return 1e9
            beta = alpha ** 2 / (2 - alpha + 1e-9)
            gamma = beta * alpha /2
            xf, _, _ = alpha_beta_gamma_filter(z_tr, alpha, beta, gamma)
            pred = xf[-1]
            return np.mean((z_val - pred) ** 2)
        
        res = minimize_scalar(cost_abg, bounds=(0.05, 0.95), method="bounded")
        alpha = float(np.clip(res.x, 0.05, 0.95))
        beta = alpha ** 2 / (2 - alpha)
        gamma = beta * alpha / 2
        return alpha, beta, gamma
    else:
        def cost_ab(alpha):
            if alpha <= 0 or alpha >= 1:
                return 1e9
            beta = alpha ** 2 / (2 - alpha + 1e-9)
            xf, _ = alpha_beta_filter(z_tr, alpha, beta)
            pred = xf[-1]
            return np.mean((z_val - pred) ** 2)
        
        res = minimize_scalar(cost_ab, bounds=(0.05, 0.95), method="bounded")
        alpha = float(np.clip(res.x, 0.05, 0.95))
        beta = alpha ** 2 / (2 - alpha)
        return alpha, beta, None


def choose_filter(y_interp: np.ndarray, t: np.ndarray):
    speed, accel = estimate_trend_acceleration(y_interp, t)
    threshold = 1e-4
    use_gamma = accel > threshold
    ftype = "a-b-g" if use_gamma else "a-b"
    alpha, beta, gamma = optimize_abg_params(y_interp, use_gamma=use_gamma)
    info = dict(type=ftype, alpha=alpha, beta=beta, gamma=gamma, speed=speed, accel=accel, threshold=threshold)
    return ftype, info


def print_quality_table(results: list[dict], best_deg: int):
    print(f"{'Deg':>4} {'AIC':>10} {'BIC':>10} {'R2':>8} {'RMSE':>8}")
    for r in results:
        star = "BEST" if r["deg"] == best_deg else ""
        print(f"{r['deg']:>4} {r['aic']:>10.2f} {r['bic']:>10.2f} {r['r2']:>8.5f} {r['rmse']:>8.4f} {star}")


if __name__=="__main__":
    main()