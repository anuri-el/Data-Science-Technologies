import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar

N = 365
a, b, c = 0.03, -6.952, 41.75
CHI2_DF = 5
C = dict(
    raw = "#78909C",
    anomaly = "#EF5350",
    clean = "#42A5F5",
    trend = "#FFCA28",
    lsm_d = "#EF5350",
    lsm_c = "#66BB6A",
    extrap = "#FF7043",
    filt = "#AB47BC",
    true = "#26C6DA",
    bg = "#1A1D27",
    panel = "#161B22",
    grid = "#30363D",
    text = "#E6EDF3",
    subtext = "#8B949E",
)

def main():
    CSV_PATH = "./outputs/l2_results.csv"
    
    t, trend, y_clean = generate_base_data(N)
    
    print(f"N={N}, chi2(df={CHI2_DF})")
    print(f"y = {a} * t**2 + {b} * t + {c}")
    print(f"Mean={y_clean.mean():.4f} sigma={y_clean.std():.4f}")

    y_noisy, anom_mask = inject_anomalies(y_clean)
    print(f"N={N}, anom_count={anom_mask.sum()} ({anom_mask.mean()*100:.1f}%)")

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

    x_filt_c, _ =run_filter(y_clean, finfo_c)
    x_filt_n, _ =run_filter(y_interp2, finfo_n)

    for tag, y_src, xf, fi in [("clean", y_clean, x_filt_c, finfo_c), ("noisy/cleaned", y_interp2, x_filt_n, finfo_n)]:
        rms_f = rmse(trend, xf)
        rms_in = rmse(trend, y_src)
        print(f"[{tag}] {fi['type']}:")
        print(f"RMSE_in: {rms_in:.4f}")
        print(f"RMSE_filter: {rms_f:.4f}")
        print(f"Improvement: {(1 - rms_f / rms_in) * 100:.2f}%")

        div = np.max(np.abs(xf - y_src))
        print(f"Max div: {div:.4f} ({'no div' if div < 5*np.std(y_clean) else "div"})")

    print(f"{'Method':<20}| {"RMSE":>10}| {"R2":>10}|")
    rows = [("lsm - dirty data", rmse(trend, np.polyval(best_d["coeffs"], t)), r_squared(trend, np.polyval(best_d["coeffs"], t))), ("lsm - clean data", rmse(trend, np.polyval(best_c["coeffs"], t)), r_squared(trend, np.polyval(best_c["coeffs"], t))), (f"{finfo_c['type']} - clean", rmse(trend, x_filt_c), r_squared(trend, x_filt_c)), (f"{finfo_n['type']} - noisy/cleaned", rmse(trend, x_filt_n), r_squared(trend, x_filt_n))]
    for name, rms_v, r2_v in rows:
        print(f"{name:<20}| {rms_v:>10.4f}| {r2_v:>10.5f}|")
    
    df_out = pd.DataFrame({
        "t": t,
        "trend": trend,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "y_interp": y_interp,
        "lsm_dirty": np.polyval(best_d["coeffs"], t),
        "lsm_clean": np.polyval(best_c["coeffs"], t),
        "filter_clean": x_filt_c,
        "filter_noisy": x_filt_n,
    })

    df_out.to_csv(CSV_PATH, index=False)
    print(f"CSV saved to {CSV_PATH}")

    plot_input_data(t, trend, y_clean, y_noisy, anom_mask, "./outputs/l2_input_data.png")
    plot_noise_histogram(trend, y_clean, "./outputs/l2_noise_histogram.png")
    plot_cleaning_result(t, y_noisy, y_interp, anom_mask, "./outputs/l2_cleaning_results.png")
    plot_bic_comparison(results_d, results_c, best_d, best_c, "./outputs/l2_bic_comparison.png")
    plot_lsm_regression_and_forecast(t, trend, y_noisy, y_interp, best_d, best_c, t_ext, y_ext_d, y_ext_c, "./outputs/l2_lsm_regression_and_forecast.png")
    plot_filter_on_clean_data(t, y_clean, trend, finfo_c, x_filt_c, "./outputs/l2_filter_on_clean_data.png")
    plot_filter_on_noisy_data(t, y_noisy2, y_interp2, trend, finfo_n, x_filt_n, "./outputs/l2_filter_on_noisy_data")
    plot_rmse_comparison(t, trend, best_d, best_c, x_filt_c, x_filt_n, "./outputs/l2_rmse_comparison.png")


def generate_base_data(n: int):
    t = np.arange(n, dtype=float)
    trend = a * t**2 + b * t + c

    noise_raw = np.random.chisquare(df=CHI2_DF, size=n)
    noise = 1.2 * (noise_raw - CHI2_DF)
    y_clean = trend + noise

    return t, trend, y_clean


def inject_anomalies(y: np.array, rate: float = 0.07, mult: float = 10):
    n = len(y)
    y_noisy = y.copy()

    diffs = np.diff(y)
    sigma = np.median(np.abs(diffs - np.median(diffs))) / 0.6745 / np.sqrt(2)

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
    coeffs_est = np.polyfit(t, y_noisy, 2)
    trend_est  = np.polyval(coeffs_est, t)
    residuals  = y_noisy - trend_est

    mask_iqr = iqr_detector(residuals)
    mask_mz  = modified_zscore_detector(residuals)
    detected = mask_iqr & mask_mz

    y_interp = y_noisy.copy()
    if detected.any():
        good_idx = np.where(~detected)[0]
        bad_idx  = np.where(detected)[0]
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


def run_filter(y: np.ndarray, finfo: dict):
    alpha, beta, gamma = finfo["alpha"], finfo["beta"], finfo["gamma"]

    if finfo["type"] == "a-b-g":
        x_filt, v_filt, a_filt = alpha_beta_gamma_filter(y, alpha, beta, gamma)
        extra = dict(v=v_filt, a=a_filt)
    else:
        x_filt, v_filt = alpha_beta_filter(y, alpha, beta)
        extra = dict(v=v_filt)

    return x_filt, extra


def print_quality_table(results: list[dict], best_deg: int):
    print(f"{'Deg':>4} {'AIC':>10} {'BIC':>10} {'R2':>8} {'RMSE':>8}")
    for r in results:
        star = "BEST" if r["deg"] == best_deg else ""
        print(f"{r['deg']:>4} {r['aic']:>10.2f} {r['bic']:>10.2f} {r['r2']:>8.5f} {r['rmse']:>8.4f} {star}")


def ax_style(ax, title=""):
    ax.set_facecolor(C["bg"])
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=5)
    ax.margins(x=0.01)
    
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, facecolor= C["bg"], edgecolor="#444444", labelcolor="#FFFFFF", loc="upper left")
    ax.grid(linestyle="--", color="#555566", alpha=0.15, linewidth=0.6)


def plot_input_data(t, trend, y_clean, y_noisy, anom_mask, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])
    
    ax.plot(t, y_clean, color=C["clean"], lw=1, alpha=0.6, label="clean")
    ax.plot(t, trend, color=C["true"], lw=1.8, ls="--", label="ideal trend")
    ax.scatter(t[anom_mask], y_noisy[anom_mask], color=C["anomaly"], s=20, zorder=5, label=f"anomalies({anom_mask.sum()})")
    
    title = "Input data and anomalies"
    ax_style(ax, title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_noise_histogram(trend, y_clean, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    noise_emp = y_clean - trend
    xr = np.linspace(noise_emp.min(), noise_emp.max(), 300)
    ax.hist(noise_emp, bins=30, density=True, color=C["clean"], alpha=0.65, label="Empirical noise")
    kde = stats.gaussian_kde(noise_emp)
    ax.plot(xr, kde(xr), color=C["true"], lw=1.8, label="KDE")
    
    title = "Noise Histograms"
    ax_style(ax, title)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_cleaning_result(t, y_noisy, y_interp, anom_mask, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    ax.plot(t, y_noisy, color=C["raw"], lw=0.8, alpha=0.55, label="Noisy")
    ax.plot(t, y_interp, color=C["clean"], lw=1.4, alpha=0.9, label="Interp")
    ax.scatter(t[anom_mask], y_noisy[anom_mask], color=C["anomaly"], s=25, zorder=5, label=f"Anomalies({anom_mask.sum()})")
    ax.scatter(t[anom_mask], y_interp[anom_mask], color=C["trend"], s=25, zorder=5, label=f"Interpolated")
    
    title = "Ensemble IQR + Modified Z-score"
    ax_style(ax, title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_bic_comparison(results_d, results_c, best_d, best_c, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    degs_d = [r["deg"] for r in results_d]
    bics_d = [r["bic"] for r in results_d]
    degs_c = [r["deg"] for r in results_c]
    bics_c = [r["bic"] for r in results_c]

    ax.plot(degs_d, bics_d, "o--", color=C["lsm_d"], lw=1.3, ms=5, label="Noisy")
    ax.plot(degs_c, bics_c, "s-", color=C["lsm_c"], lw=1.3, ms=5, label="Clean")
    ax.axvline(best_d["deg"], color=C["lsm_d"], lw=1.2, ls=":", alpha=0.8, label=f"Best (бруд) deg={best_d['deg']}")
    ax.axvline(best_c["deg"], color=C["lsm_c"], lw=1.2, ls=":", alpha=0.8, label=f"Best (чист) deg={best_c['deg']}")
    
    title = "BIC"
    ax_style(ax, title)
    ax.set_xlabel("Deg")
    ax.set_ylabel("BIC")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_lsm_regression_and_forecast(t, trend, y_noisy, y_interp, best_d, best_c, t_ext, y_ext_d, y_ext_c, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    ax.plot(t, trend, color=C["true"], lw=2, ls="--", alpha=0.7, label="Ideal trend")
    ax.plot(t, y_noisy, color=C["raw"], lw=0.9, alpha=0.4, label="Noisy")
    ax.plot(t, y_interp, color=C["clean"], lw=0.9, alpha=0.6, label="Interp")
    
    y_lsm_d = np.polyval(best_d["coeffs"], t)
    y_lsm_c = np.polyval(best_c["coeffs"], t)
    ax.plot(t, y_lsm_d, color=C["lsm_d"], lw=2.0, label=f"lsm_d deg={best_d['deg']} R2={best_d['r2']:.4f}")
    ax.plot(t, y_lsm_c, color=C["lsm_c"], lw=2.0, label=f"lsm_c deg={best_c['deg']} R2={best_c['r2']:.4f}")

    ax.axvline(t[-1], color=C["text"], lw=1.0, ls=":", alpha=0.5)
    y_min, y_max = y_noisy.min() * 0.95, y_noisy.max() * 1.05
    ax.fill_betweenx([y_min, y_max], t[-1], t_ext[-1], color=C["extrap"], alpha=0.06)
    ax.plot(t_ext, y_ext_d, "--", color=C["lsm_d"], lw=1.8, alpha=0.85, label=f"lsm_d forecast +{len(t_ext)} points")
    ax.plot(t_ext, y_ext_c, "--", color=C["lsm_c"], lw=1.8, alpha=0.85, label=f"lsm_c forecast +{len(t_ext)} points")
    
    trend_ext = a * t_ext**2 + b * t_ext + c
    ax.plot(t_ext, trend_ext, "--", color=C["true"], lw=1.4, alpha=0.5)

    title = "LSM Regression and Forecast"
    ax_style(ax, title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_filter_on_clean_data(t, y_clean, trend, finfo, x_filt, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=C["bg"])

    ax = axes[0]
    ax.plot(t, y_clean, color=C["clean"], lw=0.8, alpha=0.5, label="clean")
    ax.plot(t, trend, color=C["true"], lw=1.8, ls="--", label="ideal trend")
    ax.plot(t, x_filt, color=C["filt"], lw=1.8, label=f"{finfo['type']} alpha={finfo['alpha']:.3f} beta={finfo['beta']:.4f}" + (f" gamma={finfo['gamma']:.5f}" if finfo.get("gamma") else ""))

    title = f"Group 2: {finfo['type']} filter (clean)"
    ax_style(ax, title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    ax = axes[1]
    resid_filt = y_clean - x_filt
    ax.plot(t, resid_filt, color=C["filt"], lw=0.9, alpha=0.8, label="resid")
    ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
    ax.axhline(np.std(resid_filt), color=C["anomaly"], lw=1, ls=":", alpha=0.8, label="+sigma")
    ax.axhline(-np.std(resid_filt), color=C["anomaly"], lw=1, ls=":", alpha=0.8, label="-sigma")
    
    title = f"Group 2: {finfo['type']} filter (clean)"
    ax_style(ax, title)

    ax.set_title("Filter residues (clean)", color=C["text"])
    ax.set_xlabel("Time", color=C["text"])
    ax.set_ylabel("Residues", color=C["text"])
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.grid(True, alpha=0.3, color=C["grid"])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_filter_on_noisy_data(t, y_noisy2, y_interp2, trend, finfo2, x_filt2, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    ax.plot(t, y_noisy2, color=C["raw"], lw=0.8, alpha=0.45, label="Noisy (anomalies)")
    ax.plot(t, y_interp2, color=C["clean"], lw=0.9, alpha=0.55, label="Cleaned")
    ax.plot(t, trend, color=C["true"], lw=1.8, ls="--", label="Ideal")
    ax.plot(t, x_filt2, color=C["filt"], lw=1.8, label=f"{finfo2['type']} filter (after cleaning)")

    title = f"Group 2: {finfo2['type']} filter - noisy/cleaned data"
    ax_style(ax, title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_rmse_comparison(t, trend, best_d, best_c, x_filt, x_filt2, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])

    labels = ["lsm_d", "lsm_c", "filt_c", "filt_n"]
    rmses = [
        np.sqrt(np.mean((trend - np.polyval(best_d["coeffs"], t))**2)),
        np.sqrt(np.mean((trend - np.polyval(best_c["coeffs"], t))**2)),
        np.sqrt(np.mean((trend - x_filt)**2)),
        np.sqrt(np.mean((trend - x_filt2)**2)),
    ]
    colors_b = [C["lsm_d"], C["lsm_c"], C["filt"], C["filt"]]
    
    bars = ax.bar(labels, rmses, color=colors_b, alpha=0.8, edgecolor=C["grid"])
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", va="bottom", color=C["text"], fontsize=7.5)
        
    title = f"RMSE"
    ax_style(ax, title)
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


if __name__=="__main__":
    main()