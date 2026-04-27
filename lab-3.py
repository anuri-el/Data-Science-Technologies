import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


BANKS = [
    "ПриватБанк",
    "Ощадбанк",
    "Ukrsibbank",
    "Raiffeisen",
    "monobank",
    "ПУМБ",
    "Укргазбанк",
    "OTP Bank",
]

CRITERIA_NAMES = [
    "interest rate, %",     #min
    "term, months",         #max
    "monthly payment, UAH", #min
    "overpayment, UAH",     #min
    "down payment, %",      #min
    "approval speed, h",    #min
    "prepayment penalty, %",#min
    "insurance (0/1)",      #min
    "bank rating",          #max
    "commission fee, UAH",  #min
]

N_BANKS    = len(BANKS)
N_CRITERIA = len(CRITERIA_NAMES)

RAW_DATA = np.array([
    # C1    C2    C3       C4        C5   C6   C7   C8   C9   C10
    [13.0,  60,  21_932,  355_907,  20,   2,   2,   1,   9,   80 ],  # ПриватБанк
    [15.5,  84,  18_790,  632_742,  20,  24,   0,   1,   10,  50 ],  # Ощадбанк
    [16.8,  72,  20_100,  487_200,  15,   4,   3,   1,   8,   120],  # Ukrsibbank
    [14.5,  60,  22_300,  378_000,  25,   1,   1,   0,   9,   90 ],  # Raiffeisen
    [17.9,  48,  28_500,  408_000,  10,   0,   0,   0,   7,   0  ],  # monobank
    [15.0,  60,  22_650,  399_000,  20,   3,   2,   1,   8,   70 ],  # ПУМБ
    [12.5,  96,  13_800,  364_800,  20,  48,   1,   1,   8,   60 ],  # Укргазбанк
    [14.9,  72,  19_200,  422_400,  15,   6,   1,   0,   8,   100],  # OTP Bank
], dtype=float)

DIRECTIONS = np.array([-1, +1, -1, -1, -1, -1, -1, -1, +1, -1])

AHP_MATRIX = np.array([
    # C1    C2    C3    C4    C5    C6    C7    C8    C9    C10
    [1,    3,    1/2,  1/3,  2,    5,    7,    5,    1/2,  3  ],  # C1 interest rate
    [1/3,  1,    1/4,  1/5,  1,    3,    5,    3,    1/4,  2  ],  # C2 term
    [2,    4,    1,    1/2,  3,    6,    8,    6,    1,    4  ],  # C3 monthly payment
    [3,    5,    2,    1,    4,    7,    9,    7,    2,    5  ],  # C4 overpayment
    [1/2,  1,    1/3,  1/4,  1,    4,    6,    4,    1/3,  2  ],  # C5 down payment
    [1/5,  1/3,  1/6,  1/7,  1/4,  1,    2,    1,    1/6,  1/2],  # C6 approval speed
    [1/7,  1/5,  1/8,  1/9,  1/6,  1/2,  1,    1/2,  1/8,  1/4],  # C7 prepayment penalty
    [1/5,  1/3,  1/6,  1/7,  1/4,  1,    2,    1,    1/6,  1/2],  # C8 insurance
    [2,    4,    1,    1/2,  3,    6,    8,    6,    1,    4  ],  # C9 bank rating
    [1/3,  1/2,  1/4,  1/5,  1/2,  2,    4,    2,    1/4,  1  ],  # C10 commission fee
], dtype=float)
 
RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

C = dict(
    bg = "#1A1D27",
    panel = "#161B22",
    grid = "#30363D",
    text = "#E6EDF3",
    subtext = "#8B949E",
    b1="#2196F3",
    b2="#4CAF50", 
    b3="#FF9800", 
    b4="#E91E63",
    b5="#9C27B0", 
    b6="#00BCD4", 
    b7="#FF5722", 
    b8="#8BC34A",
    gold="#FFD700", 
    best="#66BB6A", 
    worst="#EF5350",
    accent="#F48024",
)

BANK_COLORS = [C["b1"], C["b2"], C["b3"], C["b4"], C["b5"], C["b6"], C["b7"], C["b8"]]

SEP = "=" * 67

def main():
    CAR_PRICE = 1_200_000
    LOAN_AMOUNT = 960_000

    print("Level II: Multi-criteria evaluation of loan offers")
    print(f"Car price: {CAR_PRICE:>10} UAH")
    print(f"Loan amount: {LOAN_AMOUNT:>10} UAH")

    for i, bank in enumerate(BANKS):
        mp = monthly_payment(RAW_DATA[i, 0], int(RAW_DATA[i, 1]), LOAN_AMOUNT)
        print(f"{bank:<11}: {mp:>8,.0f} UAH/month (raw: {RAW_DATA[i,2]:>8,.0f} UAH/month)")
    
    print(f"\n{SEP}")
    print("AHP")
    weights, lambda_max, cr = ahp_weights(AHP_MATRIX)
    print(f"lambda_max = {lambda_max:.4f} | CR = {cr:.4f} {"OK (CR < 0.1)" if cr < 0.1 else "! CR >= 0.1"}")
    print(f"{'Criteria':<25} {'Weight':>8} {'%':>6}")
    for name, w in zip(CRITERIA_NAMES, weights):
        bar = "=" * int(w * 100)
        print(f"{name:<25} {w:>8.4f} {w*100:>5.1f} {bar}")

    norm_mm = normilize_minmax(RAW_DATA, DIRECTIONS)
    
    print(f"\n{SEP}")
    print("Weighted Sum Model")
    scores_wsm = wsm(norm_mm, weights)
    rank_wsm = rankdata(-scores_wsm).astype(int)
    print(f"{'Bank':<15} {'WSM Score':>10} {'Rank':>5}")
    for i in np.argsort(-scores_wsm):
        marker = "BEST" if rank_wsm[i] == 1 else ""
        print(f"{BANKS[i]:<15} {scores_wsm[i]:>10.5f} {rank_wsm[i]:>5} {marker}")

    print(f"\n{SEP}")
    print("Weighted Product Model")
    scores_wpm = wpm(norm_mm, weights)
    rank_wpm = rankdata(-scores_wpm).astype(int)
    print(f"{'Bank':<15} {'WPM Score':>10} {'Rank':>5}")
    for i in np.argsort(-scores_wpm):
        marker = "BEST" if rank_wpm[i] == 1 else ""
        print(f"{BANKS[i]:<15} {scores_wpm[i]:>10.5f} {rank_wpm[i]:>5} {marker}")
    
    print(f"\n{SEP}")
    print("TOPSIS")
    scores_topsis, d_plus, d_minus = topsis(RAW_DATA, weights, DIRECTIONS)
    rank_topsis = rankdata(-scores_topsis).astype(int)
    print(f"{'Bank':<15} {'D+':>8} {'D-':>8} {'C*':>8} {'Rank':>5}")
    for i in np.argsort(-scores_topsis):
        marker = "BEST" if rank_topsis[i] == 1 else ""
        print(f"{BANKS[i]:<15} {d_plus[i]:>8.4f} {d_minus[i]:>8.4f} {scores_topsis[i]:>8.4f} {rank_topsis[i]:>5} {marker}")
    


    print(f"\n{SEP}")
    # plot_ahp_weights(weights, cr, "./outputs/l3_ahp_weights.png")
    # plot_normilized_matrix(norm_mm, "./outputs/l3_normilized_matrix.png")
    # plot_wsm_scores(scores_wsm, "./outputs/l3_wsm_scores.png")
    # plot_wpm_scores(scores_wpm, "./outputs/l3_wpm_scores.png")
    plot_topsis_distances(d_plus, d_minus, "./outputs/l3_topsis_distances.png")
    plot_topsis_scores(scores_topsis, "./outputs/l3_topsis_scores.png")


def monthly_payment(rate_annual_pct, months, principal):
    r = rate_annual_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r)**months / ((1 + r)**months - 1)


def ahp_weights(matrix: np.ndarray):
    n = matrix.shape[0]
    col_sum = matrix.sum(axis=0)
    norm = matrix / col_sum
    weights = norm.mean(axis=1)
    weights /= weights.sum()

    weighted_sum = (matrix @ weights)
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    return weights, lambda_max, cr


def normilize_minmax(data: np.ndarray, directions:np.ndarray):
    norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col    = data[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            norm[:, j] = 1.0
        elif directions[j] == +1:
            norm[:, j] = (col - mn) / (mx - mn)
        else:
            norm[:, j] = (mx - col) / (mx - mn)
    return norm

def normilize_vector(data: np.ndarray):
    denom = np.sqrt((data) ** 2).sum(axis=0)
    denom[denom == 0] = 1e-12
    return data / denom


def wsm(norm_matrix:np.ndarray, weights: np.ndarray):
    return norm_matrix @ weights


def wpm(norm_matrix: np.ndarray, weights: np.ndarray):
    safe = np.where(norm_matrix <= 0, 1e-9, norm_matrix)
    return np.prod(safe ** weights, axis=1)


def topsis(data: np.ndarray, weights: np.ndarray, directions: np.ndarray):
    norm = normilize_vector(data)
    v = norm * weights

    ideal = np.where(directions == +1, v.max(axis=0), v.min(axis=0))
    anti_ideal = np.where(directions == +1, v.min(axis=0), v.max(axis=0))

    d_plus =np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_minus =np.sqrt(((v - anti_ideal) ** 2).sum(axis=1))

    c_star = d_minus / (d_plus + d_minus + 1e-12)
    return c_star, d_plus, d_minus


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


def plot_ahp_weights(weights, cr, output_path):
    fig, ax = plt.subplots(figsize=(17, 8), facecolor=C["bg"])
    bars = ax.bar(CRITERIA_NAMES, weights * 100, color=[BANK_COLORS[i % 8] for i in range(N_CRITERIA)], alpha=0.85, edgecolor=C["grid"])
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{w*100:.1f}%", ha="center", va="bottom", color=C["text"])
    ax.axhline(100/N_CRITERIA, color=C["gold"], ls="--", lw=1.2, alpha=0.7, label=f"Equal weights ({100/N_CRITERIA} %)")
    

    title = f"Analytic Hierarchy Process (CR={cr:.4f})"
    ax_style(ax, title)
    ax.set_xlabel("Criteria", color=C["subtext"], fontsize=10)
    ax.set_ylabel("Weight, %", color=C["subtext"], fontsize=10)
    ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_normilized_matrix(norm_mm, output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])
    
    im = ax.imshow(norm_mm.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(N_BANKS))
    ax.set_xticklabels(BANKS, fontsize=7, color=C["text"])
    ax.set_yticks(range(N_CRITERIA))
    ax.set_yticklabels(CRITERIA_NAMES, fontsize=7, color=C["text"])

    for i in range(N_BANKS):
        for j in range(N_CRITERIA):
            ax.text(i, j, f"{norm_mm[i,j]:.2f}", ha="center", va="center", fontsize=6, color="black" if norm_mm[i,j] > 0.4 else "white")
    
    plt.colorbar(im, ax=ax, fraction=0.04)

    title = f"Min-Max Normilized Matrix"
    ax_style(ax, title)
    ax.grid(alpha=0.0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_wsm_scores(scores_wsm, output_path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C["bg"])
    
    idx_s = np.argsort(-scores_wsm)
    clr = [C["best"] if r == 0 else C["accent"] if r == 1 else C["worst"] if r == N_BANKS-1 else C["b1"] for r in range(N_BANKS)]
    bars = ax.barh([BANKS[i] for i in idx_s], [scores_wsm[i] for i in idx_s], color=[clr[r] for r in range(N_BANKS)], alpha=0.85, edgecolor=C["grid"])

    for bar, val in zip(bars, [scores_wsm[i] for i in idx_s]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", color=C["text"], fontsize=7.5)

    ax.tick_params(colors=C["subtext"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(alpha=0.18, color=C["grid"], ls="--", lw=0.6)

    title = f"WSM"
    ax_style(ax, title)
    ax.set_xlabel("WSM Score", color=C["subtext"], fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_wpm_scores(scores_wpm, output_path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C["bg"])
    
    idx_w = np.argsort(-scores_wpm)

    bars = ax.barh([BANKS[i] for i in idx_w], [scores_wpm[i] for i in idx_w], color=BANK_COLORS, alpha=0.85, edgecolor=C["grid"])

    for bar, val in zip(bars, [scores_wpm[i] for i in idx_w]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", color=C["text"], fontsize=7.5)

    ax.tick_params(colors=C["subtext"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(alpha=0.18, color=C["grid"], ls="--", lw=0.6)

    title = f"WPM Score"
    ax_style(ax, title)
    ax.set_xlabel("WPM score", color=C["subtext"], fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_topsis_distances(d_plus, d_minus, output_path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C["bg"])
    
    x_pos = np.arange(N_BANKS)
    w_bar = 0.35

    b1 = ax.bar(x_pos - w_bar/2, d_plus, w_bar, color=C["worst"], alpha=0.85, label="D+ (ideal)")
    b2 = ax.bar(x_pos + w_bar/2, d_minus, w_bar, color=C["best"], alpha=0.85, label="D- (anti-ideal)")

    for b, v in zip(list(b1)+list(b2), list(d_plus)+list(d_minus)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1e-4, f"{v:.3f}", ha="center", va="bottom", color=C["text"], fontsize=6.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(BANKS, fontsize=8, color=C["text"])
    
    ax.tick_params(colors=C["subtext"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(alpha=0.18, color=C["grid"], ls="--", lw=0.6)

    title = f"TOPSIS - distances to ideal/anti-ideal"
    ax_style(ax, title)
    ax.set_ylabel("Distance", color=C["subtext"], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_topsis_scores(scores_topsis, output_path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C["bg"])
    
    idx_t = np.argsort(-scores_topsis)

    bars = ax.barh([BANKS[i] for i in idx_t], [scores_topsis[i] for i in idx_t], color=BANK_COLORS, alpha=0.85, edgecolor=C["grid"])

    ax.axvline(0.5, color=C["gold"], ls="--", lw=1.2, alpha=0.8, label="threshold = 0.5")

    for bar, val in zip(bars, [scores_topsis[i] for i in idx_t]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", color=C["text"], fontsize=7.5)

    ax.tick_params(colors=C["subtext"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(alpha=0.18, color=C["grid"], ls="--", lw=0.6)

    title = f"TOPSIS C*"
    ax_style(ax, title)
    ax.set_xlabel("C*", color=C["subtext"], fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


if __name__ == "__main__":
    main()