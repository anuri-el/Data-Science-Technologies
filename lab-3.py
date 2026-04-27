import numpy as np
import matplotlib.pyplot as plt


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

def main():
    CAR_PRICE = 1_200_000
    LOAN_AMOUNT = 960_000

    print("Level II: Multi-criteria evaluation of loan offers")
    print(f"Car price: {CAR_PRICE:>10} UAH")
    print(f"Loan amount: {LOAN_AMOUNT:>10} UAH")

    for i, bank in enumerate(BANKS):
        mp = monthly_payment(RAW_DATA[i, 0], int(RAW_DATA[i, 1]), LOAN_AMOUNT)
        print(f"{bank:<11}: {mp:>8,.0f} UAH/month (raw: {RAW_DATA[i,2]:>8,.0f} UAH/month)")
    
    print("AHP")
    weights, lambda_max, cr = ahp_weights(AHP_MATRIX)
    print(f"lambda_max = {lambda_max:.4f} | CR = {cr:.4f} {"OK (CR < 0.1)" if cr < 0.1 else "! CR >= 0.1"}")
    print(f"{'Criteria':<25} {'Weight':>8} {'%':>6}")
    for name, w in zip(CRITERIA_NAMES, weights):
        bar = "=" * int(w * 100)
        print(f"{name:<25} {w:>8.4f} {w*100:>5.1f} {bar}")
    
    plot_ahp_weights(weights, cr, "./outputs/l3_ahp_weights.png")


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


if __name__ == "__main__":
    main()