import numpy as np


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
 
# Критерії (сирі дані)
# C1: interest rate, %      (min)
# C2: term, months          (max)
# C3: monthly payment, uah  (min)
# C4: overpayment, uah      (min)
# C5: down payment, %       (min)
# C6: approval speed, h     (min)
# C7: prepayment penalty, % (min)
# C8: insuranc  (0=no,1=yes; min)
# C9: bank rating           (max, 1-10)
# C10: commission fee,uah/month (min)

RAW_DATA = np.array([
    # C1    C2    C3       C4        C5   C6   C7   C8   C9   C10
    [15.5,  60,  22_800,  408_000,  20,   2,   2,   1,   9,   80 ],  # ПриватБанк
    [13.9,  84,  16_200,  400_800,  20,  24,   0,   1,   10,  50 ],  # Ощадбанк
    [16.8,  72,  20_100,  487_200,  15,   4,   3,   1,   8,   120],  # Ukrsibbank
    [14.5,  60,  22_300,  378_000,  25,   1,   1,   0,   9,   90 ],  # Raiffeisen
    [17.9,  48,  28_500,  408_000,  10,   0,   0,   0,   7,   0  ],  # monobank
    [15.0,  60,  22_650,  399_000,  20,   3,   2,   1,   8,   70 ],  # ПУМБ
    [12.5,  96,  13_800,  364_800,  20,  48,   1,   1,   8,   60 ],  # Укргазбанк
    [14.9,  72,  19_200,  422_400,  15,   6,   1,   0,   8,   100],  # OTP Bank
], dtype=float)


def main():
    CAR_PRICE = 1_200_000
    LOAN_AMOUNT = 960_000

    print("Level II: Multi-criteria evaluation of loan offers")
    print(f"Car price: {CAR_PRICE:>10} UAH")
    print(f"Loan amount: {LOAN_AMOUNT:>10} UAH")

    for i, bank in enumerate(BANKS):
        mp = monthly_payment(RAW_DATA[i, 0], int(RAW_DATA[i, 1]), LOAN_AMOUNT)
        print(f"{bank:<11}: {mp:>8,.0f} UAH/month (raw: {RAW_DATA[i,2]:>8,.0f} UAH/month)")


def monthly_payment(rate_annual_pct, months, principal):
    r = rate_annual_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r)**months / ((1 + r)**months - 1)




if __name__ == "__main__":
    main()