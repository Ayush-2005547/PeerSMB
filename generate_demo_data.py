import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

os.makedirs("demo_data", exist_ok=True)

# ---------------- Marketplace settlement CSV ----------------
start = datetime(2025, 11, 1)
weeks = 16  # ~4 months weekly settlements

rows = []
base_sales = 120000
for i in range(weeks):
    d = start + timedelta(days=7*i)
    # sales fluctuate + slight trend
    trend = 1 + (i / weeks) * 0.08
    gross_sales = base_sales * trend * np.random.uniform(0.75, 1.25)

    refund_rate = np.random.uniform(0.03, 0.14)
    refund_amount = gross_sales * refund_rate

    commission = gross_sales * np.random.uniform(0.08, 0.15)
    logistics = gross_sales * np.random.uniform(0.02, 0.06)

    payout_amount = max(gross_sales - refund_amount - commission - logistics, 0)
    orders = int((gross_sales / np.random.uniform(900, 1600)))

    rows.append([d.date().isoformat(), round(gross_sales, 2), round(refund_amount, 2), round(payout_amount, 2), orders])

marketplace = pd.DataFrame(rows, columns=["date", "gross_sales", "refund_amount", "payout_amount", "orders"])
marketplace.to_csv("demo_data/marketplace_settlement_demo.csv", index=False)

# ---------------- Bank statement CSV ----------------
# daily transactions for 120 days
days = 120
start2 = datetime(2025, 11, 1)

bank_rows = []
balance = 180000

# Settlement credits align roughly with weekly payouts
settle_dates = set(pd.to_datetime(marketplace["date"]).dt.date.tolist())

for i in range(days):
    d = start2 + timedelta(days=i)
    date_str = d.date().isoformat()

    # weekly settlement credit
    if d.date() in settle_dates:
        payout = float(marketplace.loc[marketplace["date"] == date_str, "payout_amount"].iloc[0])
        balance += payout
        bank_rows.append([date_str, "Marketplace Settlement Credit", 0.0, round(payout, 2), round(balance, 2)])

    # random daily expenses (debits)
    for _ in range(np.random.randint(1, 4)):
        amt = np.random.uniform(1500, 18000)
        balance -= amt
        bank_rows.append([date_str, "UPI/Supplier Payment", round(amt, 2), 0.0, round(balance, 2)])

    # occasional ad spend
    if np.random.rand() < 0.18:
        amt = np.random.uniform(3000, 25000)
        balance -= amt
        bank_rows.append([date_str, "Ads/Marketing Spend", round(amt, 2), 0.0, round(balance, 2)])

    # rare penalty/bounce fee
    if np.random.rand() < 0.03:
        amt = np.random.uniform(250, 1500)
        balance -= amt
        bank_rows.append([date_str, "Charges/Bounce/Return Fee", round(amt, 2), 0.0, round(balance, 2)])

bank = pd.DataFrame(bank_rows, columns=["date", "description", "debit", "credit", "balance"])
bank.to_csv("demo_data/bank_statement_demo.csv", index=False)

print("✅ Generated:")
print(" - demo_data/marketplace_settlement_demo.csv")
print(" - demo_data/bank_statement_demo.csv")