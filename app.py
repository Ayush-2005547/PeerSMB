import streamlit as st
import pandas as pd
import joblib
import os
import uuid
import time

# ============================================================
# PeerSMB – Final Demo App (Login + Better Lender UI + Compliance)
# COMPLETE + HARDENED (no KeyErrors, missing columns auto-added)
# + FIXES:
#   1) No loans match filters -> dynamic risk bands + handles NA
#   2) Compliance merge dtype error -> force loan_id to string on both sides
#   3) Safe numeric fillna(None) issue -> safe_numeric handles None
#   4) Consistent typing of loan_id across app to avoid surprises
# ============================================================

st.set_page_config(page_title="PeerSMB", layout="wide")
st.title("PeerSMB – AI Credit & Lending Platform")

# ---------------- Demo Auth (FAST + PRESENTABLE) ----------------
DEMO_USERS = {
    "Borrower": {"ayush": "1234"},
    "Lender": {"lender1": "1234"},
    "Admin": {"admin": "1234"},
}

def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def require_login():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "role": None, "user": None, "ts": None}

    if st.session_state.auth["logged_in"]:
        return

    st.markdown("## Login")
    c1, c2 = st.columns(2)
    with c1:
        role = st.selectbox("Role", ["Borrower", "Lender", "Admin"])
        user = st.text_input("Username", placeholder="e.g., ayush")
    with c2:
        pwd = st.text_input("Password", type="password", placeholder="Demo: 1234")
        st.caption("Demo password: 1234")

    if st.button("Login"):
        if user in DEMO_USERS.get(role, {}) and DEMO_USERS[role][user] == pwd:
            st.session_state.auth = {"logged_in": True, "role": role, "user": user, "ts": time.time()}
            st.success("Logged in ✅")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

require_login()

topL, topR = st.columns([0.75, 0.25])
with topL:
    st.caption(f"Logged in as **{st.session_state.auth['user']}** ({st.session_state.auth['role']})")
with topR:
    if st.button("Logout"):
        logout()

# ---------------- Helpers ----------------
def smart_path(*parts: str) -> str:
    """Prefer ./data or ./models, but fallback to current directory."""
    p = os.path.join(*parts)
    if os.path.exists(p):
        return p
    filename = parts[-1]
    if os.path.exists(filename):
        return filename
    return p

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def ensure_cols(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Ensure df has all columns listed in defaults; create missing with default value."""
    df = df.copy()
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

def safe_numeric(s, default=None):
    """pd.to_numeric + optional fill (won't call fillna(None))."""
    out = pd.to_numeric(s, errors="coerce")
    if default is None:
        return out
    return out.fillna(default)

def coerce_loan_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Force loan_id to string consistently (prevents merge + selection issues)."""
    df = df.copy()
    if "loan_id" in df.columns:
        df["loan_id"] = df["loan_id"].astype(str)
    return df

# ---------------- Load artifacts ----------------
PRED_PATH  = smart_path("data", "rejected_loans_with_predictions.csv")
RBI_PATH   = smart_path("data", "fundable_loans_post_matching.csv")
ALLOC_PATH = smart_path("data", "lender_loan_allocations.csv")
MODEL_PATH = smart_path("models", "xgb_reduced_credit_model.pkl")

pred_loans = load_csv(PRED_PATH)
loans_seed = load_csv(RBI_PATH)
alloc_seed = load_csv(ALLOC_PATH)
model = load_model(MODEL_PATH)

if not hasattr(model, "feature_names_in_"):
    st.error("Model missing feature_names_in_. Re-save model in sklearn-compatible way.")
    st.stop()

MODEL_FEATURES = list(model.feature_names_in_)  # includes one-hot states

# ---------------- Session state (live marketplace) ----------------
if "loans" not in st.session_state:
    st.session_state.loans = loans_seed.copy()

if "alloc_df" not in st.session_state:
    st.session_state.alloc_df = alloc_seed.copy()

# Normalize schema once so UI never KeyErrors
st.session_state.loans = ensure_cols(
    st.session_state.loans,
    {
        "loan_id": "",
        "loan_title": "Untitled Loan",
        "loan_amount": 0.0,
        "remaining_amount": None,   # filled from loan_amount below
        "lender_count": 0,
        "default_prob": 0.25,
        "risk_category": "NA",
        "tenure_months": 12,
        "interest_rate": 0.16,
        "addr_state": "NA",
        "purpose": "NA",            # fixes purpose KeyError everywhere
    },
)

# Force loan_id to str everywhere (prevents merge dtype error later)
st.session_state.loans = coerce_loan_ids(st.session_state.loans)

st.session_state.loans["loan_amount"] = safe_numeric(st.session_state.loans["loan_amount"], 0.0)
st.session_state.loans["remaining_amount"] = safe_numeric(st.session_state.loans["remaining_amount"], default=None)
st.session_state.loans["remaining_amount"] = st.session_state.loans["remaining_amount"].fillna(st.session_state.loans["loan_amount"])
st.session_state.loans["lender_count"] = safe_numeric(st.session_state.loans["lender_count"], 0).astype(int)
st.session_state.loans["default_prob"] = safe_numeric(st.session_state.loans["default_prob"], 0.25).clip(0, 0.99)
st.session_state.loans["interest_rate"] = safe_numeric(st.session_state.loans["interest_rate"], 0.16).clip(0, 0.99)
st.session_state.loans["tenure_months"] = safe_numeric(st.session_state.loans["tenure_months"], 12).astype(int)
st.session_state.loans["risk_category"] = st.session_state.loans["risk_category"].fillna("NA").astype(str)
st.session_state.loans["purpose"] = st.session_state.loans["purpose"].fillna("NA").astype(str)

st.session_state.alloc_df = ensure_cols(
    st.session_state.alloc_df,
    {"lender_id": "", "loan_id": "", "funded_amount": 0.0, "risk_category": "NA"},
)
# Force alloc loan_id to str too
st.session_state.alloc_df["loan_id"] = st.session_state.alloc_df["loan_id"].astype(str)
st.session_state.alloc_df["lender_id"] = st.session_state.alloc_df["lender_id"].astype(str)
st.session_state.alloc_df["funded_amount"] = safe_numeric(st.session_state.alloc_df["funded_amount"], 0.0)

if "lenders" not in st.session_state:
    lender_ids = (
        st.session_state.alloc_df["lender_id"].dropna().astype(str).unique().tolist()
        if "lender_id" in st.session_state.alloc_df.columns
        else []
    )
    if len(lender_ids) < 300:
        lender_ids = [f"L{i:03d}" for i in range(1, 301)]  # 300 lenders

    st.session_state.lenders = pd.DataFrame({
        "lender_id": lender_ids,
        "total_capacity": [2_000_000] * len(lender_ids),  # ₹20L each
    })

# ---------------- Credit scoring helpers ----------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def build_feature_row(loan_amnt: float, dti: float, credit_proxy: float, emp_years: float, state: str) -> pd.DataFrame:
    row = {c: 0.0 for c in MODEL_FEATURES}

    if "loan_amnt" in row: row["loan_amnt"] = float(loan_amnt)
    if "dti" in row: row["dti"] = float(dti)
    if "fico_avg" in row: row["fico_avg"] = float(credit_proxy)
    if "emp_length_num" in row: row["emp_length_num"] = float(emp_years)

    state_col = f"addr_state_{state}"
    if state_col in row:
        row[state_col] = 1.0

    return pd.DataFrame([row], columns=MODEL_FEATURES)

def score_borrower(loan_amnt: float, dti: float, credit_proxy: float, emp_years: float, state: str) -> float:
    X = build_feature_row(loan_amnt, dti, credit_proxy, emp_years, state)
    return float(model.predict_proba(X)[0, 1])

def altdata_to_model_inputs(
    loan_amount: float,
    monthly_revenue: float,
    stability_score: float,
    business_age: float,
    tenure_months: int = 12,
):
    """Map alt-data to model inputs (tenure-aware EMI)."""
    credit_proxy = 300.0 + float(stability_score) * 600.0   # 300..900

    n = int(tenure_months)
    assumed_rate_annual = 0.14
    r = assumed_rate_annual / 12

    if r > 1e-9:
        emi = loan_amount * (r * (1 + r) ** n) / (((1 + r) ** n) - 1)
    else:
        emi = loan_amount / max(n, 1)

    monthly_revenue = max(float(monthly_revenue), 1.0)
    emi_ratio = emi / monthly_revenue
    dti = clamp(emi_ratio * 100.0, 0.0, 65.0)

    emp_years = clamp(float(business_age), 0.0, 10.0)
    return credit_proxy, dti, emp_years

def risk_band_and_rate(prob: float):
    if prob < 0.15:
        return "AA", 0.12
    if prob < 0.25:
        return "A", 0.14
    if prob < 0.40:
        return "B", 0.16
    return "C", 0.18

# ---------------- Matching helpers (greedy + RBI caps) ----------------
def run_matching(
    loans_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    lenders_df: pd.DataFrame,
    min_lenders: int = 5,
    max_share: float = 0.20
):
    alloc_df = ensure_cols(alloc_df, {"lender_id": "", "loan_id": "", "funded_amount": 0.0, "risk_category": "NA"})
    loans = loans_df.copy()
    loans = ensure_cols(loans, {"remaining_amount": None, "lender_count": 0, "loan_amount": 0.0, "risk_category": "NA"})

    loans = coerce_loan_ids(loans)
    alloc_df["loan_id"] = alloc_df["loan_id"].astype(str)
    alloc_df["lender_id"] = alloc_df["lender_id"].astype(str)

    loans["loan_amount"] = safe_numeric(loans["loan_amount"], 0.0)
    loans["remaining_amount"] = safe_numeric(loans["remaining_amount"], default=None).fillna(loans["loan_amount"])
    loans["lender_count"] = safe_numeric(loans["lender_count"], 0).astype(int)
    loans["risk_category"] = loans["risk_category"].fillna("NA").astype(str)

    alloc_df["funded_amount"] = safe_numeric(alloc_df["funded_amount"], 0.0)

    alloc_sum = alloc_df.groupby("lender_id")["funded_amount"].sum() if len(alloc_df) else pd.Series(dtype=float)

    lenders = lenders_df.copy()
    lenders["allocated_so_far"] = lenders["lender_id"].map(alloc_sum).fillna(0.0)
    lenders["available"] = (lenders["total_capacity"] - lenders["allocated_so_far"]).clip(lower=0.0)

    new_alloc_rows = []
    open_loans = loans[loans["remaining_amount"] > 0].copy()

    for _, loan in open_loans.iterrows():
        loan_id = str(loan["loan_id"])
        loan_amount = float(loan["loan_amount"])
        remaining = float(loan["remaining_amount"])

        if loan_amount <= 0:
            continue

        per_lender_cap = max_share * loan_amount
        eligible = lenders[lenders["available"] > 0].sort_values("available", ascending=False)

        used_lenders = 0
        for _, lender in eligible.iterrows():
            if remaining <= 0:
                break

            lender_id = str(lender["lender_id"])
            avail = float(lender["available"])
            amt = min(per_lender_cap, remaining, avail)

            if amt <= 0:
                continue

            new_alloc_rows.append({
                "lender_id": lender_id,
                "loan_id": loan_id,
                "funded_amount": amt,
                "risk_category": loan.get("risk_category", "NA")
            })

            remaining -= amt
            used_lenders += 1
            lenders.loc[lenders["lender_id"] == lender_id, "available"] = avail - amt

        loans.loc[loans["loan_id"].astype(str) == loan_id, "remaining_amount"] = remaining
        loans.loc[loans["loan_id"].astype(str) == loan_id, "lender_count"] = int(used_lenders)

    if new_alloc_rows:
        alloc_df = pd.concat([alloc_df, pd.DataFrame(new_alloc_rows)], ignore_index=True)

    return loans, alloc_df

# ---------------- Doc parsing helpers ----------------
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0 else 0.0

def parse_marketplace_csv(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if "payout_amount" not in df.columns:
        raise ValueError("Marketplace CSV must contain payout_amount column")

    df["payout_amount"] = pd.to_numeric(df["payout_amount"], errors="coerce").fillna(0.0)

    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["payout_amount"].sum()

    monthly_revenue = float(monthly.mean()) if len(monthly) else float(df["payout_amount"].sum())
    payout_volatility = float(df["payout_amount"].std() / (df["payout_amount"].mean() + 1e-9))

    intervals = df["date"].diff().dt.days.dropna()
    if len(intervals) >= 2:
        interval_std = float(intervals.std())
        payout_regularity = 1.0 - min(1.0, interval_std / 10.0)
    else:
        payout_regularity = 0.5

    refund_rate = None
    if "gross_sales" in df.columns and "refund_amount" in df.columns:
        df["gross_sales"] = pd.to_numeric(df["gross_sales"], errors="coerce").fillna(0.0)
        df["refund_amount"] = pd.to_numeric(df["refund_amount"], errors="coerce").fillna(0.0)
        gross = float(df["gross_sales"].sum())
        refunds = float(df["refund_amount"].sum())
        refund_rate = safe_div(refunds, gross) if gross > 0 else 0.0

    growth = 0.0
    if len(monthly) >= 4:
        first = float(monthly.iloc[:2].mean())
        last = float(monthly.iloc[-2:].mean())
        growth = safe_div((last - first), (first + 1e-9))

    return {
        "monthly_revenue": monthly_revenue,
        "payout_volatility": min(1.0, payout_volatility),
        "payout_regularity": float(max(0.0, min(1.0, payout_regularity))),
        "refund_rate": float(refund_rate) if refund_rate is not None else None,
        "growth_momentum": float(growth),
    }

def parse_bank_csv(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for c in ["debit", "credit"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["month"] = df["date"].dt.to_period("M").astype(str)
    inflow = df.groupby("month")["credit"].sum()
    outflow = df.groupby("month")["debit"].sum()

    avg_inflow = float(inflow.mean()) if len(inflow) else float(df["credit"].sum())
    avg_outflow = float(outflow.mean()) if len(outflow) else float(df["debit"].sum())

    inflow_vol = float(inflow.std() / (inflow.mean() + 1e-9)) if len(inflow) >= 2 else 0.35

    low_balance_days = 0
    if "balance" in df.columns:
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce").ffill().fillna(0.0)
        thr = float(df["balance"].median() * 0.15)
        daily = df.groupby(df["date"].dt.date)["balance"].last()
        low_balance_days = int((daily < thr).sum())

    flags = 0
    if "description" in df.columns:
        txt = df["description"].astype(str).str.lower()
        mask = txt.str.contains(r"bounce|return|charge|charges|penalty|od|overdraft", regex=True, na=False)
        if mask.any():
            flags = int(df.loc[mask, "date"].dt.date.nunique())

    return {
        "avg_inflow": avg_inflow,
        "avg_outflow": avg_outflow,
        "inflow_volatility": min(1.0, inflow_vol),
        "low_balance_days": low_balance_days,
        "risk_flags": flags,
    }

def alt_stability_from_docs(
    gst_consistency: float,
    payout_regularity: float,
    payout_volatility: float,
    refund_rate: float,
    inflow_volatility: float,
    low_balance_days: int,
    risk_flags: int,
    business_age_years: float,
) -> float:
    gst_consistency = clamp(gst_consistency, 0.0, 1.0)
    payout_regularity = clamp(payout_regularity, 0.0, 1.0)
    payout_volatility = clamp(payout_volatility, 0.0, 1.0)
    inflow_volatility = clamp(inflow_volatility, 0.0, 1.0)
    refund_rate = clamp(refund_rate, 0.0, 0.5)
    business_age_years = clamp(business_age_years, 0.0, 10.0)

    refund_good = 1.0 - (refund_rate / 0.5)
    payout_vol_good = 1.0 - payout_volatility
    inflow_vol_good = 1.0 - inflow_volatility
    age_good = min(business_age_years / 5.0, 1.0)

    flags_good = 1.0 - min(1.0, risk_flags / 10.0)
    lowbal_good = 1.0 - min(1.0, low_balance_days / 60.0)

    score = (
        0.22 * gst_consistency +
        0.18 * payout_regularity +
        0.14 * payout_vol_good +
        0.12 * inflow_vol_good +
        0.10 * refund_good +
        0.10 * lowbal_good +
        0.08 * flags_good +
        0.06 * age_good
    )
    return clamp(score, 0.0, 1.0)

# ---------------- Tabs ----------------
role = st.session_state.auth["role"]
if role == "Borrower":
    tab1, = st.tabs(["Borrower Portal"])
elif role == "Lender":
    tab2, = st.tabs(["Lender Portal"])
else:
    tab1, tab2, tab3 = st.tabs(["Borrower Portal", "Lender Portal", "Compliance Console"])

# ============================================================
# TAB 1 — Borrower
# ============================================================
if role in ["Borrower", "Admin"]:
    with tab1:
        st.header("Borrower Application (Upload → Extract → Score → Marketplace)")

        colL, colR = st.columns([1.1, 0.9])
        with colL:
            loan_amnt = st.number_input("Loan Amount (₹)", 50_000, 1_000_000, step=10_000)
            tenure = st.selectbox("Tenure (months)", [6, 9, 12], index=2)
            purpose = st.selectbox("Purpose", ["Inventory", "Ads/Marketing", "Working Capital", "Equipment", "Other"], index=0)
            business_age = st.selectbox("Business Age", ["0–6 months", "6–12 months", "1–3 years", "3–5 years", "5+ years"], index=2)
            gst_choice = st.selectbox("GST Filing Consistency", ["Regular", "Sometimes missed", "Often missed / Not registered"], index=0)

            age_map = {"0–6 months": 0.25, "6–12 months": 0.75, "1–3 years": 2.0, "3–5 years": 4.0, "5+ years": 6.0}
            gst_map = {"Regular": 0.90, "Sometimes missed": 0.65, "Often missed / Not registered": 0.40}

            business_age_years = age_map[business_age]
            gst_consistency = gst_map[gst_choice]

            state_codes = sorted([c.replace("addr_state_", "") for c in MODEL_FEATURES if c.startswith("addr_state_")])
            default_state = "CA" if "CA" in state_codes else state_codes[0]
            state = st.selectbox("State (for model one-hot)", state_codes, index=state_codes.index(default_state))

        with colR:
            st.subheader("Upload Documents")
            mp_file = st.file_uploader("Marketplace Settlement CSV (required)", type=["csv"], key="mp_csv")
            bank_file = st.file_uploader("Bank Statement CSV (optional)", type=["csv"], key="bank_csv")
            st.caption("No document storage: features are derived in-memory; only aggregate scores are stored in marketplace.")
            st.divider()
            st.subheader("Try Demo Data")
            use_demo = st.checkbox("Use demo CSVs from demo_data/", value=False)

        mp_df = None
        bank_df = None

        if use_demo:
            try:
                mp_df = pd.read_csv("demo_data/marketplace_settlement_demo.csv")
                bank_df = pd.read_csv("demo_data/bank_statement_demo.csv")
                st.success("Loaded demo CSVs ✅")
            except Exception as e:
                st.error(f"Could not load demo CSVs: {e}")

        if mp_file is not None:
            mp_df = pd.read_csv(mp_file)
        if bank_file is not None:
            bank_df = pd.read_csv(bank_file)

        if st.button("Generate Risk Score from Documents"):
            if mp_df is None:
                st.error("Marketplace settlement CSV is required.")
                st.stop()

            mp_feat = parse_marketplace_csv(mp_df)

            if bank_df is not None:
                bank_feat = parse_bank_csv(bank_df)
            else:
                bank_feat = {
                    "avg_inflow": mp_feat["monthly_revenue"],
                    "avg_outflow": mp_feat["monthly_revenue"] * 0.7,
                    "inflow_volatility": 0.40,
                    "low_balance_days": 8,
                    "risk_flags": 1
                }

            refund_rate = mp_feat["refund_rate"] if mp_feat["refund_rate"] is not None else 0.08

            stability = alt_stability_from_docs(
                gst_consistency=gst_consistency,
                payout_regularity=mp_feat["payout_regularity"],
                payout_volatility=mp_feat["payout_volatility"],
                refund_rate=refund_rate,
                inflow_volatility=bank_feat["inflow_volatility"],
                low_balance_days=bank_feat["low_balance_days"],
                risk_flags=bank_feat["risk_flags"],
                business_age_years=business_age_years,
            )

            credit_proxy, dti_proxy, emp_years = altdata_to_model_inputs(
                loan_amount=float(loan_amnt),
                monthly_revenue=float(mp_feat["monthly_revenue"]),
                stability_score=float(stability),
                business_age=float(business_age_years),
                tenure_months=int(tenure),
            )

            prob_model = score_borrower(loan_amnt, dti_proxy, credit_proxy, emp_years, state)

            bank_stress = (
                0.6 * min(1.0, bank_feat["low_balance_days"] / 60.0) +
                0.4 * min(1.0, bank_feat["risk_flags"] / 10.0)
            )
            prob = min(0.99, prob_model + 0.10 * bank_stress)

            band, rate = risk_band_and_rate(prob)

            st.success(f"Risk Band: {band}")
            st.info(f"Predicted Default Probability (PD): {prob:.3f}")
            st.caption(f"Model PD: {prob_model:.3f} | Bank-stress overlay: +{(prob - prob_model):.3f}")
            st.info(f"Indicative Interest Rate: {int(rate*100)}% p.a.")

            with st.expander("Derived signals (for viva / judges)"):
                st.write("**Marketplace features**")
                st.json(mp_feat)
                st.write("**Bank features**")
                st.json(bank_feat)
                st.write(f"**Tenure (months):** {tenure}")
                st.write(f"**Alt-data Stability Score (0–1):** {stability:.2f}")
                st.write(f"**Derived Credit Proxy (300–900):** {credit_proxy:.0f}")
                st.write(f"**Affordability Proxy (DTI%):** {dti_proxy:.1f}")
                st.write(f"**Purpose:** {purpose}")

            st.session_state.last_scoring = {
                "loan_amount": float(loan_amnt),
                "tenure_months": int(tenure),
                "purpose": purpose,
                "state": state,
                "default_prob": prob,
                "risk_category": band,
                "rate": rate,
            }

        if "last_scoring" in st.session_state:
            st.divider()
            st.subheader("Submit to Marketplace")

            title = st.text_input("Loan title", value="Inventory Financing – E-commerce Seller")
            if st.button("Submit Application → Add to Marketplace"):
                s = st.session_state.last_scoring
                new_id = str(uuid.uuid4())[:8]

                new_row = {
                    "loan_id": str(new_id),
                    "loan_title": title,
                    "loan_amount": s["loan_amount"],
                    "remaining_amount": s["loan_amount"],
                    "lender_count": 0,
                    "default_prob": s["default_prob"],
                    "risk_category": s["risk_category"],
                    "tenure_months": s["tenure_months"],
                    "interest_rate": s["rate"],
                    "addr_state": s["state"],
                    "purpose": s.get("purpose", "NA"),
                }

                st.session_state.loans = pd.concat([st.session_state.loans, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state.loans = coerce_loan_ids(st.session_state.loans)
                st.success(f"Loan {new_id} added to marketplace ✅")

# ============================================================
# TAB 2 — Lender
# ============================================================
if role in ["Lender", "Admin"]:
    with tab2:
        st.header("Lender Marketplace")

        loans_live = ensure_cols(st.session_state.loans.copy(), {"purpose": "NA"})
        loans_live = coerce_loan_ids(loans_live)
        loans_live["risk_category"] = loans_live["risk_category"].fillna("NA").astype(str)

        alloc_live = st.session_state.alloc_df.copy()
        alloc_live["loan_id"] = alloc_live["loan_id"].astype(str)
        alloc_live["lender_id"] = alloc_live["lender_id"].astype(str)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Active Loans", str(int(len(loans_live))))
        k2.metric("Funded Loans", str(int((loans_live["remaining_amount"] <= 0).sum())))
        k3.metric("Avg PD", f"{loans_live['default_prob'].mean():.3f}")
        k4.metric("Avg Rate", f"{(loans_live['interest_rate'].fillna(0).mean() * 100):.1f}%")

        with st.expander("Filters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                # ✅ dynamic bands so "NA" doesn't break filters
                all_bands = sorted(loans_live["risk_category"].unique().tolist())
                bands = st.multiselect("Risk Bands", all_bands, default=all_bands)
            with c2:
                pd_min, pd_max = st.slider("PD Range", 0.0, 1.0, (0.0, 0.5), 0.01)
            with c3:
                amt_min, amt_max = st.slider("Loan Amount", 50_000, 1_000_000, (50_000, 1_000_000), 10_000)
            with c4:
                sort_by = st.selectbox("Sort by", ["Needs funding", "Lowest PD", "Highest PD", "Highest Amount"])

        f = loans_live[
            (loans_live["risk_category"].isin(bands)) &
            (loans_live["default_prob"].between(pd_min, pd_max)) &
            (loans_live["loan_amount"].between(amt_min, amt_max))
        ].copy()

        if sort_by == "Needs funding":
            f = f.sort_values(["remaining_amount", "default_prob"], ascending=[False, True])
        elif sort_by == "Lowest PD":
            f = f.sort_values("default_prob", ascending=True)
        elif sort_by == "Highest PD":
            f = f.sort_values("default_prob", ascending=False)
        else:
            f = f.sort_values("loan_amount", ascending=False)

        st.dataframe(
            f[["loan_id", "loan_title", "purpose", "risk_category", "default_prob", "loan_amount", "remaining_amount", "lender_count"]].head(50),
            use_container_width=True
        )

        st.divider()
        st.subheader("Loan → Lender Allocation (Syndicate View)")

        if len(f) == 0:
            st.info("No loans match your filters.")
        else:
            loan_ids = f["loan_id"].astype(str).tolist()
            selected_loan = st.selectbox("Select loan_id to view syndicate", loan_ids)

            loan_row = loans_live[loans_live["loan_id"].astype(str) == str(selected_loan)].iloc[0]
            loan_amount = float(loan_row["loan_amount"])
            alloc_sel = alloc_live[alloc_live["loan_id"].astype(str) == str(selected_loan)].copy()

            if len(alloc_sel) == 0:
                st.warning("No allocations for this loan yet. Click **Run Matching** first.")
            else:
                alloc_sel["share"] = alloc_sel["funded_amount"] / max(loan_amount, 1.0)
                alloc_sel = alloc_sel.sort_values("funded_amount", ascending=False)

                c1, c2, c3 = st.columns(3)
                c1.metric("Loan Amount", f"₹{loan_amount:,.0f}")
                c2.metric("Lenders in Syndicate", str(alloc_sel["lender_id"].nunique()))
                c3.metric("Total Funded", f"₹{alloc_sel['funded_amount'].sum():,.0f}")

                max_share = float(alloc_sel["share"].max())
                breaches = int((alloc_sel["share"] > 0.20 + 1e-9).sum())

                c4, c5 = st.columns(2)
                c4.metric("Max Lender Share", f"{max_share:.2%}")
                c5.metric("20% Cap Breaches", str(breaches))

                if breaches == 0:
                    st.success("Cap respected for this loan ✅")
                else:
                    st.error("Cap breached for this loan ❌")

                view = alloc_sel[["lender_id", "funded_amount", "share"]].copy()
                view["funded_amount"] = view["funded_amount"].round(0).astype(int)
                view["share_%"] = (view["share"] * 100).round(2)
                st.dataframe(view[["lender_id", "funded_amount", "share_%"]].head(50), use_container_width=True)
                st.caption("Share should be ≤ 20% per lender per loan (RBI diversification constraint).")

        st.divider()

        colA, colB = st.columns(2)

        with colA:
            if st.button("Run Matching (Allocate Lenders)"):
                loans_reset = st.session_state.loans.copy()
                loans_reset = coerce_loan_ids(loans_reset)

                loans_reset["remaining_amount"] = loans_reset["loan_amount"]
                loans_reset["lender_count"] = 0

                alloc_reset = pd.DataFrame(columns=["lender_id", "loan_id", "funded_amount", "risk_category"])

                updated_loans, updated_alloc = run_matching(
                    loans_reset,
                    alloc_reset,
                    st.session_state.lenders,
                    min_lenders=5,
                    max_share=0.20
                )

                st.session_state.loans = ensure_cols(updated_loans, {"purpose": "NA"})
                st.session_state.loans = coerce_loan_ids(st.session_state.loans)
                st.session_state.loans["risk_category"] = st.session_state.loans["risk_category"].fillna("NA").astype(str)
                st.session_state.loans["purpose"] = st.session_state.loans["purpose"].fillna("NA").astype(str)

                st.session_state.alloc_df = updated_alloc.copy()
                st.session_state.alloc_df["loan_id"] = st.session_state.alloc_df["loan_id"].astype(str)
                st.session_state.alloc_df["lender_id"] = st.session_state.alloc_df["lender_id"].astype(str)
                st.session_state.alloc_df["funded_amount"] = safe_numeric(st.session_state.alloc_df["funded_amount"], 0.0)

                st.success("Matching complete ✅ (Reset + Recomputed) Marketplace + Compliance updated.")

        with colB:
            loans_now = st.session_state.loans
            funded = int((loans_now["remaining_amount"] <= 0).sum())
            total = int(len(loans_now))
            rate = (funded / total * 100) if total else 0
            st.metric("Loans Fully Funded", f"{funded}/{total}")
            st.metric("Funding Success %", f"{rate:.1f}%")

# ============================================================
# TAB 3 — Compliance (Admin only)
# ============================================================
if role == "Admin":
    with tab3:
        st.header("System Health & Compliance")

        loans = ensure_cols(st.session_state.loans.copy(), {"purpose": "NA"})
        alloc_df = st.session_state.alloc_df.copy()

        loans = coerce_loan_ids(loans)
        alloc_df["loan_id"] = alloc_df["loan_id"].astype(str)
        alloc_df["lender_id"] = alloc_df["lender_id"].astype(str)

        required_loan_cols = {"loan_id", "loan_amount", "remaining_amount", "lender_count"}
        missing = required_loan_cols - set(loans.columns)
        if missing:
            st.error(f"Missing columns for RBI checks: {sorted(list(missing))}")
            st.stop()

        MIN_LENDERS = 5
        MAX_PER_LOAN = 0.20

        total_loans = len(loans)
        funded_loans = int((loans["remaining_amount"] <= 0).sum())
        funding_rate = (funded_loans / total_loans) * 100 if total_loans else 0

        funded_mask = loans["remaining_amount"] <= 0
        min_lenders_viol = int((loans.loc[funded_mask, "lender_count"] < MIN_LENDERS).sum())

        open_mask = loans["remaining_amount"] > 0
        open_needing_lenders = int((loans.loc[open_mask, "lender_count"] < MIN_LENDERS).sum())

        # ✅ dtype-safe merge (both loan_id are strings)
        alloc_merged = alloc_df.merge(loans[["loan_id", "loan_amount"]], on="loan_id", how="left")
        alloc_merged = alloc_merged.dropna(subset=["loan_amount"])
        alloc_merged["share"] = alloc_merged["funded_amount"] / alloc_merged["loan_amount"].replace(0, 1)

        cap_viol = int((alloc_merged["share"] > MAX_PER_LOAN + 1e-9).sum())
        total_violations = min_lenders_viol + cap_viol

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loans Fully Funded", f"{funded_loans}/{total_loans}")
        c2.metric("Funding Success %", f"{funding_rate:.1f}%")
        c3.metric("RBI Violations (Originated)", str(total_violations))
        c4.metric("Open Loans <5 Lenders", str(open_needing_lenders))

        if total_violations == 0:
            st.success("All diversification rules satisfied ✅")
        else:
            st.error("Some diversification rules were violated ❌")

        with st.expander("Show funded-loan min-lenders violations (sample)"):
            st.dataframe(
                loans[(loans["remaining_amount"] <= 0) & (loans["lender_count"] < MIN_LENDERS)][
                    ["loan_id", "loan_title", "risk_category", "loan_amount", "lender_count", "remaining_amount"]
                ].head(50),
                use_container_width=True
            )

        with st.expander("Show 20% cap violations (sample)"):
            st.dataframe(
                alloc_merged[alloc_merged["share"] > MAX_PER_LOAN][
                    ["loan_id", "lender_id", "funded_amount", "loan_amount", "share"]
                ].head(50),
                use_container_width=True
            )