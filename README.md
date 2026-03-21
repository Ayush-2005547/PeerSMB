# PeerSMB – AI Credit Scoring & Lending Marketplace

PeerSMB is an end-to-end machine learning–based lending platform designed for Indian SMB (Small & Medium Business) sellers. It simulates how lenders can evaluate borrower risk using alternative financial data, allocate loans across multiple lenders, and enforce diversification constraints in a marketplace setting.

## Live Demo
**App:** https://peersmb.streamlit.app

## Overview
Traditional credit systems often fail to serve small sellers with limited formal credit history. PeerSMB demonstrates how alternative data such as marketplace settlements, bank statements, and business stability signals can be used to estimate risk and support lending decisions.

The platform includes three role-based workflows:

- **Borrower Portal** – upload business documents, derive financial signals, and generate a risk score
- **Lender Portal** – browse marketplace loans, filter by risk, and view lender allocation syndicates
- **Compliance Console** – monitor funding success, diversification rules, and cap violations

## Key Features

### Borrower Workflow
- Loan application form with business inputs
- Upload of marketplace settlement CSV
- Optional bank statement CSV upload
- Demo data mode for quick testing
- Feature extraction from uploaded financial data
- Risk scoring using a trained XGBoost model
- Risk band classification and indicative interest rate generation
- Submission of scored loans into the marketplace

### Credit Scoring Engine
- Machine learning model trained for default probability prediction
- Alternative-data-driven proxy features such as:
  - monthly revenue
  - payout regularity
  - payout volatility
  - refund rate
  - inflow volatility
  - low balance days
  - risk flags
  - business age
- Mapping of derived business signals into model-ready inputs like affordability and credit proxy

### Lender Marketplace
- Marketplace view of active loans
- Filtering by:
  - risk band
  - probability of default
  - loan amount
- Sorting by funding need, lowest risk, highest risk, and amount
- Syndicate view for each loan showing lender contributions
- Allocation summaries including share percentage per lender

### Matching & Allocation Engine
- Multi-lender allocation logic for funding loans
- Greedy matching engine for lender-loan assignment
- Supports syndicate-style funding
- Tracks remaining amount and lender count per loan
- Recomputes allocations for open marketplace loans

### Compliance Console
- Funding success monitoring
- RBI-inspired diversification logic
- Cap check to ensure no lender exceeds 20% exposure on a single loan
- Minimum-lender checks on funded loans
- Violation summaries and drill-down tables

## Tech Stack
- **Language:** Python
- **Frontend / App:** Streamlit
- **Libraries:** Pandas, NumPy, Joblib, Scikit-learn, XGBoost
- **Artifacts:** CSV-based demo data, trained XGBoost model
- **Notebooks:** Training, feature engineering, risk analysis, matching engine experiments

## Project Workflow
1. Borrower uploads marketplace and optional bank statement data
2. System extracts financial signals from uploaded CSVs
3. Alternative-data stability score is derived
4. Stability metrics are converted into model inputs
5. XGBoost model predicts probability of default
6. Risk band and indicative pricing are assigned
7. Application is submitted to the marketplace
8. Lender allocation engine distributes funding across lenders
9. Compliance console checks exposure constraints and funding quality

## Repository Structure

```bash
PEERSMB/
├── app.py
├── generate_demo_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── rejected_loans_with_predictions.csv
│   ├── fundable_loans_post_risk_filter.csv
│   └── lender_loan_allocations.csv
│
├── demo_data/
│   ├── bank_statement_demo.csv
│   └── marketplace_settlement_demo.csv
│
├── models/
│   └── xgb_reduced_credit_model.pkl
│
├── notebooks/
│   ├── peersmb_model_training.ipynb
│   ├── rejected_loan_risk_analysis.ipynb
│   ├── bipartite_matching_engine.ipynb
│   └── upi_feature_engineering.ipynb

How to Run Locally
Clone the repository
git clone [https://github.com/YOUR_USERNAME/PeerSMB](https://github.com/Ayush-2005547/PeerSMB).git
cd PeerSMB
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run app.py
Demo Credentials

Use the demo credentials currently defined in app.py.

Highlights
Simulates a lending marketplace instead of a basic standalone ML model
Combines machine learning, financial feature engineering, and workflow design
Includes compliance-aware lender allocation rather than prediction-only output
Uses alternative data concepts to model lending for thin-file SMB borrowers
Publicly deployed Streamlit application with role-based workflow
Future Improvements
Add SHAP-based model explainability
Replace static CSV data with database-backed storage
Add stronger borrower/lender authentication
Improve lender appetite modelling and portfolio optimization
Extend compliance layer with richer policy constraints
Add APIs for GST / transaction ingestion in a production version
Author

Ayush Ahirwar

License

This project is for academic, portfolio, and demonstration purposes.
