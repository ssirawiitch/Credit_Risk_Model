# Quantitative Credit Risk & Multi-Model Scoring Framework

A scalable machine learning pipeline designed to assess borrower creditworthiness and optimize risk-adjusted profitability. This project evolves from traditional statistical scoring (Logistic Regression/WoE) into advanced Ensemble and Deep Learning architectures.

## Project Mission
To build a robust, end-to-end credit engine that doesn't just predict defaults but simulates business outcomes (P&L) to drive data-driven lending decisions.

---

## Phased Implementation (Roadmap)

### Phase 1: Statistical Foundation (Current)
* **Model:** Logistic Regression (Lasso & Ridge Regularization).
* **Feature Engineering:** Weight of Evidence (WoE) binning and Information Value (IV) analysis to capture monotonic risk trends.
* **Optimization:** Hyperparameter tuning ($C$ grid search) and Youden’s J Index for optimal threshold selection.

### Phase 2: Ensemble Learning (Next)
* **Objective:** Capture non-linear relationships and feature interactions without manual binning.
* **Models:** Random Forest, XGBoost, and LightGBM.
* **Technique:** Utilizing SHAP values for model explainability (XAI) in a regulated environment.

### Phase 3: Deep Learning Architecture (Future)
* **Objective:** Leverage high-dimensional data and automated feature extraction.
* **Architecture:** Tabular Neural Networks (TabNet) or Multi-Layer Perceptrons (MLP) with Dropout and Batch Normalization.
* **Focus:** Deep embedding layers for categorical data (Home Ownership, Loan Purpose).

---

## Tech Stack
- **Languages:** Python
- **Modeling:** Scikit-Learn, Imbalanced-Learn (SMOTE/Weighted Classes)
- **Math/Stats:** Pandas, NumPy, Scipy
- **Visualization:** Matplotlib, Seaborn (Custom P&L Dashboards)

---

## Business Performance & Analytics
This framework evaluates model performance through a "Quant" lens, focusing on:
* **KS Statistic & Gini:** Measuring the separation power between 'Good' and 'Bad' borrowers.
* **P&L Simulation:** A custom analytics engine that calculates:
    * **Approval Rate %** vs. **Default Rate %**
    * **Net Profit per Loan:** Based on interests gained vs. capital lost on defaults.
* **Normalized Trade-off:** Visualization of the "Sweet Spot" between aggressive lending and risk aversion.

---

## Repository Structure
```text
├── data/               # Raw and cleaned datasets
├── src/                
│   ├── iv/              # Analyzing and Fixing iv
│   ├── models/          # Cleaning and Feature Engineering + Logistic Regression, RF, and DL scripts + P&L and Business Logic
└── README.md