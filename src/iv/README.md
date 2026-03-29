# Post-Mortem Analysis & Model Improvement Strategy

This document outlines the evaluation of the initial **Credit Risk Model (v1.0)**, identifies core technical challenges, and defines a strategic roadmap for optimization.

---

## 1. Current State: The Baseline Model
The initial model utilized **Logistic Regression** with **Weight of Evidence (WoE)** binning. While the pipeline is technically sound, the performance metrics indicate that the model is not yet ready for production.

### Key Performance Indicators (KPIs):
* **AUC-ROC:** ~0.55 (Near-random performance).
* **Information Value (IV):** Most features < 0.1 (Weak predictive power).
* **Total Profit:** Negative at all thresholds (Maximum Loss: -336).

---

## 2. Identified Problems (The "Why")

### A. Low Feature Signal (The IV Problem)
The primary bottleneck is the lack of "Strong" predictors. Raw features like `age` or `income` individually do not provide enough separation between "Good" and "Bad" borrowers. Simply scaling the data (StandardScaler) does not help because the **Information Value (IV)** remains low.

### B. Linear Constraints
Logistic Regression is a linear classifier. It assumes that risk increases or decreases in a straight line relative to the input. However, credit risk is often **non-linear** and depends on **interactions** (e.g., a high loan amount is only risky *if* the income is low).

### C. Aggressive Business Constraints
The P&L structure (+10 Profit vs. -70 Loss) requires high **Precision**. To break even, the model must be at least 87.5% accurate in its "Approved" category. The current model's overlap in probability distribution makes this impossible.

---

## 3. Strategic Solutions (The "How")

### Phase 1: Feature Engineering (Boosting IV)
Instead of scaling, we will **re-engineer** features to capture higher Information Value:
* **Interaction Ratios:** Create features like `Loan_to_Income`, `DTI_x_Utilization`, and `Payment_Stress_Index`.
* **Advanced Binning:** Transition from 5 bins to **10-20 bins (Fine Binning)** to capture specific risk pockets.
* **Special Category Handling:** Isolate "Zero" values or "Missing" values into unique bins to prevent signal dilution.



### Phase 2: Algorithmic Upgrade (Ensemble Power)
To capture complex, non-linear patterns, we will move beyond Logistic Regression:
* **Gradient Boosted Trees (XGBoost/LightGBM):** These models natively handle feature interactions and non-linear boundaries.
* **Cost-Sensitive Learning:** Instead of SMOTE, we will use `scale_pos_weight` to penalize the model heavily for missing a "Default" (False Negative), aligning the model with the -70 loss penalty.

### Phase 3: Calibration for Profitability
* **Precision-Targeted Thresholds:** Shift the optimization goal from the "KS-Statistic" to **"Maximum Net Profit."**
* **Conservative Lending Policy:** Implement a stricter cutoff to ensure the "Approved" pool meets the required precision for a positive ROI.