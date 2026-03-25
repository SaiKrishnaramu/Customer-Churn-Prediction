# 📡 Telco Customer Churn Prediction

**End-to-End Machine Learning Workflow with Business Impact Analysis**

> 💡 Every churned customer = lost recurring revenue. This project builds a churn prediction model that outputs *churn probability per customer*, segments them by risk tier, and calculates the exact **monthly revenue at risk** — enabling data-driven retention strategies.

---

## 🎯 Project Overview

This notebook tackles **customer churn prediction** for a telecom company using the [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset schema. It goes beyond simple classification by translating model outputs into **actionable business insights** and **₹ revenue impact**.

### What This Project Does

| Stage | Description |
|-------|-------------|
| **Data Preprocessing** | Handles missing values, encodes categoricals, prepares features |
| **Exploratory Data Analysis** | Interactive Plotly visualizations revealing churn drivers |
| **Model Training** | HistGradientBoostingClassifier with cross-validated ROC-AUC |
| **Business Impact Analysis** | Risk-tier segmentation, revenue-at-risk calculation, ROI simulation |

---

## 📊 Key Findings

| Insight | Finding |
|---------|---------|
| **Top churn driver** | Month-to-month contracts have 3× higher churn than two-year contracts |
| **Payment risk** | Electronic check users show the highest churn rate |
| **Revenue signal** | Churned customers pay higher monthly charges on average |
| **Tenure effect** | Customers past 24 months have very low churn probability |

---

## 🏗️ Project Structure

```
├── churn_prediction.ipynb   # Main notebook (end-to-end workflow)
└── README.md                # This file
```

### Notebook Sections

1. **Section 0** — Imports & Setup
2. **Section 1** — Data Loading & Preprocessing
3. **Section 2** — Exploratory Data Analysis (EDA)
4. **Section 3** — Model Training (HistGradientBoosting + `predict_proba()`)
5. **Section 4** — Business Impact & ROI Analysis
6. **Section 5** — Conclusions & Recommendations

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | `HistGradientBoostingClassifier` (scikit-learn) |
| **Max Iterations** | 200 |
| **Learning Rate** | 0.08 |
| **Max Depth** | 4 |
| **Validation** | 5-Fold Stratified Cross-Validation |
| **Primary Metric** | ROC-AUC |
| **Output** | Churn probability (0–1) per customer |

---

## 💰 Business Impact Framework

Customers are segmented into risk tiers based on predicted churn probability:

| Tier | Churn Probability | Recommended Action |
|------|-------------------|--------------------|
| 🔴 **Critical Risk** | ≥ 80% | Immediate personal outreach, discount offer |
| 🟠 **Warning** | 50% – 80% | Proactive support, loyalty rewards |
| 🟢 **Safe** | < 50% | Regular engagement |

### ROI Simulation

The notebook includes a **Retention ROI Simulator** that estimates:
- Revenue saved through targeted interventions
- Campaign cost vs. net benefit
- ROI percentage per risk tier

---

## 📈 Visualizations

All charts are built with **Plotly** for full interactivity:

- 🍩 Overall Churn Rate (donut chart)
- 📊 Monthly Charges Distribution — Churned vs. Retained
- 📦 Tenure by Contract Type & Churn Status (box plot)
- 📉 Churn Rate by Payment Method
- 📈 ROC Curve
- 🔑 Feature Importance
- 💰 Monthly Revenue at Risk by Tier
- 📊 Churn Probability Distribution by Risk Tier

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — model training & evaluation
- **Plotly** — interactive visualizations
- **IPython** — notebook display utilities

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn plotly
```

### Run the Notebook

```bash
jupyter notebook churn_prediction_1.ipynb
```

> **Note:** The notebook includes a built-in data simulator that generates a realistic 7,043-row dataset matching the IBM Telco schema. To use the original Kaggle dataset instead, replace the simulation block in Section 1 with:
> ```python
> df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
> ```

---

## 💡 Recommended Actions (from Analysis)

1. **Critical Risk customers** → Immediate outreach + contract upgrade discount (e.g., 20% off annual plan)
2. **Warning tier** → Automated loyalty email + proactive tech support check-in
3. **Month-to-month customers** → Incentivize switching to annual contracts at onboarding
4. **Electronic check users** → Offer auto-pay discount to reduce friction
5. **New customers (tenure < 6 months)** → Dedicated onboarding specialist

### Expected Outcome

- **Churn reduction:** 5–8% absolute decrease in monthly churn rate
- **Revenue protected:** ₹25,000–₹60,000/month depending on intervention success
- **CLV increase:** Disproportionate LTV gains from retaining high-value customers

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with ❤️ by the Data Science Team</i>
</p>
