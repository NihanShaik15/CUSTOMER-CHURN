# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to
churn (stop using the service), so the business can proactively intervene with retention offers.

## 📌 Problem Statement

Customer acquisition is far more expensive than retention. This project builds a
classification model that flags high-risk customers **before** they churn, using their
account, service, and billing information.

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features after cleaning.
Includes demographics (gender, senior citizen, partner, dependents), account info
(tenure, contract, payment method, charges), and subscribed services (internet,
streaming, tech support, etc.).

## 🔍 Key EDA Findings

| Insight | Detail |
|---|---|
| Overall churn rate | **26.5%** |
| Churn by contract | Month-to-month **42.7%** vs One-year **11.3%** vs Two-year **2.8%** |
| Churn by internet service | Fiber optic **41.9%** vs DSL **19.0%** vs No internet **7.4%** |
| Tenure | Churned customers average **18 months**, retained average **37.6 months** |
| Monthly charges | Churned customers pay more on average (**$74.4** vs **$61.3**) |

**Takeaway:** Month-to-month, fiber-optic, short-tenure, higher-billed customers are the
highest-risk segment.

## 🧠 Models Trained

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.505 | 0.797 | 0.618 | 0.840 |
| **Random Forest (best)** | **0.768** | **0.545** | 0.765 | **0.636** | **0.842** |
| XGBoost | 0.758 | 0.530 | 0.767 | 0.627 | 0.840 |

All three models handle class imbalance (26% churn / 74% retained) via
`class_weight="balanced"` (Logistic Regression, Random Forest) and
`scale_pos_weight` (XGBoost).

**Why we prioritize Recall:** Missing an actual churner (false negative) costs more
than a false alarm — a retention team can afford to reach out to a happy customer by
mistake, but a missed churner is lost revenue. All models are tuned to keep recall
above 76%.

## 🔑 Top Churn Drivers (Feature Importance)

1. Contract type
2. Tenure
3. Total charges
4. Monthly charges
5. Online security subscription
6. Tech support subscription

Confirmed independently via **SHAP** values on the XGBoost model — see
`outputs/08_shap_summary.png`.

## 💡 Business Recommendations

1. Incentivize month-to-month customers to switch to annual contracts (biggest single driver).
2. Focus retention campaigns in the first 3–6 months of the customer lifecycle.
3. Investigate fiber-optic service pricing/quality — it churns 2x more than DSL.
4. Run the saved model monthly to score the active customer base and flag the top-risk segment.

## 🗂️ Project Structure

```
churn_project/
├── data/
│   └── Telco-Customer-Churn.csv       # Raw dataset
├── notebooks/
│   ├── churn_analysis.ipynb           # Main analysis notebook (run top to bottom)
│   └── churn_analysis.html            # Rendered notebook, viewable without Jupyter
├── src/
│   └── pipeline.py                    # Standalone script version (EDA + training + eval)
├── outputs/
│   ├── 01-08_*.png                    # EDA & evaluation charts
│   ├── cm_*.png                       # Confusion matrices per model
│   ├── best_model.pkl                 # Saved best-performing model
│   └── metrics_report.json            # All metrics in machine-readable form
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

```bash
pip install -r requirements.txt

# Option A: run the notebook
jupyter notebook notebooks/churn_analysis.ipynb

# Option B: run the standalone script (regenerates everything in outputs/)
python src/pipeline.py
```

## 🛠️ Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Explainability:** SHAP
- **Environment:** Jupyter Notebook

## 📈 Possible Next Steps

- Hyperparameter tuning (GridSearchCV / Optuna) for further AUC gains
- Deploy the model behind a simple Flask/Streamlit app for the retention team
- Add a monitoring pipeline to retrain the model as new customer data arrives
