 # Credit Card Churn Prediction — Direct Mail Targeting
Predicts which credit card customers are most likely to churn, so retention teams can target
direct mail offers at the accounts worth saving instead of mailing everyone.
## Problem
Credit card issuers lose revenue every time a customer closes an account. Blanket retention
campaigns waste budget mailing offers to customers who were never going to leave. This project
flags the top-decile at-risk accounts so retention spend goes where it has the most impact.
## Dataset
10,127 customer records from a US bank credit card portfolio, with 20+ features covering
demographics, account tenure, credit limit, transaction behavior, and utilization.
Source: [Credit Card Customers dataset,
Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
## Approach
1. **Framed the business problem as binary classification.** Target variable: churned vs.
active customer.
2. **Feature selection with chi-square testing.** Ran chi-square tests across 20+ candidate
features and identified Total Transaction Count and Average Utilization Ratio as the twostrongest churn drivers.
3. **Handled class imbalance with SMOTE.** The dataset has a 16% minority (churned) class.
Applied SMOTE to the training split so the model sees enough churn examples to learn the
pattern, rather than defaulting to predicting "active" for everyone.
4. **Trained and compared two classifiers.** Random Forest and Logistic Regression, tuned via
grid search on the training set.
5. **Evaluated on a held-out test split.** Selected the model configuration that gave the best
precision/recall tradeoff for a retention use case, where false positives (mailing a customer
who wasn't going to churn) cost far less than false negatives (missing one who was).
6. **Exported scored output for Power BI.** Built a dashboard tracking predicted churn rate by
customer segment, credit utilization band, and transaction frequency quartile.
## Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.96 |
| Precision (churn class) | 90% |
| Recall (churn class) | 81% |
At this precision/recall balance, the model supports high-confidence targeting with minimal
wasted mail spend, projecting a 15–20% reduction in attrition among targeted accounts versus an
untargeted campaign.
## Repo structure
```
■■■ data/
■ ■■■ README.md # where to get the dataset
■■■ churn_prediction.py # full pipeline: load, feature selection, SMOTE, train, evaluate
■■■ export_powerbi_data.py # generatepip install -r requirements.txt
python churn_prediction.py
```
This trains both models, prints the evaluation metrics, and saves the scored predictions to
`outputs/churn_predictions.csv`, which feeds directly into the Power BI dashboard.
## Tools
Python, pandas, scikit-learn, imbalanced-learn (SMOTE), scipy (chi-square testing), Power BI,s the CSV feeding the Power BI dashboard
