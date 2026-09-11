# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer will churn (`Yes`/`No`) from
account and usage data, and serves the trained model through an interactive Streamlit web app.

## Project Structure

```
├── customer_churn_data.csv   # Raw dataset
├── model_training.ipynb      # Full ML workflow: EDA, cleaning, preprocessing, training, evaluation
├── churn_pipeline.pkl        # Saved preprocessing + model pipeline (output of the notebook)
├── app.py                    # Streamlit prediction app
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dataset

17 columns covering customer demographics (`age`, `gender`, `region`), account details
(`tenure_months`, `contract_type`, `payment_method`), service usage (`internet_service`,
`tech_support`, `online_security`, `avg_monthly_usage_gb`), billing (`monthly_charges`,
`total_charges`, `paperless_billing`), and the target column `churn`.

## Approach

1. **Data cleaning** — removed 20 exact duplicate rows, filled structurally-missing service
   columns (`tech_support`, `online_security`) with `"No internet service"`, dropped the
   non-predictive `customer_id` identifier.
2. **EDA** — visualized churn distribution and its relationship to contract type, internet
   service, monthly charges, tenure, and payment method.
3. **Preprocessing** — a Scikit-Learn `ColumnTransformer` (median imputation + scaling for
   numeric columns, most-frequent imputation + one-hot encoding for categorical columns) wrapped
   in a `Pipeline`, so the exact same transformation runs at both training and prediction time.
4. **Modeling** — trained and compared Logistic Regression, Random Forest, and Decision Tree
   classifiers using Accuracy, Precision, Recall, and F1 score.
5. **Model selection** — the model with the best **F1 score** on the held-out test set was
   selected (not just training accuracy) and saved as `churn_pipeline.pkl`.

## Running Locally

```bash
pip install -r requirements.txt
jupyter notebook model_training.ipynb   # (re)train the model, produces churn_pipeline.pkl
streamlit run app.py                    # launch the prediction app
```

## Deployment

The app is deployed on **Streamlit Community Cloud**, connected to this GitHub repository.
Live app link: _add your deployed URL here_

## Notes

- The Streamlit app loads `churn_pipeline.pkl` directly — no predictions are hard-coded.
- The same preprocessing pipeline used at training time is reused automatically at prediction
  time, since it is saved as part of the single pipeline object.
