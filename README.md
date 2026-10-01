# Credit Risk Prediction Model

Guillermo Cascales Ñíguez

## Overview
The goal of this project is to build a Machine Learning model that predicts whether a person applying for a loan will default (fail to pay back the loan) or not default.
Different types of models are used, so that we can assess which model works better in this environment and actually learns the features, thus predicting correctly if the borrower defaults or not in a real situation.

## Dataset
The dataset was obtained from the website Kaggle, and it contains columns simulating credit bureau data. These are the explanatory variables we will take into account for the calculation

| Feature Name | Description |
|-----------|-----------|
| person_age	| Age
| person_income	| Annual Income
| person_home_ownership	| Home ownership
| person_emp_length |	Employment length (in years)
| loan_intent |	Loan intent
| loan_grade |	Loan grade
| loan_amnt |	Loan amount
| loan_int_rate	| Interest rate
| loan_status	| Loan status (0 is non default 1 is default)
| loan_percent_income	| Percent income
| cb_person_default_on_file	| Historical default
| cb_person_cred_hist_length |	Credit history length

It is important to note that this dataset is imbalanced, since only 21% percent of the data contains defaults. 
After the model is trained, we will also check which features were the ones that mattered the most and added more weight to the decision made by the model. This can be done easily and clearly with models that are transparent, such as decision trees. For neural networks, for example, it is not that simple, therefore not being the best option when clarity in the assessment process is needed.

## Techniques used 
 - **Data Cleaning & Sanity Checks:** Identified and filtered out impossible human entry errors (like ages over 100 or work experience over 60 years).
 - **Smart Feature Encoding:**
   Ordinal Encoding: Used ordered numbers for naturally ranked features like loan_grade (Grade A is safer than Grade B, C, D, etc.).
   One-Hot Encoding (get_dummies): Converted categories without inherent order (like home ownership type or loan intent) into binary columns (0 or 1).
 - **Data Scaling & Preventing Data Leakage:** Scaled numerical features using StandardScaler (so features on large scales don't dominate). Fitted the scaler only on training data inside cross-validation splits to ensure future test data never leaked into the training phase.
 - **Model Training & Comparison:** Evaluated three different algorithms: K-Nearest Neighbors (KNN), Logistic Regression, and XGBoost: XGBoost performed the best because tree-based gradient boosting excels at learning non-linear relationships and feature interactions.
- **Cross-Validation & Evaluation Metric:** Used 5-Fold Stratified Cross-Validation to ensure every fold maintained the 21% default balance.
Evaluated models using ROC-AUC (Receiver Operating Characteristic - Area Under Curve), because it measures how well the model ranks risky borrowers across all probability levels.
 - **Threshold Tuning & Holdout Test Evaluation:** Kept a 20% Holdout Test Set completely locked away until the very end to evaluate on unseen data. Lowered the prediction threshold from 0.50 down to 0.25. This dramatically increased the model's sensitivity to defaults (high Recall).

## Results
When evaluating on 5,727 unseen loan applications we achieved 92% Overall Accuracy and 82% Recall for Defaults, i.e. the model successfully caught 8 out of 10 defaulters (1,015 out of 1,241 actual defaults). The Precision was 80%: When the model flagged an applicant as high risk, it was correct 80% of the time. 

In retail banking, the cost of a False Negative (losing the entire principal on a defaulted loan) outweighs the opportunity cost of a False Positive (rejecting a creditworthy borrower).

By lowering the decision threshold from the default 0.50 down to 0.25, the model captured 82% of all potential non-performing loans while keeping unnecessary rejections low. This framework provides a transparent, scalable, and mathematically tool for automated loan underwriting, risk-based pricing, and capital allocation.

## Limitations
The dataset relies entirely on static borrower snapshots (income, loan amount, credit history). It doesn't include real-time economic indicators like inflation rates, interest rate changes, or unemployment trends, which heavily influence real-world default rates.

Furthermore, features like _person_home_ownership_RENT_ or _person_age_ can act as proxies for socioeconomic or age-related disparities. In strictly regulated credit markets, models must undergo legal bias/fairness audits to ensure they do not unintentionally discriminate against protected classes.

