# Churn Prediction Pipeline

## 1. Approach

### Data Preparation
- Loaded training features and labels, merged into a single DataFrame.
- Dropped columns with more than 50% missing values.
- Split features into numerical and categorical columns.

### Feature Engineering
- Imputed missing values: mean for numerical, most frequent for categorical.
- Scaled numerical features using `StandardScaler`.
- Encoded categorical features using a custom `MultiColumnLabelEncoder` (label encoding).
- Dropped highly correlated numerical features (correlation > 0.7) using a custom transformer.
- Ensured all transformations preserved original column names for interpretability.

### Feature Selection
- Trained a Random Forest model to compute feature importances.
- Selected features that together account for at least 80% of cumulative importance.

## 2. Model Selection and Rationale
- **Random Forest**: Used for initial feature importance due to its robustness and ability to handle mixed data types.
- **XGBoost**: Chosen as the main classifier for its strong performance on tabular data, ability to handle class imbalance, and support for advanced hyperparameter tuning.
- **LightGBM**: Also evaluated for comparison, as it is efficient and often competitive with XGBoost.
- **SMOTE** and **class weights**: Used to address class imbalance in the target variable.
- **Optuna**: Used for hyperparameter tuning to optimize model performance.

## 3. Evaluation: Test Set Accuracy and F1-Score Rationale
- The test set was created using a 70/30 train-test split.
- Model performance was evaluated using **F1-score** as the primary metric, along with accuracy and ROC-AUC.
- **Rationale for F1-score**: The dataset is imbalanced (churn is a minority class). F1-score balances precision and recall, making it more informative than accuracy for imbalanced classification tasks. It ensures that both false positives and false negatives are penalized, which is critical for churn prediction where both types of errors are costly.
- We have also tried varying the classification threshold to further optimize the F1-score, ensuring the best trade-off between precision and recall for the business objective.
- The best model and threshold were selected based on the highest F1-score on the test set.

---

*This pipeline is fully reproducible and all feature engineering, selection, and modeling steps are performed using scikit-learn pipelines for transparency and maintainability.*
