# Same thing as script 'elastic_net_linear.py', but we train model on v1 and test on v2 to assess if the model can generalize

# Lasso model of categorization (high completion vs low completion) drops from 70% accuracy to 50% accuracy
# Lasso model for linear regression (avg_completion_rate) has the same accuracy (MSE = 0.04 ; R2 error = 0.1 ; MAE = 0.18)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import shap

# Load dataset
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

# Create dummy variables and drop 'completion_group_no group'
imputed_complete_df = pd.get_dummies(imputed_complete_df, columns=['completion_group', 'race', 'device', 'study_arm', 'study', 'working', 'gender', 'education', 'income_satisfaction', 'marital_status', 'income_lastyear'], drop_first=False)
imputed_complete_df = imputed_complete_df.drop(columns='completion_group_no group')

# Define feature columns
# I removed race and device 
feature_columns = [
    'mean_score_phq2',
    'mean_score_phq9',
    'baseline_phq9_result',
    # 'mean_score_sleep',
    # 'gad7_sum',
    # 'alc_sum',
    'age',
    # 'mean_score_sds',
    # 'race_African-American/Black',
    # 'race_American Indian/Alaskan Native',
    # 'race_Asian',
    # 'race_Hispanic/Latino',
    # 'race_More than one',
    # 'race_Native Hawaiian/other Pacific Islander',
    # 'race_Non-Hispanic White',
    # 'race_Other',
    # 'device_Android',
    # 'device_iPhone',
    'study_arm_EVO',
    'study_arm_HealthTips',
    'study_arm_iPST',
    'working_No',
    'working_Yes',
    'gender_Female',
    'gender_Male',
    'education_Community College',
    'education_Elementary School',
    'education_Graduate Degree',
    'education_High School',
    'education_University',
    'income_satisfaction_Am comfortable',
    "income_satisfaction_Can't make ends meet",
    'income_satisfaction_Have enough to get along',
    'marital_status_Married/Partner',
    'marital_status_Separated/Widowed/Divorced',
    'marital_status_Single',
    'income_lastyear_100,000+',
    'income_lastyear_20,000-40,000',
    'income_lastyear_40,000-60,000',
    'income_lastyear_60,000-80,000',
    'income_lastyear_80,000-100,000',
    'income_lastyear_< $20,000'
]

# Drop rows with missing values in the selected feature set
imputed_complete_df = imputed_complete_df.dropna(subset=feature_columns)
print(f'Number of participants: {len(imputed_complete_df)}')

# Define features and target variable
features = feature_columns
X = imputed_complete_df[features]

# Classification task
y_class = imputed_complete_df['completion_group_high']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data based on study version for classification
X_train = imputed_complete_df[imputed_complete_df['study_Brighten-v1'] == 1][features]
y_train = imputed_complete_df[imputed_complete_df['study_Brighten-v1'] == 1]['completion_group_high']
X_test = imputed_complete_df[imputed_complete_df['study_Brighten-v2'] == 1][features]
y_test = imputed_complete_df[imputed_complete_df['study_Brighten-v2'] == 1]['completion_group_high']

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model for classification
model_class = LogisticRegression(random_state=42)
model_class.fit(X_train_scaled, y_train)

# Make predictions
y_pred_class = model_class.predict(X_test_scaled)

# Print classification report
print('When predicting completion group')
print(classification_report(y_test, y_pred_class))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f"Confusion Matrix for classification:\n{conf_matrix}")

# Explain the model with SHAP values
explainer_class = shap.Explainer(model_class, X_train_scaled)
shap_values_class = explainer_class(X_test_scaled)

# Calculate mean absolute SHAP values to determine feature importance
shap_abs_mean_class = np.abs(shap_values_class.values).mean(axis=0)
important_feature_indices_class = np.argsort(shap_abs_mean_class)[-5:]  # Get indices of top 5 features
important_features_class = [features[i] for i in important_feature_indices_class]

# Plot SHAP values for the top 5 features
shap.summary_plot(shap_values_class[:, important_feature_indices_class], X_test_scaled[:, important_feature_indices_class], feature_names=important_features_class, show=False)
plt.show()
plt.clf()  # Clear the current figure

# Regression task
y_reg = imputed_complete_df['avg_completion_rate']

# Split data for regression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

# Train ElasticNet model for regression
model_reg = ElasticNetCV(cv=5, random_state=42)
model_reg.fit(X_train, y_train)

# Make predictions
y_pred_reg = model_reg.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)

# Print accuracy metrics
print(f'Accuracy metrics for regression (when predicting average completion rate):')
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Explain the model with SHAP values
explainer_reg = shap.Explainer(model_reg, X_train)
shap_values_reg = explainer_reg(X_test)

# Calculate mean absolute SHAP values to determine feature importance
shap_abs_mean_reg = np.abs(shap_values_reg.values).mean(axis=0)
important_feature_indices_reg = np.argsort(shap_abs_mean_reg)[-5:]  # Get indices of top 5 features
important_features_reg = [features[i] for i in important_feature_indices_reg]

# Plot SHAP values for the top 5 features
shap.summary_plot(shap_values_reg[:, important_feature_indices_reg], X_test[:, important_feature_indices_reg], feature_names=important_features_reg, show=False)
plt.show()
plt.clf()  # Clear the current figure

# Plot predicted vs actual average completion rate
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_test, y_pred_reg, alpha=0.5, color='blue', edgecolor='k', label='Predicted values')
line = plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction line')
plt.xlabel('Actual Average Completion Rate')
plt.ylabel('Predicted Average Completion Rate')
plt.title('Predicted vs Actual Average Completion Rate')
plt.legend()  # Add a legend to the plot
plt.show()
plt.clf()  # Clear the current figure
