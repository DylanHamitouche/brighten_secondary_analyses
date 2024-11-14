# This script will perform a elastic regression net to categorize engaged vs withdrew participants
# We loop the script to do it again without including study version in features

# However, in this script, we will train the model on v1 and test it on v2


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap


# Load datasets
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')


# One-hot encode
imputed_complete_df = pd.get_dummies(imputed_complete_df, columns=['engagement', 'race', 'device', 'study_arm', 'study', 'working', 'gender', 'education', 'income_satisfaction', 'marital_status', 'income_lastyear', 'completion_group'], drop_first=False)


features = [
    'baseline_phq9_result',
    'age',
    'race_African-American/Black',
    'race_American Indian/Alaskan Native',
    'race_Asian',
    'race_Hispanic/Latino',
    'race_More than one',
    'race_Native Hawaiian/other Pacific Islander',
    'race_Non-Hispanic White',
    'race_Other',
    'device_Android',
    'device_iPhone',
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




# Drop rows with nan because we can't run the model with nan values
imputed_complete_df = imputed_complete_df.dropna(subset=features)
print(f'Number of participants: {len(imputed_complete_df)}')





################################## NOW WE TRY TO PREDICT ENGAGEMENT

# Split data based on study version
train_df = imputed_complete_df[imputed_complete_df['study_Brighten-v1'] == 1]
test_df = imputed_complete_df[imputed_complete_df['study_Brighten-v2'] == 1]

# Define the features and target variable for training and testing sets
X_train = train_df[features]
y_train = train_df['engagement_engaged']

X_test = test_df[features]
y_test = test_df['engagement_engaged']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Elastic Net model with cross-validation
model = ElasticNetCV(cv=5, random_state=42, max_iter=50000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_class = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to class labels

# Print classification report
print(classification_report(y_test, y_pred_class))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f"Confusion Matrix:\n{conf_matrix}")

# Explain the model with SHAP values
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Calculate mean absolute SHAP values to determine feature importance
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
important_feature_indices = np.argsort(shap_abs_mean)[-5:]  # Get indices of top 5 features
important_features = [features[i] for i in important_feature_indices]

# Plot SHAP values for the top 5 features
shap.summary_plot(shap_values[:, important_feature_indices], X_test_scaled[:, important_feature_indices], feature_names=important_features, show=False)

# Save the plot as PNG
plt.show()
plt.clf()  # Clear the current figure


# The model is terrible and classifies every v2 participant as a quitter
# When the model is applied to predict completion group, it does not converge, even if max_iter is increased





####################################### NOW WE TRY TO PREDICT COMPLETION GROUP

imputed_complete_df[imputed_complete_df['avg_completion_rate'] > 0]

features = features + ['mean_score_phq2','mean_score_phq9']

imputed_complete_df = imputed_complete_df.dropna(subset = features)


# Split data based on study version
train_df = imputed_complete_df[imputed_complete_df['study_Brighten-v1'] == 1]
test_df = imputed_complete_df[imputed_complete_df['study_Brighten-v2'] == 1]

print(f'Debug. Sample size for testing: {len(imputed_complete_df[imputed_complete_df["study_Brighten-v2"]==1])}')

# Define the features and target variable for training and testing sets
X_train = train_df[features]
y_train = train_df['completion_group_high']

X_test = test_df[features]
y_test = test_df['completion_group_high']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Elastic Net model with cross-validation
model = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_class = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to class labels

# Print classification report
print(classification_report(y_test, y_pred_class))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f"Confusion Matrix:\n{conf_matrix}")

# Explain the model with SHAP values
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Calculate mean absolute SHAP values to determine feature importance
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
important_feature_indices = np.argsort(shap_abs_mean)[-5:]  # Get indices of top 5 features
important_features = [features[i] for i in important_feature_indices]

# Plot SHAP values for the top 5 features
shap.summary_plot(shap_values[:, important_feature_indices], X_test_scaled[:, important_feature_indices], feature_names=important_features, show=False)

# Save the plot as PNG
plt.show()
plt.clf()  # Clear the current figure



##################################### NOW WE TRY TO PREDICT AVG COMPLETION RATE (DISCRETE VALUE INSTEAD OF CATEGORIZATION)


# Avg completion rate
# Define the features and target variable
X = imputed_complete_df[features]
y = imputed_complete_df['avg_completion_rate']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Elastic Net model with cross-validation
model = ElasticNetCV(cv=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")


# Explain the model with SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Calculate mean absolute SHAP values to determine feature importance
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
important_feature_indices = np.argsort(shap_abs_mean)[-5:]  # Get indices of top 5 features
important_features = [features[i] for i in important_feature_indices]

# Plot SHAP values for the top 5 features
shap.summary_plot(shap_values[:, important_feature_indices], X_test[:, important_feature_indices], feature_names=important_features, show=False)

# Save the plot as PNG
plt.show()
plt.clf()  # Clear the current figure