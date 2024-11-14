# This script will perform an elastic net regression model, but with passive data as well as demographics and questionnaire scores
# The model will be done for categorization:
#     1. High completion group vs low completion group
#     2. Engaged vs quit

# 19/08/2024: I acknowledge that it doesn't make sense to predict engagement, because the vast majority of participants who provided passive data are engaged

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import shap

# Import the passive imputed complete dataframes built in imputed_passive_dataframes.py
imputed_v1_passive_complete_df = pd.read_csv('../brighten_data/imputed_v1_passive_complete_df.csv')
imputed_v2_communication_complete_df = pd.read_csv('../brighten_data/imputed_v2_communication_complete_df.csv')
imputed_v2_mobility_complete_df = pd.read_csv('../brighten_data/imputed_v2_mobility_complete_df.csv')

# Store the dataframes and their name in lists, as we will iterate through them
list_of_df = [imputed_v1_passive_complete_df, imputed_v2_communication_complete_df, imputed_v2_mobility_complete_df]
list_names = ['imputed_v1_passive_complete_df', 'imputed_v2_communication_complete_df', 'imputed_v2_mobility_complete_df']

# This is a debug snippet
# We print the number of participants that belong to each group and see if the total matches the number of participants
# First we compare number of completion_group_high vs completion_group_low
# Then we compare number of engaged vs quit
# Output: It works, the total number of participants match the distribution between groups
for df, name in zip(list_of_df, list_names):
  print(f'Number of participants (Length of dataframe): {df["participant_id"].nunique()} ({len(df)})')
  print(f'{name}: Number of completion_group_high = {len(df[df["completion_group"]=="high"])}')
  print(f'{name}: Number of completion_group_low = {len(df[df["completion_group"]=="low"])}')
  print(f'{name}: Mean avg_completion_rate = {df["avg_completion_rate"].mean()}')
  print(f'{name}: Median avg_completion_rate = {df["avg_completion_rate"].median()}')
  print(f'{name}: Std avg_completion_rate = {df["avg_completion_rate"].std()}')



# Ok let's loop those babies in (each passive dataframe)
for df, name in zip(list_of_df, list_names):

  # Categorical features will be one-hot encoded
  df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == object and col != 'participant_id'])

  # Columns to remove from features
  columns_to_remove_from_pred = ['engagement_engaged', 'engagement_quit', 'completion_group_high', 'completion_group_low', 'participant_id', 'avg_completion_rate',
                                 'gender_Female', 'working_Yes', 'device_Android' 'alc_sum', 'gad7_sum']

  # Add columns containing 'completion' to the list
  columns_to_remove_from_pred += [col for col in df.columns if 'completion' in col]


  # Define the features and target variable
  features = [col for col in df.columns if not col in columns_to_remove_from_pred]

  # Drop rows with nan so the model can run
  df = df.dropna(subset=features)

  # Printing some metrics
  print(f'Number of participants kept in analysis for {name}: {df["participant_id"].nunique()}')
  print(f'{name}: Number of completion_group_high in analysis = {len(df[df["completion_group_high"]==1])}')
  print(f'{name}: Number of completion_group_low in analysis = {len(df[df["completion_group_low"]==1])}')
  print(f'Features for {name}: {features}')



############################# LET'S PREDICT AVG COMPLETION RATE FIRST
  # Define the features and target variable
  X = df[features]
  y = df['avg_completion_rate']

  # Standardize the features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Fit Elastic Net model with cross-validation
  model = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
  model.fit(X_train, y_train)

  # Make predictions
  y_pred = model.predict(X_test)

  # Calculate accuracy metrics
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)

  # Print accuracy metrics
  print(f'Accuracy metrics for {name}:')
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
  plt.figure(figsize=(12, 8)) 
  shap.summary_plot(shap_values[:, important_feature_indices], X_test[:, important_feature_indices], feature_names=important_features, show=False)

  # Customize tick parameters
  plt.xticks(fontsize=10)  # Adjust as needed
  plt.yticks(fontsize=10)  # Adjust as needed

  # Add title to the plot
  plt.title(f'Top 5 Features {name}:\nPredicting average completion rate')


  plt.tight_layout(pad=5)  # Increase padding to leave more space


  plt.savefig(f'../brighten_figures/passive_shap_figures/{name}_shap_avg_completion_rate_plot.png', format='png')
    
  plt.clf()  # Clear the current figure





  ####################### NOW LET'S PREDICT COMPLETION GROUP

  X = df[features]
  y = df['completion_group_high']

  # Standardize the features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Fit Elastic Net model with cross-validation
  model = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
  model.fit(X_train, y_train)

  # Make predictions
  y_pred = model.predict(X_test)

  y_pred_class = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to class labels

  # Print classification report
  print(classification_report(y_test, y_pred_class))

  # Optional: Print confusion matrix for additional insight
  conf_matrix = confusion_matrix(y_test, y_pred_class)
  print(f"Confusion Matrix for model:\n{conf_matrix}")


  # Explain the model with SHAP values
  explainer = shap.Explainer(model, X_train)
  shap_values = explainer(X_test)

  # Calculate mean absolute SHAP values to determine feature importance
  shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
  important_feature_indices = np.argsort(shap_abs_mean)[-5:]  # Get indices of top 5 features
  important_features = [features[i] for i in important_feature_indices]

  # Plot SHAP values for the top 5 features
  plt.figure(figsize=(12, 8)) 
  shap.summary_plot(shap_values[:, important_feature_indices], X_test[:, important_feature_indices], feature_names=important_features, show=False)

  # Customize tick parameters
  plt.xticks(fontsize=10)  # Adjust as needed
  plt.yticks(fontsize=10)  # Adjust as needed


  # Add title to the plot
  plt.title(f'Top 5 Features {name}:\nPredicting completion group')

  plt.tight_layout(pad=5)

  plt.savefig(f'../brighten_figures/passive_shap_figures/{name}_shap_completion_group_plot.png', format='png')
  
  plt.clf()  # Clear the current figure



