# This script performs a elastic net regression twice:
#   1. Predict completion group (high vs low)
#   2. Predict average completion rate 
# We loop the script to do it again without including study version in features



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
import shap



imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')
df_phq9 = pd.read_csv('../brighten_data/df_phq9_over_time.csv')

# Remove duplicated from df_phq9
df_phq9 = df_phq9.drop_duplicates(subset='participant_id')


# Remove participants that were not engaged, because obviously their avg completion rate is 0
imputed_complete_df = imputed_complete_df[imputed_complete_df['avg_completion_rate']>0]

# Merge improvement column into imputed_complete_df
imputed_complete_df = pd.merge(imputed_complete_df, df_phq9[['participant_id', 'improvement']], on='participant_id', how='inner')


imputed_complete_df = pd.get_dummies(imputed_complete_df, columns=['completion_group', 'race', 'device', 'study_arm', 'study', 'working', 'gender', 'education', 'income_satisfaction', 'marital_status', 'income_lastyear', 'improvement'], drop_first=False)

feature_columns = [
    'mean_score_phq2',
    'mean_score_phq9',
    'baseline_phq9_result',
    'mean_score_sleep',
    'improvement_improver',
    #'mean_score_mental', if this one is kept, participant count drops to 0
    'gad7_sum',
    'alc_sum',
    'age',
    'mean_score_sds',
    #'mean_score_pgic',
    #'mean_score_satisfaction', if this one is kept, participant count drops to 0
    #'mean_score_other', if this one is kept, participant count drops to 0
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
    'study_Brighten-v1',
    'study_Brighten-v2',
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

# We create to feature column lists, one with study version and one without to study the impact of the feature on the model's ability to predict
feature_columns_no_study = [col for col in feature_columns if col not in ['study_Brighten-v1',
    'study_Brighten-v2']]
feature_sets = [feature_columns, feature_columns_no_study]


# loop the model for both feature sets
for set in feature_sets:

    imputed_complete_df = imputed_complete_df.dropna(subset=set)
    print(f'Number of participants: {len(imputed_complete_df)}')

####################### FIRST WE DO IT FOR COMPLETION GROUP PREDICTION

    # Define the features and target variable
    features = set
    X = imputed_complete_df[features]
    y = imputed_complete_df['completion_group_high']

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

    y_pred_class = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to class labels

    # Print classification report
    print(classification_report(y_test, y_pred_class))

    # Optional: Print confusion matrix for additional insight
    conf_matrix = confusion_matrix(y_test, y_pred_class)
    if set == feature_columns:
        print(f"Confusion Matrix for model:\n{conf_matrix}")
    else:
        print(f"Confusion Matrix for model without study version:\n{conf_matrix}")
 


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

    plt.xlim(-0.21, 0.21) 
    plt.xticks(np.arange(-0.20, 0.21, 0.05))


    # Add title to the plot
    if set == feature_columns:
        plt.title('b)')
    else:
        plt.title('Top 5 Features without study:\nPredicting completion group')

    plt.tight_layout()  # Increase padding to leave more space



    if set == feature_columns:
        # Save the plot as PNG
        plt.savefig('../brighten_figures/shap_figures/shap_completion_group_high_plot.png', format='png')
    else:
        plt.savefig('../brighten_figures/shap_figures/no_study_shap_completion_group_high_plot.png', format='png')
    
    plt.clf()  # Clear the current figure




##############################################################################################################################################################################################
######################### THEN WE DO IT FOR AVG COMPLETION RATE PREDICTION


# Now let's predict average_completion_rate (instead of categorizing)

    # Define the features and target variable
    features = set
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

    # Print accuracy metrics
    if set == feature_columns:
        print(f'Accuracy metrics for all features:')
    else:
        print(f'Accuracy Metrics for all features except study:')
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
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Set x-axis limits
    plt.xlim(-0.21, 0.21) 

    plt.xticks(np.arange(-0.20, 0.21, 0.05))

    # Add title to the plot
    if set == feature_columns:
        plt.title('a)')
    else:
        plt.title('Top 5 Features without study:\nPredicting average completion rate')

    plt.tight_layout()  # Increase padding to leave more space


    if set == feature_columns:
        # Save the plot as PNG
        plt.savefig('../brighten_figures/shap_figures/shap_avg_completion_rate_plot.png', format='png')

    else:
        plt.savefig('../brighten_figures/shap_figures/no_study_shap_avg_completion_rate_plot.png', format='png')
    
    plt.clf()  # Clear the current figure
