# This script will perform a elastic regression net to categorize engaged vs withdrew participants
# We loop the script to do it again without including study version in features


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import classification_report, confusion_matrix
import shap



# Load datasets
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

print(imputed_complete_df['study_arm'].value_counts())

# Set variables as dummies. Notice that I removed study_arm because too many NaN in disengaged groups
imputed_complete_df = pd.get_dummies(imputed_complete_df, columns=['engagement', 'race', 'device', 'study', 'working', 'gender', 'education', 'income_satisfaction', 'marital_status', 'income_lastyear'], drop_first=False)

feature_columns = [
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
    #'study_Brighten-v1',
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

# Create two sets of features, one with study version and one without to study the impact of study version on the ability of the model to predict engagement
feature_columns_no_study = [col for col in feature_columns if col not in ['study_Brighten-v1', 'study_Brighten-v2']]
feature_sets = [feature_columns, feature_columns_no_study]

# Remove nan rows so we can run the model
imputed_complete_df = imputed_complete_df.dropna(subset=feature_columns)


# Loop the feature sets in the model
for set in feature_sets:

    print(f'Number of participants: {len(imputed_complete_df)}')



    # Define the features and target variable
    features = set
    X = imputed_complete_df[features]
    y = imputed_complete_df['engagement_engaged']

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
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 

    plt.xlim(-0.21, 0.21) 
    plt.xticks(np.arange(-0.20, 0.21, 0.05))

    # Add title to the plot
    if set == feature_columns:
        plt.title('c)')
    else:
        plt.title('Top 5 Features without study: Predicting engagement')
    
    plt.tight_layout()

    # Save the plot as PNG

    if set == feature_columns:
        plt.savefig('../brighten_figures/shap_figures/shap_engaged_plot.png', format='png')
    else:
        plt.savefig('../brighten_figures/shap_figures/no_study_shap_engaged_plot.png', format='png')
    
    plt.clf()  # Clear the current figure
