# This script performs a cox PH model to study retention regarding PHQ-9 assessments
# There are two groups: improvers and non improvers, calculated in imputed_complete_df.py
# To run this script, create virtual env and install lifelines so it doesn't contradict other libraries 


import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

# Load the datasets
df_phq9 = pd.read_csv('../brighten_data/df_phq9_over_time.csv')
demographics_df = pd.read_csv('../brighten_data/Baseline Demographics.csv')
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

imputed_complete_df.drop(columns='baselinePHQ9date', inplace=True)

# Merge dataframes
cox_df = pd.merge(df_phq9, imputed_complete_df, on='participant_id', how='inner')

print(cox_df.columns)

# Columns to drop
columns_to_drop = [
    'ROW_ID', 
    'ROW_VERSION', 
    'phq9_1', 
    'phq9_2', 
    'phq9_3', 
    'phq9_4', 
    'phq9_5', 
    'phq9_6', 
    'phq9_7', 
    'phq9_8', 
    'phq9_9', 
    'baseline_phq9_result_y', 
    'completion_rate_phq2',  
    'completion_rate_phq9', 
    'completion_rate_sleep', 
    'completion_rate_sds',  
    'completion_rate_pgic', 
    'completion_rate_mental', 
    'completion_rate_satisfaction', 
    'completion_rate_other', 
    'startdate',  
    'completion_group', 
    'engagement', 
    'completion_group_phq2', 
    'completion_group_phq9', 
    'completion_group_sleep', 
    'completion_group_sds', 
    'completion_group_pgic', 
    'completion_group_mental', 
    'completion_group_satisfaction', 
    'completion_group_other', 
    'quartile_avg_completion', 
    'quartile_completion_phq2', 
    'quartile_completion_phq9', 
    'quartile_completion_sleep', 
    'quartile_completion_sds', 
    'quartile_completion_pgic', 
    'quartile_completion_mental', 
    'quartile_completion_satisfaction', 
    'quartile_completion_other'
]
cox_df = cox_df.drop(columns=columns_to_drop)


# Convert date columns to datetime format
cox_df['baselinePHQ9date'] = pd.to_datetime(cox_df['baselinePHQ9date'])
cox_df['phq9Date'] = pd.to_datetime(cox_df['phq9Date'])

# Exclude participants with less than 2 unique non-baseline questionnaire dates
cox_df['is_non_baseline'] = cox_df['phq9Date'] > cox_df['baselinePHQ9date']
participants_to_keep = cox_df[cox_df['is_non_baseline']].groupby('participant_id').filter(lambda x: x['phq9Date'].nunique() >= 2)

# Extract unique rows per participant to get all potential features before encoding
participant_features = participants_to_keep.drop_duplicates(subset=['participant_id'])

#One-hot encode categorical features, excluding participant_id and date columns
participant_features_encoded = pd.get_dummies(
    participant_features, 
    columns=[col for col in participant_features.columns if participant_features[col].dtype == 'object' and col not in ['participant_id', 'phq9Date', 'baselinePHQ9date']]
)


# Define features after one-hot encoding
features = [
    # 'phq9_sum', 
    #'baseline_phq9_result_x', 
    'age', 
    # 'gad7_sum', 
    # 'alc_sum', 
    # 'mean_score_phq2', 
    # 'mean_score_phq9', 
    'mean_score_sleep', 
    # 'mean_score_sds', 
    # 'mean_score_pgic', 
    #'gender_Male', p=0.76
     #'education_Graduate Degree', p=0.17
    # 'education_High School', 
    # 'education_University', 
     #'working_Yes', p=0.54
    # "income_satisfaction_Can't make ends meet", 
    # 'income_satisfaction_Have enough to get along', 
    # 'income_lastyear_20,000-40,000', 
    # 'income_lastyear_40,000-60,000', 
    # 'income_lastyear_60,000-80,000', 
    # 'income_lastyear_80,000-100,000', 
     #'income_lastyear_< $20,000',  p =0.69
    # 'marital_status_Separated/Widowed/Divorced', 
    # 'marital_status_Single', 
    # 'race_American Indian/Alaskan Native', 
    # 'race_Asian', 
    # 'race_Hispanic/Latino', p=0.66
    # 'race_More than one', 
    # 'race_Native Hawaiian/other Pacific Islander', 
     #'race_Non-Hispanic White',  p =0.9
    # 'race_Other', 
    # 'heard_about_us_Craigslist', 
    # 'heard_about_us_Twitter/Facebook', 
    # 'heard_about_us_friend/colleague', 
    # 'heard_about_us_others', 
    # 'heard_about_us_through other studies', 
    #'device_iPhone', p=0.6
    'study_Brighten-v2', 
    #'study_arm_HealthTips'
    #  'study_arm_iPST' p=0.99
]

# Group by participant to calculate last and oldest non-baseline dates
grouped = participants_to_keep.groupby('participant_id').agg(
    last_questionnaire_date=('phq9Date', 'max'),
    earliest_non_baseline_date=('phq9Date', 'min'),
    event_count=('is_non_baseline', 'sum')
).reset_index()




#  Merge in the one-hot encoded features before preparing the final DataFrame
grouped = grouped.merge(participant_features_encoded, on='participant_id', how='left')

# Determine the event status
grouped = grouped.sort_values(by=['participant_id', 'last_questionnaire_date'])

# Determine if the last completed PHQ-9 falls within the 12-week period from their study start date
grouped['event'] = (grouped['last_questionnaire_date'] <= (grouped['earliest_non_baseline_date'] + pd.Timedelta(weeks=12))).astype(int)

# Drop duplicate rows, keeping only the first entry for each participant with their event status
grouped = grouped.drop_duplicates(subset=['participant_id'], keep='first')


# Calculate duration (days from oldest non-baseline questionnaire to last)
grouped['duration'] = (grouped['last_questionnaire_date'] - grouped['earliest_non_baseline_date']).dt.days






############################# RUNNING COX MODEL

# Prepare final dataframe for Cox model
cox_df = grouped[['duration', 'event', 'improvement_improver'] + features]

# Remove NaN rows
cox_df = cox_df.dropna()

# Print metrics
print(f'Number of unique participants: {len(cox_df)}')

# Fit the Cox model
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='duration', event_col='event')

# Print the results
cph.print_summary()

print(f'Number of features: {len(features)}')
print(f'Bonferroni corrected level of significance: {0.05/len(features)}')



################################ PLOTTING

# Create a Kaplan-Meier plot
kmf = KaplanMeierFitter()

# Fit for each group and plot
plt.figure(figsize=(10, 6))
medians = {}  # Dictionary to hold median survival times for each group

for group in cox_df['improvement_improver'].unique():
    # Set label based on the value of group
    sample_size = len(cox_df[cox_df['improvement_improver'] == group])
    label = f'Improvers (n={sample_size})' if group == 1 else f'Non Improvers (n={sample_size})'
    
    kmf.fit(durations=cox_df[cox_df['improvement_improver'] == group]['duration'],
            event_observed=cox_df[cox_df['improvement_improver'] == group]['event'],
            label=label)
    
    # Store the median survival time
    median_survival_time = kmf.median_survival_time_
    medians[label] = median_survival_time

    kmf.plot_survival_function()

# Add titles and labels
plt.title('')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid()
plt.savefig('../brighten_figures/kaplan_meier/cox_phq9.png', dpi=300)



# Print median survival times
for label, median in medians.items():
    print(f'Median survival time for {label}: {median} days')