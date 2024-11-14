# This script performs a cox PH model to study retention regarding PHQ-2 assessments
# There are two groups: improvers and non improvers, calculated (in this script) by substracting oldest score by most recent score
# To run this script, create virtual env and install lifelines so it doesn't contradict other libraries 





import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

# Load the datasets
df_phq2 = pd.read_csv('../brighten_data/PHQ-2.csv')
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

# Convert to date time
df_phq2['dt_response'] = pd.to_datetime(df_phq2['dt_response'])

# Get the earliest and most recent phq2_sum scores for each participant
earliest_scores = df_phq2.sort_values('dt_response').groupby('participant_id')['phq2_sum'].first()
latest_scores = df_phq2.sort_values('dt_response').groupby('participant_id')['phq2_sum'].last()

# Calculate the improvement condition
improvement_phq2 = (latest_scores - earliest_scores) < 0

# Create a new DataFrame for improvement: 1 (improved) or 0 (not improved)
improvement_df = pd.DataFrame({'participant_id': improvement_phq2.index, 'improvement_phq2': improvement_phq2.astype(int)})

# Reset the index of improvement_df to make participant_id a regular column
improvement_df.reset_index(inplace=True, drop=True)

# Merge back improvement_df to df_phq2
df_phq2 = df_phq2.merge(improvement_df, on='participant_id', how='left')

# Merge df_phq2 to imputed_complete_df
cox_df = pd.merge(df_phq2, imputed_complete_df, on='participant_id', how='inner')

# Drop irrelevant columns
columns_to_drop = [
    'ROW_ID', 
    'ROW_VERSION',
    'dt_yesterday',
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




# One-hot encode categorical features, excluding participant_id, date column, and improvement column
cox_df = pd.get_dummies(
    cox_df, 
    columns=[col for col in cox_df.columns if cox_df[col].dtype == 'object' and col not in ['participant_id', 'dt_response', 'improvement_phq2']]
)



# Define features 
features = [
    # 'phq9_sum', 
    # 'baseline_phq9_result_x', 
    'age', 
    # 'gad7_sum', 
    # 'alc_sum', 
    # 'mean_score_phq2', 
    # 'mean_score_phq9', 
    # 'mean_score_sleep', 
    # 'mean_score_sds', 
    # 'mean_score_pgic', 
    #   'gender_Male', p=0.36
    #  'education_Graduate Degree', p=0.11
    # 'education_High School', 
    # 'education_University', 
    #  'working_Yes', 
    # "income_satisfaction_Can't make ends meet", 
    # 'income_satisfaction_Have enough to get along', 
    #  'income_lastyear_20,000-40,000', p =0.13
    # 'income_lastyear_40,000-60,000', 
    # 'income_lastyear_60,000-80,000', 
    # 'income_lastyear_80,000-100,000', 
    # income_lastyear_< $20,000', 
    # 'marital_status_Separated/Widowed/Divorced', 
    # 'marital_status_Single', 
    # 'race_American Indian/Alaskan Native', 
    # 'race_Asian', 
    'race_Hispanic/Latino', 
    # 'race_More than one', 
    # 'race_Native Hawaiian/other Pacific Islander', 
    #  'race_Non-Hispanic White', 
    # 'race_Other', 
    # 'heard_about_us_Craigslist', 
    # 'heard_about_us_Twitter/Facebook', 
    # 'heard_about_us_friend/colleague', 
    # 'heard_about_us_others', 
    # 'heard_about_us_through other studies', 
    # 'device_iPhone', 
    'study_Brighten-v2', 
     'study_arm_HealthTips', 
    # 'study_arm_iPST'
]

# Group by participant to calculate last and oldest non-baseline dates
grouped = cox_df.groupby('participant_id').agg(
    last_questionnaire_date=('dt_response', 'max'),
    earliest_questionnaire_date=('dt_response', 'min')
).reset_index()

# Merge in the one-hot encoded features before preparing the final DataFrame
grouped = grouped.merge(cox_df, on='participant_id', how='left')

# Determine if the last completed PHQ-9 falls within the 12-week period from their study start date
grouped['event'] = (grouped['last_questionnaire_date'] <= (grouped['earliest_questionnaire_date'] + pd.Timedelta(weeks=12))).astype(int)

# Drop duplicate rows, keeping only the first entry for each participant with their event status
grouped = grouped.drop_duplicates(subset=['participant_id'], keep='first')

# Calculate duration (days from oldest non-baseline questionnaire to last)
grouped['duration'] = (grouped['last_questionnaire_date'] - grouped['earliest_questionnaire_date']).dt.days




################## RUNNING COX PH MODEL

# Prepare final dataframe for Cox model
cox_df = grouped[['duration', 'event', 'improvement_phq2'] + features]

# Remove NaN rows
cox_df = cox_df.dropna()

# Print metrics
print(f'Number of unique participants: {len(cox_df)}')

# Fit the Cox model
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='duration', event_col='event')

# Print the results
cph.print_summary()

# Print bonferonni p-value threshold for significance
print(f'Number of features used: {len(features)}')
print(f'Bonferroni level of significance: {0.05/len(features)}')



########################### PLOTTING
# Create a Kaplan-Meier plot
kmf = KaplanMeierFitter()

# Fit for each group and plot
plt.figure(figsize=(10, 6))
for group in cox_df['improvement_phq2'].unique():
    # Set label based on the value of group
    sample_size = len(cox_df[cox_df['improvement_phq2'] == group])
    label = f'Non Improvers (n={sample_size})' if group == 0 else f'Improvers (n={sample_size})'
    
    kmf.fit(durations=cox_df[cox_df['improvement_phq2'] == group]['duration'],
            event_observed=cox_df[cox_df['improvement_phq2'] == group]['event'],
            label=label)
    kmf.plot_survival_function()

# Add titles and labels
plt.title('')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.xlim(0,85)
plt.xticks(np.arange(0,85,2))
plt.legend()
plt.grid()
plt.savefig('../brighten_figures/kaplan_meier/cox_phq2.png', dpi=300)