# This script will compare score over time between two groups separated by median of avg_completion_rate
# Then we will perform a GLM repeated measures analysis
# i will take mean phq9-score of each group for each week and compare the trend 
# Individuals who have not completed any non-baseline questionnaire are excluded


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf



# Load datasets
df_phq2 = pd.read_csv('../brighten_data/PHQ-2.csv')
df_phq9 = pd.read_csv('../brighten_data/PHQ-9.csv')
df_phq9.rename(columns={'sum_phq9': 'phq9_sum'}, inplace=True)
df_sleep = pd.read_csv('../brighten_data/Sleep Quality.csv')
df_sds = pd.read_csv('../brighten_data/SDS.csv')
df_pgic = pd.read_csv('../brighten_data/Patients Global Impression of Change Scale.csv')
df_impact = pd.read_csv('../brighten_data/IMPACT Mania and Psychosis Screening.csv')
df_audit = pd.read_csv('../brighten_data/AUDIT-C.csv')
df_gad = pd.read_csv('../brighten_data/GAD-7.csv')
demographics_df = pd.read_csv('../brighten_data/Baseline Demographics.csv')
df_mental_service = pd.read_csv('../brighten_data/Mental Health Services.csv')
df_study_satisfaction = pd.read_csv('../brighten_data/Study App Satisfaction.csv')
df_other = pd.read_csv('../brighten_data/Other Health-related Apps Used.csv')
df_baseline_phq9 = pd.read_csv('../brighten_data/Baseline PHQ9 Survey.csv')
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

# Need to create a "sum" column for relevant dataframes if they don't have one
df_sleep['sleep_sum'] = df_sleep['sleep_1'] + df_sleep['sleep_2'] + df_sleep['sleep_3']
df_sds['sds_sum'] = df_sds['sds_1'] + df_sds['sds_2'] + df_sds['sds_3'] + df_sds['stress'] + df_sds['support']

print(df_impact.head())
df_impact['impact_sum'] = 0
for i in range(1,5):
  df_impact['impact_sum'] += df_impact[f'screen_{i}']

df_pgic['pgic_sum'] = df_pgic['mood_1']


# This dictionnary represents total number of each questionnaire that was submitted to participants
dict_of_questionnaires = {
    'phq2': 12*7,  # daily for 21 weeks
    'phq9': 8,     # weekly for 4 first weeks, then biweekly for 8 more weeks
    'sleep': 12,   # weekly for 12 weeks
    'sds': 8,      # weekly for 4 first weeks, then biweekly for 8 more weeks
    'pgic': 12,     # weekly for 12 weeks
    'mental': 12,   # weekly for 12 weeks
    'satisfaction': 3, # Weeks 4,8,12
    'other': 4 # Weeks 1,4,8,12
}

# We use the original questionnaire dataframes to keep time in consideration for glm repeated measures
# For each questionnaire, we group by score for each time unit
for key in dict_of_questionnaires.keys():
    if key == 'phq2':
        df_questionnaire = df_phq2
    elif key == 'phq9':
        df_questionnaire = df_phq9
    elif key == 'sleep':
        df_questionnaire = df_sleep
    elif key == 'sds':
        df_questionnaire = df_sds
    elif key == 'pgic':
        df_questionnaire = df_pgic
    else:
        continue
    df_with_groups = pd.merge(df_questionnaire, imputed_complete_df[['participant_id', 'completion_group']], on='participant_id', how='left')
    df_with_groups = df_with_groups[df_with_groups['completion_group'] != 'no group']

    if key == 'phq2':
        mean_sum_questionnaire = df_with_groups.groupby(['day', 'completion_group'])[f'{key}_sum'].agg(['mean', 'count', 'std']).reset_index()
        mean_sum_questionnaire = mean_sum_questionnaire[mean_sum_questionnaire['day'] <= 84] # We remove questionnaires that exceed the 12 weeks period (there are some overachievers!)
    elif key == 'phq9':
        mean_sum_questionnaire = df_with_groups.groupby(['week', 'completion_group'])[f'{key}_sum'].agg(['mean', 'count', 'std']).reset_index()
        mean_sum_questionnaire = mean_sum_questionnaire[mean_sum_questionnaire['week'] <= 12] # We remove questionnaires that exceed the 12 weeks period (there are some overachievers!)
        weeks_to_remove = [5, 7, 9, 11]
        mean_sum_questionnaire = mean_sum_questionnaire[~mean_sum_questionnaire['week'].isin(weeks_to_remove)]

    elif key != 'phq2':
        mean_sum_questionnaire = df_with_groups.groupby(['week', 'completion_group'])[f'{key}_sum'].agg(['mean', 'count', 'std']).reset_index()
        mean_sum_questionnaire = mean_sum_questionnaire[mean_sum_questionnaire['week'] <= 12] # We remove questionnaires that exceed the 12 weeks period (there are some overachievers!)


    # Calculate SEM
    mean_sum_questionnaire['SEM'] =  mean_sum_questionnaire['std'] / ( mean_sum_questionnaire['count'] ** 0.5)
    

    # Rename mean column
    mean_sum_questionnaire.rename(columns={'mean': f'average_score_{key}'}, inplace=True)
    


    # Create the figure and axis objects
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Define color mapping
    color_mapping = {'high': 'green', 'low': 'red'}
    color_mapping_sample_size = {'high': 'darkgreen', 'low': 'darkred'}

    if key == 'phq2':
        time_unit = 'day'
    else: 
        time_unit = 'week'

    # Plot the mean scores on the first y-axis (ax1) and store handles for legend
    handles = []
    labels = []  # Store labels for the legend
    for group, color in color_mapping.items():
        group_data = mean_sum_questionnaire[mean_sum_questionnaire['completion_group'] == group]
        line, = ax1.plot(
            group_data[time_unit],
            group_data[f'average_score_{key}'],
            color=color,
            marker='o',
            linestyle='-'  # Use solid lines
        )
        handles.append(line)  # Store the line for legend
        labels.append(f'Mean Score ({group})')  # Add label for mean score

    # Adding error bars manually to include SEM
    for group, color in color_mapping.items():
        group_data = mean_sum_questionnaire[mean_sum_questionnaire['completion_group'] == group]
        ax1.errorbar(
            group_data[time_unit],
            group_data[f'average_score_{key}'],
            yerr=group_data['SEM'],
            fmt='none',  # No marker, just error bars
            color=color,  # Match error bar color to line color
            capsize=5
        )

    # Create a secondary y-axis for sample size
    ax2 = ax1.twinx()

    # Plot the sample size on the secondary y-axis (ax2) and create handles for legend
    for group, color in color_mapping_sample_size.items():
        group_data = mean_sum_questionnaire[mean_sum_questionnaire['completion_group'] == group]
        sample_size_line, = ax2.plot(
            group_data[time_unit],
            group_data['count'],
            color=color,
            linestyle='-',  # Use solid line for sample size
            marker=None,
        )
        handles.append(sample_size_line)  # Store sample size line for legend
        labels.append(f'Sample Size ({group})')  # Add label for sample size

    # Set axis labels
    ax1.set_xlabel(f'{time_unit.capitalize()}')
    ax1.set_ylabel(f'Mean {key.capitalize()} Score')
    ax2.set_ylabel('Sample Size')

    # Set the title
    ax1.set_title(f'Mean {key.capitalize()} Score Over Time by Completion Group (determined by average completion rate)')

    # Set a single legend for both axes
    ax1.legend(handles, labels, loc='best')

    # Grid and layout adjustments
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(f'../brighten_figures/median_separation_glm_analysis/{key}_score_over_time_median_separation.png', dpi=300)
    


    # Convert categorical variables to dummy variables
    mean_sum_questionnaire['completion_group'] = mean_sum_questionnaire['completion_group'].astype('category')
    mean_sum_questionnaire = pd.get_dummies(mean_sum_questionnaire, columns=['completion_group'], drop_first=True)

    # Fit the GLM
    model = smf.glm(
        formula= f'average_score_{key} ~ {time_unit} * completion_group_low',
        data=mean_sum_questionnaire,
        family=sm.families.Gaussian()
    ).fit()

    # Print the summary
    print(model.summary())
    


    # Let's check if the rate of sampling size decrease over time is the same between high and low completion groups
    # Fit a linear model with interaction term
    interaction_model = smf.ols(formula=f'count ~ {time_unit} * completion_group_low', data=mean_sum_questionnaire).fit()

    # Print the summary of the model
    print(f'Effect of time on sample size between high and low completion groups for questionnaire {key}')
    print(interaction_model.summary())