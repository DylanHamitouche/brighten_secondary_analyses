# This script creates a csv file that compares demographics parameters between two groups:
#   1. Engaged (At least 1 non baseline questionnaire was completed)
#   2. Quit (No non-baseline questionnaire was completed)




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind # for phq9 baseline, which is represented by a mean instead of frequency

# Load datasets
df_baseline_phq9 = pd.read_csv('../brighten_data/Baseline PHQ9 Survey.csv')
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')



# Store all the possible groups in a dictionnary
unique_values = {
    'gender': ['Male', 'Female'],
    'education': ['Graduate Degree', 'High School', 'University', 'Community College', 'Elementary School'],
    'working': ['Yes', 'No'],
    'income_lastyear':  ['< $20,000', '40,000-60,000', '20,000-40,000', '60,000-80,000','80,000-100,000', '100,000+'],
    'income_satisfaction': ["Can't make ends meet", 'Am comfortable', 'Have enough to get along'],
    'marital_status': ['Single', 'Married/Partner', 'Separated/Widowed/Divorced'],
    'race': ['Asian', 'African-American/Black', 'Non-Hispanic White', 'Hispanic/Latino',
             'More than one', 'American Indian/Alaskan Native', 'Other','Native Hawaiian/other Pacific Islander'],
    'heard_about_us': ['Craigslist', 'others', 'Advertisement', 'friend/colleague',
                       'through other studies', 'Twitter/Facebook'],
    'device': ['iPhone', 'Android'],
    'study_arm': ['iPST','EVO', 'HealthTips'],
    'study': ['Brighten-v1','Brighten-v2']
}

# Initialize summary_data list
summary_data = []

# Check how many participants are engaged vs disengaged (quitters)
total_engaged = len(imputed_complete_df[imputed_complete_df['engagement'] == 'engaged'])
total_quitters = len(imputed_complete_df[imputed_complete_df['engagement'] == 'quit'])

# Loop the unique_dictionnary and check for each category how many engaged and quitters belong to each group
# Then, in the same loop, divide by total number to get the percentage
# Append everything to summary_data list
for column, values in unique_values.items():
    for value in values:
        count_engaged = imputed_complete_df[(imputed_complete_df[column] == value) & (imputed_complete_df['engagement'] == 'engaged')].shape[0]
        count_quitters = imputed_complete_df[(imputed_complete_df[column] == value) & (imputed_complete_df['engagement'] == 'quit')].shape[0]
        
        percentage_engaged = (count_engaged / total_engaged) * 100 if total_engaged > 0 else 0
        percentage_quitters = (count_quitters / total_quitters) * 100 if total_quitters > 0 else 0
        
        summary_data.append({
            'Variable': f"{value}",
            f'Engaged (N={total_engaged})': f"{count_engaged} ({percentage_engaged:.2f}%)",
            f'Quitters (N={total_quitters})': f"{count_quitters} ({percentage_quitters:.2f}%)"
        })


# Calculate mean age for engaged and quitter groups
mean_age_engaged = imputed_complete_df[imputed_complete_df['engagement'] == 'engaged']['age'].mean()
mean_age_quitters = imputed_complete_df[imputed_complete_df['engagement'] == 'quit']['age'].mean()


# Convert 'age' column to integers
imputed_complete_df['age'] = imputed_complete_df['age'].astype(int)

# Add rows for 'age'
bins = pd.IntervalIndex.from_tuples([(18, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 130)], closed='both')
labels = ['18-30 years', '31-40 years', '41-50 years', '51-60 years', '61-70 years', '71 years and over']
imputed_complete_df['age_group'] = pd.cut(imputed_complete_df['age'], bins=bins, labels=labels)



for age_group in bins:
    count_engaged = len(imputed_complete_df[(imputed_complete_df['age_group'] == age_group) & (imputed_complete_df['engagement'] == 'engaged')])
    count_quitters = len(imputed_complete_df[(imputed_complete_df['age_group'] == age_group) & (imputed_complete_df['engagement'] == 'quit')])
    
    percentage_engaged = (count_engaged / total_engaged) * 100
    percentage_quitters = (count_quitters / total_quitters) * 100
    
    summary_data.append({
        'Variable': f"{age_group}",
        f'Engaged (N={total_engaged})': f"{count_engaged} ({percentage_engaged:.2f}%)",
        f'Quitters (N={total_quitters})': f"{count_quitters} ({percentage_quitters:.2f}%)"
    })


# Create a dataframe from the summary data list
df_summary = pd.DataFrame(summary_data)


# Extract counts from the summary data
counts_engaged = [int(row[f'Engaged (N={total_engaged})'].split()[0]) for row in summary_data]
counts_quitters = [int(row[f'Quitters (N={total_quitters})'].split()[0]) for row in summary_data]

# Calculate p-values using Chi-square test for each row
p_values = []
for counts_eng, counts_quit in zip(counts_engaged, counts_quitters):
    total_counts = counts_eng + counts_quit
    expected_engaged = total_engaged * (counts_eng / total_counts)
    expected_quitters = total_quitters * (counts_quit / total_counts)
    
    # Create the contingency table
    observed = np.array([[counts_eng, counts_quit], [total_engaged - counts_eng, total_quitters - counts_quit]])
    expected = np.array([[expected_engaged, expected_quitters], 
                         [total_engaged - expected_engaged, total_quitters - expected_quitters]])
    
    # Perform Chi-square test
    _, p_value, _, _ = chi2_contingency(observed)
    p_values.append(p_value)

# Add the p-values to the dataframe
df_summary['p_value'] = p_values



# Compare phq9_sum between engaged and quitters
imputed_complete_df['phq9_sum'] = df_baseline_phq9[[col for col in df_baseline_phq9.columns if 'phq9' in col]].sum(axis=1)

mean_phq9_sum_engaged = imputed_complete_df[imputed_complete_df['engagement'] == 'engaged']['phq9_sum'].mean()
mean_phq9_sum_quitters = imputed_complete_df[imputed_complete_df['engagement'] == 'quit']['phq9_sum'].mean()
t_stat, phq9_p_value = ttest_ind(mean_phq9_sum_engaged, mean_phq9_sum_quitters, nan_policy='omit')

# Ensure the 'Mean PHQ-9 Sum' row is added to df_summary
baseline_phq9_row = pd.DataFrame([{
    'Variable': 'Mean PHQ-9 Sum',
    f'Engaged (N={total_engaged})': f"{mean_phq9_sum_engaged:.2f}",
    f'Quitters (N={total_quitters})': f"{mean_phq9_sum_quitters:.2f}",
    'p_value': p_value
}])

df_summary = pd.concat([df_summary, baseline_phq9_row], ignore_index=True)


# We apply bonferroni correction to all p-values based on total number of comparisons
number_of_tests = len(summary_data) #49
df_summary['p_value'] = df_summary['p_value'] * number_of_tests

print(f'Number of observations: {number_of_tests}')
#df_summary = df_summary[df_summary['p_value'] < 0.05]

# Display the summary dataframe
print('test')
print(df_summary)

# Save our precious baby
df_summary.to_csv('../brighten_data/comparison_demographics_engaged_vs_withdrew.csv', index=False)
