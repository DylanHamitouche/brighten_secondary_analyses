# This script attempts to study the score differences between two groups separated by the median avg_completion_rate
# The comparison does not include time as a variable, so only average score is taken into consideration

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


# Load datasets
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

# Create a dataframe to compare scores

rows = ['mean_score_phq2','mean_score_phq9','mean_score_sleep', 'mean_score_sds', 'mean_score_pgic','baseline_phq9_result','alc_sum','gad7_sum']
number_of_tests = len(rows)
columns = ['high completion rate', 'high completion rate (std)', 'low completion rate', 'low completion rate (std)']
mean_score_df = pd.DataFrame(index=rows, columns=columns)

for row in rows:
    mean_score_df.at[row, 'high completion rate'] = imputed_complete_df[imputed_complete_df['completion_group'] == 'high'][row].mean()
    mean_score_df.at[row, 'high completion rate (std)'] = imputed_complete_df[imputed_complete_df['completion_group'] == 'high'][row].std()
    mean_score_df.at[row, 'low completion rate'] = imputed_complete_df[imputed_complete_df['completion_group'] == 'low'][row].mean()
    mean_score_df.at[row, 'low completion rate (std)'] = imputed_complete_df[imputed_complete_df['completion_group'] == 'low'][row].std()


# Add t-score and p-value columns
mean_score_df['t-score'] = np.nan
mean_score_df['p-value'] = np.nan

for row in rows:
    high_group = imputed_complete_df[imputed_complete_df['completion_group'] == 'high'][row]
    low_group = imputed_complete_df[imputed_complete_df['completion_group'] == 'low'][row]
    
    t_score, p_value = ttest_ind(high_group.dropna(), low_group.dropna(), equal_var=False)
    mean_score_df.at[row, 't-score'] = t_score
    mean_score_df.at[row, 'p-value'] = min(p_value * number_of_tests, 1)

print(mean_score_df)

mean_score_df.to_csv('../brighten_data/median_separation_score_analysis.csv', index=True)






