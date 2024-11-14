# This script compares average completion rate between 4 groups:
    # low phq9 low pgic
    # low phq9 high pgic
    # high phq9 low pgic
    # high phq9 high pgic


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
df_pgic = pd.read_csv('../brighten_data/Patients Global Impression of Change Scale.csv')
df_phq9 = pd.read_csv('../brighten_data/PHQ-9.csv')
df_phq9.rename(columns={'sum_phq9': 'phq9_sum'}, inplace=True)
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

median_score_pgic = df_pgic['mood_1'].median()
median_score_phq9 = df_phq9['phq9_sum'].median()

print(f'Median score PGIC: {median_score_pgic}')
print(f'Median score PHQ-9: {median_score_phq9}')

# Only keep engaged participants
imputed_complete_df = imputed_complete_df[imputed_complete_df['engagement']=='engaged']


imputed_complete_df['score_group_pgic'] = np.where(imputed_complete_df['mean_score_pgic'] > median_score_pgic, 'high', 'low')
imputed_complete_df['score_group_phq9'] = np.where(imputed_complete_df['mean_score_phq9'] > median_score_pgic, 'high', 'low')



# Define conditions for score_group_pgic and score_group_phq9
groups = [
    (imputed_complete_df['score_group_pgic'] == 'low') & (imputed_complete_df['score_group_phq9'] == 'low'),
    (imputed_complete_df['score_group_pgic'] == 'low') & (imputed_complete_df['score_group_phq9'] == 'high'),
    (imputed_complete_df['score_group_pgic'] == 'high') & (imputed_complete_df['score_group_phq9'] == 'low'),
    (imputed_complete_df['score_group_pgic'] == 'high') & (imputed_complete_df['score_group_phq9'] == 'high'),
]

# Define the corresponding values
values = [1, 2, 3, 4]

# Create the new column based on the conditions
imputed_complete_df['score_combination'] = np.select(groups, values)

print('Value count for groups:')
print(imputed_complete_df['score_combination'].value_counts())

# Now we have our dataframe, let's see if there is a difference in average completion rate between the groups!

# Group the data by score_combination and calculate the avg_completion_rate
groups = [group['avg_completion_rate'].values for name, group in imputed_complete_df.groupby('score_combination')]

# Perform the Kruskal-Wallis test
kruskal_stat, kruskal_p = stats.kruskal(*groups)

print(f"Kruskal-Wallis H-statistic: {kruskal_stat}, p-value: {kruskal_p}")

# Perform Dunn's test
dunn_results = sp.posthoc_dunn(imputed_complete_df, val_col='avg_completion_rate', group_col='score_combination', p_adjust='bonferroni')

# Display the results
print(dunn_results)

sample_sizes = imputed_complete_df.groupby('score_combination')['avg_completion_rate'].count()

# Let's plot it
# Set the style of the visualization
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='score_combination', y='avg_completion_rate', data=imputed_complete_df, palette='Set2')

# Add title and labels
plt.title('Average Completion Rate by Score Combination', fontsize=16)
plt.xlabel('Score Combination', fontsize=14)
plt.ylabel('Average Completion Rate', fontsize=14)

# Annotate the sample sizes on the plot
for i, size in enumerate(sample_sizes):
    ax.text(i, ax.get_ylim()[1] * 0.95, f'n={size}', 
            horizontalalignment='center', size='medium', color='black', weight='semibold')

# Show the plot
plt.xticks(ticks=[0, 1, 2, 3], labels=['Low PGIC, Low PHQ-9 (1)', 'Low PGIC, High PHQ-9 (2)', 'High PGIC, Low PHQ-9 (3)', 'High PGIC, High PHQ-9 (4)'])
plt.show()