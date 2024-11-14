# This script will perform a violin plot of the avg_completion_rate and phq9_completion_rate for the groups separated by improvement status
# Then, we will assess Mann-Withney U test to assess if there are differences in avg_completion_rate, phq9_completion_rate, and baseline_phq9_result across improvement groups
# We will perform RLM model
# Then, we will plot the mean baseline_phq9_result for each group and compare (errorbar = SEM)

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import statsmodels.formula.api as smf

# Import imputed_complete_df and rename a column so it looks better in the cox_phq9 kaplan meier curve in cox_phq9.py (trust the process)
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')
imputed_complete_df.loc[imputed_complete_df['improvement'] == 'non_improver', 'improvement'] = 'non-improver'


print(imputed_complete_df.head())
print(imputed_complete_df.columns)


# Now let's plot phq9 completion rate groups separated by improvement status
imputed_complete_df = imputed_complete_df.dropna(subset='improvement')

# Define the updated color mapping for improvement categories
color_mapping = {'improver': 'lightgreen', 'non-improver': 'lightcoral'}

# Calculate sample sizes for each improvement category
sample_sizes = imputed_complete_df['improvement'].value_counts()

# Create a legend label string for sample sizes
legend_labels = [f"{key}: {sample_sizes[key]}" for key in color_mapping.keys()]
legend_handles = [plt.Line2D([0], [0], color='none', label=label) for label in legend_labels]

# Plot the comparison for avg_completion_rate
plt.figure(figsize=(8, 6))
sns.violinplot(x='improvement', y='avg_completion_rate', data=imputed_complete_df, palette=color_mapping)
plt.title('Comparison of Avg Completion Rate by Improvement')
plt.xlabel('Improvement')
plt.ylabel('Avg Completion Rate')
plt.ylim(-0.2, 1)  # Set y-axis limits
plt.yticks(np.arange(0, 1.1, 0.2))
plt.legend(handles=legend_handles, title='Sample Size', loc='upper right', frameon=True, handletextpad=0.5, handlelength=0)
plt.tight_layout()
plt.savefig(f'../brighten_figures/improvement_completion_rate/avg_completion_rate_improvement_separation.png', dpi=300)

# Plot the comparison for completion_rate_phq9
plt.figure(figsize=(8, 6))
sns.violinplot(x='improvement', y='completion_rate_phq9', data=imputed_complete_df, palette=color_mapping)
plt.title('Comparison of PHQ-9 Completion Rate by Improvement')
plt.xlabel('Improvement')
plt.ylabel('PHQ-9 Completion Rate')
plt.ylim(-0.2, 1.4)  # Set y-axis limits
plt.yticks(np.arange(0, 1.1, 0.2))
plt.legend(handles=legend_handles, title='Sample Size', loc='upper right', frameon=True, handletextpad=0.5, handlelength=0)
plt.tight_layout()
plt.savefig(f'../brighten_figures/improvement_completion_rate/completion_rate_phq9_improvement_separation.png', dpi=300)



############################################### 
# We will perform u-test to compare improvers and non improvers on...
  # 1. avg_completion_rate
  # 2. phq9_completion_rate
  # 3. baseline_phq9_score

# Extract the data for each category
improver = imputed_complete_df[imputed_complete_df['improvement'] == 'improver']['avg_completion_rate']
nonimprover = imputed_complete_df[imputed_complete_df['improvement'] == 'non-improver']['avg_completion_rate']

# Perform Mann-Whitney U test for avg_completion_rate
u_stat, p_value = mannwhitneyu(improver, nonimprover, alternative='two-sided')
print('Comparing average completion rate between groups:')
print(f"Mann-Whitney U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Extract for PHQ-9 completion rate
improver = imputed_complete_df[imputed_complete_df['improvement'] == 'improver']['completion_rate_phq9']
nonimprover = imputed_complete_df[imputed_complete_df['improvement'] == 'non-improver']['completion_rate_phq9']

# Perform  Mann-Whitney U test for phq9_completion_rate
u_stat, p_value = mannwhitneyu(improver, nonimprover, alternative='two-sided')
print('\nComparing PHQ-9 completion rate between groups:')
print(f"Mann-Whitney U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Extract the baseline_phq9_result for each improvement category
improver = imputed_complete_df[imputed_complete_df['improvement'] == 'improver']['baseline_phq9_result']
nonimprover = imputed_complete_df[imputed_complete_df['improvement'] == 'non-improver']['baseline_phq9_result']

# Perform  Mann-Whitney U test to compare baseline_phq9_result across groups
u_stat, p_value = mannwhitneyu(improver, nonimprover, alternative='two-sided')
print('\nComparing baseline_phq9_result between improvement groups:')
print(f"Mann-Whitney U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")

print(f'Mean baseline improver: {imputed_complete_df[imputed_complete_df["improvement"] == "improver"]["baseline_phq9_result"].mean()}')
print(f'SD baseline improver: {imputed_complete_df[imputed_complete_df["improvement"] == "improver"]["baseline_phq9_result"].std()}')
print(f'Mean baseline non-improver: {imputed_complete_df[imputed_complete_df["improvement"] == "non-improver"]["baseline_phq9_result"].mean()}')
print(f'SD baseline non-improver: {imputed_complete_df[imputed_complete_df["improvement"] == "non-improver"]["baseline_phq9_result"].std()}')

print(f'Mean +/- STD for baseline PHQ-9 result for the cohort: {imputed_complete_df["baseline_phq9_result"].mean()} +/- {imputed_complete_df["baseline_phq9_result"].std()}')


# RLM Model
model = smf.rlm('avg_completion_rate ~ baseline_phq9_result + C(improvement)', data=imputed_complete_df).fit()
print(model.summary())







# Now let's plot mean baseline PHQ9 result for each group

# Group data by 'improvement' and calculate the mean and SEM for baseline_phq9_result
grouped_df = imputed_complete_df.groupby('improvement').agg(
    mean=('baseline_phq9_result', 'mean'),
    sem=('baseline_phq9_result', sem),
    sample_size=('baseline_phq9_result', 'count')
).reset_index()

print(grouped_df)

# Plotting the mean with SEM error bars
phq9_zones = {'Minimal': (0, 5), 'Mild': (5, 10), 'Moderate': (10, 15), 'Moderately Severe': (15, 20), 'Severe': (20, 27)}
phq9_colors = ['green', 'yellow', 'orange', 'red', 'darkred']

plt.figure(figsize=(10, 6))

for idx, (zone, limits) in enumerate(phq9_zones.items()):
  plt.axhspan(limits[0], limits[1], color=phq9_colors[idx], alpha=0.3, label=zone)

plt.errorbar(grouped_df['improvement'], grouped_df['mean'], yerr=grouped_df['sem'], fmt='o', capsize=5, color='b', ecolor='r')

# Add annotations for PHQ-9 values
for i, sample_size in enumerate(grouped_df['sample_size']):
  if pd.notna(sample_size):
      plt.annotate(f'n={sample_size:.0f}', (grouped_df['improvement'].iloc[i], grouped_df['mean'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Mean Baseline PHQ-9 Result Across Improvement Groups')
plt.xlabel('Improvement Group')
plt.ylabel('Mean Baseline PHQ-9 Result')
plt.ylim(0,27)
plt.yticks(np.arange(0,28,3))
plt.tight_layout()
plt.savefig(f'../brighten_figures/improvement_completion_rate/mean_baseline_phq9_result_across_improvement_groups.png', dpi=300)








