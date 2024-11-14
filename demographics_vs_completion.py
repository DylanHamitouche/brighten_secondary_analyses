# This script will attempt to study the association between average completion rate and demographics parameters
# Mann-Withney U test will be performed when there are only two groups being compareed
# Kriskall-Wallis test will be performed when there are more than two groups being compared
# If Kriskall_Wallis is significant, Dunn'stet will be performed as a post-hoc test

# Latino vs Non-Latino will be performed to better study the association between this population and completion rate

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import scikit_posthocs as sp


# Read demographics data
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')


# Create age_group column
bins = [18, 30, 40, 50, 60, 70]
labels = [
    '18-30 years',
    '31-40 years',
    '41-50 years',
    '51-60 years',
    '61-70 years'
]
imputed_complete_df['age_group'] = pd.cut(imputed_complete_df['age'].astype(int), bins=bins, labels=labels, include_lowest=True, right=True)



# List of columns to plot
columns_to_plot = ['gender', 'education', 'working', 'income_satisfaction', 'income_lastyear', 
                   'marital_status', 'race', 'heard_about_us', 'device', 'study_arm', 'study', 'age_group']

# Set a new dataframes to store the significant results
significant_comparisons_df = pd.DataFrame(columns=['comparison', 'u-statistic', 'p-value'])

# For every column in columns_to_plot, we will check how many groups there are and compare each combination of groups together
# If column has 2 groups: u-test
# If column has >2 groups: kruskal-wallis
# In both instances, we will store the results in the significant_comparisons_df if the bonferroni-corrected p-value < 0.05

for column in columns_to_plot:
    group_list = imputed_complete_df[column].unique().tolist()
    group_list = [x for x in group_list if not pd.isna(x)]
    group_stats = []
    number_of_groups = len(group_list)
    group_completion_rates = []
    print(f' How many people for each age group in column {column}: {imputed_complete_df[column].value_counts()}')
    for group in group_list:
        group_data = imputed_complete_df[imputed_complete_df[column] == group]['avg_completion_rate']
        group_mean = group_data.mean()
        group_std = group_data.std()
        group_count = group_data.count()
        group_stats.append((group, group_mean, group_std, group_count))
    group_stats_df = pd.DataFrame(group_stats, columns=['group', 'mean', 'std', 'count'])

    # Define the custom order if the column is 'age_group'
    if column == 'age_group':
        desired_order = ['18-30 years', '31-40 years', '41-50 years', '51-60 years', '61-70 years']
        group_stats_df['group'] = pd.Categorical(group_stats_df['group'], categories=desired_order, ordered=True)
        group_stats_df = group_stats_df.sort_values('group').reset_index(drop=True)
        group_list = desired_order  # Use this custom order for plotting

    if number_of_groups == 2:
        for i in range(len(group_stats_df)):
            for j in range(i + 1, len(group_stats_df)):
                group_i = group_stats_df.iloc[i]['group']
                group_j = group_stats_df.iloc[j]['group']
                
                u_test_result = mannwhitneyu(
                    imputed_complete_df[imputed_complete_df[column] == group_i]['avg_completion_rate'],
                    imputed_complete_df[imputed_complete_df[column] == group_j]['avg_completion_rate']
                )
                
                group_stats_df.loc[i, f'u-statistic_vs_{group_j}'] = u_test_result.statistic
                group_stats_df.loc[i, f'p-value_vs_{group_j}'] = u_test_result.pvalue

                if u_test_result.pvalue < 0.05:
                    comparison_string = f'{group_i}_vs_{group_j}'
                    new_row = pd.DataFrame({
                    'comparison': [comparison_string],
                    'u-statistic': [u_test_result.statistic],
                    'p-value': [u_test_result.pvalue]
                    })
                    significant_comparisons_df = pd.concat([significant_comparisons_df, new_row], ignore_index=True)

    elif number_of_groups > 2:
        # Gather the completion rates for each group
        for i in range(len(group_stats_df)):
            group = group_stats_df.iloc[i]['group']
            completion_rates = imputed_complete_df[imputed_complete_df[column] == group]['avg_completion_rate']
            group_completion_rates.append(completion_rates)

        # Perform the Kruskal-Wallis test
        h_statistic, p_value = kruskal(*group_completion_rates)

        # Add the Kruskal-Wallis result to the dataframe
        group_stats_df['H-statistic'] = h_statistic
        group_stats_df['p-value'] = p_value

        # Check if the result is significant (for Kruskall-wallis, we set significance = 0.05)
        if p_value < 0.05:
            # Perform a post-hoc test (Dunn's test)
            posthoc_results = sp.posthoc_dunn(group_completion_rates, p_adjust='bonferroni')

            # Process posthoc_results to add significant comparisons
            for i in range(len(posthoc_results)):
                for j in range(i + 1, len(posthoc_results)):
                    if posthoc_results.iloc[i, j] < 0.05 / (number_of_groups * (number_of_groups - 1) / 2):
                        comparison_string = f'{group_stats_df.iloc[i]["group"]} vs {group_stats_df.iloc[j]["group"]}'
                        new_row = pd.DataFrame({
                            'comparison': [comparison_string],
                            'H-statistic': [h_statistic],  # Not applicable for pairwise comparisons
                            'p-value': [posthoc_results.iloc[i, j]]
                        })
                        significant_comparisons_df = pd.concat([significant_comparisons_df, new_row], ignore_index=True)


 
    print(f'Group list is the following: {group_list}')
    print(group_stats_df)
    print(significant_comparisons_df)
    



############## PLOTTING FOR EVERY COLUMN IN COLUMNS_TO_PLOT
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imputed_complete_df, x=column, y='avg_completion_rate', errorbar='se')
    plt.xlabel(column)
    plt.ylabel('Average Completion Rate')
    plt.title(f'Average Completion Rate by {column}')
    plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
    if column != 'age':
        if number_of_groups == 2:
            # Extract u-statistic and p-value:
            u_statistic_column = [col for col in group_stats_df.columns if 'u-statistic' in col][0]
            u_statistic_value = group_stats_df[u_statistic_column].dropna().iloc[0]
            p_value_column = [col for col in group_stats_df.columns if 'p-value' in col][0]
            p_value = group_stats_df[p_value_column].dropna().iloc[0]
            # Add a legend or explanatory text to the plot
            plt.gca().text(
                0.95, 0.05,
                f'U-statistic = {u_statistic_value:.0f}\n'
                f'P-value = {p_value:.3e}',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
                )
        if number_of_groups > 2:
            h_statistic_column = [col for col in group_stats_df.columns if 'H-statistic' in col][0]
            h_statistic_value = group_stats_df[h_statistic_column].dropna().iloc[0]
            p_value_column = [col for col in group_stats_df.columns if 'p-value' in col][0]
            p_value = group_stats_df[p_value_column].dropna().iloc[0]
            # Add a legend or explanatory text to the plot
            plt.gca().text(
                0.95, 0.05,
                f'H-statistic = {h_statistic_value:.0f}\n'
                f'p-value = {p_value:.3e}',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
                )

        plt.tight_layout()
    plt.savefig(f'../brighten_figures/demographics_completion_excluded_figures/excluded_completion_rate_demographics_{column}.png', dpi=300)


# Saving the dataframe as csv
significant_comparisons_df.to_csv('../brighten_data/significant_comparisons_demographics_completion_rate.csv', index=False)



#######################################################################################

# ADDITIONAL: COMPARISON OF COMPLETION RATE BETWEEN LATINO VS NON-LATINO 

latino_imputed_complete_df = imputed_complete_df

for index, row in latino_imputed_complete_df.iterrows():
    if row['race'] != 'Hispanic/Latino':
        latino_imputed_complete_df.loc[index, 'race'] = 'Non-Hispanic/Latino'

# Perform T-test
non_latino_group = latino_imputed_complete_df.loc[latino_imputed_complete_df['race'] == 'Non-Hispanic/Latino', 'avg_completion_rate']
latino_group = latino_imputed_complete_df.loc[latino_imputed_complete_df['race'] == 'Hispanic/Latino', 'avg_completion_rate']
u_test_result = mannwhitneyu(non_latino_group, latino_group)
latino_comparison_u_statistic = u_test_result.statistic
latino_comparison_p_value = u_test_result.pvalue


# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=latino_imputed_complete_df['race'], y=imputed_complete_df['avg_completion_rate'], errorbar='se')
plt.xlabel('Race')
plt.ylabel('Average Completion Rate')
plt.title('Average Completion Rate in Hispanic/Latino vs Non-Hispanic/Latino')
plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
plt.tight_layout()
        # Add a legend or explanatory text to the plot
plt.gca().text(
          0.95, 0.05,
          f'u-statistic = {latino_comparison_u_statistic:.0f}\n'
          f'P-value = {latino_comparison_p_value:.3f}',
          horizontalalignment='right',
          verticalalignment='bottom',
          transform=plt.gca().transAxes,
          fontsize=12,
          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
plt.savefig(f'../brighten_figures/demographics_completion_excluded_figures/latino_excluded_completion_rate.png', dpi=300)





