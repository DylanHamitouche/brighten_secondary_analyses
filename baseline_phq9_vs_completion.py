import pandas as pd
import statsmodels.api as sm
from scipy import stats

imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

####################### baseline phq9 result vs engagement


# Group by engagement status and calculate the average baseline PHQ-9 result and standard error of the mean (SEM)
avg_phq9_results = imputed_complete_df.groupby('engagement')['baseline_phq9_result'].agg(['mean', 'std'])

# Display the results
print(avg_phq9_results)

# Optionally, perform a t-test to see if there's a significant difference
engaged_results = imputed_complete_df[imputed_complete_df['engagement'] == 'engaged']['baseline_phq9_result']
quitters_results = imputed_complete_df[imputed_complete_df['engagement'] == 'quit']['baseline_phq9_result']

print(engaged_results.count())
print(quitters_results.count())

t_stat, p_value = stats.ttest_ind(engaged_results, quitters_results)

print(f"T-test result: t-statistic = {t_stat}, p-value = {p_value}")


############### baseline phq9 results vs avg completion rate


imputed_complete_df = imputed_complete_df[imputed_complete_df['avg_completion_rate'] > 0]


correlation = imputed_complete_df['baseline_phq9_result'].corr(imputed_complete_df['avg_completion_rate'])
print(f'Correlation between baseline PHQ-9 and average completion rate: {correlation:.3f}')

# Fit a linear regression model
X = sm.add_constant(imputed_complete_df['baseline_phq9_result'])  # Add a constant for the intercept
y = imputed_complete_df['avg_completion_rate']
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())