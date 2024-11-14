# This script is the backbone of the project, it aims to create the super mega dataframe that will be used everywhere else
# This script uses available data from demographics and questionnaire dataframes to create a complete imputed dataframe containing the following information:
      # Demographics
      # Mean questionnaire score throughout the study
      # Average completion rate of the whole study
      # Completion rate for each non-baseline questionnaire
      # Average completion rate of the study
      # Completion rate group (high vs low) (separated by median average completion rate)
      # Engagement group (engaged vs quit) (To be engaged, they must have completed at least one non-baseline questionnaire)
      # PHQ9 Improvement group (better, worse, or same) (Determined by most recent phq-9 score substracted by oldest phq-9 score)
            # Clinically significant difference is 5 or more:
            # https://www-ncbi-nlm-nih-gov.proxy3.library.mcgill.ca/pmc/articles/PMC3281149/#:~:text=Scores%20of%205%2C%2010%2C%2015,less%20than%205%20represents%20remission.
            # https://link.springer.com/article/10.1007/s00127-022-02402-y 

# The complete dataframe will then be imputed with miss forest
# Only columns with less than 30% missing data will be imputed
# Columns that were imputed: ['baseline_phq9_result', 'gender', 'education', 'working', 'income_satisfaction', 'income_lastyear', 'marital_status', 'race', 'age', 'heard_about_us', 'device', 'study']





import pandas as pd
import numpy as np
from missforest.missforest import MissForest


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

# Initialize combined_df
combined_df = pd.DataFrame()

# Need to create a "sum" column for relevant dataframes if they don't have one
df_sleep['sleep_sum'] = df_sleep['sleep_1'] + df_sleep['sleep_2'] + df_sleep['sleep_3']
df_sds['sds_sum'] = df_sds['sds_1'] + df_sds['sds_2'] + df_sds['sds_3'] + df_sds['stress'] + df_sds['support']

print(df_impact.head())
df_impact['impact_sum'] = 0
for i in range(1,5):
  df_impact['impact_sum'] += df_impact[f'screen_{i}']

df_pgic['pgic_sum'] = df_pgic['mood_1']


# Create improvement column and its logic
# To do it, we will add the improvement column directly in dataframe df_phq9
# I will merge it with df_baseline_phq9 to use it as initial score
# Then, latest score available will be substracted by the baseline score
# if value positive: improvement: Yes
# if value negative: improvemen: No



phq9_questionnaires_list = [df_phq9, df_baseline_phq9]

# Create the 'phq9_sum' column in df_baseline_phq9
df_baseline_phq9['baseline_phq9_result'] = df_baseline_phq9.filter(like='phq9_').sum(axis=1)

# Merge the 'baseline_phq9_result' column and its date into df_phq9 on 'participant_id' using an outer join
df_phq9 = df_phq9.merge(df_baseline_phq9[['participant_id', 'baseline_phq9_result', 'baselinePHQ9date']], on='participant_id', how='outer')

# Remove participants that have baseline phq9 score but no non-baseline phq9 score, because we won't be able to calculate improvement for them
df_phq9 = df_phq9.dropna(subset='phq9_sum')

# Remove participants if they have no baseline phq9 score and only one non-baseline phq9 score, because we won't be able to calculate improvement for them
# This removes 9 participants
merged_phq9_participant_counts = df_phq9['participant_id'].value_counts()
df_phq9= df_phq9[~((df_phq9['participant_id'].isin(merged_phq9_participant_counts[merged_phq9_participant_counts == 1].index)) & 
                        (df_phq9['baseline_phq9_result'].isna()))]


# Convert phq9Date and baselinePHQ9date to datetime
df_phq9['phq9Date'] = pd.to_datetime(df_phq9['phq9Date'])
df_phq9['baselinePHQ9date'] = pd.to_datetime(df_phq9['baselinePHQ9date'])

print('DEBUG')
print(df_phq9['baseline_phq9_result'].value_counts())

# Define a function to calculate improvement
def calculate_improvement_phq9(group):
    # Sort by date to find the most recent and oldest records
    group = group.sort_values(by='phq9Date')
    
    if pd.notna(group['baseline_phq9_result'].iloc[0]):
        # Use baseline_phq9_result if available
        most_recent_score = group['phq9_sum'].iloc[-1] # This will take the most recent score (first one in the sorted list)
        baseline_score = group['baseline_phq9_result'].iloc[0] # This will take the baseline score
        difference = most_recent_score - baseline_score
    else:
        # Use most recent and oldest phq9_sum, if baseline score is not available
        most_recent_score = group['phq9_sum'].iloc[-1]
        oldest_score = group['phq9_sum'].iloc[0]
        difference = most_recent_score - oldest_score
    
    # Determine improvement status, according to MCID = 5
    if difference <= -5:
        improvement = 'improver'
    else:
        improvement = 'non_improver'
    
    # Assign improvement to all rows in the group
    group['improvement'] = improvement
    return group

# Apply the function to each participant group and update the DataFrame
df_phq9 = df_phq9.groupby('participant_id').apply(calculate_improvement_phq9).reset_index(drop=True)

# Save for cox model analysis (see cox_phq9.py)
df_phq9.to_csv(f'../brighten_data/df_phq9_over_time.csv', index=False)


# Create a dictionnary were each questionnaire has its number of submissions to participants (how many questionnaires were sent for the 12 week duration of the study)
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

# Define a function to process each questionnaire
def process_questionnaire(df_questionnaire, key, value):
    # Take all the unique participants for the questionnaire
    participants = pd.DataFrame({'participant_id': df_questionnaire['participant_id'].unique()})

    # Count how many time they appear in the questionnaire (that amount equals the number of questionnaires they completed)
    questionnaire_counts_df = participants.merge(
        df_questionnaire.groupby('participant_id').size().reset_index(name='number of questionnaires'),
        on='participant_id', how='left'
    )

    # Create a column completion_rate_ for the questionnaire, where the number of questionnaires they completed is divided by the number of questionnaires they were supposed to complete
    questionnaire_counts_df[f'completion_rate_{key}'] = (questionnaire_counts_df['number of questionnaires'] / value).clip(upper=1)

    # Discard other irrelevant columns
    questionnaire_counts_df = questionnaire_counts_df[['participant_id', f'completion_rate_{key}']]
    
    # For the questionnaires 'mental', 'satisfaction', 'other', we won't calculate score, as it won't be useful for analysis (also, completion rates for those questionnaires are very low)
    # For the other questionnaires, we calculate the mean of all sum columns, which will render the mean score for the 12 weeks.
    if key in ['mental', 'satisfaction', 'other']:
        mean_scores = pd.DataFrame({'participant_id': df_questionnaire['participant_id'].unique()})
    else:
        sum_columns = [column for column in df_questionnaire.columns if 'sum' in column]
        mean_scores = df_questionnaire.groupby('participant_id')[sum_columns].mean().reset_index()
    
    mean_score_column = f'mean_score_{key}'
    if key not in ['mental', 'satisfaction', 'other']:
        mean_scores[mean_score_column] = mean_scores[sum_columns].mean(axis=1)
    else:
        mean_scores[mean_score_column] = np.nan

    # Discard irrelevant columns
    mean_scores = mean_scores[['participant_id', mean_score_column]]
    
    # Return the questionnaire count, which contains participant id and completion rate
    # Return mean_scores, which contains participant id and mean score
    result_df = pd.merge(questionnaire_counts_df, mean_scores, on='participant_id')
    return result_df

# mean_score_by_completion_rate_dict = {}

# Process each questionnaire from the dictionnary, run the function process_questionnaire() and merge results
for key, value in dict_of_questionnaires.items():
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
    elif key == 'mental':
        df_questionnaire = df_mental_service 
    elif key == 'satisfaction':
        df_questionnaire = df_study_satisfaction
    elif key == 'other':
        df_questionnaire = df_other
    
    # Process the questionnaire to return the completion rate and the mean score for each participant
    result_df = process_questionnaire(df_questionnaire, key, value)
    
    # Store all the results in combined_df
    if combined_df.empty:
        combined_df = result_df
    else:
        combined_df = pd.merge(combined_df, result_df, on='participant_id', how='outer')

    # # Calculate mean and std score for a given completion rate
    # mean_scores = combined_df.groupby(f'completion_rate_{key}').agg(
    # mean_score_mean=(f'mean_score_{key}', 'mean'),
    # mean_score_std=(f'mean_score_{key}', 'std'),
    # count=(f'mean_score_{key}', 'count')).reset_index()

    # # Calculate the standard error
    # mean_scores['mean_score_sem'] = mean_scores['mean_score_std'] / np.sqrt(mean_scores['count'])
    # mean_score_by_completion_rate_dict[key] = mean_scores


# Calculate avg_completion_rate
combined_df['avg_completion_rate'] = combined_df.loc[:, combined_df.columns.str.startswith('completion_rate_')].mean(axis=1, skipna=True)


# Add baseline phq9 to combined_df
combined_df = pd.merge(df_baseline_phq9, combined_df, on='participant_id', how='outer')

# Add Audit to combined_df (audit is baseline questionnaire)
combined_df = pd.merge(df_audit, combined_df, on='participant_id', how='outer')

# Add gad-7 to combined_df (gad-7 is a baseline questionnaire)
combined_df = pd.merge(df_gad, combined_df, on='participant_id', how='outer')

# Print metrics
print("Final combined_df")
print(combined_df.head())
print(combined_df.columns)


# Drop irrelevant columns from combined_df
combined_df = combined_df.drop(columns=[
'ROW_ID',
'ROW_VERSION', 
'dt_response_x',
'week_x',
'gad7_1',
'gad7_2',
'gad7_3',
'gad7_4',
'gad7_5',
'gad7_6',
'gad7_7',
'gad7_8',
'phq9_1', 
'phq9_2', 
'phq9_3', 
'phq9_4', 
'phq9_5', 
'phq9_6', 
'phq9_7', 
'phq9_8', 
'phq9_9',
'ROW_ID_x',
'ROW_VERSION_x',
'dt_response_y',
'week_y',
'alc_1',
'alc_2',
'alc_3',
'study' # We drop this one so it won't be duplicated when merging with demographics
])




# Merge demographics dataframe with scores dataframes
complete_df = pd.merge(combined_df, demographics_df, on='participant_id', how='outer') # There are 8 participants with no demographics data

# We have to drop ROW_ID and ROW_VERSION again, because they were added again when merging with demographics data
# These columns are useless
complete_df = complete_df.drop(columns=['ROW_ID','ROW_VERSION'], axis=1)

# I have noticed that some people have NaN in completion rate columns 
# Since they have not been calculated, it simply means that they are absent from the dataframes of the questionnaires
# Because they are absent, they haven't completed any questionnaire of that type
# Thus, we must set their value to 0 instead of NaN
# Let's fix this issue:
completion_rate_columns = [col for col in complete_df.columns if 'completion_rate' in col]
complete_df[completion_rate_columns] = complete_df[completion_rate_columns].fillna(0)



################### Now we will create completion_group column and engagement column

# Filter the DataFrame to include only rows where 'avg_completion_rate' > 0 (select engaged participants)
filtered_complete_df = complete_df[complete_df['avg_completion_rate'] > 0]

# Calculate the median of 'avg_completion_rate' for the filtered DataFrame
median_avg_completion_rate = filtered_complete_df['avg_completion_rate'].median()
print(f'Median avg_completion_rate: {median_avg_completion_rate}')

# Create the columns
complete_df['completion_group'] = ''
complete_df['engagement'] = ''

# Classify participants in groups
for index, row in complete_df.iterrows():
    if row['avg_completion_rate'] > median_avg_completion_rate:
        complete_df.at[index, 'completion_group'] = 'high'
        complete_df.at[index, 'engagement'] = 'engaged'
    elif 0 < row['avg_completion_rate'] <= median_avg_completion_rate:
        complete_df.at[index, 'completion_group'] = 'low'
        complete_df.at[index, 'engagement'] = 'engaged'
    else:
        complete_df.at[index, 'completion_group'] = np.nan
        complete_df.at[index, 'engagement'] = 'quit'

print('Number of participants in each completion group')
print(complete_df['completion_group'].value_counts())
print(f'Mean avg_completion_rate for high completion group: {complete_df[complete_df["completion_group"] == "high"]["avg_completion_rate"].mean()}')
print(f'SD avg_completion_rate for high completion group: {complete_df[complete_df["completion_group"] == "high"]["avg_completion_rate"].std()}')
print(f'Mean avg_completion_rate for low completion group: {complete_df[complete_df["completion_group"] == "low"]["avg_completion_rate"].mean()}')
print(f'Mean avg_completion_rate for low completion group: {complete_df[complete_df["completion_group"] == "low"]["avg_completion_rate"].std()}')

# Create a specific completion group column for each questionnaire
for key, value in dict_of_questionnaires.items():
    complete_df[f'completion_group_{key}'] = np.nan
    specific_median_completion_rate = complete_df.loc[complete_df[f'completion_rate_{key}'] > 0, f'completion_rate_{key}'].median()

    for index, row in complete_df.iterrows():
        if row[f'completion_rate_{key}'] > specific_median_completion_rate:
            complete_df.at[index, f'completion_group_{key}'] = 'high'
        elif 0 < row[f'completion_rate_{key}'] < specific_median_completion_rate:
            complete_df.at[index, f'completion_group_{key}'] = 'low'
        elif row[f'completion_rate_{key}'] == specific_median_completion_rate:
            complete_df.at[index, f'completion_group_{key}'] = 'low'
        else:
            complete_df.at[index, f'completion_group_{key}'] = np.nan



##################### NOW WE'LL DO SOMETHING SIMILAR, BUT WE WILL SEPARATE PARTICIPANTS IN QUARTILES BASED ON COMPLETION RATES FOR EACH QUESTIONNAIRE AND FOR AVERAGE COMPLETION RATE
# Not very useful, i didn't end up using quartiles in the analyses, so Youcef you can skip this part!
# Create quartiles only for participants with avg_completion_rate > 0
complete_df['quartile_avg_completion'] = np.nan
complete_df.loc[complete_df['avg_completion_rate'] > 0, 'quartile_avg_completion'] = pd.qcut(
    complete_df['avg_completion_rate'][complete_df['avg_completion_rate'] > 0], 
    q=4,  # Specify 4 quartiles
    labels=('Q1', 'Q2', 'Q3', 'Q4')  # Use integer labels (0 to 3)
)


# We create quartiles of completion rate for every questionnaire
# Because there are a lot of participants with the same value of completion rate, we have to add some noise
# Example, there are 288 participants with completion_rate_phq9 == 1
for key, value in dict_of_questionnaires.items():
    complete_df[f'quartile_completion_{key}'] = np.nan
    filtered_data = complete_df[complete_df[f'completion_rate_{key}'] > 0]
    
    # Proceed only if there are enough unique values
    if len(filtered_data[f'completion_rate_{key}'].unique()) >= 4:
        try:
            # Add a tiny random noise to help avoid ties
            noisy_values = filtered_data[f'completion_rate_{key}'] + np.random.normal(0, 1e-5, size=len(filtered_data))
            
            # Try to create quartiles using pd.qcut
            bins = pd.qcut(
                noisy_values,
                q=4,  # Specify 4 quantiles
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'  # Allow duplicates to be dropped
            )
            
            # Check if we got exactly 4 bins
            if len(bins.cat.categories) == 4:
                complete_df.loc[complete_df[f'completion_rate_{key}'] > 0, f'quartile_completion_{key}'] = bins
            else:
                raise ValueError("Not enough bins created")
        except ValueError as e:
            # Fallback: Use pd.cut to manually create 4 bins based on percentiles
            try:
                complete_df.loc[complete_df[f'completion_rate_{key}'] > 0, f'quartile_completion_{key}'] = pd.cut(
                    filtered_data[f'completion_rate_{key}'],
                    bins=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4']
                )
            except Exception as fallback_error:
                print(f"Error with {key}: {fallback_error}")
    else:
        print(f"Not enough unique values for {key} to create quartiles.") # satisfaction falls under that category

complete_df = complete_df.drop_duplicates(subset='participant_id', keep='first')



print('DEBUG DIDI')
print(complete_df.head())
print(complete_df.columns)
print(complete_df['completion_group_phq9'].value_counts())



######################## IMPUTATION VIA MISS FOREST

# Identify columns with missing rate <= 30% for imputation
columns_to_impute = [col for col in complete_df.columns if 0 < complete_df[col].isna().sum() / len(complete_df) < 0.3 and col not in ['participant_id', 'startdate', 'completion_group']]
print(f'COLUMNS THAT WILL BE IMPUTED: {columns_to_impute}')

# Identify categorical columns to impute (excluding participant_id)
categorical_columns = [col for col in columns_to_impute if complete_df[col].dtype == 'object']

# Dictionary to store the mapping for each categorical column
category_mappings = {}

# Convert categorical columns to category codes and store mappings
for col in categorical_columns:
    complete_df[col] = pd.Categorical(complete_df[col])
    category_mappings[col] = dict(enumerate(complete_df[col].cat.categories))
    complete_df[col] = complete_df[col].cat.codes.replace(-1, pd.NA)  # Replace -1 with NA for missing values

# Debug: Check mappings
for col, mapping in category_mappings.items():
    print(f"Mapping for column '{col}': {mapping}")

# Impute numeric columns
miss_forest = MissForest()
numeric_columns = [col for col in columns_to_impute if col not in categorical_columns and complete_df[col].dtype != 'object']
imputed_numeric_data = miss_forest.fit_transform(complete_df[numeric_columns])
df_imputed_numeric = pd.DataFrame(imputed_numeric_data, columns=numeric_columns, index=complete_df.index)

# Impute categorical columns separately
imputed_categorical_data = miss_forest.fit_transform(complete_df[categorical_columns])
df_imputed_categorical = pd.DataFrame(imputed_categorical_data, columns=categorical_columns, index=complete_df.index)

# Restore categorical columns to original categories
for col, mapping in category_mappings.items():
    df_imputed_categorical[col] = df_imputed_categorical[col].round().astype(int)  # Round imputed values
    df_imputed_categorical[col] = df_imputed_categorical[col].map(lambda x: mapping.get(x, pd.NA))
    df_imputed_categorical[col] = pd.Categorical(df_imputed_categorical[col], categories=mapping.values())

# Combine imputed dataframes with non-imputed columns
df_final = pd.concat([df_imputed_numeric, df_imputed_categorical, complete_df.drop(columns=columns_to_impute)], axis=1)


print(df_final.head())
print(df_final.columns)
print(df_final['participant_id'].nunique())
print(len(df_final))
df_final.to_csv(f'../brighten_data/imputed_complete_df.csv', index=False)








####################################################################################
# VERIFICATION



# We will print a participant with missing age before and after imputation to see what happened
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

participant_with_missing_age = complete_df.loc[complete_df['age'].isna(), 'participant_id'].iloc[0]
print(f'The following participant has missing age: {participant_with_missing_age}')

print(f'This is their new age: {imputed_complete_df.loc[imputed_complete_df["participant_id"]==participant_with_missing_age, "age"]}') # 59.58 years old

# Let's compare if the values for that participant are the same before and after imputation
before_imputation = complete_df[complete_df['participant_id'] == participant_with_missing_age]
after_imputation = imputed_complete_df[imputed_complete_df['participant_id'] == participant_with_missing_age]

for col in before_imputation.columns:
    print(f"DEBUG: value before imputation for {col}: {before_imputation[col].iloc[0]}")

for col in after_imputation.columns:
    print(f"DEBUG: value after imputation for {col}: {after_imputation[col].iloc[0]}")


# IT ALL WORKS JUST FINE!




###########################################################################################
# Let's print some key values to get a nice summary
print(f'Total number of participants: {imputed_complete_df["participant_id"].nunique()}, or length: {len(imputed_complete_df)}')
print(f'Number of engaged: {len(imputed_complete_df[imputed_complete_df["engagement"]=="engaged"])}')
print(f'Number of disengaged: {len(imputed_complete_df[imputed_complete_df["engagement"]=="quit"])}')
print(f'Number of completion group high: {len(imputed_complete_df[imputed_complete_df["completion_group"]=="high"])}')
print(f'Number of completion group low: {len(imputed_complete_df[imputed_complete_df["completion_group"]=="low"])}')
print(f'Mean avg_completion_rate: {imputed_complete_df["avg_completion_rate"].mean()}')
print(f'std avg_completion_rate: {imputed_complete_df["avg_completion_rate"].std()}')
print(f'Median avg_completion_rate: {imputed_complete_df["avg_completion_rate"].median()}')
print(f'Mean avg_completion_rate for engaged participants: {imputed_complete_df[imputed_complete_df["engagement"] == "engaged"]["avg_completion_rate"].mean()}')
print(f'std avg_completion_rate for engaged participants: {imputed_complete_df[imputed_complete_df["engagement"] == "engaged"]["avg_completion_rate"].std()}')
print(f'Median avg_completion_rate for engaged participants: {imputed_complete_df[imputed_complete_df["engagement"] == "engaged"]["avg_completion_rate"].median()}')


