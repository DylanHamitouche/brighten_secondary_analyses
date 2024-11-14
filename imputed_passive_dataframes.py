# The script will create 3 dataframes:
#       1. V1 dataframe for passive data in V1 cohort
#       2. V2 datafame for passive data in V2 cohort that contain both mobility and communication
#       3. V2 dataframe for passive data in V2 cohort that contains only mobility
#
#       It is necessary to create 2 v2 dataframes, because a lot of participants who provided data on communication provided data on mobility, but the opposite is not true
#
# For the passive dataframes, I will drop parameters that have more than 30% missing values. The rest will be imputed by missing forest

# For Youcef: I didn't end up using passive data a lot, so don't spend too much time on this



import pandas as pd
from missforest.missforest import MissForest


# Read data
passive_features_v1_df = pd.read_csv('../brighten_data/Passive Features Brighten V1.csv')
passive_mobility_features_v2_df = pd.read_csv('../brighten_data/Passive mobility Features Brighten v2.csv')
passive_comms_features_v2_df = pd.read_csv('../brighten_data/Passive Phone Communication Features Brighten v2.csv')
imputed_complete_df = pd.read_csv('../brighten_data/imputed_complete_df.csv')

print('Number of participants in each raw dataframe:')
print(f'Number of participants in raw passive_features_v1_df: {passive_features_v1_df["participant_id"].nunique()}')
print(f'Number of participants in raw passive_mobility_features_v2_df: {passive_mobility_features_v2_df["participant_id"].nunique()}')
print(f'Number of participants in raw passive_comms_features_v2_df: {passive_comms_features_v2_df["participant_id"].nunique()}')

print('COLUMNS FOR EACH PASSIVE DATAFRAME:')
print(f'passive_features_v1_df: {passive_features_v1_df.columns}')
print(f'passive_mobility_features_v2_df: {passive_mobility_features_v2_df.columns}')
print(f'passive_comms_features_v2_df: {passive_comms_features_v2_df.columns}')

# We select the relevant numerical columns for each passive dataframe (weather and passive cluster excluded because found not relevant for analysis)
# We must exclude categorical data because we can't take the average of that

columns_to_remove = ['ROW_ID', 'week', 'ROW_VERSION']

passive_features_v1_df_numeric_cols = [col for col in passive_features_v1_df.columns if pd.api.types.is_numeric_dtype(passive_features_v1_df[col]) or col == 'participant_id']
passive_features_v1_df = passive_features_v1_df[passive_features_v1_df_numeric_cols]
passive_features_v1_df.drop(columns_to_remove, axis=1, inplace=True)

passive_mobility_features_v2_df_numeric_cols = [col for col in passive_mobility_features_v2_df.columns if pd.api.types.is_numeric_dtype(passive_mobility_features_v2_df[col]) or col == 'participant_id']
passive_mobility_features_v2_df = passive_mobility_features_v2_df[passive_mobility_features_v2_df_numeric_cols]
passive_mobility_features_v2_df.drop(columns_to_remove, axis=1, inplace=True)

passive_comms_features_v2_df_numeric_cols = [col for col in passive_comms_features_v2_df.columns if pd.api.types.is_numeric_dtype(passive_comms_features_v2_df[col]) or col == 'participant_id']
passive_comms_features_v2_df = passive_comms_features_v2_df[passive_comms_features_v2_df_numeric_cols]
passive_comms_features_v2_df.drop(columns_to_remove, axis=1, inplace=True)


# We group by participant and apply mean() to keep only the average value for a given participant
passive_features_v1_df = passive_features_v1_df.groupby('participant_id').mean().reset_index()
passive_mobility_features_v2_df = passive_mobility_features_v2_df.groupby('participant_id').mean().reset_index()
passive_comms_features_v2_df = passive_comms_features_v2_df.groupby('participant_id').mean().reset_index()



# Merge complete dataframe containing scores and demographics with dataframes containing passive data
imputed_complete_df = pd.merge(imputed_complete_df, passive_features_v1_df, on='participant_id', how='outer')
imputed_complete_df = pd.merge(imputed_complete_df, passive_mobility_features_v2_df, on='participant_id', how='outer')
imputed_complete_df = pd.merge(imputed_complete_df, passive_comms_features_v2_df, on='participant_id', how='outer')

# Let's create v2 dataframe for mobility, by removing people who don't have one of its feature
v2_mobility_complete_df = imputed_complete_df.dropna(subset=['came_to_work'])

# Remove columns with more than 30% missing data, by removing people who don't have one of its feature
number_of_removed_columns_v2 = 0
for col in v2_mobility_complete_df.columns:
    if v2_mobility_complete_df[col].isna().sum() > 0.5 * len(v2_mobility_complete_df):
        v2_mobility_complete_df = v2_mobility_complete_df.drop(columns=col)
        print(f'{col} has been removed')
        number_of_removed_columns_v2 += 1
print(f'Number of columns that were removed because too many values are missing: {number_of_removed_columns_v2}')
print(v2_mobility_complete_df.head())
print(v2_mobility_complete_df.columns)




# Let's create v2 dataframe for communication,  by removing people who don't have one of its feature
v2_communication_complete_df = imputed_complete_df.dropna(subset=['callDuration_incoming'])

# Remove columns with more than 30% missing data
number_of_removed_columns_v2 = 0
for col in v2_communication_complete_df.columns:
    if v2_communication_complete_df[col].isna().sum() > 0.5 * len(v2_communication_complete_df):
        v2_communication_complete_df = v2_communication_complete_df.drop(columns=col)
        print(f'{col} has been removed')
        number_of_removed_columns_v2 += 1
print(f'Number of columns that were removed because too many values are missing: {number_of_removed_columns_v2}')
print(v2_communication_complete_df.head())
print(v2_communication_complete_df.columns)



# Create v2 dataframe for passive features
common_ids_df = passive_features_v1_df[['participant_id']].drop_duplicates()

v1_passive_complete_df = imputed_complete_df.merge(common_ids_df, on='participant_id')

# Remove column 'day' from v1 passive dataframe
v1_passive_complete_df.drop('day', axis=1, inplace=True)

# Remove columns with more than 30% missing data
number_of_removed_columns_v2 = 0
for col in v1_passive_complete_df.columns:
    if v1_passive_complete_df[col].isna().sum() > 0.5 * len(v1_passive_complete_df):
        v1_passive_complete_df = v1_passive_complete_df.drop(columns=col)
        print(f'{col} has been removed')
        number_of_removed_columns_v2 += 1
print(f'Number of columns that were removed because too many values are missing: {number_of_removed_columns_v2}')
print(v1_passive_complete_df.head())
print(v1_passive_complete_df.columns)


list_of_df = [v1_passive_complete_df, v2_communication_complete_df, v2_mobility_complete_df]
df_names = ['v1_passive_complete_df', 'v2_communication_complete_df', 'v2_mobility_complete_df']
imputed_columns_dict = {}


# Process each dataframe
for df, name in zip(list_of_df, df_names):

    df.drop('study', axis=1, inplace=True)

    if 'startdate' in df.columns:
        df.drop('startdate', axis=1, inplace=True)
    

    completion_rate_columns = [col for col in df.columns if 'completion_rate' in col]
    df[completion_rate_columns] = df[completion_rate_columns].fillna(0) 
    
    # Identify columns with missing rate <= 30% for imputation
    columns_to_impute = [col for col in df.columns if 0 < df[col].isna().sum() / len(df) < 0.5 and col != 'participant_id']

    # Store the list of columns to impute in the dictionary
    imputed_columns_dict[name] = columns_to_impute


    # Identify categorical columns to impute (excluding participant_id)
    categorical_columns = [col for col in columns_to_impute if df[col].dtype == 'object']
    
    # Dictionary to store the mapping for each categorical column
    category_mappings = {}

    # Convert categorical columns to category codes and store mappings
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col])
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes.replace(-1, pd.NA)  # Replace -1 with NA for missing values

    # Debug: Check mappings
    for col, mapping in category_mappings.items():
        print(f"Mapping for column '{col}': {mapping}")

    # Impute numeric columns
    miss_forest = MissForest()
    numeric_columns = [col for col in columns_to_impute if col not in categorical_columns and df[col].dtype != 'object']
    imputed_numeric_data = miss_forest.fit_transform(df[numeric_columns])
    df_imputed_numeric = pd.DataFrame(imputed_numeric_data, columns=numeric_columns, index=df.index)

    # Impute categorical columns separately
    imputed_categorical_data = miss_forest.fit_transform(df[categorical_columns])
    df_imputed_categorical = pd.DataFrame(imputed_categorical_data, columns=categorical_columns, index=df.index)

    # Restore categorical columns to original categories
    for col, mapping in category_mappings.items():
        df_imputed_categorical[col] = df_imputed_categorical[col].round().astype(int)  # Round imputed values
        df_imputed_categorical[col] = df_imputed_categorical[col].map(lambda x: mapping.get(x, pd.NA))
        df_imputed_categorical[col] = pd.Categorical(df_imputed_categorical[col], categories=mapping.values())

    # Combine imputed dataframes with non-imputed columns
    df_final = pd.concat([df_imputed_numeric, df_imputed_categorical, df.drop(columns=columns_to_impute)], axis=1)

    print(f'Number of participants in {name}: {df_final["participant_id"].nunique()} ')
    print(f'Length of dataframe {name}: {len(df_final)}')

    # Debug: Check for missing values and incorrect categories
    for col in df_final.columns:
        print(f"Missing values in column '{col}' after imputation: {df_final[col].isna().sum()}")

    # Save the dataframes
    df_final.to_csv(f'../brighten_data/imputed_{name}.csv', index=False)

    print('FINAL DEBUG')
    print(f'Number of participants in {name}: {df["participant_id"].nunique()}')
    print(f'Make sure there are no duplicates: {len(df)}')
    print(df.columns)

    # Everything works!




# print('Imputed columns for each dataframe:')
# print(imputed_columns_dict['v1_passive_complete_df'])
# print(imputed_columns_dict['v2_communication_complete_df'])
# print(imputed_columns_dict['v2_mobility_complete_df'])




