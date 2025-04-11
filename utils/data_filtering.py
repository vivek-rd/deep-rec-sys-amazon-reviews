import pandas as pd

def iterative_filter_dfs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Explanation:

    Iterative Filtering for Training Data:
    We start by computing user_count and item_count for each row in filtered_train.
    Rows that do not have at least 2 records for both the user and the item are filtered out.
    The process repeats until no additional rows are removed. This ensures that if removing a row causes another user or item to drop below the threshold, it will be removed in the next iteration.
    
    Filtering the Validation Data:
    After obtaining the final filtered training set, we extract the unique user_id and item_id values.
    The validation set is then filtered so that only records with users and items present in the training set are retained.
    This method ensures that the training data is sufficiently dense even if records need to be dropped due to interdependencies, and returns new filtered dataframes for both training and validation.
    
    """
    # Make a copy to avoid modifying original dataframe
    filtered_train = train_df.copy()
    
    while True:
        # Compute record counts for each user and item
        filtered_train['user_count'] = filtered_train.groupby('user_id')['rating'].transform('count')
        filtered_train['item_count'] = filtered_train.groupby('parent_asin')['rating'].transform('count')
        
        # Identify rows that meet the criteria
        mask = (filtered_train['user_count'] >= 2) & (filtered_train['item_count'] >= 2)
        new_filtered = filtered_train[mask].copy()
        
        # If the number of rows hasn't changed, break out of the loop
        if new_filtered.shape[0] == filtered_train.shape[0]:
            break
        filtered_train = new_filtered.copy()
    
    # Filter the validation set: keep only those rows whose user_id and item_id are in the filtered training set
    valid_users = filtered_train['user_id'].unique()
    valid_items = filtered_train['parent_asin'].unique()
    
    filtered_val = val_df[
        (val_df['user_id'].isin(valid_users)) &
        (val_df['parent_asin'].isin(valid_items))
    ].copy()
    
    filtered_test = test_df[
        (test_df['user_id'].isin(valid_users)) &
        (test_df['parent_asin'].isin(valid_items))
    ].copy()
    
    return filtered_train, filtered_val, filtered_test

# Example usage:
import pandas as pd

full_train = pd.read_csv('../data/train_df_with_text.csv')
full_val = pd.read_csv('../data/val_df_with_text.csv')
full_test = pd.read_csv('../data/test_df_with_text.csv')

new_train_df, new_val_df, new_test_df = iterative_filter_dfs(full_train, full_val, full_test)

new_train_df.to_csv('../data/train_df_filtered.csv', index=False)
new_val_df.to_csv('../data/val_df_filtered.csv', index=False)
new_test_df.to_csv('../data/test_df_filtered.csv', index=False)

print(len(new_train_df), len(new_val_df))
print(f'min user count - {min(new_train_df["user_id"].value_counts())}')
print(f'min item count - {min(new_train_df["parent_asin"].value_counts())}')