import pandas as pd

train_df= pd.read_csv("train_df_filtered")
val_df= pd.read_csv("val_df_filtered")
test_df= pd.read_csv("test_df_filtered")

min_rating = train_df['rating'].min()
max_rating = train_df['rating'].max()
denom = max_rating - min_rating

train_df = train_df.apply(lambda row: (row['rating'] - min_rating)/denom, axis = 1 )
val_df = val_df.apply(lambda row: (row['rating'] - min_rating)/denom, axis = 1 )
test_df = test_df.apply(lambda row: (row['rating'] - min_rating)/denom, axis = 1 )

train_df.to_csv('train_df_filtered.csv', index=False)
val_df.to_csv('val_df_filtered.csv', index=False)
test_df.to_csv('test_df_filtered.csv', index=False)