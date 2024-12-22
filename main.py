from src.data_preparation import load_data, preprocess_data

filepath = "data/Student Depression Dataset.csv"
df = load_data(filepath)
df_clean, scaler = preprocess_data(df)
df_clean.to_csv('clean_data.csv', index=False)

print(df.head())