import pandas as pd

df = pd.read_csv("/home/patricija/Desktop/GifModelRe/matching_captions.csv")
# Ensure the column 'caption' exists in the DataFrame before accessing it
if 'caption' in df.columns:
    print(df['caption'].value_counts().head(10))
else:
    print("Column 'caption' does not exist in the DataFrame.")

