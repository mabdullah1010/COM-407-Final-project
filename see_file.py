import pandas as pd
file="complete_navigator_combined_output.csv"
df = pd.read_csv(file)

num_rows = len(df)
print(f"Number of rows: {num_rows}")
