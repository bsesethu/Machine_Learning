import pandas as pd

# Creating a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'], 
        'Score': [85, 93, 78, 90]}
df = pd.DataFrame(data)

# Randomizing rows
randomized_df = df.sample(frac=1).reset_index(drop=True)
print(randomized_df)