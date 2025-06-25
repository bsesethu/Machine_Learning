from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initial analysis, there's a name for it though
df = pd.read_csv('housing.csv')
print(df.head())
print(df.shape)
print(df.info()) # prints columns and datatypes

fig, axs = plt.subplots(3,3, figsize = (10,5))
plt1 = sns.boxplot(df['longitude'], ax = axs[0,0])
plt2 = sns.boxplot(df['latitude'], ax = axs[0,1])
plt3 = sns.boxplot(df['housing_median_age'], ax = axs[0,2])
plt1 = sns.boxplot(df['total_rooms'], ax = axs[1,0])
plt2 = sns.boxplot(df['population'], ax = axs[1,1])
plt3 = sns.boxplot(df['households'], ax = axs[1,2])
plt1 = sns.boxplot(df['median_income'], ax = axs[2,0])
plt2 = sns.boxplot(df['median_house_value'], ax = axs[1,1])
plt3 = sns.boxplot(df['ocean_proximity'], ax = axs[2,2])

plt.tight_layout()
plt.show()

plt.boxplot(df.median_house_value) # Didn't know we could do this, .column_name like that. It works
Q1 = df.median_house_value.quantile(0.25)
Q3 = df.median_house_value.quantile(0.75)
IQR = Q3 - Q1
housing = df[(df.median_house_value >= Q1 - 1.5*IQR) & (df.median_house_value <= Q3 + 1.5*IQR)]
plt.show()

# Find max, min, avg
print(df['median_house_value'].max())
print(df['median_house_value'].min())
print(df['median_house_value'].mean()) # Use mean instead of avg

#NOTE This process is called Exploratory data analysis. 
# sns.pairplot(df) # It's originally 'housing' instead of df. It also works. But I don't understand what either of them means
# plt.show()

# Heat Map
df = df.drop('ocean_proximity', axis= 1) # Function heatmap doesn't accept string values
df.describe()
plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show() #NOTE [Important] Result is a heat map. The more positive the number, the stronger the relationship between the two.

# Predict Housing price
df = df.dropna() # Has to have the df = , otherwise it doesn't save the new df
x = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population' ,'households' ,'median_income']]
y = df['median_house_value']

model = LinearRegression()
model.fit(x, y)

predictionA = model.predict(x)[2] # Interesting way of doing it. And simple. Just use x.
print(round(predictionA, 2))