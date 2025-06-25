from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Input features
X = [ [50, 70, 80], 
      [80, 85, 90], 
      [85, 87, 88], 
      [90, 95, 100],       
      [60, 75, 85], 
      [70, 80, 90], 
      [75, 82, 85], 
      [95, 100, 105], 
      [65, 78, 82], 
      [88, 92, 95],
      [400000, 550000, 660000], # Changing a single set of values here doesn't change the predictions much. We can just consider it an outlier.
      [10, 15, 9],
      [20, 16, 12]]  
# Target values
y = [65, 40, 98, 92, 70, 85, 90, 100, 75, 88, 60, 15, 60] 
# Split the data into training and testing sets. NOTE I don't know what this means
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # test size of 0.5 and 7 give the same result
# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Output the coefficients and intercept
print("Coefficients:", model.coef_)       
print("Intercept:", model.intercept_)
# # Make predictions using the model
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual values:", y_test)
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Working on the data.csv dataset
df = pd.read_csv('data.csv')
# print('\n', df)

x = df[['StudyHours', 'Attendance', 'ProjectsCompleted', 'InternshipMonths']] #[[]] If we're making it an array
y = df['GPA']

# Creating the model
model = LinearRegression() # The first model variable is overwritten
model.fit(x, y) # Features

# For one individual student
kate_cv = pd.DataFrame([{
    'StudyHours': 39,
    'Attendance': 90,
    'ProjectsCompleted': 3.87,
    'InternshipMonths': 2.75
}])

predicted_gpa = model.predict(kate_cv)[0]  #NOTE [0] is the 0th value in the total number of predictions
print(f"Predicted GPA for Kate: {predicted_gpa:.2f}")
print("Coefficients:", model.coef_)       
print("Intercept:", model.intercept_)

# Extrapolate all columns except GPA. I used excel to extrapolate
df3 = pd.read_csv('data3.csv', delimiter= ';')
print('\nExtrapolated DF') 
print(df3.head())

# Replace NaN GPA values with the predicted values from the model
df3_dropGPA = df3.drop(columns= 'GPA')
print('\n', df3_dropGPA.head())

# Find the predicted GPAs
predicted_GPAs = model.predict(df3_dropGPA)[87] #NOTE [87] is the 87th value in the total number of predictions. It's amazing that it works so simply
print(f'Predicted GPA for one of the extrapolated students: {predicted_GPAs:.2f}')

# Put these GPA's into one df column and merge with the extrapolated df
length_df = df3_dropGPA.shape[0]
list_GPA = []
for i in range(length_df):
      list_GPA.append(round(model.predict(df3_dropGPA)[i], 2))

# print(list_GPA)
df_GPA = pd.DataFrame({'Predicted GPA': list_GPA}) #NOTE This is how to create a df from a list, so each column is literally the column as 'key' and a list
print(df_GPA)

# Joining the dfs. Not merging, merging means combining based on a shared column. Which is the opposite in SQL
joined_df = df3_dropGPA.join(df_GPA)
print('\n')
print(joined_df)

# Save joined_df as csv
# joined_df.to_csv('predicted_GPA.csv')