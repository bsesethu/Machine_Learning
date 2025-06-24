from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# x = [[50, 70, 80], [55, 45, 70], [65, 52, 80], [95, 67, 55]]
# y = [65, 40, 23, 65] # 4 values here correspond to the 4 individuals in x


# model = LinearRegression()
# model.fit(x, y)
# print("Coefficients:", model.coef_)       # Slope for each input feature
# print("Intercept:", model.intercept_)

# prediction = model.predict(x)
# print('Predictions', prediction)


# Same thing as above
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
      [0, 5, 6], 
      [10, 15, 9],
      [20, 16, 12]]  
# Target values
y = [65, 40, 98, 92, 70, 85, 90, 100, 75, 88, 60, 15, 60] 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Output the coefficients and intercept
print("Coefficients:", model.coef_)       
print("Intercept:", model.intercept_)
# Make predictions using the model
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual values:", y_test)