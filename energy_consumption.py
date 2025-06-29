import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Building a model for test_energy_data.csv. Link to original https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression?resource=download

df = pd.read_csv('test_energy_data.csv')
print(df)

x = df[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']] #NOTE Seems like it can't handle string values
y = df['Energy Consumption']

# Creating model
model = LinearRegression()
model.fit(x, y)

# Check model accuracy using test cases
test_case = pd.DataFrame([{ # Can't handle string values
    'Square Footage': 36720, # index = 4
    'Number of Occupants': 58,
    'Appliances Used': 47,
    'Average Temperature': 17.88,
}])

predicted_consumption = model.predict(test_case)[0]
print('\nActual consumption (@index = 4) = 4820; whereas predicted value =', round(predicted_consumption, 2))
print('Not a very accurate prediction, hence not a very accurate model')
#---------------------------------------------------------------------------------------------------------------------

# Use the larger dataset 'train_energy_data.csv', doing the exact same thing
df = pd.read_csv('train_energy_data.csv')
print(df)

x = df[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']] #NOTE Seems like it can't handle string values
y = df['Energy Consumption']

# Creating model
model = LinearRegression()
model.fit(x, y)

# Check model accuracy using test cases
test_case = pd.DataFrame([{ # Can't handle string values
    'Square Footage': 15813, # index = 999
    'Number of Occupants': 57,
    'Appliances Used': 11,
    'Average Temperature': 31.4,
}])

predicted_consumption = model.predict(test_case)[0]
print('\nActual consumption (@index = 999) = 3423.63; whereas predicted value =', round(predicted_consumption, 2)) # Index 999 is dead on accurate, but it seems most predictions are significantly off
print('-----------------------------------------------------------------------------------------------------------------------------------')

# Automate the above to find the degree of deviation from actual values
df_test_case_1 = df[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']]
length_df = df_test_case_1.shape[0]
list_predConsumption = []

for i in range(length_df):
    list_predConsumption.append(round(model.predict(df_test_case_1)[i], 2)) # Returns list of predicted values

# Convert list to df
df_prediction = pd.DataFrame({'Predicted Energy Consumption': list_predConsumption})
df_Energy_Cons = df['Energy Consumption']
# print('\n', df_prediction, df_Energy_Cons)

# Find percentage difference of predicted value from the actual value
df_joined = df_prediction.join(df_Energy_Cons) #NOTE Having the join applied the other way around doesn't work though. Interesting.
print('\n', df_joined)

def find_perc_diff(df): # Function for finding the percentage difference
    df_percentage = round(((df['Energy Consumption'] - df['Predicted Energy Consumption']) / df['Energy Consumption']) * 100.0, 2)
    return df_percentage

df_joined['Percentage Difference'] = df_joined.apply(find_perc_diff, axis= 1) # Creating a new column 'Percentage Difference'
print('\n', df_joined) # Degree of deviation final result. As it stands, the model is not very accurate.
# avg_percentDiff = df_joined['Percentage Difference'].apply(np.average) # Find the overall average
# print(avg_percentDiff)
#----------------------------------------------------------------------------------------------------------------------------------------------------

# Another way to potentially make the model more accurate is to include 'Building Type' and 'Day of Week', assign numbers to each value.
def convert_Building_Type(df):
    for i in range(length_df):
        if df.loc[i, 'Building Type'] == 'Residential': # Self explanetory
            df.loc[i, 'Building Type'] = 1
        elif df.loc[i, 'Building Type'] == 'Commercial':
            df.loc[i, 'Building Type'] = 2
        elif df.loc[i, 'Building Type'] == 'Industrial':
            df.loc[i, 'Building Type'] = 3
    return df

def convert_DayofWeek(df):
    for i in range(length_df):
        if df.loc[i, 'Day of Week'] == 'Weekday': # Self explanetory
            df.loc[i, 'Day of Week'] = 1
        elif df.loc[i, 'Day of Week'] == 'Weekend':
            df.loc[i, 'Day of Week'] = 2
    return df

df_new = convert_Building_Type(df)
df_new = convert_DayofWeek(df_new)
print(df_new)

# Apply the model including the two new parameters
# Copy paste the model again, from above
x = df_new[['Building Type', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']]
y = df_new['Energy Consumption']

# Creating model
model = LinearRegression()
model.fit(x, y)

# Check model accuracy using test cases
test_case = pd.DataFrame([{ # Can't handle string values
    'Building Type': 1,                      
    'Square Footage': 7063, # index = 0
    'Number of Occupants': 76,
    'Appliances Used': 10,
    'Average Temperature': 29.84,
    'Day of Week': 1
}])

predicted_consumption = model.predict(test_case)[0]
print('\nActual consumption (@index = 0) = 2713.96; whereas predicted value =', round(predicted_consumption, 2)) 

# Test this model using the previously defined test(We should use functions rather)

df_test_case_1 = df_new[['Building Type', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']]
length_df = df_test_case_1.shape[0]
list_predConsumption = []

for i in range(length_df):
    list_predConsumption.append(round(model.predict(df_test_case_1)[i], 2)) # Returns list of predicted values

# Convert list to df
df_prediction = pd.DataFrame({'Predicted Energy Consumption': list_predConsumption}) # Predicted Energy Consumption 
df_Energy_Cons = df['Energy Consumption'] # Original Energy Consumption column
# print('\n', df_prediction, df_Energy_Cons)

# Find percentage difference of predicted value from the actual value
df_joined = df_prediction.join(df_Energy_Cons) 
print('\n', df_joined)

def find_perc_diff(df): # Function for finding the percentage difference
    df_percentage = round(((df['Energy Consumption'] - df['Predicted Energy Consumption']) / df['Energy Consumption']) * 100.0, 2)
    return df_percentage

df_joined['Percentage Difference'] = df_joined.apply(find_perc_diff, axis= 1) # Creating a new column 'Percentage Difference'
print('\n', df_joined) # Degree of deviation final result. The model is now very accurate.
