import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data from a CSV file (assuming the file is named 'stock_data.csv')
data = pd.read_csv('INR=X.csv')

# Convert the date column to datetime format (if it's not already in datetime)
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date (if it's not already sorted)
data = data.sort_values(by='Date')

# Feature Engineering - You can add more features here if required
data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())

# Define the feature columns and target column
feature_cols = ['DateOrdinal']
target_col = 'Close'

# Split the data into features (X) and target (y)
X = data[feature_cols]
y = data[target_col]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using metrics (Root Mean Squared Error and Mean Absolute Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
