import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Step 1: Data Collection
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_housing_df = pd.read_csv(url)

# Display the data in Streamlit
st.write("Boston Housing Data:")
st.write(boston_housing_df)

# Step 2: Data Preprocessing
# Separating features and target variable
X = boston_housing_df.drop('medv', axis=1)  # Features
y = boston_housing_df['medv']                # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the datasets
st.write(f'Train set shape: {X_train.shape}, {y_train.shape}')
st.write(f'Test set shape: {X_test.shape}, {y_test.shape}')

# Step 3: Build Model using Lasso
lasso = Lasso(alpha=0.1)  # You can adjust alpha as needed
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results in Streamlit
st.write("Lasso Regression Results:")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line for perfect prediction
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values (medv)')
plt.ylabel('Predicted Values (medv)')
st.pyplot(plt)
