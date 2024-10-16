import pandas as pd
from sklearn.model_selection import train_test_split
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

# Example of plotting the relationship between features and target
plt.figure(figsize=(10, 6))
plt.scatter(X['rm'], y, alpha=0.5)
plt.title('Average Number of Rooms vs. Median Value of Homes')
plt.xlabel('Average Number of Rooms (rm)')
plt.ylabel('Median Value of Homes (medv)')
st.pyplot(plt)
