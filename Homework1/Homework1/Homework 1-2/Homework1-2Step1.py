import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Fetch the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_housing_df = pd.read_csv(url)

# Display the data
st.write(boston_housing_df)

# Example of plotting
plt.figure(figsize=(10, 6))
plt.scatter(boston_housing_df['rm'], boston_housing_df['medv'], alpha=0.5)
plt.title('Average Number of Rooms vs. Median Value of Homes')
plt.xlabel('Average Number of Rooms (rm)')
plt.ylabel('Median Value of Homes (medv)')
st.pyplot(plt)
