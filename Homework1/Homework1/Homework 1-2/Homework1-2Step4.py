import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import optuna

# Step 1: Data Collection
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_housing_df = pd.read_csv(url)

# Step 2: Data Preprocessing
X = boston_housing_df.drop('medv', axis=1)
y = boston_housing_df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build Model using Lasso
def lasso_objective(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e2, log=True)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Optimize using Optuna
study = optuna.create_study(direction="minimize")
study.optimize(lasso_objective, n_trials=100)
best_alpha = study.best_params['alpha']

# Train the final model
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.write("Lasso Regression Results:")
st.write(f"Best Alpha: {best_alpha:.5f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot training and testing curves
train_errors, test_errors = [], []
alpha_values = np.logspace(-5, 2, 100)

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(alpha_values, train_errors, label='Train MSE', color='blue')
plt.plot(alpha_values, test_errors, label='Test MSE', color='orange')
plt.xscale('log')
plt.title('Training and Test MSE vs Alpha')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.legend()
st.pyplot(plt)
