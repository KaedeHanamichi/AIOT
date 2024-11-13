import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Step 1: Generate 300 random values for X(i) in the range [0, 1000]
np.random.seed(42)  # For reproducibility
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)

# Step 2: Define Y(i) based on the condition 500 < X(i) < 800
Y = np.where((X > 500) & (X < 800), 1, 0).reshape(-1)

# Step 3: Train Logistic Regression and SVM models
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
y1 = log_reg.predict(X)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, Y_train)
y2 = svm.predict(X)

# Step 4: Plotting the results

# Plot 1: X vs Y and X vs Y1 (Logistic Regression)
plt.figure(figsize=(12, 6))

# Plot X vs Y (True labels)
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='True Labels (Y)', alpha=0.6)
plt.plot(X, y1, color='red', label='Logistic Regression Prediction (Y1)', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression: X vs Y and X vs Y1')
plt.legend()

# Plot 2: X vs Y and X vs Y2 (SVM)
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', label='True Labels (Y)', alpha=0.6)
plt.plot(X, y2, color='green', label='SVM Prediction (Y2)', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM: X vs Y and X vs Y2')
plt.legend()

plt.tight_layout()
plt.show()
