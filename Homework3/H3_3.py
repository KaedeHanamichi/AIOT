import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons  # 用於生成月亮形狀的數據
from sklearn.model_selection import train_test_split

# 生成月亮形狀的數據
X, y = make_moons(n_samples=300, noise=0.1)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用 SVM 進行分類
svm = SVC(kernel='rbf')  # 使用RBF核函數，這對於非線性數據有很好的分類效果
svm.fit(X_train, y_train)

# 預測
y_pred = svm.predict(X_test)

# Streamlit UI
st.title("SVM with Moon Dataset and 2D Visualization")
st.write("這個應用展示了 SVM 在月亮形狀數據集上的分類效果，並在 2D 平面中可視化分類結果。")

# 顯示原始數據與預測結果的 2D 圖
fig, ax = plt.subplots(figsize=(10, 7))

# 繪製 SVM 預測結果的 2D 散點圖
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired)

# 設定 2D 圖形的標題和坐標軸標籤
ax.set_title("2D Visualization of SVM Predictions")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# 在 Streamlit 中顯示圖像
st.pyplot(fig)

# 顯示 SVM 模型的準確度
accuracy = svm.score(X_test, y_test)
st.write(f"SVM 模型準確度: {accuracy:.2f}")
