import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# 生成圓形分佈的數據
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用 SVM 進行分類
svm = SVC(kernel='rbf')  # 使用RBF核函數，對於圓形分佈的數據來說，這是一個合適的選擇
svm.fit(X_train, y_train)

# 預測
y_pred = svm.predict(X_test)

# Streamlit UI
st.title("SVM with Circle Dataset and 3D Visualization")
st.write("這個應用展示了 SVM 在圓形數據集上的分類效果，並在 3D 空間中可視化分類結果。")

# 顯示原始數據與預測結果
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 為了方便視覺化，我們將所有數據點的 z 軸設置為0，這樣就可以在 3D 中顯示它們
z = np.zeros(X_test.shape[0])  # 設置 z 為零，所有資料點都在同一個平面

# 繪製 SVM 預測結果的 3D 散點圖
ax.scatter(X_test[:, 0], X_test[:, 1], z, c=y_pred, cmap=plt.cm.Paired)

# 設定 3D 圖形的標題和坐標軸標籤
ax.set_title("3D Visualization of SVM Predictions")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Decision")

# 在 Streamlit 中顯示圖像
st.pyplot(fig)

# 顯示 SVM 模型的準確度
accuracy = svm.score(X_test, y_test)
st.write(f"SVM 模型準確度: {accuracy:.2f}")
