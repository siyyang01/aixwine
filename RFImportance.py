# plot_feature_importance.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 디렉토리 생성 (image)
os.makedirs("image", exist_ok=True)

# 모델 로드
model = joblib.load("models/random_forest_model.pkl")

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

# 중요도 추출
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# 시각화
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_d')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig("image/rf_feature_importance.png")
plt.show()
