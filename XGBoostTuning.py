import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import numpy as np

# 디렉토리 생성
os.makedirs("image", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 데이터 로드
df_red = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
df_white = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# 라벨링
df_red['label'] = df_red['quality'].apply(lambda x: 1 if x >= 6 else 0)
df_white['label'] = df_white['quality'].apply(lambda x: 1 if x >= 6 else 0)

X_red = df_red.drop(['quality', 'label'], axis=1)
y_red = df_red['label']
X_white = df_white.drop(['quality', 'label'], axis=1)
y_white = df_white['label']

# 스케일링
scaler = StandardScaler()
X_red_scaled = scaler.fit_transform(X_red)
X_white_scaled = scaler.transform(X_white)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_red_scaled, y_red, test_size=0.2, random_state=42)

# 하이퍼파라미터 범위 정의
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 1, 5]
}

# 모델 정의 (GPU 대신 hist 방식 사용)
xgb = XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42)

# 랜덤 서치
rs = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

rs.fit(X_train, y_train)

# 최적 모델 저장
joblib.dump(rs.best_estimator_, "models/xgboost_tuned_model.pkl")

# 예측
y_pred_red = rs.best_estimator_.predict(X_test)
y_pred_white = rs.best_estimator_.predict(X_white_scaled)

# 평가
report_red = classification_report(y_test, y_pred_red, output_dict=True)
report_white = classification_report(y_white, y_pred_white, output_dict=True)

# 혼동 행렬
cm_red = confusion_matrix(y_test, y_pred_red)
cm_white = confusion_matrix(y_white, y_pred_white)

# 시각화 함수
def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Pred: Bad', 'Pred: Good'],
                yticklabels=['True: Bad', 'True: Good'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# 혼동 행렬 저장
plot_confusion_matrix(cm_red, "Tuned XGBoost Confusion Matrix (Red Test)", "image/xgb_tuned_confusion_red.png")
plot_confusion_matrix(cm_white, "Tuned XGBoost Confusion Matrix (White Generalization)", "image/xgb_tuned_confusion_white.png")

# Feature Importance
feature_names = X_red.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rs.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=True)

palette = sns.color_palette("Oranges_r", n_colors=len(importance_df))
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette=palette)
plt.title('Tuned XGBoost Feature Importance')
plt.tight_layout()
plt.savefig("image/xgb_tuned_feature_importance.png")
plt.close()
