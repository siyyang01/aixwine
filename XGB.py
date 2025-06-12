import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# 디렉토리 생성
os.makedirs("image", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 데이터 로드 및 전처리
df_red = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
df_white = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# 라벨링
df_red['label'] = df_red['quality'].apply(lambda x: 1 if x >= 6 else 0)
df_white['label'] = df_white['quality'].apply(lambda x: 1 if x >= 6 else 0)

X_red = df_red.drop(['quality', 'label'], axis=1)
y_red = df_red['label']
X_white = df_white.drop(['quality', 'label'], axis=1)
y_white = df_white['label']

# 스케일링 (RandomForest와 동일 기준 사용)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_red_scaled = scaler.fit_transform(X_red)
X_white_scaled = scaler.transform(X_white)

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_red_scaled, y_red, test_size=0.2, random_state=42)

# XGBoost 모델 학습
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# 모델 저장
joblib.dump(xgb, "models/xgboost_model.pkl")

# 예측
y_pred_red = xgb.predict(X_test)
y_pred_white = xgb.predict(X_white_scaled)

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

# Confusion Matrix 저장
plot_confusion_matrix(cm_red, "XGBoost Confusion Matrix (Red Test)", "image/xgb_confusion_red.png")
plot_confusion_matrix(cm_white, "XGBoost Confusion Matrix (White Generalization)", "image/xgb_confusion_white.png")

# Feature Importance 시각화
feature_names = X_red.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Oranges_r')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig("image/xgb_feature_importance.png")
plt.close()
