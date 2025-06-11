import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("image", exist_ok=True)



# 데이터 로드
df_red = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
df_white = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# 이진 라벨 생성 (quality >= 6 → good(1), else bad(0))
df_red['label'] = df_red['quality'].apply(lambda x: 1 if x >= 6 else 0)
df_white['label'] = df_white['quality'].apply(lambda x: 1 if x >= 6 else 0)

# 특성과 라벨 분리
X_red = df_red.drop(['quality', 'label'], axis=1)
y_red = df_red['label']
X_white = df_white.drop(['quality', 'label'], axis=1)
y_white = df_white['label']

# 표준화
scaler = StandardScaler()
X_red_scaled = scaler.fit_transform(X_red)
X_white_scaled = scaler.transform(X_white)  # red 기준으로 transform만

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X_red_scaled, y_red, test_size=0.2, random_state=42)

# 모델 학습
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


# 저장
joblib.dump(clf, "models/random_forest_model.pkl")

# 예측
y_pred_red = clf.predict(X_test)
y_pred_white = clf.predict(X_white_scaled)

# 평가 리포트
report_red = classification_report(y_test, y_pred_red, output_dict=True)
report_white = classification_report(y_white, y_pred_white, output_dict=True)

# 혼동 행렬
cm_red = confusion_matrix(y_test, y_pred_red)
cm_white = confusion_matrix(y_white, y_pred_white)

# 시각화 함수
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Bad', 'Pred: Good'],
                yticklabels=['True: Bad', 'True: Good'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return plt

plt1 = plot_confusion_matrix(cm_red, "Random Forest Confusion Matrix (Red Test)")
plt2 = plot_confusion_matrix(cm_white, "Random Forest Confusion Matrix (White Generalization)")
plt1.savefig("image/rf_confusion_red.png")
plt2.savefig("image/rf_confusion_white.png")
plt1.close()
plt2.close()
