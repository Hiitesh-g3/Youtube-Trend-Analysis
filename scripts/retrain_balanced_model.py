# 1. Import Libraries
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Data
data = pd.read_csv('../data/processed_data.csv')
print(f"✅ Data Loaded: {data.shape}")

# 3. Feature Engineering
data['title_length'] = data['title'].apply(lambda x: len(str(x)))
data['num_tags'] = data['tags'].apply(lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
data['upload_hour'] = pd.to_datetime(data['publish_time']).dt.hour

# New Feature: Video Length (optional, simulate if missing)
if 'video_length' not in data.columns:
    np.random.seed(42)
    data['video_length'] = np.random.uniform(1, 20, size=len(data))

print("✅ Features created: title_length, num_tags, upload_hour, video_length")

# 4. Target Variable
if 'is_trending' not in data.columns:
    threshold = data['views'].median()
    data['is_trending'] = data['views'].apply(lambda x: 1 if x > threshold else 0)
    print("✅ Created 'is_trending' based on views median.")

# 5. Define Features and Target
features = ['title_length', 'num_tags', 'upload_hour', 'video_length']
X = data[features]
y = data['is_trending']

# 6. Oversampling (Balance the classes)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("✅ After balancing:")
print(pd.Series(y_resampled).value_counts())

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 8. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Random Forest Report")
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

# 9. Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n📊 XGBoost Report")
print(classification_report(y_test, xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))

# 10. Save the Better Model (Choose XGBoost if working well)
os.makedirs('../models', exist_ok=True)

with open('../models/recommendation_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("\n✅ New model saved successfully at '../models/recommendation_model.pkl'!")

# 11. (Optional) ROC Curve Plot
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.show()

