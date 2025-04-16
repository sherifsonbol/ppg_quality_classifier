
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

features_path = "data/ppg_features.csv"
labels_path = "data/quality-hr-ann.csv"

df_features = pd.read_csv(features_path)
df_labels = pd.read_csv(labels_path)
df = pd.merge(df_features, df_labels, on="ID")
df = df.dropna()

X = df.drop(columns=["ID", "Quality", "HR"])
y = df["Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, eval_metric='logloss', random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

y_pred = clf.predict(X_test)
print("=== XGBoost Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Poor", "Good"]))

model_path = "models/ppg_quality_model_xgboost.joblib"
joblib.dump(clf, model_path)
print(f"✅ Model saved to: {model_path}")
