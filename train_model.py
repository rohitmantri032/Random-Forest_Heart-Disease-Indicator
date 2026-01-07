import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "heart_2022_no_nans.csv")

data = pd.read_csv(DATA_PATH)

X = data.drop("HadHeartAttack", axis=1)
y = data["HadHeartAttack"].map({"Yes": 1, "No": 0})

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            max_samples=0.7
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_dist = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 15],
    "classifier__min_samples_split": [5],
    "classifier__max_features": ["sqrt"],
    "classifier__min_samples_leaf": [2, 5]
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=8,
    cv=cv,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

# Default threshold (0.5)
y_pred = search.predict(X_test)

# Probabilities
y_prob = search.predict_proba(X_test)[:, 1]

# Custom threshold
y_pred_custom = (y_prob >= 0.20).astype(int)

print("=== Default Threshold (0.5) ===")
print(classification_report(y_test, y_pred))

print("=== Custom Threshold (0.20) ===")
print(classification_report(y_test, y_pred_custom))

print("ROC-AUC:", roc_auc_score(y_test,  y_prob))

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)


# Extract trained pipeline
best_pipeline = search.best_estimator_

# Get preprocessor and model
preprocessor = best_pipeline.named_steps["preprocessor"]
rf_model = best_pipeline.named_steps["classifier"]

# Feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Feature importance
importances = rf_model.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Save top features for business / app usage
importance_df.head(20).to_csv("top_feature_importance.csv", index=False)

print("✅ Feature importance saved")

# Save model
joblib.dump(search.best_estimator_, "heart_attack_rf_model.pkl")

joblib.dump({
    "model": search.best_estimator_,
    "features": X.columns.tolist(),
    "threshold": 0.20
}, "model_bundle.pkl")

print("✅ Model trained, evaluated & saved")


# cd "C:\Users\rohit\OneDrive\Desktop\Data Science\Algorithms\RF\Indicators of Heart disease\training"
# python train_model.py