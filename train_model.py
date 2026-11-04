from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "aqi_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df = pd.read_csv(DATA_PATH).dropna()
X = df[FEATURES]
y_aqi = df["AQI"]
y_level = df["Pollution_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_aqi, test_size=0.2, random_state=42
)
regressor = RandomForestRegressor(n_estimators=200, random_state=42)
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
print("AQI regression results")
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2 score:", round(r2_score(y_test, pred), 3))
joblib.dump(regressor, MODEL_DIR / "aqi_regressor.joblib")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_level, test_size=0.2, random_state=42, stratify=y_level
)
classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
classifier.fit(X_train_c, y_train_c)
class_pred = classifier.predict(X_test_c)
print("\\nPollution classification results")
print("Accuracy:", round(accuracy_score(y_test_c, class_pred), 3))
print(classification_report(y_test_c, class_pred, zero_division=0))
joblib.dump(classifier, MODEL_DIR / "aqi_classifier.joblib")
print("Models saved in:", MODEL_DIR)
