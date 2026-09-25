import kagglehub
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from pathlib import Path
import os

le = LabelEncoder()

# Resolve paths relative to this script so the file works from any working directory
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data-sets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Download dataset into predictor/data-sets/
os.environ["KAGGLE_CACHE_FOLDER"] = str(DATA_DIR)
dataset_path = kagglehub.dataset_download("haseebindata/student-performance-predictions")
print(f"Dataset downloaded to: {dataset_path}")

# Locate the CSV inside the versioned download directory
csv_files = list(Path(dataset_path).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in downloaded dataset at: {dataset_path}")
student_performance_predictions_path = csv_files[0]

df = pd.read_csv(student_performance_predictions_path)
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"All columns: {df.columns.to_list()}")


# 2. Create function to map exam scores
def assign_cbc_band(score:int)->str:
    """
    Map Exam_Score to CBC Bands

    Args:
        score (int): Students profile score.
    """
    if score >= 80:
        return "Exceeding Expectation"
    elif score >= 60:
        return "Meeting Expectation"
    elif score >= 40:
        return "Approaching Expectation"
    else:
        return "Below Expectation"

df["CBC_Band"] = df["FinalGrade"].apply(assign_cbc_band)

# Check class distribution
print("\nClass distribution:")
print(df["CBC_Band"].value_counts())



# 3. Define Feature Subsets
# This information will be logged continously as a student uses the app

target_column = "FinalGrade"

# Numerical Features
numerical_features = [
    "StudyHoursPerWeek",     # Practice time
    "Attendance (%)",        # Attendance rate
    "PreviousGrade",         # Running average of previous quiz marks
    "Online Classes Taken"   # Online/digital interaction count
]

# 3. Categorical Features
categorical_features = [
    "ParentalSupport",       # Maps to parental involvement
    "ExtracurricularActivities",
    "Gender"
]

X = df[numerical_features + categorical_features]
y = df["CBC_Band"]



# 4. Map targets to ordinal integer labels
target_mapping = {
    "Below Expectation": 0,
    "Approaching Expectation": 1,
    "Meeting Expectation": 2,
    "Exceeding Expectation": 3
}
inv_target_mapping = {v: k for k, v in target_mapping.items()}

y_encoded = y.map(target_mapping)
y_encoded = le.fit_transform(y)

label_mapping = {index: label for index, label in enumerate(le.classes_)}
print("Automatic consecutive class mapping:", label_mapping)

# Stratified split to ensure all 4 classes are evenly represented in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# test_size = 0.2: tells scikit-learn to allocate 20% of rows to the test set and 80% to the training set
# random_state = 42: standard fixed seed that makes results reproducible
# stratify=y_encoded:


# 5. Preprocessing Pipelines
# Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Impute missing categories with most frequent value, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. Instantiate the Classifier Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        eval_metric="mlogloss"
    ))
])

# Train the model
print("\nTraining XGBoost pipeline...")
model_pipeline.fit(X_train, y_train)


# 7. Evaluate on the Test Set
y_pred = model_pipeline.predict(X_test)

print("\n--- Model Evaluation Report ---")
target_names = [inv_target_mapping[i] for i in sorted(inv_target_mapping.keys())]
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. Save the trained pipeline and mapping dict to disk
artifact = {
    "pipeline": model_pipeline,
    "label_mapping": inv_target_mapping,
    'label_encoder':le,
    "expected_columns": numerical_features + categorical_features
}

output_path = SCRIPT_DIR / "cbc_predictor_bundle.pkl"
joblib.dump(artifact, output_path)
print(f"\nSaved trained model artifact to '{output_path}' successfully.")
