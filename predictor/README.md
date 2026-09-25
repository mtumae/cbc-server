# CBC Student Competency Predictor

A lightweight machine learning classification pipeline that evaluates student learning telemetry and environmental factors to predict CBC competency performance bands (*EE, ME, AE, BE*).

---

## 1. System Overview
* **Role in Architecture:** Evaluates student profile attributes and behavioral quiz metrics (<5 ms CPU inference) to output a predicted competency band, which will be used to call the Generative AI / RAG quiz generation engine.
* **Core Libraries:** `pandas`, `scikit-learn`, `xgboost`, `joblib`.

---

## 2. Data Pipeline & Features
* **Target Variable:** Continuous final scores (`FinalGrade`) are grouped into standard KICD CBC bands:
  * $\ge 80 \rightarrow$ Exceeding Expectation (EE)
  * $60\text{--}79 \rightarrow$ Meeting Expectation (ME)
  * $40\text{--}59 \rightarrow$ Approaching Expectation (AE)
  * $< 40 \rightarrow$ Below Expectation (BE)
* **Numerical Features:** `StudyHoursPerWeek`, `Attendance (%)`, `PreviousGrade`, `Online Classes Taken`.
* **Categorical Features:** `ParentalSupport`, `ExtracurricularActivities`, `Gender`.
* **Excluded Identifiers:** `StudentID` and `Name` dropped to prevent overfitting.

---

## 3. Preprocessing & Data Leakage Prevention
* **Encapsulation via `ColumnTransformer`:**
  * **Numeric Pipeline:** `SimpleImputer(strategy="median")` followed by `StandardScaler()`.
  * **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
* **Label Encoding:** Scikit-learn's `LabelEncoder` guarantees contiguous integer target indices (`[0, 1, 2...]`), avoiding missing-class value errors in XGBoost.
* **Stratified Splitting:** 80/20 train-test split (`train_test_split`) using `stratify=y_encoded` to preserve minority-class distributions across both subsets.

---

## 4. Model Training & Export
* **Classifier:** `XGBClassifier` configured with shallow tree depth (`max_depth=5`), moderate estimators (`n_estimators=120`), cautious step size (`learning_rate=0.08`), and `eval_metric="mlogloss"`.
* **Complete Pipeline Packaging:** Preprocessing steps and the trained boosting model are wrapped into a single `Pipeline` instance.
* **Artifact Generation:** Saved via `joblib.dump()` into `cbc_predictor_bundle.pkl`, containing:
  * The fitted scikit-learn `Pipeline` object.
  * Bidirectional class label mapping dictionaries.
  * Expected input column schemas.

---

## 5. Usage & Inference
1. **Train Model:** Run `python train_model.py` to process the dataset, evaluate classification metrics (Macro F1 & Confusion Matrix), and serialize the bundle.
2. **Predict:** Load `cbc_predictor_bundle.pkl` directly inside FastAPI server startup memory:
   ```python
   artifact = joblib.load("cbc_predictor_bundle.pkl")
   pipeline = artifact["pipeline"]
   predicted_band = artifact["label_mapping"][pipeline.predict(student_payload)[0]]
