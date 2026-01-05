# Student Dropout Predictor ✅

A lightweight machine learning project to predict the probability of student dropout from enrollment data. This repository contains data ingestion, preprocessing, model training, and a FastAPI-based prediction endpoint for batch CSV predictions.

---

## 🚀 Quick overview

- **Goal:** Predict student dropout probability (percentage) and store prediction results.
- **Primary model:** Stacked ensemble (LightGBM, XGBoost, CatBoost, RandomForest) with a LogisticRegression meta-estimator.
- **API:** FastAPI server located in `app.py` exposes `/predict-csv` and `/get-results/{file_name}` endpoints.

---

## 📁 Project structure (key files)

- `app.py` — FastAPI app for batch CSV predictions and storage to MongoDB.
- `notebook/data/dataset.csv` — Main dataset used for training.
- `src/components/` — Core pipeline components:
  - `data_ingestion.py` — Reads dataset, creates train/test splits, writes to `artifacts/`.
  - `data_transformation.py` — Preprocessing pipeline and saving `artifacts/proprocessor.pkl`.
  - `model_trainer.py` — Model training and saving `artifacts/model.pkl`.
- `src/pipeline/predict_pipeline.py` — Loads preprocessor & model and returns dropout percentage for input DataFrame.
- `artifacts/` — Output artifacts (e.g., `data.csv`, `train.csv`, `test.csv`, `model.pkl`, `proprocessor.pkl`).
- `uploads/` and `outputs/` — CSV upload input and prediction results output.
- `requirements.txt` — Python package dependencies.

---

## ⚙️ Requirements

- Python 3.10+
- Create a virtual environment and install dependencies:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧭 Setup & Usage

### 1) Prepare data

Ensure the dataset is present at `notebook/data/dataset.csv`. The ingestion component will create `artifacts/data.csv`, `artifacts/train.csv`, and `artifacts/test.csv`.

### 2) Train the model

The simplest way to run the pipeline end-to-end is executing the data ingestion module which triggers transformation and training (see `if __name__ == "__main__"` block in `src/components/data_ingestion.py`):

```bash
python src/components/data_ingestion.py
```

This will generate `artifacts/proprocessor.pkl` and `artifacts/model.pkl`.

> Tip: You can also import and call the classes directly in Python if you want programmatic control.

### 3) Run the API server

Start the FastAPI server (install `uvicorn` if not already in `requirements.txt`):

```bash
uvicorn app:app --reload
```

The server exposes two endpoints:

- POST `/predict-csv` — Upload a CSV file (same columns as training inputs, without target). Returns success message and writes predictions to `outputs/predicted_<filename>`. Also stores records in MongoDB.
- GET `/get-results/{file_name}` — Retrieves stored prediction results for a previously processed upload.

Example: Use `curl` to POST a CSV file:

```bash
curl -X POST "http://127.0.0.1:8000/predict-csv" -F "file=@uploads/student_prediction_input_10_students.csv"
```

### 4) Programmatic prediction (Python)

```python
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

df = pd.read_csv("uploads/student_prediction_input_10_students.csv")
predictor = PredictPipeline()
probs = predictor.predict_percentage(df)
df["dropout_probability_%"] = probs
```

---

## 🔐 Environment variables

- `MONGO_URI` — MongoDB connection string used by `app.py` to store prediction results. If not set, the API will attempt to connect and may fail — set it before running the server:

```bash
# PowerShell
$env:MONGO_URI = "mongodb://localhost:27017"
```

---

## ♻️ Outputs & Artifacts

- `artifacts/model.pkl` — Trained model (stacked classifier).
- `artifacts/proprocessor.pkl` — Preprocessing pipeline saved for inference.
- `outputs/predicted_<filename>` — CSVs produced by the API with added `dropout_probability_%`, `file_name` and `created_at`.
- `logs/` — Application logs.

---

## 🛠 Development & testing

- Follow standard Python packaging and development practices.
- Add unit tests for the pipeline components (currently not included).
- If you change column names, update the numerical feature list in `src/components/data_transformation.py`.

---

## 🤝 Contributing

Contributions are welcome. Please open issues or pull requests with a clear description of changes and tests where applicable.

---

## 📄 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this software with proper attribution.

---

## 📞 Contact

**Hritviz Manral**  
📧 Email: hritvizmanral66@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/hritvizmanral 
💻 GitHub: https://github.com/Hritviz666

For questions, suggestions, or collaboration opportunities, feel free to reach out.


---

**Happy modeling!** 🎯
