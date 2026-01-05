from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil
from pymongo import MongoClient
from datetime import datetime

from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

db = client["student_db"]
collection = db["dropout_predictions"]


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

 
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   
    df = pd.read_csv(input_path)

   
    predictor = PredictPipeline()
    probabilities = predictor.predict_percentage(df)

    
    df["dropout_probability_%"] = probabilities

    
    df["file_name"] = file.filename
    df["created_at"] = datetime.utcnow()

    
    output_path = os.path.join(OUTPUT_DIR, f"predicted_{file.filename}")
    df.to_csv(output_path, index=False)

    
    collection.insert_many(df.to_dict(orient="records"))

    return JSONResponse({
        "message": "Prediction successful",
        "file_name": file.filename,
        "students_stored": len(df)
    })


@app.get("/get-results/{file_name}")
async def get_results(file_name: str):

    data = list(collection.find(
        {"file_name": file_name},
        {"_id": 0}
    ))

    return {
        "file_name": file_name,
        "total_students": len(data),
        "students": data
    }
