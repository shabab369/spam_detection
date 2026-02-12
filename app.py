from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI
app = FastAPI(title="SMS Spam Detection API")

# Load model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input schema
class Message(BaseModel):
    text: str

# Health check
@app.get("/")
def home():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(message: Message):

    # Transform input
    vector = vectorizer.transform([message.text])

    # Predict
    prediction = model.predict(vector)[0]

    result = "spam" if prediction == 1 else "ham"

    return {
        "input_text": message.text,
        "prediction": result
    }
