# SMS Spam Detection – Mini AI Project

##  Project Overview

This project builds an end-to-end AI solution to classify SMS messages as **Spam** or **Ham (Not Spam)**.  
It covers data preprocessing, machine learning model training, evaluation, and deployment using FastAPI.

The system accepts raw SMS text and returns predictions through a REST API.

---

##  Objective

- Understand data preprocessing for text data
- Build and evaluate multiple ML models
- Explain AI logic and feature impact
- Deploy model using FastAPI
- Follow professional Git + documentation practices

---

##  Project Structure

sms-spam-detection/
│
├── app.py # FastAPI application
├── train.py # Model training script
├── spam_model.pkl # Saved Logistic Regression model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt
└── README.md


---

##  Models Used

- Logistic Regression (final deployed model)
- Random Forest (for comparison)

Logistic Regression was selected because it performs efficiently on high-dimensional sparse TF-IDF features and gives consistent results for text classification.

---

##  Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert SMS text into numerical vectors.

Words such as:

- free
- win
- offer
- urgent

receive higher weights, strongly influencing spam prediction.

---

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

These metrics ensure balanced evaluation, especially important for spam detection where class imbalance exists.

---

##  Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/shabab369/spam_detection
cd sms-spam-detection

    2. Install Dependencies
 pip install fastapi uvicorn joblib scikit-learn pandas
    3. Train Model
 python train.py
    4. Run API
 uvicorn app:app --reload
    5. Test API in browser
 http://127.0.0.1:8000/docs
    

sample input:
  {
  "text": "Congratulations! You won a free prize"
}

example for this response:
{
  "input_text": "Congratulations! You won a free prize",
  "prediction": "spam"
}

