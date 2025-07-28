import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel

# Load trained model
rf_model = joblib.load("trained_model1.pkl")

# Load CodeBERT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_bert = AutoModel.from_pretrained("microsoft/codebert-base")

class CodeRequest(BaseModel):
    code: str

# Function to extract features
def extract_features(code):
    if not isinstance(code, str) or not code.strip():  # Skip empty code
        return None

    # Convert code to CodeBERT embeddings
    tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

    # Ensure feature vector has 771 dimensions (Padding if necessary)
    expected_features = 771
    embedding = np.pad(embedding, (0, expected_features - embedding.shape[0]), mode='constant')

    return embedding.tolist()

app = FastAPI()

@app.post("/predict/")
def predict_unsafe_code(request: CodeRequest):
    feature_vector = extract_features(request.code)
    if feature_vector is None:
        return {"error": "Invalid code input"}
    
    feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    # Debugging: Print expected vs actual features
    print(f"Expected features: {rf_model.n_features_in_}, Input features: {feature_vector.shape[1]}")

    # Make prediction
    prediction = rf_model.predict(feature_vector)[0]
    return {"result": "unsafe" if prediction == 1 else "safe"}
