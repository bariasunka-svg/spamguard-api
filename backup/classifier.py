# classifier.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = os.environ.get('MODEL_PATH', r'E:\V1_Upgraded_Thesis_Experiment\models\p3_full_fgm_ml256_s456\best')

# Load once at module import time (not per request)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set to evaluation mode (disables dropout)

def classify(text: str) -> dict:
    inputs = tokenizer(text, return_tensors='pt',
                        truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    p_spam = float(probs[1])
    return {
        'result': 'spam' if p_spam > 0.7 else 'ham' if p_spam < 0.3 else 'uncertain',
        'confidence': round(p_spam, 4),
        'tier_used': 2
    }
