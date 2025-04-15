import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from utils.mfcc import extract_mfcc_from_audio
from whisper.audio import load_audio, SAMPLE_RATE

def predict_imam(AUDIO_PATH: str, model_path: str, jsonl_path: str, sourate_id: int, label_path: str):
    X, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["sourate"] == sourate_id:
                X.append(data["mfcc"])
                y.append(data["imam"])
    if not X:
        return []

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

    model = load_model(model_path)
    y_audio, sr = load_audio(AUDIO_PATH), SAMPLE_RATE
    mfcc = extract_mfcc_from_audio(y=y_audio, sr=sr)
    preds = model.predict(np.array([mfcc]))[0]

    top_indices = preds.argsort()[-3:][::-1]
    return [(label_encoder.inverse_transform([idx])[0], preds[idx]) for idx in top_indices]
    # return [(label_encoder.inverse_transform([idx])[0], preds[idx]*100) for idx in top_indices]
