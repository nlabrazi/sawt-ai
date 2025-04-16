import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from utils.mfcc import extract_mfcc_from_audio
from whisper.audio import load_audio, SAMPLE_RATE

def predict_imam(audio_path, model_path, label_path):
    try:
        y_audio, sr = load_audio(audio_path), SAMPLE_RATE
        mfcc = extract_mfcc_from_audio(y=y_audio, sr=sr)

        model = load_model(model_path)

        with open(label_path, "rb") as f:
            label_encoder = pickle.load(f)

        preds = model.predict(np.array([mfcc]))[0]
        top_indices = preds.argsort()[-3:][::-1]
        return [(label_encoder.inverse_transform([idx])[0], preds[idx]) for idx in top_indices]
    except Exception as e:
        print(f"⚠️ Erreur prédiction imam : {e}")
        return []
