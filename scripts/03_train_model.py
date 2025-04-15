import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 🔇 Supprime les logs TensorFlow
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 📥 Charger les données
with open("mfcc_data/mfcc_data_v2.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

X = np.array([d["mfcc"] for d in data])
y_labels = [d["imam"] for d in data]

# 🏷️ Encoder les labels
unique_labels = sorted(list(set(y_labels)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_index[label] for label in y_labels])
y = to_categorical(y, num_classes=len(unique_labels))

# 🔄 Redimensionner pour CNN
X = np.expand_dims(X, axis=-1)

# ✂️ Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🧠 Définir le modèle CNN avec 3 couches Conv1D
from keras.layers import BatchNormalization, GlobalAveragePooling1D

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1), padding='same'),

    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    GlobalAveragePooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(unique_labels), activation='softmax')
])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ⏱️ EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 🚀 Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=128,
    verbose=2,
    callbacks=[early_stop]
)

# 🎯 Évaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🎯 Précision test : {accuracy * 100:.2f}%")

# 💾 Sauvegarde du modèle
# model.save("models/model_cnn_imam_fixed.keras")
model.save("models/model_cnn_imam_v3.keras", save_format="keras")
print("✅ Modèle sauvegardé dans models/model_cnn_imam_fixed.keras")
