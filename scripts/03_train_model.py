import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from collections import Counter

# 📁 Chemin des données
JSONL_PATH = "mfcc_data/mfcc_data_v2.jsonl"

# 📥 Chargement et filtrage des données
X, y_labels = [], []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        mfcc = data["mfcc"]

        # ⚠️ On ignore les entrées incorrectes
        if len(mfcc) != 26:
            continue

        # Facultatif : filtrer les extraits courts uniquement
        if "augmentation" in data and data["augmentation"] == "original":
            continue

        X.append(mfcc)
        y_labels.append(data["imam"])

print(f"✅ Données chargées : {len(X)} échantillons valides.")

# 🎯 Vérification de la distribution
counter = Counter(y_labels)
print("\n📊 Répartition des imams :")
for imam, count in sorted(counter.items()):
    print(f"  ➤ {imam:<25} : {count}")

# 🏷️ Encodage des labels
unique_labels = sorted(list(set(y_labels)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_index[label] for label in y_labels])
y = to_categorical(y, num_classes=len(unique_labels))

# 🔄 Mise en forme pour CNN
X = np.array(X)
X = np.expand_dims(X, axis=-1)

# ✂️ Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧠 Définition du modèle CNN (amélioré)
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ⏱️ EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# 🚀 Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

# 🎯 Évaluation finale
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n🎯 Précision test : {accuracy * 100:.2f}%")

# 💾 Sauvegarde
model.save("models/model_cnn_imam_v4.keras", save_format="keras")
print("✅ Modèle sauvegardé : model_cnn_imam_v4.keras")

# 📊 Courbes accuracy / loss
os.makedirs("output", exist_ok=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("output/accuracy_loss.png")
print("🖼️ Courbes sauvegardées dans output/accuracy_loss.png")
