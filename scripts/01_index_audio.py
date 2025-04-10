import os
import librosa
import numpy as np
import json
import gc
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from opensearchpy import OpenSearch

# 🔹 Connexion à OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "MonSuperMotDePasse123!"),
    use_ssl=True,
    verify_certs=True,
    ca_certs="/etc/opensearch/certs/ca.crt",
    ssl_show_warn=False,
)

# 🔹 Vérification de la connexion
print(client.info())

# 🔹 Chemins et configurations
index_name = "quran_audio"
base_path = "/mnt/e/DIN/data"
mfcc_output_file = "/mfcc_data/mfcc_data.jsonl"
BATCH_SIZE = 50  # Réduction pour éviter les crashs

# 🔹 Vérifier si l’index existe
if not client.indices.exists(index=index_name):
    print(f"❌ Erreur : L'index {index_name} n'existe pas. Crée-le avant d'indexer.")
    exit()

# 🔹 Récupération des fichiers déjà indexés
indexed_files = set()
query = {"size": 10000, "_source": ["file_path"]}
response = client.search(index=index_name, body=query)
if "hits" in response and "hits" in response["hits"]:
    for doc in response["hits"]["hits"]:
        indexed_files.add(doc["_source"]["file_path"])

print(f"👂 {len(indexed_files)} fichiers déjà indexés.")

# 🔹 Fonction pour traiter un fichier audio
def process_audio(file_path, imam, sourate):
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convertir en Mo
        print(f"📂 Traitement en cours : {file_path} ({file_size:.2f} Mo)")

        # Chargement du fichier audio
        y, sr = librosa.load(file_path, sr=None)

        # Extraction des MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        # Stocker dans un fichier JSONL
        with open(mfcc_output_file, "a") as f:
            json.dump({"imam": imam, "sourate": sourate, "file_path": file_path, "mfcc": mfcc_mean}, f)
            f.write("\n")

        # 🔹 Indexation dans OpenSearch
        document = {
            "imam": imam,
            "sourate": sourate,
            "file_path": file_path
        }
        client.index(index=index_name, body=document)

        print(f"✅ Indexé : {file_path}")
        return True
    except Exception as e:
        print(f"⚠️ Erreur lors du traitement de {file_path} : {e}")
        return False

# 🔹 Liste des fichiers à indexer
files_to_process = []
for imam in os.listdir(base_path):
    imam_path = os.path.join(base_path, imam)
    if os.path.isdir(imam_path):
        for filename in os.listdir(imam_path):
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                file_path = os.path.join(imam_path, filename)
                if file_path not in indexed_files:
                    try:
                        sourate = int(filename.split("-")[0])
                        files_to_process.append((file_path, imam, sourate))
                    except ValueError:
                        print(f"⚠️ Nom de fichier invalide ignoré : {filename}")

print(f"🚀 {len(files_to_process)} fichiers à indexer.")

# 🔹 Traitement par batch
for i in range(0, len(files_to_process), BATCH_SIZE):
    batch = files_to_process[i:i + BATCH_SIZE]
    print(f"💧 Traitement du batch {i//BATCH_SIZE + 1}/{len(files_to_process)//BATCH_SIZE + 1}...")

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_audio, *args): args for args in batch}
        for future in concurrent.futures.as_completed(future_to_file, timeout=120):  # Timeout de 120 sec max
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de {future_to_file[future]} : {e}")

    # 🔹 Libération de la mémoire après chaque batch
    gc.collect()

print("🚀 Indexation terminée !")
