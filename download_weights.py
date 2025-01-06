import os
import gdown
import subprocess

# URL du dossier Google Drive ou ID du fichier (remplacer par votre URL)
MODEL_URL = "https://drive.google.com/file/d/1SiMTxuk-PY6m3B4sAT1ieaK8waaelqTE/view?usp=sharing"
OUTPUT_DIR = "./src/saved_models"
OUTPUT_FILE = "film_genre_classifier.pth"

# Créer le répertoire de destination si nécessaire
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Installer gdown (si nécessaire)
try:
    subprocess.run(['pip', 'install', 'gdown'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Erreur lors de l'installation de gdown: {e}")
    exit(1)

# Télécharger le fichier
print(f"Téléchargement des poids du modèle depuis Google Drive vers {OUTPUT_DIR}/{OUTPUT_FILE}...")
gdown.download(MODEL_URL, os.path.join(OUTPUT_DIR, OUTPUT_FILE), quiet=False)

# Vérification si le fichier a bien été téléchargé
if os.path.isfile(os.path.join(OUTPUT_DIR, OUTPUT_FILE)):
    print(f"Téléchargement terminé : {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
else:
    print("Échec du téléchargement.")
    exit(1)
