#!/bin/bash

# Définir les variables
MODEL_URL="https://drive.google.com/file/d/1SiMTxuk-PY6m3B4sAT1ieaK8waaelqTE/view?usp=sharing"

# Remplacez <YOUR_FILE_ID> par l'ID du fichier Google Drive
OUTPUT_DIR="./data/saved_models"
OUTPUT_FILE="film_genre_classifier18.pth"

# Créer le répertoire si nécessaire
mkdir -p "$OUTPUT_DIR"
# Télécharger les poids avec gdown

echo "Téléchargement des poids du modèle depuis Google Drive..."
pip install gdown --quiet
gdown --fuzzy "$MODEL_URL" -O "$OUTPUT_DIR/$OUTPUT_FILE"
# Vérification

if [ -f "$OUTPUT_DIR/$OUTPUT_FILE" ]; then
    echo "Téléchargement terminé : $OUTPUT_DIR/$OUTPUT_FILE"
else
    echo "Échec du téléchargement."
    exit 1
fi
