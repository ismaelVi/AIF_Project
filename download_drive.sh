#!/bin/bash

# Définition des variables
DATA_DIR="/app"
ZIP_FILE="$DATA_DIR/fichier.zip"
MARKER_FILE="$DATA_DIR/data/RINE.txt"
FILE_ID="1JYfJaqkkHz6eD5MiJQF2dtj77XnuCGKx"

# Vérifier et créer le répertoire si nécessaire
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR" || { echo "Erreur : Impossible de créer $DATA_DIR"; exit 1; }
fi

# Téléchargement du fichier ZIP depuis Google Drive
echo "Téléchargement du fichier .zip depuis Google Drive..."
gdown --fuzzy "https://drive.google.com/uc?id=$FILE_ID" -O "$ZIP_FILE"

# Vérifier si le téléchargement a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors du téléchargement."
    exit 1
else
    echo "Téléchargement réussi."
fi

# Décompression du fichier ZIP
echo "Décompression du fichier .zip..."
unzip -o -q "$ZIP_FILE" -d "$DATA_DIR"

# Vérifier si la décompression a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de la décompression."
    exit 1
else
    echo "Décompression réussie. Les données sont disponibles dans $DATA_DIR."
fi

# Créer un fichier indicateur pour signaler la fin du processus
touch "$MARKER_FILE" || { echo "Erreur : Impossible de créer le fichier indicateur."; exit 1; }

echo "Processus terminé avec succès !"
