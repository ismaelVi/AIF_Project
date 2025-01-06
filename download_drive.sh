#!/bin/bash

# Chemin où les données seront téléchargées
DATA_DIR="/app"

# Assurez-vous que le répertoire de destination existe
mkdir -p $DATA_DIR

# ID du fichier Google Drive (extrait de l'URL)
FILE_ID="11JzOWNLua9Tg-pS9XvRu6vcSx1-40lvo"

# Utiliser gdown pour télécharger le fichier .zip
echo "Téléchargement du fichier .zip depuis Google Drive..."
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$DATA_DIR/fichier.zip"

# Vérifiez si le téléchargement a réussi
if [ $? -eq 0 ]; then
    echo "Téléchargement réussi."
else
    echo "Erreur lors du téléchargement."
    exit 1
fi

# Décompresser le fichier .zip téléchargé
echo "Décompression du fichier .zip..."
unzip "$DATA_DIR/fichier.zip" -d "$DATA_DIR"

# Créer un fichier indicateur pour signaler la fin du processus
touch /app/data/download_done.txt

# Vérifiez si la décompression a réussi
if [ $? -eq 0 ]; then
    echo "Décompression réussie. Les données sont disponibles dans $DATA_DIR."
else
    echo "Erreur lors de la décompression."
    exit 1
fi
