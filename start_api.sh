#!/bin/bash
# Vérifier que le fichier "download_done.txt" existe
echo "Attente que le téléchargement soit terminé..."
while [ ! -f /app/data/DONE.txt ]; do
  echo "Téléchargement non terminé. Attente..."
  sleep 15
done

# Supprimer le fichier RIN.txt
echo "Suppression du fichier DONE.txt..."
rm /app/data/DONE.txt

# Lancer l'API
echo "Téléchargement terminé, démarrage de l'API..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
