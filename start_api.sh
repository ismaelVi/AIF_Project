#!/bin/bash
# Vérifier que le fichier "download_done.txt" existe
echo "Attente que le téléchargement soit terminé..."
while [ ! -f /app/data/RINE.txt ]; do
  echo "Téléchargement non terminé. Attente..."
  sleep 15
done
# Lancer l'API
echo "Téléchargement terminé, démarrage de l'API..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
