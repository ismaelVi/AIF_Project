# Étape 1 : Utiliser une image Python légère
FROM python:3.9-slim-buster


# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Mettez à jour les dépôts et installez les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Étape 5 : Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Étape 3 : Copier le code source et les fichiers nécessaires
COPY data /app/data
COPY src /app/src
COPY download_weights.sh /app/download_weights.sh


# Étape 4 : Rendre le script téléchargeable exécutable
RUN chmod +x /app/download_weights.sh


# Étape 6 : Télécharger les poids du modèle
RUN /bin/bash /app/download_weights.sh

# Ajouter /app au PYTHONPATH
ENV PYTHONPATH=/app

RUN python src/data/annoy_rep_extractor.py


# Étape 7 : Exposer le port de l'application
EXPOSE 8000

# Étape 8 : Définir la commande pour lancer l'API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
