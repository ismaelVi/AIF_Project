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
COPY src/api/app.py /app/src/api/app.py
COPY src/models/models.py /app/src/models/models.py
COPY src/param/param.py /app/src/param/param.py

# Ajouter /app au PYTHONPATH
ENV PYTHONPATH=/app

# Étape 7 : Exposer le port de l'application
EXPOSE 8000

# Copier le script de démarrage
COPY start_api.sh /app/start_api.sh

# Étape 4 : Rendre le script exécutable
RUN chmod +x /app/start_api.sh

# Étape 5 : Exécuter le script de démarrage
CMD ["/bin/bash", "/app/start_api.sh"]
