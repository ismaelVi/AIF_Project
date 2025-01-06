# Étape 1 : Utiliser une image Python légère
FROM python:3.9-slim

# Étape 2 : Installer rclone et d'autres outils nécessaires
RUN apt-get update && apt-get install -y \
    bash \
    unzip

RUN pip install gdown

# Étape 3 : Définir le répertoire de travail
WORKDIR /app

# Étape 4 : Copier le script de téléchargement dans le conteneur
COPY download_drive.sh /app/download_drive.sh

# Étape 5 : Rendre le script exécutable
RUN chmod +x /app/download_drive.sh

# Point de montage pour les données
VOLUME /app/data

# Étape 6 : Exécuter le script au démarrage
CMD ["/bin/bash", "/app/download_drive.sh"]