# Étape 1 : Utiliser une image Python légère
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier le fichier requirements.txt (les dépendances pour Gradio)
COPY gradio/requirements.txt /app/requirements.txt

# Étape 4 : Installer les dépendances Python (Gradio et autres)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Étape 5 : Copier uniquement le fichier gradio_app.py dans le conteneur
COPY gradio/gradio_app.py /app/gradio_app.py

COPY data /app/data



# Étape 6 : Exposer le port de l'application Gradio
EXPOSE 7860

# Étape 8 : Lancer l'application Gradio
CMD ["python", "gradio_app.py"]

