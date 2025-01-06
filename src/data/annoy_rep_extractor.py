import torch
from torchvision import transforms
from PIL import Image
from annoy import AnnoyIndex
import json
import os

from src.models.models import FilmGenreClassifier18
from src.param.param import DATA_PATH, NUM_CLASSES, WEIGHTS_PATH, SHORT_DATA_PATH



# Model config
model_weights = WEIGHTS_PATH['RESNET18']

model = FilmGenreClassifier18(NUM_CLASSES)
model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
model.eval()


# Initialiser l'index Annoy
annoy_index = AnnoyIndex(NUM_CLASSES, 'angular')
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convertit les images en tenseurs PyTorch et les normalise entre 0 et 1
    ])

# Charger les métadonnées
metadata = []  # Pour stocker les informations des films
current_id = 0


for genre_folder in os.listdir(DATA_PATH):
    genre_path = os.path.join(DATA_PATH, genre_folder)
    if os.path.isdir(genre_path):
        for image_file in os.listdir(genre_path):

            image_path = os.path.join(genre_path, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            # Extraire le vecteur
            with torch.no_grad():
                vector = model(image_tensor).squeeze().numpy()

            # Ajouter le vecteur à Annoy
            annoy_index.add_item(current_id, vector)

            metadata.append({
                        "id": current_id, # Id unique
                        "title": image_file.split('.')[0],  # Nom de l'image comme titre
                        "poster_url": os.path.join(SHORT_DATA_PATH, genre_folder,image_file),  # Chemin dynamique ou URL du poster
                        "genre": genre_folder              # Genre correspondant au dossier
                    })

                    # Incrémenter l'ID

            current_id += 1
            
# Construire l'index Annoy
annoy_index.build(75)  # Le nombre d'arbres peut être ajusté (ex. : 10 ou 20)
annoy_index.save("data/annoy/movie_posters_resnet18.ann")

# Sauvegarder les métadonnées en JSON
output_metadata_path = "data/annoy/metadata18.json"
with open(output_metadata_path, 'w') as json_file:
    json.dump(metadata, json_file, indent=4)

print(f"Fichier JSON des métadonnées généré : {output_metadata_path}")
