from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from annoy import AnnoyIndex
import json
import pandas as pd
import joblib

from pydantic import BaseModel
import numpy as np

from transformers import DistilBertTokenizer, DistilBertModel
from gensim.models import KeyedVectors

from src.models.models import FilmGenreClassifier18

from src.param.param import WEIGHTS_PATH, NUM_CLASSES, GENRES, OUTPUT_PATH,OUTPUT_PATH2, OUTPUT_FILE, GLOVE_PATH, DISTILLBERT_MODEL_NAME


### Initialisation de l'app ###
app = FastAPI()

### Chargement du modèle pour la prédiction de genre ###
model_weights = WEIGHTS_PATH['RESNET18']
model = FilmGenreClassifier18(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
model.eval()


# Transformations d'entrée à appliquer à l'image pour le modèle de prédiction de genre
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

### Chargement de l'index annoy pour la recommandation de film via poster ###
annoy_index = AnnoyIndex(NUM_CLASSES, 'angular')
annoy_index.load("data/annoy/movie_posters_resnet18.ann")  # Charger l'index Annoy préconstruit

# Chargement des métadonnées
with open("data/annoy/metadata18.json", "r") as f:
    metadata = json.load(f)

### Point d'entrée pour vérifier que l'API fonctionne ###
@app.get("/")
def read_root():
    return {"message": "API de prédiction de genre de film en cours d'exécution."}


### Endpoint pour la prédiction de genre ###
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Charger l'image et appliquer les transformations
    image = Image.open(file.file).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Ajouter une dimension pour le batch
    with torch.no_grad():
        output = model(image)  # Faire la prédiction
        _, predicted = torch.max(output, 1)  # Obtenir la classe prédite
    return {"genre": GENRES[predicted.item()]}  # Retourner le genre prédit



### Endpoint pour la recommandation de genre via poster ###
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):

    # Charger et prétraiter l'image
    image = Image.open(file.file).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Ajouter une dimension pour le batch

    with torch.no_grad():
        # Extraire les vecteurs à partir du modèle
        vector = model(image).squeeze()

    # Trouver les 5 voisins les plus proches
    indices = annoy_index.get_nns_by_vector(vector, 6, include_distances=False)
    recommendations = [metadata[idx] for idx in indices[1:]]
    return {"recommendations": [dct['poster_url'] for dct in recommendations]} #retourner les url des poster des 5 plus proches voisins


###### Recommandation basé sur la description du film ########


#Fonction pour charger l'index Annoy et les metadata
def load_annoy_index_and_metadata(embedding_type):
    if embedding_type != "bert_embeddings":
        df = pd.read_parquet(OUTPUT_PATH)
    else:
        df = pd.read_parquet(OUTPUT_PATH2)
    embedding_dim = len(df[embedding_type][0])
    annoy_index2 = AnnoyIndex(embedding_dim, 'angular')
    annoy_index2.load(f"data/{embedding_type}_movie_posters.ann")
    with open(f"data/{embedding_type}_metadata.json", "r") as f:
        metadata2 = json.load(f)
    return annoy_index2, metadata2

# Fonction pour calculer les films les plus "proches" et obtenir les films recommandés
def recommend_movies(plot, embedding_type="bow"):
    # Charger l'index Annoy et les métadonnées
    annoy_index2, metadata2 = load_annoy_index_and_metadata(embedding_type)    
    # Calculer l'embedding du plot selon le type d'embedding choisi
    plot = plot.lower()

    if embedding_type == "bow_embeddings":
        save_path = OUTPUT_FILE + "/saved_models/bow_model"
        vectorizer = joblib.load(save_path)                   # Chargement du model bow
        bow_embedding = vectorizer.transform([plot]).toarray() # embedding du nouveau plot
        vector = bow_embedding[0]

    elif embedding_type == "glove_embeddings":
        glove_model = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False, no_header=True)  # Chargement du model glove
        words = plot.split()
        word_vectors = [glove_model[word] for word in words if word in glove_model]
        if word_vectors:
            vector = np.mean(word_vectors, axis=0)                              # embedding du nouveau plot
        else:
            vector = np.zeros(glove_model.vector_size)

    elif embedding_type == "bert_embeddings":
        tokenizer = DistilBertTokenizer.from_pretrained(DISTILLBERT_MODEL_NAME)
        model = DistilBertModel.from_pretrained(DISTILLBERT_MODEL_NAME)            # Chargement du model
        inputs = tokenizer(plot, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()              # embedding du nouveau plot 


    # Obtenir les indices des 6 films les plus similaires ( les vecteurs les plus proche dans l'index annoy)
    indices = annoy_index2.get_nns_by_vector(vector, 6, include_distances=False)
    
    # Récupérer les films recommandés correspondant
    recommendations = [metadata2[idx] for idx in indices]
    
    # Extraire les résultats : titre du film et poster URL
    recommended_titles = [rec['title'] for rec in recommendations]

    
    return recommended_titles

# Modèle de données pour l'input de l'utilisateur
class MovieRequest(BaseModel):
    plot: str
    embedding_type: str = "bow_embeddings"  # par défaut, on utilise BOW


### Endpoint pour la recommandation via description du film ###

@app.post("/recommendation_plot")
async def get_recommendations(request: MovieRequest):
    titles = recommend_movies(request.plot, request.embedding_type)
    return {"titles": titles}




