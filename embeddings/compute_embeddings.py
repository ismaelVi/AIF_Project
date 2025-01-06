import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import os
import sys
import joblib

# Ajouter le répertoire parent de 'src' au chemin pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.param.param import CSV_PATH, OUTPUT_PATH, GLOVE_PATH, OUTPUT_FILE



# Charger les modèles et tokenizer pour GloVe et DistillBERT
DISTILLBERT_MODEL_NAME = "distilbert-base-uncased"

# 1. Charger les données
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    # Garder uniquement les colonnes nécessaires
    return df[["id", "title", "overview"]].dropna(subset=["overview"]).reset_index(drop=True)

# 2. Nettoyer le texte
def clean_text(text):
    # Supprime ponctuation, transforme en minuscule, etc. (simplifié ici)
    return text.lower()

# 3. Embedding Bag of Words
def compute_bow_embeddings(overviews):
    vectorizer = CountVectorizer(max_features=5000)  # Limite à 5000 mots les plus fréquents
    bow_matrix = vectorizer.fit_transform(overviews)
    return bow_matrix.toarray(), vectorizer


# 4. Embedding GloVe
def load_glove_model(glove_path):
    return KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

def compute_glove_embeddings(overviews, glove_model):
    embeddings = []
    for text in overviews:
        words = text.split()
        word_vectors = [glove_model[word] for word in words if word in glove_model]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))  # Moyenne des vecteurs
        else:
            embeddings.append(np.zeros(glove_model.vector_size))  # Vecteur nul si aucun mot connu
    return np.array(embeddings)


# 5. Embedding DistillBERT
def compute_bert_embeddings(overviews, tokenizer, model):
    embeddings = []
    for text in overviews:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())  # Moyenne sur la séquence
    return np.array(embeddings)


# Pipeline principal
def main():

    # Chargement des données
    print("Chargement des données...")
    df = load_data(CSV_PATH)

    # Cleaning des données
    df["cleaned_overview"] = df["overview"].apply(clean_text)

    # Calcul des embeddings avec bow
    print("Calcul des embeddings Bag of Words...")
    bow_embeddings, vectorizer = compute_bow_embeddings(df["cleaned_overview"])
    df["bow_embeddings"] = list(bow_embeddings)

    # Sauvegare du model bow
    print("sauvegarde du model")
    save_path = OUTPUT_FILE + "/saved_models/bow_model"
    joblib.dump(vectorizer, save_path)
    print("model sauvé")

    # Calcul des embeddings avec Glove
    print("Chargement du modèle GloVe...")
    glove_model = load_glove_model(GLOVE_PATH)
    print("Calcul des embeddings GloVe...")
    glove_embeddings = compute_glove_embeddings(df["cleaned_overview"], glove_model)
    df["glove_embeddings"] = list(glove_embeddings)

    #Calcul des embeddings avec Bert
    # print("Chargement du modèle DistillBERT...")
    # tokenizer = DistilBertTokenizer.from_pretrained(DISTILLBERT_MODEL_NAME)
    # model = DistilBertModel.from_pretrained(DISTILLBERT_MODEL_NAME)
    # print("Calcul des embeddings DistillBERT...")
    # bert_embeddings = compute_bert_embeddings(df["cleaned_overview"], tokenizer, model)
    # df["bert_embeddings"] = list(bert_embeddings)

    # Sauvegarder le dataframe avec les embeddings
    print("Sauvegarde des résultats...")
    # Installer pyarrow si nécessaire : pip install pyarrow
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Embeddings sauvegardés dans {OUTPUT_PATH}.")




# print(f"Embeddings calculés et sauvegardés dans {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()
