import numpy as np
from annoy import AnnoyIndex
import pandas as pd
import json

import os
import sys

# Ajouter le répertoire parent de 'src' au chemin pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.param.param import OUTPUT_PATH, OUTPUT_FILE

# Chargement du Dataframe
df = pd.read_parquet(OUTPUT_PATH)


# Fonction générique pour construire un index Annoy
def build_annoy(df, embedding_type="bow_embeddings"):
    
    # Obtenir la dimension des embeddings
    embedding_dim = len(df[embedding_type][0])

    print(f"Dimension des embeddings :'{embedding_dim}'")
    
    # Créer l'index Annoy pour cet embedding
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    
    # Créer une liste de métadonnées
    metadata = []
    
    #Ajouter tout les embeddings à notre index et metre à jour metadata pour pouvoir lié les embeddings avec les noms des films
    for i, (embedding, title) in enumerate(zip(df[embedding_type], df['title'])):

        if not isinstance(embedding, (list, np.ndarray)):
            print(f"Erreur : l'embedding pour '{title}' n'est pas un vecteur valide : {embedding}")
        elif len(embedding) != embedding_dim:
            print(f"Erreur : l'embedding pour '{title}' a une dimension incorrecte : {len(embedding)}")
        else:
            
            annoy_index.add_item(i, embedding)
            
            metadata.append({
                "id": i,
                "title": title,
            })

    print(f"Nombre d'items ajoutés à l'index : {annoy_index.get_n_items()}")
    
    if annoy_index.get_n_items() == 0:
        print("Aucun item ajouté à l'index. Vérifiez vos données.")
    

    # Construire l'index
    try:
        print("start build")
        annoy_index.build(75)
        print("Index construit avec succès.")
    except Exception as e:
        print(f"Erreur lors de la construction de l'index : {e}")
    
    # Sauvegarder l'index Annoy
    print("sauvegarde")
    file_name = embedding_type + "_movie_posters.ann"
    print(os.path.join(OUTPUT_FILE,file_name))
    annoy_index.save(os.path.join(OUTPUT_FILE,file_name))
    
    # Sauvegarder les métadonnées dans un fichier JSON
    meta_name = embedding_type + "_metadata.json"
    with open(os.path.join(OUTPUT_FILE,meta_name), 'w') as f:
        json.dump(metadata, f)

def build_all_annoy_indexes():
    # Créer et enregistrer les index Annoy pour chaque embedding
    build_annoy(df, embedding_type="glove_embeddings")
    build_annoy(df, embedding_type="bow_embeddings")
    #build_annoy(df, embedding_type="bert_embeddings")

# Appeler la fonction pour créer tous les index
build_all_annoy_indexes()
