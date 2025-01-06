import os
import gradio as gr
import requests
from PIL import Image
from urllib.parse import urljoin

# URL de votre API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # Par défaut, utilise localhost pour le dev local



# Fonction de prédiction via l'API
def predict(image):
    # Envoyer l'image à l'API
    with open(image, "rb") as img:
        response = requests.post(API_URL+"/predict", files={"file": img})
    # Vérifier la réponse
    if response.status_code == 200:
        return response.json().get("genre")
    else:
        return "Erreur lors de la prédiction"
    

# Fonction pour la recommandation par poster
def recommend_movies(image):

    try:
        # Envoyer l'image à l'API pour obtenir des recommandations
        with open(image, "rb") as img:
            response = requests.post(API_URL+"/recommend", files={"file": img})
        
        # Vérifier la réponse
        if response.status_code == 200:
            data = response.json()
            # Extraire les chemins des images des films recommandés
            recommended_paths = data.get("recommendations", [])
            if recommended_paths:
                # Charger les images depuis les chemins absolus
                images = [Image.open(os.path.abspath(__file__ +'/../'+path)) for path in recommended_paths]
                return images
            else:
                return "Aucune recommandation disponible."
        else:
            return f"Erreur {response.status_code} : Impossible d'obtenir les recommandations."
    except Exception as e:
        return f"Erreur : {str(e)}"
    

# Fonction pour la recommandation par plot
def get_recommendations_from_api(plot, embedding_type="bow_embeddings"):
    # Créer la payload pour la requête
    payload = {"plot": plot, "embedding_type": embedding_type}
    
    # Faire une requête POST à l'API FastAPI
    response = requests.post(API_URL+"/recommendation_plot", json=payload)
    
    if response.status_code == 200:
        # Extraire les résultats
        data = response.json()
        return data['titles']
    else:
        return []
    



# Interface combinée
with gr.Blocks() as demo:
    gr.Markdown("# Prédicteur de Genre de Film avec Recommandations")
    with gr.Row():
        with gr.Column():
            # Interface pour prédire le genre
            genre_image = gr.Image(type="filepath", label="Télécharger l'affiche du film")
            genre_output = gr.Textbox(label="Genre prédit")
            predict_button = gr.Button("Prédire le Genre")
            
        with gr.Column():
            # Interface pour les recommandations
            recommendation_image = gr.Image(type="filepath", label="Télécharger l'affiche pour des recommandations")
            #recommendations_output = gr.Textbox(label="film recomadé")
            recommendations_output = gr.Gallery(type = "filepath", label="Films Recommandés") # gr.Textbox(label="Genre prédit")  # 
            recommend_button = gr.Button("Obtenir des Recommandations")
    with gr.Row():
        with gr.Column():
            # Interface pour les recommandations basées sur le plot
            api_plot_input = gr.Textbox(label="Entrez la description du film pour des recommandations basées sur l'API")
            
            # Ajout du menu déroulant pour choisir l'Embedding Type
            embedding_type_dropdown = gr.Dropdown(
                label="Choisir le type d'Embedding", 
                choices=["bow_embeddings", "glove_embeddings", "bert_embeddings"], 
                value="bow_embeddings"  # Valeur par défaut
            )
            
            api_recommend_button = gr.Button("Obtenir Recommandations")
            api_recommendations_output = gr.Textbox(label="Films recommandés par l'API")

        
    # Connecter les fonctions aux boutons
    predict_button.click(fn=predict, inputs=genre_image, outputs=genre_output)
    recommend_button.click(fn=recommend_movies, inputs=recommendation_image, outputs=recommendations_output)
    api_recommend_button.click(fn=get_recommendations_from_api, inputs=[api_plot_input, embedding_type_dropdown], outputs=api_recommendations_output)



if __name__ == "__main__":
    print("web app running on http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860,show_error=True)

