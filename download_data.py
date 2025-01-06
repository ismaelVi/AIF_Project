import gdown
import os

# URL du fichier Google Drive (assurez-vous qu'il est partagé publiquement avec "Anyone with the link")
DATA_URL = "https://drive.google.com/uc?id=<YOUR_FILE_ID>"
OUTPUT_DIR = "./data"

def download_data():
    """Télécharge les données d'entraînement depuis Google Drive."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "training_data.zip")
    print(f"Téléchargement des données d'entraînement depuis {DATA_URL}...")
    gdown.download(DATA_URL, output_file, quiet=False)
    print(f"Données téléchargées et sauvegardées dans {output_file}.")

    # Décompresser si nécessaire
    print(f"Décompression des données dans {OUTPUT_DIR}...")
    import zipfile
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)
    print("Décompression terminée.")

if __name__ == "__main__":
    download_data()
