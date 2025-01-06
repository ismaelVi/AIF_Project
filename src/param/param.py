import os 

# Convertir en chemin absolu
DATA_PATH = os.path.abspath(__file__ + '/../../../data/raw/sorted_movie_posters_paligema')
SHORT_DATA_PATH = 'data/raw/sorted_movie_posters_paligema'
CSV_PATH = os.path.abspath(__file__ + '/../../../data/movies_metadata.csv')
OUTPUT_PATH = os.path.abspath(__file__ + '/../../../data/movies_with_embeddings.parquet')
OUTPUT_FILE = os.path.abspath(__file__ + '/../../../data')
GLOVE_PATH = os.path.abspath(__file__ + '/../../../data/glove.6B.100d.txt')


WEIGHTS_PATH = {'RESNET18': os.path.abspath(__file__ + '/../../../data/saved_models/film_genre_classifier18.pth'),
                'RESNET50': os.path.abspath(__file__ + '/../../../data/saved_models/film_genre_classifier50.pth')
                }

NUM_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.001

GENRES = os.listdir(DATA_PATH)
NUM_CLASSES = len(GENRES)

DISTILLBERT_MODEL_NAME = "distilbert-base-uncased"