import pandas as pd
import os
import sys

# Ajouter le répertoire parent de 'src' au chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.param.param import CSV_PATH, OUTPUT_PATH

df = pd.read_csv(CSV_PATH)

# Obtenir la première ligne
first_row = df.iloc[0]

# Afficher chaque colonne avec son contenu
print("Contenu de la première ligne :")
for column, value in first_row.items():
    print(f"{column}: {value}")