# Study Performance Predictor
# Autor: Mohamed Mehdi Abid
# Beschreibung:
# Dieses Projekt untersucht, wie Lernzeit und Schlafdauer das Prüfungsergebnis beeinflussen.
# Es kombiniert Datenanalyse, Visualisierung und ein einfaches Machine-Learning-Modell.

# ===============================
# 1. Bibliotheken importieren
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ===============================
# 2. Beispieldaten erstellen
# ===============================
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep_Hours": [8, 7.5, 7, 6.5, 6, 6, 5.5, 5, 5, 4.5],
    "Exam_Score": [45, 48, 51, 55, 61, 67, 72, 78, 85, 90]
}

df = pd.DataFrame(data)
print("Datensatz:")
print(df.head())


# ===============================
# 3. Datenanalyse
# ===============================
#TODO
# Visualisierung

#TODO

# ===============================
# 4. Machine-Learning-Modell
# ===============================

#TODO

# ===============================
# 5. Fazit
# ===============================
