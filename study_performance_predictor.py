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
print("\nStatistische Übersicht:")
print(df.describe())

print("\nKorrelationen:")
print(df.corr())

# Visualisierung
plt.scatter(df["Hours_Studied"], df["Exam_Score"], color="blue")
plt.title("Einfluss der Lernzeit auf das Prüfungsergebnis")
plt.xlabel("Stunden gelernt")
plt.ylabel("Prüfungsergebnis")
plt.grid(True)
plt.show()

# ===============================
# 4. Machine-Learning-Modell
# ===============================
X = df[["Hours_Studied", "Sleep_Hours"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\nModellergebnisse:")
print(f"RMSE: {rmse:.2f}")
print(f"Koeffizienten: {model.coef_}")
print(f"Achsenabschnitt: {model.intercept_}")

# ===============================
# 5. Fazit
# ===============================
print("\nFazit:")
print("Mehr Lernzeit verbessert das Ergebnis, während zu wenig Schlaf einen negativen Einfluss hat.")