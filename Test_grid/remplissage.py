import pandas as pd
import matplotlib.pyplot as plt

# Liste des fichiers à inclure (sans 'facile')
fichiers = {
    "moyen": "data_moyen.csv",
    "difficile": "data_difficile.csv",
    "expert": "data_expert.csv",
    "impossible": "data_impossible.csv"
}

# Couleurs personnalisées pour correspondre à ton histogramme
couleurs = {
    "moyen": "#5DA5DA",       # bleu
    "difficile": "#FAA43A",   # orange
    "expert": "#60BD68",      # vert
    "impossible": "#F17CB0"   # rose
}

# Calcul des moyennes
moyennes = {}
for niveau, fichier in fichiers.items():
    df = pd.read_csv(fichier)
    moyennes[niveau] = df["clues"].mean()

# Création du barplot
plt.figure(figsize=(8, 5))
niveaux = list(moyennes.keys())
valeurs = [moyennes[niv] for niv in niveaux]
barres = plt.bar(niveaux, valeurs, color=[couleurs[niv] for niv in niveaux])

# Ajout des valeurs au-dessus des barres
for bar in barres:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}", ha='center', va='bottom')

# Style
plt.title("Moyenne des cases remplies par difficulté")
plt.ylabel("Nombre moyen d'indices'")
plt.ylim(0, max(valeurs) + 10)

plt.show()
