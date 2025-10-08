import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Télécharger les ressources nécessaires
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# 1. Charger le dataset
df = pd.read_csv("Avis_Client.csv", sep=",", encoding="utf-8", quotechar='"', engine="python")

print("=" * 60)
print("APERÇU DES DONNÉES")
print("=" * 60)
print(df.head())
print(f"\nNombre total d'avis : {len(df)}")

# 2. CRÉER DE VRAIS LABELS AVEC VADER (analyse de sentiment)
print("\n" + "=" * 60)
print("CRÉATION DES LABELS AVEC VADER")
print("=" * 60)

sia = SentimentIntensityAnalyzer()


def get_sentiment_vader(text):
    """Utilise VADER pour déterminer le sentiment"""
    if not isinstance(text, str):
        return 'neutre'

    scores = sia.polarity_scores(text)
    compound = scores['compound']

    # Classification basée sur le score compound
    if compound >= 0.05:
        return 'positif'
    elif compound <= -0.05:
        return 'negatif'
    else:
        return 'neutre'


# Générer les labels réels
df['Sentiment_reel'] = df['Avis'].apply(get_sentiment_vader)

print("\nDistribution des sentiments réels :")
print(df['Sentiment_reel'].value_counts())
print("\nPourcentage :")
print(df['Sentiment_reel'].value_counts(normalize=True) * 100)

# 3. Prétraitement du texte AMÉLIORÉ
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Ajouter des mots-clés de sentiment français (si vos avis sont en français)
mots_positifs_fr = {'bon', 'excellent', 'super', 'génial', 'parfait', 'satisfait',
                    'content', 'merveilleux', 'top', 'recommande'}
mots_negatifs_fr = {'mauvais', 'nul', 'horrible', 'décevant', 'médiocre',
                    'insatisfait', 'problème', 'arnaque', 'éviter'}


def clean_text_advanced(text):
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    # Conserver certains patterns importants avant nettoyage
    text_lower = re.sub(r"pas bon|pas bien|pas satisfait", "negatif_expression", text_lower)
    text_lower = re.sub(r"très bon|très bien|très satisfait", "positif_expression", text_lower)

    # Nettoyage standard
    text_lower = re.sub(r"http\S+|www\S+|https\S+", '', text_lower)
    text_lower = re.sub(r'\d+', '', text_lower)
    text_lower = text_lower.translate(str.maketrans('', '', string.punctuation))

    # Tokenization et lemmatisation
    words = text_lower.split()

    # Ne pas supprimer les stopwords négatifs importants
    important_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                       'pas', 'jamais', 'rien', 'aucun'}
    words = [word for word in words if word not in stop_words or word in important_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


df['clean_avis'] = df['Avis'].apply(clean_text_advanced)

print("\n" + "=" * 60)
print("EXEMPLES DE TEXTE NETTOYÉ")
print("=" * 60)
print(df[['Avis', 'clean_avis', 'Sentiment_reel']].head(10))

# 4. Préparation des données
X = df['clean_avis']
y = df['Sentiment_reel']  # ⭐ UTILISER LES VRAIS LABELS

# Vérifier s'il y a assez de données par classe
print(f"\nDistribution des classes :\n{y.value_counts()}")

# Split stratifié pour garder la même proportion dans train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTaille du train : {len(X_train)}")
print(f"Taille du test : {len(X_test)}")

# 5. TF-IDF avec paramètres optimisés
vectorizer = TfidfVectorizer(
    max_features=2000,  # Réduit pour éviter l'overfitting avec peu de données
    ngram_range=(1, 3),  # Inclure les trigrammes
    min_df=2,  # Ignorer les mots qui apparaissent dans moins de 2 documents
    max_df=0.8,  # Ignorer les mots trop fréquents
    sublinear_tf=True  # Utiliser le log du TF
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nDimensions de la matrice TF-IDF : {X_train_vec.shape}")

# 6. GESTION DU DÉSÉQUILIBRE DES CLASSES avec SMOTE
print("\n" + "=" * 60)
print("APPLICATION DE SMOTE POUR ÉQUILIBRER LES CLASSES")
print("=" * 60)

try:
    smote = SMOTE(random_state=42, k_neighbors=min(5, min(y_train.value_counts()) - 1))
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    print("SMOTE appliqué avec succès")
    print(f"Distribution après SMOTE :\n{pd.Series(y_train_balanced).value_counts()}")
except Exception as e:
    print(f" SMOTE non applicable : {e}")
    print("On continue sans SMOTE...")
    X_train_balanced, y_train_balanced = X_train_vec, y_train

# 7. ENTRAÎNEMENT DE PLUSIEURS MODÈLES
print("\n" + "=" * 60)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=10,  # Régularisation plus forte
        class_weight='balanced'  #  Important pour classes déséquilibrées
    ),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        class_weight='balanced'
    )
}

results = {}

for name, model in models.items():
    print(f"\n Entraînement : {name}")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\nRapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

# 8. SÉLECTIONNER LE MEILLEUR MODÈLE
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n" + "=" * 60)
print(f" MEILLEUR MODÈLE : {best_model_name}")
print(f" Accuracy : {results[best_model_name]:.4f}")
print("=" * 60)

# 9. Prédiction sur tous les avis
X_all_vec = vectorizer.transform(X)
df['Sentiment_predit'] = best_model.predict(X_all_vec)

# 10. Comparaison VADER vs Modèle
print("\n" + "=" * 60)
print("COMPARAISON DES PRÉDICTIONS")
print("=" * 60)

print("\nSentiments VADER (labels d'entraînement) :")
print(df['Sentiment_reel'].value_counts(normalize=True) * 100)

print("\nSentiments prédits par le modèle :")
print(df['Sentiment_predit'].value_counts(normalize=True) * 100)

# 11. Aperçu final
print("\n" + "=" * 60)
print("APERÇU FINAL DES PRÉDICTIONS")
print("=" * 60)
print(df[['Avis', 'Sentiment_reel', 'Sentiment_predit']].head(20))

# 12. Analyse des erreurs
print("\n" + "=" * 60)
print("ANALYSE DES ERREURS")
print("=" * 60)

errors = df[df['Sentiment_reel'] != df['Sentiment_predit']]
print(f"\nNombre d'erreurs : {len(errors)} / {len(df)} ({len(errors) / len(df) * 100:.2f}%)")

if len(errors) > 0:
    print("\nExemples d'erreurs :")
    print(errors[['Avis', 'Sentiment_reel', 'Sentiment_predit']].head(10))

# 13. Sauvegarder les résultats
df.to_csv("Avis_Client_avec_sentiments.csv", index=False, encoding='utf-8')
print("\n Résultats sauvegardés dans 'Avis_Client_avec_sentiments.csv'")

# 14. CONSEILS POUR AMÉLIORER DAVANTAGE
print("\n" + "=" * 60)
print("💡 RECOMMANDATIONS POUR AMÉLIORER LE MODÈLE")
print("=" * 60)
print("""
1.  LABELLISER MANUELLEMENT : Avec seulement 100 avis, la meilleure 
   approche est de créer un fichier Excel avec une colonne 'Sentiment' 
   que vous remplissez manuellement (positif/negatif/neutre).

2.  LANGUE : Si vos avis sont en français, utilisez :
   - stopwords.words('french') au lieu de 'english'
   - Un modèle pré-entraîné français (CamemBERT, FlauBERT)

3.  PLUS DE DONNÉES : 100 avis est très peu. Essayez d'en collecter plus.

4.  MODÈLES AVANCÉS : Utilisez des transformers pré-entraînés :
   - from transformers import pipeline
   - classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

5.  CLASSES DÉSÉQUILIBRÉES : Si une classe domine, utilisez :
   - class_weight='balanced' (déjà fait)
   - SMOTE pour sur-échantillonner (déjà fait)
""")