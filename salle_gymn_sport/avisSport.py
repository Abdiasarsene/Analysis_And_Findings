# Importation des librairies
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import seaborn as sns
from nrclex import NRCLex
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation

# Téléchargement des autres packages importants
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

# Importation des données textuelles
avissportif = pd.read_csv(r"D:\Projets\IT\Datascience\datascience\nlp\AvisSportif_Data.csv")

# Tokenization
def tokenize(text):
    tokens= re.findall(r'\b\w+\b', text.lower())
    return tokens

# Application et affichage
avissportif['tokens']= avissportif['Avis Sportif'].apply(tokenize)

# Supression des stopwords
def remove_stopwords(tokens, stop_word):
    stop_words = [word for word in tokens if word not in stop_word]
    return stop_words

# Listes des stop_words
stop_word =["le", "la", "les", "de", "des", "et", "en", "un", "une", "c", "est",'j','l','sont','que','ai']

# Application et affichage
avissportif['stop_words'] = avissportif['tokens'].apply(lambda tokens : remove_stopwords(tokens, stop_word))

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

# Application et affichage 
avissportif['lemmatized'] = avissportif['stop_words'].apply(lemmatize_tokens)
avissportif['lemmatized'] 

# Conversion des tokens en textes 
avissportif['lemmatized_text'] = avissportif['lemmatized'].apply(lambda x : ' ' .join(x))

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(avissportif['lemmatized_text'])

# Récupration des termes et des scores
terms = tfidf.get_feature_names_out()
score = tfidf_matrix.toarray()

# Création de DataFrame
tfidf_data = pd.DataFrame(score, columns=terms)
tfidf_data = tfidf_data.max().reset_index()
tfidf_data.columns=['Termes','Scores']

# Trier les termes importants
tfidf_data = tfidf_data.sort_values(by='Scores', ascending=False).head(10)

# Visualisation graphiques des termes importants
sns.set(style='whitegrid', palette='deep',font_scale=1.1)
plt.figure(figsize=(12,6))
sns.barplot(x='Termes', y='Scores', data=tfidf_data, color='purple')

# Initialisation de l'analyse des sentiments
sentiment = SentimentIntensityAnalyzer()

def analysis_sentiment(text):
    analyse = sentiment.polarity_scores(text)
    return analyse['compound']

# Application et affichage
avissportif['polarity'] =avissportif['stop_words'].apply(analysis_sentiment)

# Interprétation des résultats
def interpretation(polarity):
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else :
        return 'Neutral'

# Application au cluster
avissportif['cluster'] =avissportif['polarity'].apply(interpretation)

# Compter les avis par catégoris
cluster_count = avissportif['cluster'].value_counts()

# Visualisation de la segmentation des polarités
plt.figure(figsize=(8, 6))
cluster_count.plot(kind='bar', color=['orange', 'purple', 'green'])
plt.xlabel('Catégories')
plt.ylabel("Nombre d'Avis")
plt.title("Analyse des sentiments")
plt.xticks(rotation=0)
plt.show()

# Modélisation des sujets avec LDA
# Vectorisation des avis pour LDA
count_vectorizer = CountVectorizer(stop_words=stop_word, max_features=1000)  # Utilisez stop_words comme argument nommé

count_matrix = count_vectorizer.fit_transform(avissportif['stop_words'])  # Correction de l'argument

# Modèle LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)  # 3 thèmes
lda.fit(count_matrix)

# Extraction des thèmes
terms = count_vectorizer.get_feature_names_out()

# Créer un tableau pour les mots-clés des thèmes
topics = []
for idx, topic in enumerate(lda.components_):
    top_terms = [terms[i] for i in topic.argsort()[-10:]]
    topics.append(f"Thème {idx + 1}: {', '.join(top_terms)}")

# Graphique des thèmes (poids des mots)
for idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-10:]
    plt.figure(figsize=(10, 5))
    plt.barh([terms[i] for i in top_indices], topic[top_indices], color='skyblue')
    plt.title(f"Thème {idx + 1}")
    plt.xlabel("Importance")
    plt.ylabel("Termes")
    plt.show()

# Analyse des émotions
# Fonction pour analyser les émotions
def analyze_emotions(text):
    emotion_analysis = NRCLex(text)
    return emotion_analysis.raw_emotion_scores

# Application sur les avis (Exemple fictif)
avissportif['emotions'] = avissportif['stop_words'].apply(analyze_emotions)

# Agrégation des émotions
emotion_counts = Counter()
for emotions in avissportif['emotions']:
    emotion_counts.update(emotions)

# Visualisation des émotions
plt.figure(figsize=(15, 6))
plt.bar(emotion_counts.keys(), emotion_counts.values(), color='green')
plt.xlabel('Émotions')
plt.ylabel("Nombre d'occurrences")
plt.title("Analyse des émotions")
plt.xticks(rotation=45)
plt.show()