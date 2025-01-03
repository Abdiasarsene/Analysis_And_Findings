# Importation deslibrairies
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Assurez-vous d'avoir téléchargé les ressources nécessaires de nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Importation de la base de données
sentiment = pd.read_csv(r"D:\Projets\IT\Datascience\reelproject\analyse- sentiments\avis_consommateurs.csv")

# Tokenization
def tokenize(text):
    tokens = re.findall(r'\b\w+\b',text.lower())
    return tokens

# Application de la tokénization au DataFrame
sentiment['tokens'] = sentiment['Avis'].apply(tokenize)

# Affichage des résultats
sentiment['tokens']

# Stop-word
def remove_stop_word(tokens, stop_word):
    filtered_stop_word = [word for word in tokens if word not in stop_word]
    return filtered_stop_word

# Les mots à supprimer
stop_word =["le", "la", "les", "de", "des", "et", "en", "un", "une", "c", "est",'j','l']

# Application de la suppression des stop-words au DataFrame
sentiment['filtered_stop_word'] = sentiment["tokens"].apply(lambda tokens : remove_stop_word(tokens, stop_word))

# Affichage des résultats
print('Stop Word')
sentiment['filtered_stop_word']

# Initialisation du lemmatizer
lemmatizer = WordNetLemmatizer()

# Fonction de lemmatisation
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Application de la lemmatisation
sentiment['lemmatized_tokens'] = sentiment['filtered_stop_word'].apply(lemmatize_tokens)

# Affichage des résultats
print(sentiment['lemmatized_tokens'])

# Calcul du TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentiment['lemmatized_tokens'])

# Récupérer les termes et les scores TF-IDF
terms = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# Filtrer les termes avec un score TF-IDF inférieur à 0,35
filtered_terms = {term: tfidf_scores[:, idx] for idx, term in enumerate(terms) if tfidf_scores[:, idx].max() >= 0.35}

# Affichage des termes filtrés
print('Termes avec un score TF-IDF >= 0,35:')
for term, scores in filtered_terms.items():
    print(f'Terme: {term}, Score max: {scores.max()}')



# Importation de la base de données
sentiment = pd.read_csv(r"D:\Projets\IT\Datascience\reelproject\analyse- sentiments\avis_consommateurs.csv")

# Tokenization
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Application de la tokénization au DataFrame
sentiment['tokens'] = sentiment['Avis'].apply(tokenize)

# Stop-word
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('french'))
    return [word for word in tokens if word not in stop_words]

# Application de la suppression des stop-words au DataFrame
sentiment['filtered_tokens'] = sentiment["tokens"].apply(remove_stop_words)

# Initialisation du lemmatizer
lemmatizer = WordNetLemmatizer()

# Fonction de lemmatisation
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Application de la lemmatisation
sentiment['lemmatized_tokens'] = sentiment['filtered_tokens'].apply(lemmatize_tokens)

# Convertir les tokens lemmatizés en texte pour TF-IDF
sentiment['lemmatized_text'] = sentiment['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

# Calcul du TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentiment['lemmatized_text'])

# Récupérer les termes et les scores TF-IDF
terms = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# Créer un DataFrame avec les termes et leurs scores TF-IDF
tfidf_df = pd.DataFrame(tfidf_scores, columns=terms)
tfidf_df = tfidf_df.max().reset_index()
tfidf_df.columns = ['Terme', 'Score_TFIDF']

# Filtrer les termes avec un score TF-IDF supérieur ou égal à 0,35
filtered_tfidf_df = tfidf_df[tfidf_df['Score_TFIDF'] >= 0.20]

# Trier les termes par score TF-IDF décroissant
filtered_tfidf_df = filtered_tfidf_df.sort_values(by='Score_TFIDF', ascending=False)

filtered_tfidf_df