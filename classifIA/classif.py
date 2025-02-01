import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import backend_projet_6
from fpdf import FPDF
from io import BytesIO

# Configurer la page
st.set_page_config(page_title="VisionPro", layout="wide")

# CSS pour positionner les logos avec un z-index élevé
st.markdown(
    """
    <style>
        .left-logo {
            position: fixed; /* Fixe le logo pour qu'il reste visible */
            top: 0; /* Positionne le logo en haut */
            left: -10px;
            top: 10px;
            z-index: 1000; /* Force le logo à rester au-dessus des autres éléments */
        }
        .right-logo {
            position: fixed; /* Fixe le logo pour qu'il reste visible */
            top: 10; /* Positionne le logo en haut */
            right: 10px;
            top: 10px;
            z-index: 1000; /* Force le logo à rester au-dessus des autres éléments */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Fonction pour convertir une image en base64 (pour HTML)
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Ajouter les logos avec CSS
try:
    logo_left = Image.open("ise.png").resize((200, 200))
    logo_left_base64 = image_to_base64(logo_left)
    st.markdown(
        f'<img class="left-logo" src="data:image/png;base64,{logo_left_base64}">',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.write("Logo gauche non disponible.")

try:
    logo_right = Image.open("eneam.png").resize((100, 100))
    logo_right_base64 = image_to_base64(logo_right)
    st.markdown(
        f'<img class="right-logo" src="data:image/png;base64,{logo_right_base64}">',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.write("Logo droit non disponible.")

# Chargement et affichage du fichier CSS
with open("style.css") as css_file:
    css = css_file.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Chargement et affichage du fichier HTML
with open("index.html") as html_file:
    html_content = html_file.read()
st.markdown(html_content, unsafe_allow_html=True)

# Zone d'importation d'images
uploaded_files = st.file_uploader("Importer des images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)


# Affichage des images importées et bouton pour la classification
if uploaded_files:
    st.session_state["images"] = uploaded_files

    # Texte en gras et en tant que sous-titre
    st.markdown("## **Images importées :**")

    cols = st.columns(5)  # Limiter à 5 colonnes
    img_files = []  # Liste pour stocker les chemins des fichiers

    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        
        # Redimensionner l'image
        resized_img = img.resize((150, 150))  # Ajuster selon vos besoins

        # Affichage des images
        cols[i % 5].image(
            resized_img,
            caption=file.name,
            width=150  # Définir explicitement la largeur
        )

        # Ajouter le fichier téléchargé à la liste img_files
        img_files.append(file)

    # CSS pour les bordures et les coins arrondis
    st.markdown(
        """
        <style>
            .stImage img {
                border: 2px solid black;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Bouton de classification
    if st.button("Classer les Images"):
        try:
            # Appel de la fonction backend pour classifier les images
            predictions = backend_projet_6.classify_images(img_files)
            st.write("## Résultats de la Classification")

            # Affichage des prédictions
            for file, pred in zip(img_files, predictions):
                st.image(file, caption=f"Classe prédite : {pred}", use_column_width=True)

            # Afficher le bouton "Réinitialiser" après la classification
            if st.button("Réinitialiser"):
                st.session_state.clear()  # Effacer les images importées et autres variables
                st.stop()  # Arrêter le script et recharger la page pour recommencer

        except Exception as e:
            st.error(f"Erreur lors de la classification : {e}")
