# ===============================================
# tfidf_vectorizer.py — version pro basée sur vectorizer.py
# ===============================================

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from  frequence import preprocess_and_lemmatize, load_model_for_text  

# --------------------------------------------------------
# 1. Construction des vecteurs TF-IDF
# --------------------------------------------------------

def build_tfidf_vectors(
    text: str,
    top_n: int = 20,
) -> Tuple[List[str], List[float]]:
    """
    Calcule les scores TF-IDF pour un texte.
    Utilise la lemmatisation et le filtrage définis dans frequence.py.
    Retourne (mots, scores) triés par importance décroissante.
    """
    # Charger le modèle pour découper le texte en phrases (pseudo-corpus)
    nlp = load_model_for_text(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) < 2:
        sentences = [text]

    # TF-IDF avec ton tokenizer custom (lemmatisation spaCy)
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_and_lemmatize,
        lowercase=False,
        token_pattern=None,
    )

    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Moyenne des scores sur toutes les phrases
    scores = np.asarray(X.mean(axis=0)).ravel()
    idx_sorted = np.argsort(-scores)[:top_n]

    top_words = [feature_names[i] for i in idx_sorted]
    top_scores = [float(scores[i]) for i in idx_sorted]

    return top_words, top_scores


# --------------------------------------------------------
# 2. Exemple d’utilisation
# --------------------------------------------------------
if __name__ == "__main__":
    text = """
    Lire en anglais est une excellente habitude pour améliorer son niveau de langue,
    de même qu’écouter la radio anglophone ou regarder des films en anglais.
    Cette activité permet de se familiariser avec des tournures de phrase
    et d’assimiler de nouveaux mots de vocabulaire dans leur contexte.
    """

    mots, scores = build_tfidf_vectors(text, top_n=15)

    print("\n Top mots TF-IDF :")
    for mot, score in zip(mots, scores):
        print(f"{mot:15s} → {score:.4f}")

# les scores correspond au TF=sa fréquence dans le texte , IDF=sa rarété dans  le corpus global 