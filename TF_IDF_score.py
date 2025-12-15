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
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from frequence import preprocess_and_lemmatize, load_model_for_text  

def build_tfidf_vectors(
    text: str,
    top_n: int = 20,
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Calcule TF, IDF et TF-IDF pour un texte.
    Utilise la lemmatisation et le filtrage définis dans frequence.py.

    Retourne 4 listes parallèles (triées par TF-IDF décroissant) :
      - mots
      - tf_moyen
      - idf
      - tfidf_moyen
    """

    # 1) Découper le texte en phrases (pseudo-corpus = phrases du texte)
    nlp = load_model_for_text(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) < 2:
        sentences = [text]

    # 2) TF-IDF avec ton tokenizer custom, SANS normalisation (norm=None)
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_and_lemmatize,
        lowercase=False,
        token_pattern=None,
        norm=None,        # <--- important : pas de normalisation L2
        use_idf=True,
    )

    X = vectorizer.fit_transform(sentences)  # TF * IDF sur chaque phrase
    feature_names = vectorizer.get_feature_names_out()

    # IDF appris sur ton "corpus" (les phrases du texte)
    idf = vectorizer.idf_  # tableau de même taille que feature_names

    # 3) Moyenne des scores TF-IDF sur toutes les phrases
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()  # TF-IDF moyen par mot

    # 4) En déduire TF moyen : TF = (TF-IDF) / IDF   (élément par élément)
    #    (idf n'est jamais 0 grâce au smoothing de scikit-learn)
    tf_mean = tfidf_mean / idf

    # 5) On trie selon TF-IDF moyen décroissant
    idx_sorted = np.argsort(-tfidf_mean)[:top_n]

    top_words   = [feature_names[i]   for i in idx_sorted]
    top_tfs     = [float(tf_mean[i])  for i in idx_sorted]
    top_idfs    = [float(idf[i])      for i in idx_sorted]
    top_tfidfs  = [float(tfidf_mean[i]) for i in idx_sorted]

    return top_words, top_tfs, top_idfs, top_tfidfs



# les scores correspond au TF=sa fréquence dans le texte , IDF=sa rarété dans  le corpus global 