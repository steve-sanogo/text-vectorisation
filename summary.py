# summary.py
from __future__ import annotations
from typing import List, Tuple
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from frequence import load_model_for_text, preprocess_and_lemmatize


# -----------------------------
# 1. Découpage en phrases
# -----------------------------
def split_sentences(text: str) -> List[str]:
    """
    Découpe un texte en phrases en utilisant spaCy
    (via load_model_for_text de frequence.py).
    """
    nlp = load_model_for_text(text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# -----------------------------
# 2. Calcul du nombre de phrases à garder
# -----------------------------
def compute_k_from_percentage(
    n_sentences: int,
    percentage: float,
    min_sentences: int = 1,
) -> int:
    """
    Transforme un pourcentage utilisateur en nombre de phrases à garder.

    Exemple :
      10 % sur 25 phrases -> ceil(25 * 10 / 100) = 3

    Garantit au moins min_sentences phrases, et ne dépasse pas n_sentences.
    """
    if n_sentences == 0:
        return 0

    # Sécurisation des valeurs
    if percentage <= 0:
        return min_sentences
    if percentage > 100:
        percentage = 100

    k = math.ceil(n_sentences * (percentage / 100.0))
    k = max(k, min_sentences)
    k = min(k, n_sentences)
    return k


# -----------------------------
# 3. MÉTHODE 1 : premières phrases
# -----------------------------
def summarize_first_sentences(
    text: str,
    percentage: float = 10,
    min_sentences: int = 1,
) -> str:
    """
    Méthode 1 : Résumé = les premières phrases du texte
    selon le pourcentage choisi par l'utilisateur.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    k = compute_k_from_percentage(
        n_sentences=len(sentences),
        percentage=percentage,
        min_sentences=min_sentences,
    )

    selected = sentences[:k]
    return " ".join(selected)


# -----------------------------
# 4. OUTIL COMMUN POUR MÉTHODES 2 & 3
# -----------------------------
def _most_similar_sentences(
    text: str,
    percentage: float,
    min_sentences: int = 1,
    include_stopwords: bool = False,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """
    Calcule les similarités entre chaque phrase et le texte complet via TF-IDF + cosinus.
    Retourne :
      - sentences : liste des phrases
      - similarities : tableau de scores cosinus (une valeur par phrase)
      - top_idx : indices des k phrases les plus similaires
    """
    sentences = split_sentences(text)
    if not sentences:
        return [], np.array([]), []

    k = compute_k_from_percentage(
        n_sentences=len(sentences),
        percentage=percentage,
        min_sentences=min_sentences,
    )
    if k == 0:
        return sentences, np.array([]), []

    # Corpus = texte complet + une entrée par phrase
    docs = [text] + sentences

    vectorizer = TfidfVectorizer(
        tokenizer=lambda txt: preprocess_and_lemmatize(
            txt,
            include_stopwords=include_stopwords,
        ),
        lowercase=False,
        token_pattern=None,      # on laisse le tokenizer custom gérer
        ngram_range=ngram_range,
    )

    X = vectorizer.fit_transform(docs)

    # Vecteur du texte complet (1 x d)
    full_vec = normalize(X[0], norm="l2", axis=1)
    # Vecteurs des phrases (n x d)
    sent_vecs = normalize(X[1:], norm="l2", axis=1)

    # Similarités cosinus : (n x d) @ (d x 1) = (n x 1)
    similarities = (sent_vecs @ full_vec.T).toarray().ravel()

    # Indices des k phrases les plus similaires
    top_idx = np.argsort(-similarities)[:k]

    return sentences, similarities, top_idx


# -----------------------------
# 5. MÉTHODE 2 : classée par importance
# -----------------------------
def summarize_most_similar_ranked(
    text: str,
    percentage: float = 10,
    min_sentences: int = 1,
    include_stopwords: bool = False,
    ngram_range: Tuple[int, int] = (1, 1),
) -> str:
    """
    Méthode 2 :
    - Résumé = X % des phrases les plus similaires au texte complet
      (TF-IDF + cosinus),
    - Triées par score de similarité décroissant (phrases les plus
      importantes en premier).
    """
    sentences, similarities, top_idx = _most_similar_sentences(
        text=text,
        percentage=percentage,
        min_sentences=min_sentences,
        include_stopwords=include_stopwords,
        ngram_range=ngram_range,
    )

    if len(top_idx) == 0:
        return ""

    # Tri par importance (score de similarité décroissant)
    ranked_indices = sorted(top_idx, key=lambda i: similarities[i], reverse=True)
    selected = [sentences[i] for i in ranked_indices]

    return " ".join(selected)


# -----------------------------
# 6. MÉTHODE 3 : mêmes phrases mais ordre du texte
# -----------------------------
def summarize_most_similar_in_text_order(
    text: str,
    percentage: float = 10,
    min_sentences: int = 1,
    include_stopwords: bool = False,
    ngram_range: Tuple[int, int] = (1, 1),
) -> str:
    """
    Méthode 3 :
    - On reprend les X % de phrases les plus similaires au texte complet
      (même sélection que la méthode 2),
    - Mais on les affiche dans l'ordre d'apparition dans le texte,
      pour un résumé plus fluide.
    """
    sentences, similarities, top_idx = _most_similar_sentences(
        text=text,
        percentage=percentage,
        min_sentences=min_sentences,
        include_stopwords=include_stopwords,
        ngram_range=ngram_range,
    )

    if len(top_idx) == 0:
        return ""

    # Tri dans l'ordre d'apparition (indices croissants)
    ordered_indices = sorted(top_idx)
    selected = [sentences[i] for i in ordered_indices]

    return " ".join(selected)




    