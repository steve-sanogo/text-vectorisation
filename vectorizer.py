# vectorizer.py
from __future__ import annotations
from collections import Counter
from typing import List, Tuple
import re

# ------------------------------
# 1. Stopwords de base (FR + EN)
# ------------------------------
BASIC_STOPWORDS = {
    # Français
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "ce", "cet", "cette", "ces",
    "que", "qui", "quoi", "dont", "où",
    "ne", "pas", "plus", "moins",
    # clitiques d'une lettre (français parlé)
    "d", "c", "j", "t", "m", "s", "n",

    # Anglais basique
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "to",
    "for", "and", "or", "that", "this", "with", "on", "at", "by"
}

# Regex pour extraire les mots (lettres/chiffres)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Regex pour séparer les apostrophes : l', d', j', t', m', s', n', c'
APOS_CLITIC_RE = re.compile(r"\b([cdjlmnst])['’]", flags=re.IGNORECASE)


# ------------------------------
# 2. Tokenisation / prétraitement
# ------------------------------

def tokenize(text: str) -> List[str]:
    """
    Découpe le texte en mots simples, en gérant les apostrophes.
    Exemple :
        "L'intelligence artificielle" -> ["l", "intelligence", "artificielle"]
    """
    # uniformiser les apostrophes
    text = text.replace("’", "'")
    # séparer les clitiques (l', d', j', etc.)
    text = APOS_CLITIC_RE.sub(r"\1 ", text)
    # extraire les mots et passer en minuscule
    return [t.lower() for t in TOKEN_RE.findall(text)]


def filter_stopwords(tokens: List[str], include_stopwords: bool) -> List[str]:
    """
    Si include_stopwords == False, enlève les mots vides (stopwords).
    Sinon, retourne les tokens tels quels.
    """
    if include_stopwords:
        return tokens
    return [t for t in tokens if t not in BASIC_STOPWORDS]


# ------------------------------
# 3. Fonction principale
# ------------------------------

def build_frequency_vector(
    text: str,
    include_stopwords: bool = False,
) -> Tuple[List[int], List[str]]:
    """
    Construit le vecteur complet des fréquences des mots présents dans le texte.

    :param text: texte à analyser
    :param include_stopwords: True -> garde les stopwords
                              False -> les enlève

    :return: (vector, words)
             - vector : liste des fréquences
             - words  : liste des mots correspondants (même ordre)
    """
    # 1) Nettoyage et découpe
    tokens = tokenize(text)

    # 2) Filtrage des stopwords si besoin
    tokens = filter_stopwords(tokens, include_stopwords=include_stopwords)

    # 3) Comptage des fréquences
    counter = Counter(tokens)

    # 4) Tri par fréquence décroissante puis alphabétique
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # 5) Construction du vecteur complet (tous les mots)
    words = [w for w, _ in items]
    vector = [freq for _, freq in items]

    return vector, words
