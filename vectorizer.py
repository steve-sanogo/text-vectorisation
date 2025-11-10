# ===============================================
# vectorizer.py — version robuste avec fallback
# ===============================================

from __future__ import annotations
from collections import Counter
from typing import List, Tuple
from langdetect import detect
import spacy
from spacy.cli import download
import re

# --------------------------------------------
# 1. Dictionnaire des modèles spaCy disponibles
# --------------------------------------------
LANG_MODELS = {
    "fr": "fr_core_news_sm",  # Français
    "en": "en_core_web_sm",   # Anglais
}

_loaded_models: dict[str, spacy.Language] = {}

APOS_CLITIC_RE = re.compile(r"\b([cdjlmnst])['’]", flags=re.IGNORECASE)


# --------------------------------------------------------
# 2. Détection automatique de la langue et chargement modèle
# --------------------------------------------------------

def load_model_for_text(text: str) -> spacy.Language:
    """
    Détecte la langue du texte, charge le modèle spaCy correspondant.
    Si le modèle n’est pas installé, il est téléchargé automatiquement.
    Si le téléchargement échoue, un modèle vide est utilisé pour ne pas planter.
    """
    try:
        lang_code = detect(text)
    except Exception:
        lang_code = "en"  # Par défaut, anglais si détection impossible

    model_name = LANG_MODELS.get(lang_code, "en_core_web_sm")

    if model_name not in _loaded_models:
        try:
            _loaded_models[model_name] = spacy.load(model_name)
            print(f" Modèle chargé : {model_name}")
        except OSError:
            print(f"⚠️ Modèle {model_name} introuvable. Tentative de téléchargement...")
            try:
                download(model_name)
                _loaded_models[model_name] = spacy.load(model_name)
                print(f" Modèle {model_name} téléchargé et chargé avec succès.")
            except Exception as e:
                print(f"Impossible de charger ou télécharger le modèle {model_name}.")
                print(f" Erreur : {e}")
                print("⚙️ Utilisation d’un modèle linguistique vide (sans lemmatisation).")
                _loaded_models[model_name] = spacy.blank(lang_code if lang_code in LANG_MODELS else "en")

    return _loaded_models[model_name]


# ---------------------------------------
# 3. Prétraitement + lemmatisation complète
# ---------------------------------------

def preprocess_and_lemmatize(text: str) -> List[str]:
    """
    Nettoie le texte, détecte la langue, lemmatise et filtre les stopwords.
    Fonctionne automatiquement pour le français et l’anglais.
    """
    # Uniformiser les apostrophes et séparer les clitiques
    text = text.replace("’", "'")
    text = APOS_CLITIC_RE.sub(r"\1 ", text)

    # Charger le modèle adapté
    nlp = load_model_for_text(text)
    doc = nlp(text)

    tokens: List[str] = []
    for token in doc:
        if not token.is_alpha:
            continue
        if token.is_stop:
            continue
        tokens.append(token.lemma_.lower() if token.lemma_ else token.text.lower())

    return tokens


# -------------------------------------
# 4. Construction du vecteur de fréquence
# -------------------------------------

def build_frequency_vector(text: str) -> Tuple[List[int], List[str]]:
    """
    Construit le vecteur des fréquences de mots lemmatisés.
    Compatible français / anglais et robuste en cas d'erreur modèle.
    """
    tokens = preprocess_and_lemmatize(text)
    counter = Counter(tokens)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words = [w for w, _ in items]
    vector = [freq for _, freq in items]
    return vector, words


# -------------------------------------
# 5. Exemple d'utilisation directe
# -------------------------------------
if __name__ == "__main__":
    text = """

    The children eat red apples.
    """

    vector, words = build_frequency_vector(text)
    print("\n🔍 Mots lemmatisés les plus fréquents :")
    for w, f in zip(words[:10], vector[:10]):
        print(f"{w:15s} → {f}")