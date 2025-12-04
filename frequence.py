# ===============================================
# vectorizer.py — version robuste + nettoyage complet
# ===============================================

from __future__ import annotations
from collections import Counter
from typing import List, Tuple
from langdetect import detect
import spacy
from spacy.cli import download
import re
import unicodedata
import html

# --------------------------------------------
# 1) Config langue et modèles
# --------------------------------------------
LANG_MODELS = {
    "fr": "fr_core_news_sm",  # Français
    "en": "en_core_web_sm",   # Anglais
}
_loaded_models: dict[str, spacy.Language] = {}

# --------------------------------------------
# 2) Regex utilitaires de nettoyage
# --------------------------------------------
# clitiques français : l', d', j', t', m', s', n', c'
APOS_CLITIC_RE = re.compile(r"\b([cdjlmnst])['’]", flags=re.IGNORECASE)
# Liste des clitiques à filtrer après tokenisation
CLITIC_STOPWORDS = {"d", "l", "j", "t", "m", "s", "n", "c"}
# URLs, emails, @mentions
URL_RE   = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
AT_RE    = re.compile(r"@\w+")
# chiffres isolés / nombres
DIGIT_RE = re.compile(r"\b\d+\b")
# tirets multiples -> espace (évite “pomme-rouge” en un seul token, spaCy gère aussi)
MULTIDASH_RE = re.compile(r"[-–—]{2,}")

# --------------------------------------------
# 3) Helpers de nettoyage (avant spaCy)
# --------------------------------------------
def strip_accents(s: str) -> str:
    # NFKD puis suppression des diacritiques
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def basic_clean(text: str) -> str:
    """
    Nettoyage brut AVANT spaCy:
      - unescape HTML (&nbsp;, &amp;, …)
      - normalisation apostrophes, clitiques
      - suppression URLs/emails/@mentions
      - suppression chiffres isolés
      - réduction tirets multiples
      - trim espaces
    On NE supprime pas toute la ponctuation ici: spaCy gère déjà is_alpha.
    """
    text = html.unescape(text)
    text = text.replace("’", "'")
    text = APOS_CLITIC_RE.sub(r"\1 ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = AT_RE.sub(" ", text)
    text = MULTIDASH_RE.sub(" ", text)
    text = DIGIT_RE.sub(" ", text)
    # espaces propres
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------
# 4) Chargement modèle (auto-download + fallback)
# --------------------------------------------
def load_model_for_text(text: str) -> spacy.Language:
    try:
        lang_code = detect(text)
    except Exception:
        lang_code = "en"  # fallback si détection échoue

    model_name = LANG_MODELS.get(lang_code, "en_core_web_sm")

    if model_name not in _loaded_models:
        try:
            _loaded_models[model_name] = spacy.load(model_name)
            print(f" Modèle chargé : {model_name}")
        except OSError:
            print(f" Modèle {model_name} introuvable. Tentative de téléchargement...")
            try:
                download(model_name)
                _loaded_models[model_name] = spacy.load(model_name)
                print(f" Modèle {model_name} téléchargé et chargé avec succès.")
            except Exception as e:
                print(f"Impossible de charger ou télécharger le modèle {model_name}.")
                print(f" Erreur : {e}")
                print("⚙️ Utilisation d’un modèle linguistique vide (sans lemmatisation).")
                # modèle vide: pas de POS/lemma mais évite le crash
                # on choisit la langue détectée si FR/EN, sinon EN
                _loaded_models[model_name] = spacy.blank(lang_code if lang_code in LANG_MODELS else "en")
    return _loaded_models[model_name]

# --------------------------------------------
# 5) Prétraitement + lemmatisation + filtres
# --------------------------------------------
def preprocess_and_lemmatize(
    text: str,
    remove_accents: bool = True,
    min_len: int = 2,
    include_stopwords: bool = False ,   # <--- AJOUT
) -> List[str]:
    """
    Pipeline pro:
      1) Nettoyage brut (URLs, emails, clitiques, chiffres…)
      2) (option) suppression des accents
      3) spaCy: tokenisation + lemmatisation
      4) filtres: alpha, stopwords, clitiques résiduels, longueur min
    Retour: liste de lemmes utiles.
    """
    text = basic_clean(text)
    if remove_accents:
        text = strip_accents(text)

    nlp = load_model_for_text(text)
    doc = nlp(text)

    tokens: List[str] = []
    for token in doc:
        if not token.is_alpha:
            continue
        lemma = (token.lemma_ or token.text).lower()
        # filtre stopwords seulement si include_stopwords == False
        if not include_stopwords and token.is_stop:
            continue
        if lemma in CLITIC_STOPWORDS:
            continue
        if len(lemma) < min_len:
            continue
        tokens.append(lemma)
    return tokens
# --------------------------------------------
# 6) Fréquences
# --------------------------------------------
def build_frequency_vector(
    text: str,
    include_stopwords: bool = False,
) -> Tuple[List[int], List[str]]:
    """
    Construit le vecteur des fréquences de mots lemmatisés (FR/EN).
    """
    tokens = preprocess_and_lemmatize(text, include_stopwords=include_stopwords)
    counter = Counter(tokens)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words = [w for w, _ in items]
    vector = [freq for _, freq in items]
    return vector, words

