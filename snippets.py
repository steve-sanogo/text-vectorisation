# ===============================================
# snippets.py — recherche des meilleurs snippets (extraits)
# basé sur ta pipeline frequence.py + TF-IDF_score.py
# ===============================================

from __future__ import annotations
from typing import List, Tuple, Dict
import re

from frequence import preprocess_and_lemmatize
from TF_IDF_score import build_tfidf_vectors  # adapte si le nom du fichier change


# ------------------------------------------------
# 0. Découpage en phrases (très simple)
# ------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """
    Découpe un texte brut en phrases en se basant sur la ponctuation (. ? !).
    Ce n'est pas parfait comme un vrai modèle NLP, mais suffisant pour nos snippets.
    """
    if not text:
        return []

    # On découpe après ., ? ou ! suivis d'au moins un espace
    phrases = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    # On enlève les chaînes vides éventuelles
    return [p.strip() for p in phrases if p.strip()]


# ------------------------------------------------
# 1. Utilitaires : tokeniser / lemmatiser un texte
# ------------------------------------------------
def tokenize_text_for_snippet(
    text: str,
    include_stopwords: bool = False,
) -> List[str]:
    """
    Utilise ta fonction preprocess_and_lemmatize pour transformer le texte
    en une liste de lemmes propres (FR/EN).

    On s'en sert pour :
      - le texte des fenêtres (pour le scoring)
      - la requête (query) à chercher
    """
    return preprocess_and_lemmatize(
        text,
        include_stopwords=include_stopwords,
    )


# ------------------------------------------------
# 2. Construction d'un dictionnaire de poids TF-IDF
# ------------------------------------------------
def build_tfidf_weight_dict(
    text: str,
    top_n: int = 100,
) -> Dict[str, float]:
    """
    Utilise ta fonction build_tfidf_vectors pour calculer les mots les plus
    importants du texte + leur score TF-IDF moyen.

    Retourne un dict : {mot: score_tfidf_moyen}
    """

    words, tfs, idfs, tfidfs = build_tfidf_vectors(text, top_n=top_n)

    weight_dict: Dict[str, float] = {}
    for w, score in zip(words, tfidfs):
        weight_dict[w] = float(score)

    return weight_dict


# ------------------------------------------------
# 3. Fonctions de scoring pour une fenêtre
# ------------------------------------------------
def score_window_overlap(
    window_tokens: List[str],
    query_tokens: List[str],
) -> float:
    """
    Score très simple :
      = nombre de mots de la requête présents dans la fenêtre.
    """
    if not window_tokens:
        return 0.0

    query_set = set(query_tokens)
    return sum(1.0 for tok in window_tokens if tok in query_set)


def score_window_tfidf(
    window_tokens: List[str],
    query_tokens: List[str],
    tfidf_weights: Dict[str, float],
) -> float:
    """
    Score pondéré par TF-IDF :
      = somme des scores TF-IDF des mots qui sont à la fois
        dans la fenêtre ET dans la requête.
    """
    if not window_tokens:
        return 0.0

    query_set = set(query_tokens)
    score = 0.0
    for tok in window_tokens:
        if tok in query_set:
            score += tfidf_weights.get(tok, 0.0)
    return score


# ------------------------------------------------
# 4. Fonction principale : trouver LE meilleur snippet (TEXTE BRUT)
# ------------------------------------------------
def find_best_snippet(
    document_text: str,
    query_text: str,
    window_size: int = 50,
    include_stopwords: bool = False,
    use_tfidf: bool = True,
    tfidf_top_n: int = 100,
) -> Tuple[str, int, float]:
    """
    Trouve le "meilleur" snippet (= fenêtre de window_size mots BRUTS)
    dans document_text par rapport à la requête query_text.

    ⚠️ IMPORTANT :
      - On retourne un extrait en TEXTE BRUT (pas lemmatisé).
      - La lemmatisation est utilisée UNIQUEMENT pour calculer le score.

    Retourne :
      - best_snippet_raw : str  -> meilleur extrait brut
      - best_start_word  : int  -> index du premier mot (brut)
      - best_score       : float
    """

    # 0) Découper le document en mots BRUTS (pour l'extrait final)
    raw_words: List[str] = document_text.split()

    if not raw_words:
        return "", 0, 0.0

    # 1) Tokens lemmatisés de la requête (pour le scoring)
    query_tokens = tokenize_text_for_snippet(
        query_text,
        include_stopwords=False,  # on veut garder les mots informatifs
    )

    if not query_tokens:
        # pas de requête utile => on renvoie le début du texte brut
        snippet_words = raw_words[:window_size]
        best_snippet_raw = " ".join(snippet_words)
        return best_snippet_raw, 0, 0.0

    # 2) Prépare éventuellement les poids TF-IDF à partir du texte complet
    tfidf_weights: Dict[str, float] = {}
    if use_tfidf:
        tfidf_weights = build_tfidf_weight_dict(
            document_text,
            top_n=tfidf_top_n,
        )

    # 3) Si le texte a moins de mots que la fenêtre, on retourne tout
    if len(raw_words) <= window_size:
        snippet_words = raw_words
        snippet_raw = " ".join(snippet_words)

        # on calcule quand même un score pour info
        window_tokens = tokenize_text_for_snippet(
            snippet_raw,
            include_stopwords=include_stopwords,
        )
        if use_tfidf:
            score = score_window_tfidf(window_tokens, query_tokens, tfidf_weights)
        else:
            score = score_window_overlap(window_tokens, query_tokens)

        return snippet_raw, 0, score

    # 4) Fenêtre glissante SUR LES MOTS BRUTS,
    #    scoring SUR LES TOKENS LEMMATISÉS de chaque fenêtre.
    best_score = -1.0
    best_start_word = 0

    # on fait glisser la fenêtre sur raw_words
    for start in range(0, len(raw_words) - window_size + 1):
        end = start + window_size

        # extrait brut de la fenêtre (ce qu'on affichera SI c'est le meilleur)
        window_raw_words = raw_words[start:end]
        window_raw_text = " ".join(window_raw_words)

        # version lemmatisée de la fenêtre pour le scoring
        window_tokens = tokenize_text_for_snippet(
            window_raw_text,
            include_stopwords=include_stopwords,
        )

        if use_tfidf:
            score = score_window_tfidf(window_tokens, query_tokens, tfidf_weights)
        else:
            score = score_window_overlap(window_tokens, query_tokens)

        if score > best_score:
            best_score = score
            best_start_word = start

    # 5) Construire le meilleur snippet en TEXTE BRUT
    best_snippet_words = raw_words[best_start_word:best_start_word + window_size]
    best_snippet_raw = " ".join(best_snippet_words)

    return best_snippet_raw, best_start_word, best_score


# ------------------------------------------------
# 5. NOUVEAU : trouver les k meilleurs snippets (phrases)
# ------------------------------------------------
def find_top_k_snippets(
    document_text: str,
    query_text: str,
    k: int = 5,
    max_words: int = 50,
    include_stopwords: bool = False,
    use_tfidf: bool = True,
    tfidf_top_n: int = 100,
) -> List[Tuple[str, int, float]]:
    """
    Retourne les k meilleurs snippets sous forme de phrases.

    Approche :
      - on découpe le texte en phrases (split_sentences)
      - chaque phrase est tronquée à max_words mots
      - on calcule un score pour chaque phrase (overlap ou TF-IDF)
      - on retourne les k phrases avec le meilleur score

    Paramètres :
      - document_text : texte complet
      - query_text    : requête (mots-clés, phrase...)
      - k             : nombre de snippets à retourner (par ex. 5)
      - max_words     : nombre max de mots par snippet (par ex. 50)

    Retour :
      Liste de tuples (snippet_brut, index_phrase, score)
      triés par score décroissant.
    """

    # 0) Découpage en phrases brutes
    sentences = split_sentences(document_text)

    if not sentences:
        return []

    # 1) Tokens lemmatisés de la requête
    query_tokens = tokenize_text_for_snippet(
        query_text,
        include_stopwords=False,
    )
    if not query_tokens:
        # si la requête ne donne rien, on retourne les premières phrases brutes
        results: List[Tuple[str, int, float]] = []
        for idx, s in enumerate(sentences[:k]):
            words = s.split()
            snippet_raw = " ".join(words[:max_words])
            results.append((snippet_raw, idx, 0.0))
        return results

    # 2) TF-IDF global sur le texte si demandé
    tfidf_weights: Dict[str, float] = {}
    if use_tfidf:
        tfidf_weights = build_tfidf_weight_dict(
            document_text,
            top_n=tfidf_top_n,
        )

    # 3) On score chaque phrase
    scored_snippets: List[Tuple[str, int, float]] = []

    for idx, sent in enumerate(sentences):
        if not sent.strip():
            continue

        # limitation en nombre de mots
        sent_words = sent.split()
        if max_words is not None and len(sent_words) > max_words:
            snippet_raw = " ".join(sent_words[:max_words])
        else:
            snippet_raw = " ".join(sent_words)

        # tokens lemmatisés pour le scoring
        window_tokens = tokenize_text_for_snippet(
            snippet_raw,
            include_stopwords=include_stopwords,
        )

        if use_tfidf:
            score = score_window_tfidf(window_tokens, query_tokens, tfidf_weights)
        else:
            score = score_window_overlap(window_tokens, query_tokens)

        scored_snippets.append((snippet_raw, idx, score))

    # 4) On trie par score décroissant, puis par index de phrase
    scored_snippets.sort(key=lambda x: (-x[2], x[1]))

    # 5) On retourne les k meilleurs
    return scored_snippets[:k]