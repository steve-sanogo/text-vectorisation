from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

#  Adapte ces imports à ton projet.
# On suppose que tu as déjà ces fonctions dans ton pipeline "frequence.py"
# - preprocess_and_lemmatize(text, include_stopwords=False) -> List[str] (tokens)
# - load_model_for_text(text) (si besoin côté TF-IDF Wikipédia)
from frequence import preprocess_and_lemmatize  # ton tokenizer spaCy + options



# -----------------------------
# Utilitaires
# -----------------------------
def _cosine_sim_dense(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity pour vecteurs denses (1D)."""
    # Éviter division par zéro
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (na * nb))


def _top_contributing_terms(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    feature_names: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Renvoie les termes les plus "contributeurs" à la similarité cosinus, via le
    produit terme-à-terme des poids TF-IDF : w_a[t] * w_b[t].
    """
    # Contribution par dimension
    contrib = vec_a * vec_b  # produit Hadamard
    if contrib.ndim > 1:
        contrib = contrib.ravel()

    idx = np.argsort(-contrib)
    out = []
    count = 0
    for i in idx:
        if contrib[i] <= 0.0:
            continue
        out.append((feature_names[i], float(contrib[i])))
        count += 1
        if count >= top_k:
            break
    return out


# -----------------------------
# Similarité TF-IDF (principal)
# -----------------------------
def text_similarity_tfidf(
    text_a: str,
    text_b: str,
    ngram_range: Tuple[int, int] = (1, 1),
    sublinear_tf: bool = True,
    use_idf: bool = True,
    smooth_idf: bool = True,
    prefit_vectorizer: Optional[TfidfVectorizer] = None,
    return_details: bool = False,
    top_terms: int = 10,
) -> Dict:
    """
    Calcule la similarité cosinus entre deux textes via TF-IDF + lemmatisation.

    - include_stopwords : conserve ou enlève les stopwords dans le tokenizer
    - ngram_range       : (1,1) pour unigrams, (1,2) pour uni+bi-grams, etc.
    - sublinear_tf      : log(1 + tf) si True -> réduit l'effet des mots très fréquents
    - use_idf/smooth_idf: config IDF
    - prefit_vectorizer : si fourni, on l'utilise (déjà fit sur un gros corpus, ex Wikipédia)
                          sinon on fit sur [text_a, text_b] (corpus local minimal)
    - return_details    : si True, retourne des infos sur les termes les plus contributeurs
    - top_terms         : nombre de termes contributeurs à renvoyer

    Retourne un dict :
    {
        "similarity": float in [0,1],
        "method": "tfidf_cosine",
        "details": {
            "top_terms": List[(term, contribution)],   # si return_details=True
            "feature_count": int,
            "ngram_range": (n1, n2),
            "include_stopwords": bool,
            "idf_source": "prefit" | "pair_fit"
        }
    }
    """
    # 1) Construire (ou réutiliser) le vectorizer
    if prefit_vectorizer is None:
        # Vectorizer fondé sur TON tokenizer
        vectorizer = TfidfVectorizer(
            tokenizer=lambda txt: preprocess_and_lemmatize(
                txt
            ),
            lowercase=False,      # déjà géré dans ton preprocess
            token_pattern=None,   # on passe un tokenizer custom -> ignorer le pattern par défaut
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            norm=None,            # on fera la normalisation à la main (pour détails)
        )
        X = vectorizer.fit_transform([text_a, text_b])
        idf_source = "pair_fit"
    else:
        vectorizer = prefit_vectorizer
        X = vectorizer.transform([text_a, text_b])
        idf_source = "prefit"

    # 2) Récup matrices -> vecteurs 1D denses
    #    (on normalise ensuite pour un cosinus propre)
    a = X[0].toarray().ravel()
    b = X[1].toarray().ravel()

    # 3) Normalisation L2 puis cosinus
    a = normalize(a.reshape(1, -1), norm="l2").ravel()
    b = normalize(b.reshape(1, -1), norm="l2").ravel()
    sim = _cosine_sim_dense(a, b)

    result = {
        "similarity": sim,
        "method": "tfidf_cosine",
    }

    if return_details:
        feats = vectorizer.get_feature_names_out()
        top = _top_contributing_terms(a, b, feats, top_k=top_terms)
        result["details"] = {
            "top_terms": top,
            "feature_count": int(len(feats)),
            "ngram_range": ngram_range,
            # "include_stopwords": include_stopwords,
            "idf_source": idf_source,
        }

    return result


# -----------------------------
# Baseline optionnelle (Jaccard)
# -----------------------------
def jaccard_similarity(
    text_a: str,
    text_b: str,
    include_stopwords: bool = False,
) -> float:
    """
    Similarité de Jaccard sur ensembles de lemmes (baseline interprétable).
    score = |A ∩ B| / |A ∪ B|
    """
    toks_a = set(preprocess_and_lemmatize(text_a, include_stopwords=include_stopwords))
    toks_b = set(preprocess_and_lemmatize(text_b, include_stopwords=include_stopwords))
    union = len(toks_a | toks_b)
    if union == 0:
        return 0.0
    return len(toks_a & toks_b) / union



def similarity_score(
    text_a: str,
    text_b: str,
    ngram_range: Tuple[int, int] = (1, 1),
) -> float:
    """
    Renvoie directement un score numérique de similarité TF-IDF (cosinus)
    entre deux textes, sans les détails.
    """
    res = text_similarity_tfidf(
        text_a,
        text_b,
        ngram_range=ngram_range,
        return_details=False,
    )
    return res["similarity"]