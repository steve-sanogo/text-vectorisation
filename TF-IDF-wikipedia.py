from datasets import load_dataset
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from frequence import preprocess_and_lemmatize  # ta fonction

def load_small_wiki_corpus_fr(
    max_articles: int = 1000,
    dump_id: str = "20231101.fr",  # version FR de novembre 2023
):
    """
    Charge un petit corpus Wikipédia FR depuis Hugging Face.
    Retourne une liste de textes (un article = une string).
    """
    ds = load_dataset("wikimedia/wikipedia", dump_id, split="train", streaming=True)
    corpus = []

    for i, example in enumerate(ds):
        # Chaque example a normalement un champ "text" avec le contenu de l'article
        corpus.append(example["text"])
        if i + 1 >= max_articles:
            break

    return corpus

def build_tfidf_vectors_wikipedia(
    text: str,
    wiki_corpus: List[str],
    top_n: int = 20,
) -> Tuple[List[str], List[float]]:
    """
    Calcule les scores TF-IDF de `text` par rapport à un corpus Wikipédia.
    """
    # 1) Apprendre IDF sur le corpus Wikipédia
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_and_lemmatize,
        lowercase=False,
        token_pattern=None,
    )
    vectorizer.fit(wiki_corpus)

    # 2) Appliquer au texte utilisateur
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(X.toarray()).ravel()

    idx_sorted = np.argsort(-scores)[:top_n]
    top_words = [feature_names[i] for i in idx_sorted]
    top_scores = [float(scores[i]) for i in idx_sorted]
    return top_words, top_scores