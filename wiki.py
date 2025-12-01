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

def build_tfidf_on_wikipedia_corpus(
    wiki_corpus: List[str],
    top_n: int = 20,
) -> Tuple[List[str], List[float]]:
    """
    Calcule les TF-IDF des mots SUR le corpus Wikipédia.
    On ne regarde pas un texte utilisateur, juste Wikipédia.
    On renvoie les mots les plus importants dans le corpus wiki.
    """

    # 1) TF + IDF apprises sur Wikipédia (et appliquées sur Wikipédia)
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_and_lemmatize,
        lowercase=False,
        token_pattern=None,
    )

    # Ici on utilise fit_transform parce qu'on travaille UNIQUEMENT sur wiki
    X = vectorizer.fit_transform(wiki_corpus)

    feature_names = vectorizer.get_feature_names_out()

    # 2) On fait la moyenne des scores TF-IDF sur tous les articles wiki
    scores = np.asarray(X.mean(axis=0)).ravel()

    # 3) On prend les top_n mots les plus importants
    idx_sorted = np.argsort(-scores)[:top_n]

    top_words = [feature_names[i] for i in idx_sorted]
    top_scores = [float(scores[i]) for i in idx_sorted]

    return top_words, top_scores
