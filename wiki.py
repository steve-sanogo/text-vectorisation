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
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Calcule TF, IDF et TF-IDF des mots SUR le corpus Wikipédia.
    On renvoie les mots les plus importants du corpus wiki, avec :
      - TF moyen
      - IDF
      - TF-IDF moyen
    """

    # 1) TF + IDF apprises sur Wikipédia (et appliquées sur Wikipédia)
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_and_lemmatize,
        lowercase=False,
        token_pattern=None,
        norm=None,      # <--- important : pas de normalisation L2 automatique
        use_idf=True,
    )

    # fit_transform sur le corpus wiki -> TF * IDF
    X = vectorizer.fit_transform(wiki_corpus)

    feature_names = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_  # IDF de chaque mot

    # 2) Moyenne des scores TF-IDF sur tous les articles wiki
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()

    # 3) En déduire le TF moyen : TF = (TF-IDF) / IDF
    tf_mean = tfidf_mean / idf

    # 4) On prend les top_n mots selon TF-IDF moyen
    idx_sorted = np.argsort(-tfidf_mean)[:top_n]

    top_words   = [feature_names[i]      for i in idx_sorted]
    top_tfs     = [float(tf_mean[i])     for i in idx_sorted]
    top_idfs    = [float(idf[i])         for i in idx_sorted]
    top_tfidfs  = [float(tfidf_mean[i])  for i in idx_sorted]

    return top_words, top_tfs, top_idfs, top_tfidfs