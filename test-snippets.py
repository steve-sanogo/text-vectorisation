# ===============================================
# test_snippets.py — petits tests pour find_best_snippet
# ===============================================

from snippets import find_best_snippet


def test_simple_fr():
    document = """
    Dans ce travail, nous appliquons des méthodes classiques de pondération
    comme le TF-IDF sur un grand corpus de textes. Le corpus choisi est
    Wikipedia en français, car il contient de très nombreux articles variés
    et mis à jour régulièrement. Nous comparons ensuite différentes mesures
    de similarité sur ces représentations vectorielles pour évaluer la
    pertinence des résumés générés.
    """

    query = "tf-idf wikipedia similarité"

    snippet, pos, score = find_best_snippet(
        document_text=document,
        query_text=query,
        window_size=30,      # ~30 tokens lemmatisés
        use_tfidf=True,      # utilise les poids TF-IDF
        tfidf_top_n=50,
    )

    print("===== TEST FR =====")
    print("Requête :", query)
    print("Position de début (tokens) :", pos)
    print(f"Score : {score:.4f}")
    print("Snippet lemmatisé :")
    print(snippet)
    print()


def test_simple_en():
    document = """
    Artificial intelligence is widely used to analyse medical images.
    In this work, we apply TF-IDF and cosine similarity on a collection
    of radiology reports to detect similar cases. Our experiments show
    that TF-IDF features combined with lemmatization significantly improve
    retrieval quality compared to simple bag-of-words.
    """

    query = "tf-idf cosine similarity medical images"

    snippet, pos, score = find_best_snippet(
        document_text=document,
        query_text=query,
        window_size=25,
        use_tfidf=True,
        tfidf_top_n=40,
    )

    print("===== TEST EN =====")
    print("Requête :", query)
    print("Position de début (tokens) :", pos)
    print(f"Score : {score:.4f}")
    print("Snippet lemmatisé :")
    print(snippet)
    print()


def test_without_tfidf():
    document = """
    Le TF-IDF est une mesure très utilisée pour pondérer les mots
    dans un texte. Cependant, dans certains cas simples, on peut se
    contenter de compter le nombre de mots de la requête présents
    dans une fenêtre donnée. Ce test illustre la version sans TF-IDF.
    """

    query = "tf-idf mots requête"

    snippet, pos, score = find_best_snippet(
        document_text=document,
        query_text=query,
        window_size=20,
        use_tfidf=False,   # ici on désactive TF-IDF
    )

    print("===== TEST SANS TF-IDF =====")
    print("Requête :", query)
    print("Position de début (tokens) :", pos)
    print(f"Score (overlap) : {score:.4f}")
    print("Snippet lemmatisé :")
    print(snippet)
    print()


if __name__ == "__main__":
    test_simple_fr()
    test_simple_en()
    test_without_tfidf()