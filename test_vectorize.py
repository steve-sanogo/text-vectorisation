from frequence import build_frequency_vector
def test():
    # Petit texte de test
    text = """
    L'intelligence artificielle est un domaine fascinant de l'informatique.
    L'intelligence artificielle permet de créer des systèmes capables d'apprendre
    et de résoudre des problèmes complexes. Les applications de l'intelligence
    artificielle sont nombreuses : reconnaissance d'images, traitement du langage
    naturel, véhicules autonomes, et bien plus encore. L'apprentissage automatique
    est une branche importante de l'intelligence artificielle.
    """

    # Appel de ta fonction
    mots, tfs, idfs, tfidfs = build_frequency_vector(text, top_n=20)

    print("\nTop mots avec TF, IDF et TF-IDF :\n")
    for mot, tf, idf, tfidf in zip(mots, tfs, idfs, tfidfs):
        print(f"{mot:20s}  TF={tf:.4f}  IDF={idf:.4f}  TF-IDF={tfidf:.4f}")
