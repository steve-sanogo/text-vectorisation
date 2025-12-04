from wiki import load_small_wiki_corpus_fr, build_tfidf_on_wikipedia_corpus

if __name__ == "__main__":
    print("Chargement d'un petit corpus Wikipédia FR depuis Hugging Face...")
    wiki_corpus = load_small_wiki_corpus_fr(max_articles=200)
    print(f" Corpus chargé avec {len(wiki_corpus)} articles.")

    # Ton texte à analyser
    texte = """L'intelligence artificielle est un domaine fascinant de l'informatique.
    L'intelligence artificielle permet de créer des systèmes capables d'apprendre
    et de résoudre des problèmes complexes. Les applications de l'intelligence
    artificielle sont nombreuses : reconnaissance d'images, traitement du langage
    naturel, véhicules autonomes, et bien plus encore. L'apprentissage automatique
    est une branche importante de l'intelligence artificielle."""

    print(" Calcul des TF / IDF / TF-IDF pour le texte (IDF Wikipédia)...")
    top_words, top_tfs, top_idfs, top_tfidfs = build_tfidf_on_wikipedia_corpus(
        texte,        # 1er argument : text
        wiki_corpus,  # 2e argument : wiki_corpus
        top_n=30,
    )

    print("\n Mots du texte avec TF (texte), IDF (wiki) et TF-IDF :")
    for w, tf, idf, tfidf in zip(top_words, top_tfs, top_idfs, top_tfidfs):
        print(f"{w:25s}  TF={tf:.4f}  IDF={idf:.4f}  TF-IDF={tfidf:.4f}")