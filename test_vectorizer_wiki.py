from wiki import (
    load_small_wiki_corpus_fr,
    build_tfidf_on_wikipedia_corpus,
)
def testest():
    # 1) Charger un petit corpus Wikipédia (tu peux ajuster max_articles)
    print(" Chargement d'un petit corpus Wikipédia FR depuis Hugging Face...")
    wiki_corpus = load_small_wiki_corpus_fr(max_articles=200)

    print(f"✅ Corpus chargé avec {len(wiki_corpus)} articles.")

    # 2) Calculer les TF-IDF sur ce corpus
    print("⚙️ Calcul des TF-IDF sur le corpus Wikipédia...")
    top_words, top_tfs, top_idfs, top_tfidfs = build_tfidf_on_wikipedia_corpus(
    wiki_corpus,
    top_n=30,
)

    for w, tf, idf, tfidf in zip(top_words, top_tfs, top_idfs, top_tfidfs):
        print(f"{w:25s}  TF={tf:.4f}  IDF={idf:.4f}  TF-IDF={tfidf:.4f}")



testest()