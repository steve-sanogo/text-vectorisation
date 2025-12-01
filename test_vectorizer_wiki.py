
from wiki import (
    load_small_wiki_corpus_fr,
    build_tfidf_on_wikipedia_corpus,
)
if __name__ == "__main__":
    # 1) Charger un petit corpus Wikipédia (tu peux ajuster max_articles)
    print(" Chargement d'un petit corpus Wikipédia FR depuis Hugging Face...")
    wiki_corpus = load_small_wiki_corpus_fr(max_articles=200)

    print(f"✅ Corpus chargé avec {len(wiki_corpus)} articles.")

    # 2) Calculer les TF-IDF sur ce corpus
    print("⚙️ Calcul des TF-IDF sur le corpus Wikipédia...")
    top_words, top_scores = build_tfidf_on_wikipedia_corpus(
        wiki_corpus,
        top_n=30,  # nombre de mots à afficher
    )

    # 3) Afficher les résultats
    print("\n Mots les plus importants dans ce mini-corpus Wikipédia :")
    for w, s in zip(top_words, top_scores):
        print(f"{w:25s} → {s:.4f}")