from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from vectorizer import build_frequency_vector
import json
from similarity import similarity_score
# from wiki import (
#     load_small_wiki_corpus_fr,
#     build_tfidf_on_wikipedia_corpus,
# )
from summary import (
    summarize_first_sentences,
    summarize_most_similar_ranked,
    summarize_most_similar_in_text_order,
)

from wiki import load_small_wiki_corpus_fr, build_tfidf_on_wikipedia_corpus

from snippets import find_best_snippet, find_top_k_snippets


# texte = """L'intelligence artificielle est un domaine fascinant de l'informatique.
#     L'intelligence artificielle permet de créer des systèmes capables d'apprendre
#     et de résoudre des problèmes complexes. Les applications de l'intelligence
#     artificielle sont nombreuses : reconnaissance d'images, traitement du langage
#     naturel, véhicules autonomes, et bien plus encore. L'apprentissage automatique
#     est une branche importante de l'intelligence artificielle."""
# vec, mots = build_frequency_vector(texte)


app=Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False


@app.route("/tfidf",methods=['POST'])
def hello():
    data = request.get_json()
    message = data.get('message')
    vec, mots = build_frequency_vector(message)
    dictionnaire=dict(zip(mots,vec))
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )

@app.route("/similarite",methods=['POST'])
def similarite():
    data = request.get_json()
    texte1 = data.get('text1')
    texte2 = data.get('text2')
    score = similarity_score(texte1,texte2,ngram_range=(1,2))
    dictionnaire={"similarity":score}
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )

@app.route("/resumer_simple",methods=['POST'])
def resumerSimple():
    data = request.get_json()
    texte = data.get('message')
    pourcentage = float(10) if data.get('pourcentage')=="" else float(data.get('pourcentage'))
    print("dssssssssssssss",pourcentage)
    resumer=summarize_first_sentences(texte,pourcentage)

    dictionnaire={"resumer":resumer}
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )

@app.route("/resumer_par_ordre",methods=['POST'])
def resumerParOrdre():
    data = request.get_json()
    texte = data.get('message')
    pourcentage = float(10) if data.get('pourcentage')=="" else float(data.get('pourcentage'))
    print("dssssssssssssss",pourcentage)
    resumer=summarize_most_similar_ranked(texte,pourcentage)

    dictionnaire={"resumer":resumer}
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )

@app.route("/resumer_par_similarite",methods=['POST'])
def resumerParSimilarite():
    data = request.get_json()
    texte = data.get('message')
    pourcentage = float(10) if data.get('pourcentage')=="" else float(data.get('pourcentage'))
    print("dssssssssssssss",pourcentage)
    resumer=summarize_most_similar_in_text_order(texte,pourcentage)

    dictionnaire={"resumer":resumer}
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )


@app.route("/TF_IDF",methods=['POST'])
def TfIDF():

    data = request.get_json()
    texte = data.get('message')

    wiki_corpus = load_small_wiki_corpus_fr(max_articles=200)
    top_words, top_tfs, top_idfs, top_tfidfs = build_tfidf_on_wikipedia_corpus(
        texte,        # 1er argument : text
        wiki_corpus,  # 2e argument : wiki_corpus
        top_n=30,
    )

    dictionnaire={"mot":top_words,"tf":top_tfs,"idf":top_idfs,"TF_IDF":top_tfidfs}
    # print(dictionnaire)
    return Response(
        json.dumps(dictionnaire, sort_keys=False),
        mimetype='application/json'
    )

############################
@app.route("/best_snippet", methods=['POST'])
def get_best_snippet():
    """
    Attend un JSON :
    {
        "document": "Texte complet...",
        "query": "Mots clés...",
        "window_size": 50 (optionnel),
        "use_tfidf": true (optionnel)
    }
    """
    data = request.get_json()
    
    # Récupération des données avec valeurs par défaut
    document = data.get('document', '')
    requete = data.get('query', '')
    fenetre = data.get('window_size', 50)
    use_tfidf = data.get('use_tfidf', True) # Par défaut on utilise TF-IDF

    # Appel de ta fonction logique
    snippet, pos, score = find_best_snippet(
        document_text=document,
        query_text=requete,
        window_size=fenetre,
        use_tfidf=use_tfidf,
        tfidf_top_n=200
    )

    # Construction du dictionnaire de réponse
    resultat = {
        "type": "best_snippet",
        "method": "TF-IDF" if use_tfidf else "Simple Count",
        "position_debut": pos,
        "score": round(score, 4),
        "snippet": snippet
    }

    return Response(
        json.dumps(resultat, sort_keys=False, ensure_ascii=False),
        mimetype='application/json'
    )

# ---------------------------------------------------------
# 2. Route pour les TOP K SNIPPETS (Phrases clés)
# ---------------------------------------------------------
@app.route("/top_k_snippets", methods=['POST'])
def get_top_k_snippets():
    """
    Attend un JSON :
    {
        "document": "Texte complet...",
        "query": "Mots clés...",
        "k": 5 (optionnel)
    }
    """
    data = request.get_json()

    document = data.get('document', '')
    requete = data.get('query', '')
    k_val = data.get('k', 5)
    max_words = data.get('max_words', 50)

    # Appel de ta fonction logique
    top_snippets = find_top_k_snippets(
        document_text=document,
        query_text=requete,
        k=k_val,
        max_words=max_words,
        use_tfidf=True,
        tfidf_top_n=200
    )

    # Transformation de la liste de tuples en liste de dictionnaires pour le JSON
    # top_snippets est une liste de tuples : (texte, index, score)
    resultats_formates = []
    
    if top_snippets:
        for snippet_text, idx_phrase, score in top_snippets:
            resultats_formates.append({
                "index_phrase": idx_phrase,
                "score": round(score, 4),
                "texte": snippet_text
            })

    response_data = {
        "requete": requete,
        "nombre_resultats": len(resultats_formates),
        "resultats": resultats_formates
    }

    return Response(
        json.dumps(response_data, sort_keys=False, ensure_ascii=False),
        mimetype='application/json'
    )

# print(" Chargement d'un petit corpus Wikipédia FR depuis Hugging Face...")
# wiki_corpus = load_small_wiki_corpus_fr(max_articles=200)

# print(f"✅ Corpus chargé avec {len(wiki_corpus)} articles.")

# # 2) Calculer les TF-IDF sur ce corpus
# print("⚙️ Calcul des TF-IDF sur le corpus Wikipédia...")
# top_words, top_tfs, top_idfs, top_tfidfs = build_tfidf_on_wikipedia_corpus(
# wiki_corpus,
# top_n=30,
# )

# for w, tf, idf, tfidf in zip(top_words, top_tfs, top_idfs, top_tfidfs):
#     print(f"{w:25s}  TF={tf:.4f}  IDF={idf:.4f}  TF-IDF={tfidf:.4f}")

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)