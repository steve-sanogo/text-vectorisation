# ===============================================
# test_snippets.py — tests pour les snippets
# ===============================================

from snippets import find_best_snippet, find_top_k_snippets


def afficher_resultats_best(titre, document, requete, fenetre):
    """
    Lance find_best_snippet avec et sans TF-IDF
    et affiche les résultats pour comparaison.
    """
    print("=" * 80)
    print(f"{titre} (fenêtre = {fenetre} mots)")
    print(f"Requête : {requete}")
    print()

    # -------- SANS TF-IDF --------
    snippet_simple, pos_simple, score_simple = find_best_snippet(
        document_text=document,
        query_text=requete,
        window_size=fenetre,
        use_tfidf=False,
    )

    print("[MEILLEUR SNIPPET SANS TF-IDF]")
    print(f"Position de début (mot) : {pos_simple}")
    print(f"Score : {score_simple:.4f}")
    print("Snippet :")
    print(snippet_simple)
    print()

    # -------- AVEC TF-IDF --------
    snippet_tfidf, pos_tfidf, score_tfidf = find_best_snippet(
        document_text=document,
        query_text=requete,
        window_size=fenetre,
        use_tfidf=True,
        tfidf_top_n=200,
    )

    print("[MEILLEUR SNIPPET AVEC TF-IDF]")
    print(f"Position de début (mot) : {pos_tfidf}")
    print(f"Score : {score_tfidf:.4f}")
    print("Snippet :")
    print(snippet_tfidf)
    print()
    print("-" * 80)
    print()


def afficher_top_k(titre, document, requete, k=5, max_words=50):
    """
    Affiche les k meilleurs snippets (phrases) avec TF-IDF.
    """
    print("=" * 80)
    print(f"{titre} — TOP {k} SNIPPETS (max {max_words} mots)")
    print(f"Requête : {requete}")
    print()

    top_snippets = find_top_k_snippets(
        document_text=document,
        query_text=requete,
        k=k,
        max_words=max_words,
        use_tfidf=True,
        tfidf_top_n=200,
    )

    if not top_snippets:
        print("Aucun snippet trouvé.")
        print()
        return

    for i, (snippet, idx_phrase, score) in enumerate(top_snippets, start=1):
        print(f"Snippet #{i} (index phrase = {idx_phrase}, score = {score:.4f})")
        print(snippet)
        print()


def get_texte_wiki_style_ia() -> str:
    return (
        "L'intelligence artificielle est un domaine de l'informatique qui vise à créer des systèmes "
        "capables de reproduire certains comportements considérés comme intelligents. Au fil des décennies, "
        "les chercheurs ont proposé de nombreuses approches différentes, allant des systèmes à base de règles "
        "jusqu'aux méthodes modernes d'apprentissage automatique et d'apprentissage profond. "
        "Les grandes bases de données en ligne, comme les encyclopédies collaboratives, fournissent aujourd'hui "
        "une quantité considérable de textes qui peuvent être utilisés pour entraîner des modèles statistiques. "
        "Dans ce contexte, la pondération TF-IDF a longtemps été une méthode de référence pour représenter les documents. "
        "Chaque document est vu comme un sac de mots, et l'importance relative de chaque terme est évaluée à partir "
        "de sa fréquence dans le document et de sa rareté dans l'ensemble du corpus. "
        "Cette représentation permet ensuite de comparer des textes à l'aide de mesures comme la similarité cosinus. "
        "Avec l'arrivée des modèles de langage de grande taille, de nouvelles représentations ont été proposées, "
        "mais TF-IDF reste encore utilisé pour des tâches de base, notamment lorsqu'on souhaite un modèle simple, "
        "interprétable et peu coûteux en calcul. "
        "Dans de nombreux projets pédagogiques, TF-IDF sert de point d'entrée pour comprendre la vectorisation des textes. "
        "Les étudiants peuvent ainsi expérimenter avec des résumés automatiques, la recherche de documents similaires "
        "ou encore la génération de snippets à partir d'articles longs. "
        "Un snippet bien choisi permet à l'utilisateur de se faire rapidement une idée du contenu d'un article "
        "sans avoir à lire l'intégralité du texte. "
        "Les moteurs de recherche exploitent cette idée depuis longtemps en affichant quelques lignes de contexte "
        "autour des mots-clés de la requête."
    )


def get_texte_wiki_style_climat() -> str:
    return (
        "Le changement climatique désigne la modification durable des paramètres statistiques du climat "
        "à l'échelle de la planète ou d'une région donnée. Depuis le début de l'ère industrielle, "
        "les activités humaines ont entraîné une augmentation rapide des émissions de gaz à effet de serre, "
        "notamment le dioxyde de carbone et le méthane. Ces gaz s'accumulent dans l'atmosphère et renforcent "
        "l'effet de serre naturel, ce qui provoque une hausse progressive de la température moyenne globale. "
        "De nombreux rapports scientifiques se basent sur des séries de données longues et sur des modèles "
        "numériques complexes afin de prévoir l'évolution du climat au cours des prochaines décennies. "
        "Les conséquences du réchauffement comprennent la fonte des glaciers, l'élévation du niveau des mers, "
        "l'augmentation de la fréquence des vagues de chaleur et des événements météorologiques extrêmes. "
        "Dans ce contexte, les textes de vulgarisation jouent un rôle important pour expliquer les mécanismes "
        "scientifiques au grand public. Les bibliothèques numériques et les encyclopédies en ligne regroupent "
        "un grand nombre d'articles consacrés aux enjeux climatiques, aux politiques publiques de réduction des émissions "
        "et aux stratégies d'adaptation. "
        "Pour analyser ces documents, on peut utiliser des techniques de traitement automatique des langues, "
        "comme la vectorisation par TF-IDF, afin de repérer les termes les plus caractéristiques. "
        "On peut ainsi extraire des résumés, comparer des rapports issus de différentes organisations, "
        "ou construire des snippets qui mettent en avant les phrases clés d'un article consacré au climat. "
        "Les décideurs et les citoyens peuvent alors naviguer plus efficacement dans une grande quantité de textes "
        "et identifier les informations essentielles sur le changement climatique."
    )


def test_articles_wikipedia_style():
    texte_ia = get_texte_wiki_style_ia()
    requete_ia = "tf-idf similarité cosinus représentation documents"

    texte_climat = get_texte_wiki_style_climat()
    requete_climat = "réchauffement climatique gaz effet serre tf-idf articles"

    # Meilleur snippet (fenêtre glissante)
    afficher_resultats_best(
        titre="Article style Wikipédia : intelligence artificielle",
        document=texte_ia,
        requete=requete_ia,
        fenetre=50,
    )

    afficher_resultats_best(
        titre="Article style Wikipédia : changement climatique",
        document=texte_climat,
        requete=requete_climat,
        fenetre=50,
    )

    # Top 5 phrases/snippets (par phrase)
    afficher_top_k(
        titre="Article IA",
        document=texte_ia,
        requete=requete_ia,
        k=5,
        max_words=50,
    )

    afficher_top_k(
        titre="Article climat",
        document=texte_climat,
        requete=requete_climat,
        k=5,
        max_words=50,
    )


if __name__ == "__main__":
    test_articles_wikipedia_style()