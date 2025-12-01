from similarity import similarity_score

t1 = "L'intelligence artificielle analyse des images médicales."
t2 = "L’IA est utilisée pour l’analyse d’images en médecine."

score = similarity_score(t1, t2, ngram_range=(1, 2))
print(f"Score de similarité : {score:.4f}")