from frequence import build_frequency_vector

texte = """L'intelligence artificielle est un domaine fascinant de l'informatique.
    L'intelligence artificielle permet de créer des systèmes capables d'apprendre
    et de résoudre des problèmes complexes. Les applications de l'intelligence
    artificielle sont nombreuses : reconnaissance d'images, traitement du langage
    naturel, véhicules autonomes, et bien plus encore. L'apprentissage automatique
    est une branche importante de l'intelligence artificielle."""
vec, mots = build_frequency_vector(
    text=texte,
    include_stopwords=False,  # on enlève les mots vides
                    
)

print("Mots  :", mots)
print("Vecteur :", vec)
