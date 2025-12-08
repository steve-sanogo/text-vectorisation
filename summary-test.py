# summary-test.py
from summary import (
    summarize_first_sentences,
    summarize_most_similar_ranked,
    summarize_most_similar_in_text_order,
)


text = """
L'intelligence artificielle (IA) transforme progressivement le monde dans lequel nous vivons. 
Elle est désormais utilisée dans de nombreux secteurs, tels que la santé, l'éducation, la finance et les transports. 
Dans le domaine médical, l'IA aide au diagnostic rapide des maladies et à la personnalisation des traitements. 
Les entreprises exploitent l'IA pour analyser de grandes quantités de données et optimiser leurs stratégies commerciales. 
Cependant, cette technologie soulève aussi des questions éthiques et sociales importantes. 
Par exemple, la protection des données personnelles et le risque de biais algorithmique sont des préoccupations majeures. 
L'éducation doit évoluer pour préparer les étudiants aux métiers de demain, où l'IA jouera un rôle central. 
Enfin, les gouvernements et les organisations internationales doivent mettre en place des régulations adaptées pour encadrer l'utilisation de l'IA.
"""



# Demander le pourcentage à l'utilisateur
percentage = float(input(
    "\nPourcentage de résumé (ex : 10, 20, 30) : "
))

print("\n========== MÉTHODE 1 : premières phrases ==========")
print(summarize_first_sentences(text, percentage=percentage))


print("\n========== MÉTHODE 2 : phrases les + similaires (TF-IDF + cosinus) ==========")
print(summarize_most_similar_ranked(text, percentage=percentage))

print("\n========== MÉTHODE 3 : phrases avec mots les + fréquents ==========")
print(summarize_most_similar_in_text_order(text, percentage=percentage))



