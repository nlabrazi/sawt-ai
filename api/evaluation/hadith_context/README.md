# Diagnostic « la couronne » et correction du découpage

Le hadith demandé par l’utilisateur concerne les parents qui reçoivent une couronne de lumière grâce à l’apprentissage et à la mise en pratique du Coran par leur enfant. Il n’a pas été retrouvé dans les 1 790 fiches de notre instantané. Cela ne conclut ni à son absence de toutes les sources ni à son authenticité.

## Pourquoi les résultats 58226 et 4181 remontaient

Le classement conserve le meilleur score de chaque hadith parmi ses passages. Le découpage au budget de tokens laissait parfois une fin de texte presque vide ou une attribution seule dans un passage indépendant. Ces passages sans contexte recevaient des scores élevés. Le défaut vient de notre préparation des documents.

| ID | Passage gagnant exact avant correction | Cosinus | Rang avant | Rang après |
| --- | --- | ---: | ---: | ---: |
| 58226 | `passage:  »` | 0.824137 | 2 | 1576 |
| 4181 | `passage: . » Hadith rapporté par al-Bukhârî et Muslim.` | 0.823883 | 3 | 196 |

Ces scores sont des similarités, pas des probabilités que le résultat soit correct. Le résultat 3413 remonte par son titre et sa catégorie ; le lien entre « couronne » et « trône » est une interprétation plausible de cette proximité, pas une explication prouvée du modèle.

## Correction

La stratégie `multi_context` rééquilibre les deux derniers segments quand la fin contient moins d’un quart de la fenêtre de tokens. Les citations et références restent avec du contexte. Aucun caractère de la source n’est supprimé et tous les passages restent sous 512 tokens. Les hadiths courts complets restent conservés.

434 passages ont été réencodés pour 216 hadiths ; les 4 369 embeddings inchangés ont été réutilisés. Le modèle E5-base, le tokenizer de segmentation, le corpus, les IDs attendus et le maximum par hadith sont inchangés. Les amorces de recherche sont nettoyées dans les deux branches.

| Mesure, mêmes 20 requêtes | Avant | Après |
| --- | ---: | ---: |
| Top-1 | 70% | 85% |
| Top-3 | 95% | 95% |
| Latence moyenne à chaud, hors HTTP | 25.89 ms | 24.61 ms |
| Documents tronqués | 0 | 0 |

Les associations attendues restent provisoires et non validées humainement. La requête « Le conseil de parler seulement pour dire du bien sinon se taire » reste en échec : 5437 passe du rang 91 au rang 87. Aucun nouveau modèle ni filtre de pertinence n’a été ajouté.

## Couverture et limite restante

La recherche exacte de « couronne » / « couronnes » dans les champs français et de تاج et ses formes nominales dans les champs arabes du corpus ne trouve aucune fiche. Le seul radical français proche est « couronné de succès », dans un texte sans rapport avec les parents et le Coran. Les recherches complémentaires sur le site HadeethEnc n’ont pas permis d’identifier une fiche cible ; elles ne prouvent pas l’absence sur tout le site.

La correction retire les deux résultats parasites du Top 3 mais ne rend pas la requête « la couronne » satisfaisante. Le moteur renvoie encore trois voisins même si aucun ne correspond. Ce cas reste un test de couverture hors des pourcentages de réussite. Il faut encore traiter les requêtes sans réponse dans la collection et disposer du texte attendu dans une source autorisée.

Après correction :

1. [3413](https://hadeethenc.com/fr/browse/hadith/3413) — (« Al-Kursî ») est, comparé au trône, tel un anneau de fer jeté dans une terre déserte.
2. [3601](https://hadeethenc.com/fr/browse/hadith/3601) — Allah m’a ordonné de te réciter la sourate : « Al-Bayyinah » (la Preuve Évidente).
3. [4568](https://hadeethenc.com/fr/browse/hadith/4568) — Chaque jour où le soleil se lève, la personne doit s’acquitter d’une aumône

## Vérifier et reproduire

Depuis le dossier `api`, le lanceur emploie désormais le découpage corrigé :

```bash
bash scripts/search_hadith.sh "Je cherche le hadith sur la couronne"
bash scripts/search_hadith.sh --variant benchmark-original "Je cherche le hadith sur la couronne"
```

Pour reconstruire la correction et ses mesures dans l’environnement Docker du lanceur :

```bash
.cache/hadith-cli-venv/bin/python scripts/evaluate_hadith_context.py
```

Les anciens rapports et matrices sont conservés. [comparison.json](comparison.json) contient les passages gagnants exacts avant/après, les scores, les rangs des deux résultats parasites et les autres essais libres. [before.json](before.json) et [le rapport corrigé](../hadith_retrieval/multilingual-e5-base_multi_context.json) conservent le diagnostic des vingt requêtes et les audits de tokens.
