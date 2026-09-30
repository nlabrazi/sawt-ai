# Diagnostic du retrieval Hadith — comparaison finale

**Conclusion : E5-base avec plusieurs embeddings par hadith est la meilleure variante de ces essais pour le Top-3 : 19/20.** Aucun endpoint ni UI n’a été activé. Le modèle et le constructeur par défaut n’ont pas été changés.

**Statut des labels : provisoires, non validés humainement.** Les chiffres mesurent l’accord avec les candidats explicites du corpus de relecture, pas une exactitude religieuse validée. Les 20 formulations ont été préparées pour ce diagnostic, elles ne proviennent pas de journaux utilisateurs.

## Résultats

| Variante | Top-1 | Top-3 | Latence moyenne | Embeddings | Documents tronqués |
| --- | ---: | ---: | ---: | ---: | ---: |
| [E5-small · document initial](multilingual-e5-small_original.md) | 50% (10/20) | 65% (13/20) | 10.12 ms | 1790 | 796 |
| [E5-small · champs sémantiques en tête](multilingual-e5-small_semantic_first.md) | 50% (10/20) | 60% (12/20) | 9.66 ms | 1790 | 796 |
| [E5-small · plusieurs embeddings](multilingual-e5-small_multi.md) | 65% (13/20) | 80% (16/20) | 10.61 ms | 4803 | 0 |
| [E5-base · mêmes embeddings textuels](multilingual-e5-base_multi.md) | 65% (13/20) | 95% (19/20) | 25.26 ms | 4803 | 0 |

Latence à chaud sur CPU, quatre threads PyTorch, mêmes dépendances : encodage de la query + cosinus + maximum par ID. Le téléchargement, le chargement du modèle et les appels HadeethEnc sont exclus. Les mesures finales ont été réalisées successivement à partir des matrices en cache, sans construction d’index concurrente.

## Ce que le diagnostic montre

1. Les préfixes `passage:` et `query:` sont présents partout. Les embeddings des IDs candidats reproduisent ceux de l’index initial. Le pooling et la normalisation de sentence-transformers donnent les mêmes vecteurs que la formule de référence E5 sur les entrées contrôlées (écart maximal : 0).
2. **796/1 790 documents dépassent 512 tokens** dans le format initial. L’audit y compte 1 096 entrées de catégorie, 529 enseignements et 117 explications entièrement hors de la fenêtre, en plus des sections partiellement coupées. Ces comptes portent sur les sections, pas sur des hadiths distincts.
3. Le réordonnancement protège les catégories et davantage d’enseignements, mais conserve 796 documents trop longs. Il ne résout pas les échecs : Top-3 de 65 % à 60 % ; le cas de l’arbre progresse, les cas du frère et des parents régressent.
4. Les passages séparés conservent tout le contenu sans troncature. Le regroupement utilise le maximum par ID, avec un seul résultat par ID. Le meilleur rang candidat passe notamment de 943 à 1 pour les intentions, de 13 à 1 pour la facilité, et de 55 à 2 pour la miséricorde avec E5-small. Cela soutient l’hypothèse qu’un vecteur unique mélange parfois trop de thèmes ; ce n’est pas une preuve d’une cause unique.
5. **La troncature n’explique pas tous les échecs.** Le document initial 4709 contient 471 tokens et 5437 en contient 500, mais ils échouent déjà. E5-base corrige le cas 4709 à documents identiques ; le cas 5437 reste difficile même avec tous ses textes disponibles.

## A/B contrôlé

- Corpus officiel figé : `0991a4f74ba6884361b26cc42fb2a6f2f7c4e106997ae8e1eb709b773ddd3ce5`.
- Requêtes et IDs candidats identiques : `5b2a39cbef4496b478c94ac7af066566c1b8ff1e2aa6bd857d599ef606322779`.
- Les 4 803 documents de l’A/B ont la même empreinte : `dbcbc905c6e296e7fae87f2f368970040328f1c5faff273026593fc20173ad84`.
- Segmentation figée avec le tokenizer E5-small ; encodage avec le tokenizer et le modèle de chaque variante. Le contrôle refuse tout changement de document ou dépassement de 512 tokens.
- E5-base utilise environ 1,1 Gio de cache modèle, contre 471 Mio pour E5-small. Les matrices restent locales et petites ; aucun service vectoriel externe n’est introduit.

## Échec restant avec E5-base

> Le conseil de parler seulement pour dire du bien sinon se taire

ID candidat attendu : **5437**, meilleur rang : **91**.

| Rang | ID | Cosinus | Document gagnant |
| --- | --- | ---: | --- |
| 1 | [6988](https://hadeethenc.com/fr/browse/hadith/6988) | 0.830918 | title_categories |
| 2 | [5835](https://hadeethenc.com/fr/browse/hadith/5835) | 0.826042 | title_categories |
| 3 | [65101](https://hadeethenc.com/fr/browse/hadith/65101) | 0.825603 | hints |
| 4 | [58226](https://hadeethenc.com/fr/browse/hadith/58226) | 0.818382 | hadith_explanation:3 |
| 5 | [5501](https://hadeethenc.com/fr/browse/hadith/5501) | 0.817751 | title_categories |

Le modèle classe plus haut d’autres textes liés à la parole, au silence ou aux comportements. Les trois groupes du candidat attendu tiennent tous sous la limite : ce cas ne peut pas être attribué à une coupure du texte. Le rapport détaillé montre les contenus exacts ; aucune reformulation de la query ni modification du label n’a été faite pour améliorer ce score.

Voir le [diagnostic complet E5-base](multilingual-e5-base_multi.md) pour les textes exacts et le Top 5 avec les titres. Les rapports JSON conservent également les détails des cas réussis.

## Rang du meilleur candidat pour chaque requête

| Requête | Initial | Champs en tête | Small multi | Base multi |
| --- | ---: | ---: | ---: | ---: |
| Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère | 7 | 22 | 7 | 1 |
| Le vrai fort c'est celui qui arrive à se contrôler quand il s'énerve | 1 | 1 | 1 | 1 |
| Un hadith qui dit que nos actes comptent selon l'intention qu'on avait | 943 | 270 | 1 | 3 |
| Il faut souhaiter pour son frère ce qu'on souhaite pour soi | 2 | 4 | 1 | 2 |
| Le conseil de parler seulement pour dire du bien sinon se taire | 232 | 151 | 451 | 91 |
| Qui mérite le plus ma bonne compagnie ? Il répond trois fois la mère | 1 | 1 | 1 | 1 |
| Une femme est pardonnée après avoir donné à boire à un chien assoiffé | 1 | 1 | 1 | 1 |
| Une femme punie parce qu'elle avait enfermé un chat sans le nourrir | 1 | 2 | 1 | 1 |
| Planter un arbre dont les animaux mangent rapporte une récompense | 7 | 1 | 3 | 2 |
| Ne pas mépriser les petits gestes de bien même accueillir quelqu'un avec le sourire | 2 | 1 | 1 | 1 |
| Je cherche le conseil de faciliter les choses et de ne pas faire fuir les gens | 13 | 14 | 1 | 1 |
| Qui est vraiment ruiné le jour du jugement malgré ses prières et ses bonnes actions ? | 1 | 1 | 2 | 1 |
| Trois choses accompagnent le mort à la tombe mais seules ses actions restent | 1 | 1 | 1 | 1 |
| Quelqu'un demande un conseil sur l'islam et on lui dit de croire puis de rester droit | 2 | 2 | 1 | 1 |
| Le hadith qui explique que la religion est la sincérité | 1 | 1 | 1 | 1 |
| Les meilleurs sont ceux qui apprennent le Coran et l'enseignent | 1 | 1 | 1 | 1 |
| La récompense des parents qui ont perdu trois enfants avant la puberté | 1 | 4 | 10 | 2 |
| Enlever un obstacle de la route fait partie des branches de la foi | 28 | 6 | 33 | 2 |
| Ne pas apprendre la religion pour se vanter devant les savants ou se disputer | 1 | 1 | 1 | 1 |
| Allah accorde sa miséricorde à ceux qui sont miséricordieux | 55 | 36 | 2 | 3 |

## Limites avant validation de la bêta

- Les labels restent à relire dans [`hadith_search_corpus.json`](../hadith_search_corpus.json). Certaines variantes proches peuvent être des réponses acceptables : par exemple 8871 apparaît dans le Top-3 Small pour les parents, alors que le label candidat actuel est 8875. Les résultats n’ont pas été relabellisés après observation.
- Les probes « parents recevant une couronne » et « bonnes actions après la mort » ne sont pas incluses dans ces taux : elles restent sans IDs attendus validés. Aucune occurrence de « couronne » n’a été trouvée dans les textes des 1 790 fiches téléchargées ; cela ne prouve pas l’absence du thème dans toute la source.
- Vingt cas ne suffisent pas à mesurer la qualité en conditions réelles. Relecture humaine, ajout de nouvelles formulations indépendantes et examen de 5437 restent nécessaires avant de déclarer la V0 validée.

Recommandation : conserver **E5-base + passages séparés + maximum par ID** comme candidat pour l’étape suivante. Le coût mesuré est environ 2,4 fois celui de Small en latence de recherche, avec un Top-3 supérieur de 15 points sur ce corpus provisoire. Le Top-1 reste identique. Pas de seuil, boost lexical, reranker ou génération ajouté.

## Reproduction et tests

Les commandes et le protocole se trouvent dans [`HADITH_SEARCH.md`](../HADITH_SEARCH.md). Les tests backend passent : **215 tests**. Les tests unitaires utilisent des doubles de modèle et de réseau ; les mesures ci-dessus utilisent les vrais modèles et l’instantané de contenu officiel.
