# Hadith Search V0 — validation du moteur

## Périmètre et source

Recherche sémantique française dans HadeethEnc, sans génération de contenu.
Le client utilise uniquement l'[API officielle](https://documenter.getpostman.com/view/5211979/TVev3j7q).
Les catégories racines sont paginées puis les IDs sont dédupliqués. Chaque fiche
fournit directement `hadeeth`, `hadeeth_ar`, `attribution`, `grade`, `explanation`
et `hints`. Aucun rapporteur ou numéro de recueil n'est déduit d'un texte.
Les chaînes affichées restent celles de la source, y compris ses éventuelles coquilles.

## Essayer une phrase depuis le terminal

Depuis le dossier `api` (Docker doit être démarré) :

```bash
bash scripts/search_hadith.sh "Je cherche le hadith sur la colère"
```

Depuis la racine `sawt-ai`, utiliser `bash api/scripts/search_hadith.sh`.
Remplacer uniquement la phrase entre guillemets pour faire un autre essai.
Le lanceur affiche trois titres classés et leurs liens officiels : ouvrir les
liens permet de lire les hadiths et de vérifier s'ils répondent à la demande.
Le lanceur retire désormais les amorces reconnues comme « Je cherche le hadith sur »
ou « Donnez moi hadith qui parle de », puis affiche `Recherche utilisée` lorsque
la phrase a changé. Il conserve les détails et les négations du sujet. Les formes
inconnues ou incomplètes restent intactes. `--raw-query` permet de comparer avec
la phrase d'origine ; `--json` affiche aussi les textes complets, la phrase utilisée
et les scores de diagnostic.

Cette commande utilise **E5-base avec le découpage corrigé**, avec
les modèles et la matrice déjà présents dans `.cache`. La [correction du découpage](hadith_context/README.md) empêche les fins de citation et attributions isolées de devenir des passages indépendants. `--variant benchmark-original` permet de reproduire le découpage précédent. Elle contrôle les
empreintes du corpus et des passages avant de rechercher. Elle ne reconstruit
pas l'index. Pour une nouvelle machine dépourvue de ces caches, reproduire d'abord
les expériences décrites plus bas. Le modèle fonctionne localement ; les textes
des résultats sont récupérés sur HadeethEnc et nécessitent Internet.

Le lanceur crée un conteneur temporaire à partir de l'image API existante et un
environnement Python persistant dans `api/.cache/hadith-cli-venv`. La première
utilisation installe les dépendances de recherche nécessaires. Il réutilise le
PyTorch de l'image et laisse les dépendances du serveur API en place. Le conteneur
est supprimé à la fin ; les dépendances et les modèles en cache sont conservés.
Le serveur API et le site n'ont pas besoin d'être lancés pour cet essai.

Le Python système (`py` ou `python`) peut manquer de bibliothèques comme
`pydantic` : utiliser le lanceur ci-dessus évite de les installer sur l'hôte.
Depuis `api`, le chemin Python serait `scripts/search_hadith.py`, sans répéter
`api/`, mais cette invocation directe exige un environnement Python préparé.

Les 95 % en Top-3 ne couvrent que les 20 phrases préparées et leurs labels
provisoires. Sans nettoyage, « Je cherche le hadith sur la colère » renvoyait
4181, 8266, puis 3287. Avec le nettoyage, la recherche « la colère » renvoie
4709, 3743, puis 3287 : le conseil de ne pas se mettre en colère devient premier.
Le benchmark initial reste à 95 % en Top-3 et passe de 65 % à 70 % en Top-1.
Les textes indexés, le modèle et le classement sont inchangés. Les requêtes très
courtes peuvent toutefois rester ambiguës, et certaines donnent des résultats
moins précis après nettoyage (par exemple « la mère »).
La [comparaison avant/après](hadith_query_cleanup/README.md) conserve tous les
résultats, ainsi que 60 variations synthétiques des mêmes 20 sujets, sans modifier
les labels ni les rapports du benchmark initial.
Pour l'instant, un retour utile consiste simplement à conserver la phrase testée
et à indiquer si un des trois liens correspond à ce qui était recherché.

Après cette correction, les mêmes vingt requêtes avec nettoyage des amorces
atteignent 85 % en Top-1 et 95 % en Top-3. Ces chiffres restent provisoires.
La requête « couronne » reste sans réponse attendue identifiée dans le corpus ;
le moteur renvoie encore ses voisins les plus proches. Les résultats de la section
précédente décrivent l'étape antérieure au rééquilibrage des passages.

Pour une nouvelle machine, après avoir reconstruit les matrices de l'A/B initial,
exécuter `scripts/evaluate_hadith_context.py` dans l'environnement Python préparé
pour créer la matrice corrigée utilisée par le lanceur.

## Construire et essayer le moteur initial

Après installation de `api/requirements.txt`, depuis la racine :

```bash
python api/scripts/build_hadith_index.py
python api/scripts/search_hadith.py "Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère"
python api/scripts/evaluate_hadith_search.py --preview
```

Dans le conteneur :

```bash
docker compose exec -e HF_HOME=/app/.cache/huggingface -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 api python scripts/build_hadith_index.py
docker compose exec -e HF_HOME=/app/.cache/huggingface -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 api python scripts/search_hadith.py "Ne pas se mettre en colère"
```

La première construction télécharge le modèle et les fiches officielles.
`--fetch-only` récupère uniquement les fiches. Le cache permet de reprendre un
téléchargement interrompu ; `--refresh` recharge toutes les fiches lors d'une mise
à jour. Une erreur de récupération empêche de publier un index partiel.
Les fichiers du cache ne sont jamais utilisés comme contenu de résultat.

L'index contient un vecteur par ID, construit à partir du titre, du texte français,
de l'explication, des bénéfices et des catégories. E5 utilise `passage:` pour les
documents et `query:` pour les recherches, même en français. Les textes de plus
de 512 tokens sont tronqués par le modèle ; c'est une limite à mesurer avant toute
amélioration de la préparation du texte.

Les chemins et le modèle sont configurables dans `api/.env.example`. Hors Docker,
omettre les chemins `/app/...` pour utiliser les chemins relatifs au backend.
Le constructeur écrit `api/assets/hadith_index.npz` et `hadith_index_meta.json`.
Les deux fichiers doivent être distribués ensemble, puis le processus API redémarré
pour charger une nouvelle version. Ils sont ignorés par Git.
La checksum lie les métadonnées à la matrice ; le modèle, sa révision, la langue,
la dimension, les dates de récupération et l'empreinte des sources sont enregistrés.
L'API consultée n'expose pas de version globale du corpus : `corpus_version` reste
`null`, les dates de récupération ne sont pas présentées comme une version officielle.

## Relecture humaine du benchmark

`hadith_search_corpus.json` contient 20 formulations utilisateur. Les titres et
liens des `candidates_for_review` proviennent de fiches effectivement téléchargées.
Ces candidats ne constituent **pas** des labels validés par un humain.
Les probes de couverture, dont « couronne », sont séparées des cas à labelliser.

Pour chaque requête, lire les fiches officielles, vérifier les variantes acceptables,
puis renseigner :

```json
{
  "expected_hadeethenc_ids": ["ID_LU_ET_VALIDÉ"],
  "review_status": "human_reviewed",
  "reviewed_by": "nom du relecteur",
  "reviewed_at": "date ISO 8601"
}
```

Ne pas choisir les IDs attendus en fonction des résultats du moteur. Après relecture
de tous les cas :

```bash
python api/scripts/evaluate_hadith_search.py
```

Sans cette relecture, la commande refuse de calculer Top-1/Top-3 accuracy.
`--preview` liste les classements et leur latence sans accuracy. La latence mesurée
comprend l'encodage de la query et la similarité cosinus à chaud, hors HTTP HadeethEnc.
Le chargement initial est mesuré séparément. Le script `search_hadith.py` mesure
aussi la récupération des fiches officielles, avec le chargement initial sur la
première requête.

## Diagnostic demandé et comparaison A/B

Le diagnostic détaillé est dans [`hadith_retrieval/`](hadith_retrieval/).
Chaque rapport Markdown contient, pour **chaque échec Top-3**, la requête,
les IDs attendus provisoires, le Top 5 avec les scores, le rang de chaque
candidat attendu et son `search_text` exact. Le JSON contient ces informations
pour tous les cas, y compris les réussites, avec l'audit des tokens par section.

Les expériences utilisent le même instantané local figé, les mêmes 20 requêtes
et les mêmes IDs candidats. `--provisional-labels` est un choix explicite de
mesurer l'accord avec ces candidats ; cela ne marque aucun label comme validé
humainement et ne remplace pas la relecture religieuse. Les probes de couverture
restent hors des métriques tant qu'aucune réponse attendue n'est identifiée.

Dans l'environnement Python contenant les dépendances API :

```bash
python api/scripts/diagnose_hadith_retrieval.py --strategy original --reuse-original-index --provisional-labels
python api/scripts/diagnose_hadith_retrieval.py --strategy semantic_first --provisional-labels
python api/scripts/diagnose_hadith_retrieval.py --strategy multi --provisional-labels
python api/scripts/diagnose_hadith_retrieval.py --strategy multi --model intfloat/multilingual-e5-base --provisional-labels --compare-to api/evaluation/hadith_retrieval/multilingual-e5-small_multi.json
```

Pour reproduire les mesures CPU, utiliser quatre threads PyTorch/OpenMP/MKL
(`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`) et un thread OpenBLAS
(`OPENBLAS_NUM_THREADS=1`). Dans Docker, les scripts sont sous `scripts/` et
les rapports sous `evaluation/`, puisque le répertoire de travail est `/app`.
Le cache Hugging Face peut être conservé dans `/app/.cache/huggingface` via `HF_HOME`.
Les caches d'expérimentation restent exclus de l'image par `api/.dockerignore`.

Trois constructions sont comparées :

- `original` : titre, hadith, explication, enseignements, catégories ;
- `semantic_first` : titre, catégories, enseignements, hadith, explication ;
- `multi` : titre + catégories ; enseignements ; hadith + explication.

Pour `multi`, les groupes trop longs sont découpés en segments sous 512 tokens,
sans suppression de texte. Un hadith peut donc avoir plus de trois segments.
Chaque segment reçoit `passage:` ; la query reçoit `query:`. On calcule la
similarité cosinus, conserve le **maximum par ID HadeethEnc**, puis trie les IDs.
Il n'y a ni moyenne des scores, ni seuil, ni boost lexical.

`--compare-to` vérifie les empreintes du corpus, des documents exacts et du
benchmark avant d'encoder. Il reprend le tokenizer de segmentation de l'expérience
de référence : les tokenizers E5-small et E5-base diffèrent notamment sur l'espace
final du préfixe. Changer le tokenizer de segmentation aurait modifié les
documents et faussé l'A/B. Les textes figés sont ensuite encodés par le tokenizer
et le modèle de l'expérience courante ; aucun ne doit dépasser sa limite.

Les embeddings sont conservés dans `.cache/hadith_retrieval/` pour permettre de
relancer les diagnostics sans recalculer le corpus. Le cache est lié aux textes,
au modèle et à sa révision. La version initiale de l'index a été reproduite
numériquement pour tous les IDs candidats du benchmark avant de servir de référence.
Le calcul `sentence-transformers` a aussi été comparé à la moyenne masquée suivie
de normalisation de la documentation E5 : aucun écart sur les vecteurs contrôlés.

Ces expériences n'activent ni endpoint ni UI. Le constructeur d'index initial
et le modèle par défaut restent inchangés jusqu'au choix explicite de la variante.

## Première mesure locale avec E5-small

Le corpus téléchargé contient 1 790 fiches et 418 catégories. L'index compressé
occupe 2,41 Mio et le cache du modèle environ 471 Mio. Dans le conteneur existant,
les six distributions d'embedding installées occupent environ 124 Mio, dont
`tokenizers` remplace une version déjà présente. Ce n'est pas une mesure de l'écart
entre deux images Docker entièrement reconstruites. PyTorch et NumPy sont réutilisés.
`transformers==4.48.3` est fixé pour rester compatible avec `huggingface_hub==0.25.2`
et éviter une longue résolution de versions ; le serveur existant a été conservé
pendant ces essais, réalisés dans un environnement temporaire.

Sur les premières requêtes, la fiche `4709` est première avec l'apostrophe typographique
dans « quelqu’un », mais septième avec l'apostrophe droite dans « quelqu'un ».
Plusieurs paraphrases produisent également des résultats insuffisants. Le pipeline
fonctionne techniquement, mais **la qualité d'E5-small n'est pas validée**.
Ni seuil ni boost lexical n'a été ajouté. Les cas « couronne » et « actions après
la mort » demandent aussi une vérification humaine de la couverture du corpus.

## Tests sans accès réseau

```bash
api/.venv/bin/python -m pytest -c api/pytest.ini api/tests/services/test_hadeethenc_client.py api/tests/services/test_hadith_index.py api/tests/evaluation/test_hadith_search.py
```

Les tests mockent HTTP et le modèle ; ils n'appellent ni HadeethEnc ni Hugging Face.
