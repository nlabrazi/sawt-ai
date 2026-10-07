# Traduction française et tafsir : intégration progressive

## Architecture actuelle

| Donnée ou parcours | Emplacement | Comportement |
| --- | --- | --- |
| Versets arabes | `api/assets/quran_versets.json` | 114 sourates, 6 236 versets ; chargés en mémoire par `app/core/model_loader.py`. |
| Catalogue des sourates | `app/services/quran_catalog_service.py` | Métadonnées et nombre de versets réutilisables pour contrôler les références. |
| Reconnaissance | `app/services/inference_pipeline.py`, `app/schemas/recognize.py` | `POST /recognize` retourne une sourate et une plage `start_verse` / `end_verse`. Le résultat n'est pas persisté. |
| Résultat côté navigateur | `ui/app/composables/useRecognition.ts` | Appel `$fetch` vers FastAPI et résultat dans une `ref` Vue. |
| Tajwid | `api/assets/quran_tajwid.json`, `app/services/tajwid_service.py` | Snapshot local, puis URL de sauvegarde, puis AlQuran Cloud ; cache mémoire. |
| Détails d'un passage | `ui/app/components/VerseDetailsSheet.vue`, `ui/app/composables/useTajwid.ts` | Détails et tajwid chargés à la demande. |
| Stockage modifiable | `app/services/feedback_store.py` | Retours utilisateurs dans Supabase via REST ; aucune couche ORM ou connexion SQL. |
| Accès interne | Aucun actuellement | Une protection serveur est nécessaire avant d'activer la review. |

## Étape 1 : contrats de données

Cette étape ajoute uniquement des schémas Pydantic, leurs tests et ce plan.
Elle ne crée ni stockage, ni import, ni route, ni interface.

`QuranTranslation`, dans `app/schemas/quran.py`, conserve :

- `surah_id`, `ayah`, `text` ;
- `source`, `translator`, `version`, `source_url` ;
- `footnotes`, si la source fournit des notes.

Le texte et les notes sont conservés sans normalisation. Une traduction issue
d'une source validée n'a pas de statut de review.

`TafsirEntry`, dans `app/schemas/tafsir.py`, conserve :

- `surah_id`, `ayah`, `text_fr` ;
- `source` : uniquement `ibn_kathir` ou `as_saadi` ;
- `source_reference` : référence précise de la source, URL ou édition et page ;
- `version` : version du contenu source utilisée ;
- `status` : `need_review` par défaut ou `verified` ;
- `reviewed_at` : absent avant review, obligatoire avec fuseau horaire après validation.

`TafsirImportEntry` restreint ces deux derniers champs à `need_review` et `null`.
Tous les futurs imports et toutes les sorties de génération devront passer par
ce schéma. Un fichier d'import qui revendique `verified` sera rejeté.

Les enregistrements sont immuables et refusent les champs inconnus. Les futures
écritures devront reconstruire un modèle validé avec `model_validate`, plutôt
que modifier ses attributs ou utiliser `model_copy(update=...)`, qui ne valide
pas les modifications.

Les nouveaux contrats emploient `surah_id`, déjà utilisé par le tajwid, et `ayah`
pour désigner un verset individuel. Le frontend convertira `sourate_id` du
résultat de reconnaissance et parcourra sa plage de versets. Les modèles bornent
les identifiants ; l'existence du couple exact sera contrôlée lors de l'import
et des lectures via le catalogue actuel, sans dupliquer la liste des sourates.

Un modèle interne contenant `verified` ne prouve pas à lui seul une validation
humaine. Le contrôle d'accès, la transition et le filtrage public seront assurés
par les services et routes des étapes suivantes.

## Étape 2 : import pilote de la traduction

Le snapshot `api/assets/quran_translation_fr.json` contient 13 traductions de
Rachid Maach, fournies par QuranEnc : Al-Fatiha 1:1–7, Al-Baqara 2:1–5 et 2:255.
La version importée est `1.0.3`, lue depuis les métadonnées du fournisseur.

Son format comporte :

- `meta` : version du format, date UTC d'import, source, traducteur, clé QuranEnc,
  version de traduction, lien vers les conditions et réponse JSON de métadonnées ;
- `source_responses` : URL et réponse JSON d'origine pour chacun des 13 versets ;
- `translations` : entrées `QuranTranslation` prêtes à être lues localement.

Les réponses d'origine conservent également les autres informations retournées
par QuranEnc, notamment le texte arabe. Le texte français et les notes ne sont
ni réécrits, ni nettoyés, ni complétés.

Depuis la racine du dépôt, pour actualiser ce pilote :

```bash
api/.venv/bin/python api/scripts/import_quran_translation.py
```

Le script télécharge uniquement ces 13 versets et leurs métadonnées. Il contrôle
que chaque réponse correspond au verset demandé et que les références existent
dans le catalogue local. Il relit les métadonnées après téléchargement pour
rejeter un changement de version ou de date de mise à jour pendant l'import.
Le fichier est remplacé atomiquement après validation complète ; un échec
réseau ou une réponse invalide conserve le snapshot précédent.

Pour écrire dans un autre emplacement :

```bash
api/.venv/bin/python api/scripts/import_quran_translation.py --output /tmp/quran_translation_fr.json
```

`app/services/quran_translation_service.py` fournit
`fetch_quran_translations(surah_id, start_verse, end_verse)`. Le service valide
le snapshot entier avant de le mettre en cache. Il refuse les doublons, les
références invalides et les divergences de provenance avec les métadonnées.
La lecture utilise le couple `(surah_id, ayah)` et retourne les entrées présentes
dans l'ordre des versets. Un verset valide absent du pilote ne retourne aucune
traduction ; une référence invalide déclenche une erreur.

Le chemin est configurable avec `QURAN_TRANSLATION_PATH`. Une valeur vide utilise
le snapshot fourni dans `api/assets`. La lecture ne fait aucun appel réseau et
ne charge pas la traduction au démarrage de FastAPI. Après un nouvel import,
redémarrer le backend pour renouveler son cache ; les tests peuvent utiliser
`clear_quran_translation_cache()`.

Pour vérifier cette étape sans réseau :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/schemas/test_quran_content.py api/tests/services/test_quran_translation_service.py
```

Les tests contrôlent le couple sourate/verset, les plages partielles, la
conservation des textes et notes du snapshot livré, et la préservation du
fichier précédent en cas d'échec d'import. Les fixtures d'import sont fictives ;
le test du snapshot livré compare ses entrées aux réponses QuranEnc archivées.

[Les conditions de QuranEnc](https://quranenc.com/en/home/api) demandent notamment
de préserver le contenu, d'identifier la source et sa version, de conserver les
informations du document et de suivre les mises à jour. Les futures réponses
publiques et leur affichage devront conserver l'attribution et la version.
Ce jeu partiel sert à valider le pipeline ; ce n'est pas un corpus complet.

La route publique et l'affichage Nuxt font partie des prochaines étapes.
Commit proposé : `feat: import french Quran translation dataset`.

## Étape 3 : identification des sources tafsir

`app/services/tafsir_sources.py` référence deux ressources originales arabes
du catalogue Quran Foundation, contrôlées le 7 octobre 2026 :

| Source Sawt-AI | Ressource fournisseur | Slug | Consultation |
| --- | --- | --- | --- |
| `ibn_kathir` | `14` | `ar-tafsir-ibn-kathir` | [Ibn Kathir sur Quran.com](https://quran.com/al-fatihah/1/tafsirs/ar-tafsir-ibn-kathir) |
| `as_saadi` | `91` | `ar-tafseer-al-saddi` | [As-Sa‘di sur Quran.com](https://quran.com/al-fatihah/1/tafsirs/ar-tafseer-al-saddi) |

Les définitions sont immuables et séparées. Le résolveur de métadonnées exige
une ressource unique, avec l'identifiant entier, le slug et la langue attendus.
Un ouvrage absent, dupliqué ou remplacé provoque une erreur, sans choix implicite
d'une autre ressource. Il conserve les métadonnées retournées sans modification.
Le futur import devra appeler ce résolveur avant de télécharger du contenu.

Le paramètre `language=fr` du [catalogue officiel](https://api-docs.quran.com/docs/content_apis_versioned/4.0.0/tafsirs/)
traduit les **libellés**, pas le texte des tafsirs. La langue réelle est
`language_name`. La ressource anglaise `169`, « Ibn Kathir (Abridged) », ne
remplace pas la ressource arabe `14`. Les notes de la traduction QuranEnc
ne constituent pas non plus un tafsir Ibn Kathir ou As-Sa‘di.

### Conditions de conservation avant l'import

[Les conditions Quran Foundation](https://api-docs.quran.com/legal/developer-terms/),
datées du 4 octobre 2026, limitent le stockage des réponses ordinaires à une
semaine. Pour les ressources disponibles via Content Sync, la copie hors ligne
doit être obtenue et maintenue par ce mécanisme, avec synchronisation au moins
tous les sept jours lorsque la connexion est disponible. Un backend interne
destiné à l'affichage dans l'application est distingué d'une redistribution
de données. Une redistribution comme dataset ou service de données exige
une licence distincte.

En conséquence, cette étape ajoute le référencement et ses contrôles uniquement.
Elle n'archive aucun texte tafsir et ne crée aucun appel réseau au démarrage,
pendant la reconnaissance ou lors d'une lecture publique. Un import permanent
à partir des endpoints ordinaires ne convient pas au stockage prévu.

La prochaine étape doit intégrer Content Sync avec les accès fournisseur,
ou utiliser un corpus local dont la provenance et les droits de réutilisation
sont établis. Aucun texte religieux de test n'est livré comme donnée réelle.
Les éditions françaises physiques de review (éditeur, traducteur, année et
éventuelle version abrégée) restent à renseigner ; la version d'une édition
source n'est pas déduite de son identifiant API.

Le pilote reste Al-Fatiha 1:1–7 et Al-Baqara 2:1–5, 2:255. Si une source
regroupe plusieurs versets, conserver le groupe et son texte original :
ne pas inventer une découpe par verset. Toute future sortie française devra
passer par `TafsirImportEntry`, avec `need_review` et `reviewed_at = null`.

Vérification hors ligne :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/services/test_tafsir_sources.py api/tests/schemas/test_quran_content.py api/tests/services/test_quran_translation_service.py
```

Commit proposé : `feat: add Ibn Kathir and As-Saadi tafsir sources`.

## Plan d'intégration

1. **Traduction pilote — réalisée.** Ajouter `api/scripts/import_quran_translation.py`,
   `app/services/quran_translation_service.py` et le snapshot local
   `api/assets/quran_translation_fr.json`. Importer Al-Fatiha et un petit
   échantillon d'Al-Baqara, par exemple 2:1–5 et 2:255. Vérifier les couples
   sourate/verset contre le catalogue, les doublons et les métadonnées. Garder
   les réponses source et la version obtenue au moment de l'import.
2. **Sources identifiées ; import tafsir à réaliser.** Les deux ressources arabes
   Quran Foundation sont référencées. Confirmer les éditions françaises de
   review et choisir Content Sync ou un corpus local réutilisable pour l'import.
   Conserver le texte source
   et sa provenance dans des fichiers internes séparés. Commencer par l'import
   de français existant ; une éventuelle génération doit traduire uniquement
   le texte source fourni, sans enrichissement ni mélange entre ouvrages.
   Chaque sortie passe par `TafsirImportEntry`. Une réimportation ne doit jamais
   remplacer silencieusement un texte déjà relu.
3. **Persistance et validation.** Ajouter une table Supabase `tafsir_entries`,
   une contrainte unique `(surah_id, ayah, source)` et des contraintes de statut
   et date. Reprendre le mode d'accès REST de `feedback_store.py` dans un service
   dédié, sans couche générique ni ORM. L'accès anonyme direct à la table doit
   être interdit. Toute modification d'un texte validé impose une nouvelle
   review ; seule l'action interne « Valider » écrit `verified` et la date.
4. **Interface interne.** Ajouter des routes FastAPI protégées côté serveur
   pour lister, corriger et valider ; interface Nuxt avec filtres sourate,
   source et statut. Réutiliser `GET /surahs` pour le filtre. Décider du mécanisme
   d'authentification interne avant cette étape ; CORS ou une URL discrète ne
   constituent pas une protection. Aucun secret dans `runtimeConfig.public`.
5. **Lecture publique.** Ajouter une route de lecture par sourate et plage de
   versets, avec schéma de réponse dédié. Retourner la traduction locale et
   uniquement les tafsirs `status = verified`, filtrés dès la requête de stockage.
   Préserver la référence de chaque verset et chaque source. Aucun appel à la
   génération ou à un import lors d'une lecture publique. Ne pas conserver en
   cache un tafsir redevenu `need_review`.
6. **Affichage.** Étendre `VerseDetailsSheet.vue` avec un composable de lecture
   utilisant `$fetch` et `runtimeConfig.public.apiBaseUrl`. Afficher le texte
   arabe, la traduction et les onglets Ibn Kathir / As-Sa‘di pour chaque verset
   du passage. Aucun contenu tafsir lorsqu'une entrée validée manque.

Ce plan réutilise les schémas Pydantic, le catalogue, les snapshots locaux,
Supabase REST et le composant de détails déjà présents. La reconnaissance audio
ne dépendra pas du chargement des traductions ou tafsirs.

## Source de traduction retenue pour le pilote

[La documentation officielle QuranEnc](https://quranenc.com/en/home/api)
décrit la liste des traductions, les lectures par sourate et par verset, ainsi
que les champs `sura`, `aya`, `translation` et `footnotes`. Elle référence la
traduction française de Rachid Maach sous la clé `french_rashid`.
La version sera lue depuis les métadonnées du fournisseur, sans valeur inventée.

Les ressources arabes des tafsirs sont référencées à l'étape 3. Leur mécanisme
d'import et les éditions physiques françaises utilisées pour la review restent
à préciser avant d'importer le contenu. Aucun contenu religieux fictif n'est
ajouté aux données du projet.

## Validation progressive

Tests de l'étape 1, depuis la racine :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/schemas/test_quran_content.py
```

Les tests couvrent la conservation du texte et de la provenance, les sources
permises, les imports obligatoirement en attente et la cohérence statut/date.
Les textes des fixtures sont explicitement fictifs.

L'étape 2 ajoute les tests des couples sourate/verset pour les traductions.
Les étapes suivantes ajouteront les tests de la séparation effective des sources
tafsir au stockage, de la transition manuelle et du filtrage public.
Tester le filtrage avec un stockage contenant à la fois des
entrées `need_review` et `verified`, et vérifier toute la réponse HTTP publique.
Les tests de modèles ne remplacent pas ces tests de service et d'API.

Commit proposé pour cette première étape :
`feat: add translation and tafsir data models`.
