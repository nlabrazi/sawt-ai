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

## Plan des étapes suivantes

1. **Traduction pilote.** Ajouter `api/scripts/import_quran_translation.py`,
   `app/services/quran_translation_service.py` et le snapshot local
   `api/assets/quran_translation_fr.json`. Importer Al-Fatiha et un petit
   échantillon d'Al-Baqara, par exemple 2:1–5 et 2:255. Vérifier les couples
   sourate/verset contre le catalogue, les doublons et les métadonnées. Garder
   les réponses source et la version obtenue au moment de l'import.
2. **Sources et import tafsir.** Identifier une édition fiable pour chacun des
   deux ouvrages et ses conditions de réutilisation. Conserver le texte source
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

Le choix des éditions sources des tafsirs et des éditions physiques françaises
utilisées pour la review reste à préciser avant leur import. Aucun contenu
religieux fictif n'est ajouté aux données du projet.

## Validation progressive

Tests de l'étape 1, depuis la racine :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/schemas/test_quran_content.py
```

Les tests couvrent la conservation du texte et de la provenance, les sources
permises, les imports obligatoirement en attente et la cohérence statut/date.
Les textes des fixtures sont explicitement fictifs.

Les étapes suivantes ajouteront les tests des couples sourate/verset, de la
séparation effective des sources au stockage, de la transition manuelle et du
filtrage public. Tester le filtrage avec un stockage contenant à la fois des
entrées `need_review` et `verified`, et vérifier toute la réponse HTTP publique.
Les tests de modèles ne remplacent pas ces tests de service et d'API.

Commit proposé pour cette première étape :
`feat: add translation and tafsir data models`.
