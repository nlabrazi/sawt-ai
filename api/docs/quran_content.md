# Traduction française et tafsir : intégration progressive

## Architecture actuelle

| Donnée ou parcours | Emplacement | Comportement |
| --- | --- | --- |
| Versets arabes | `api/assets/quran_versets.json` | 114 sourates, 6 236 versets ; chargés en mémoire par `app/core/model_loader.py`. |
| Catalogue des sourates | `app/services/quran_catalog_service.py` | Métadonnées et nombre de versets réutilisables pour contrôler les références. |
| Reconnaissance | `app/services/inference_pipeline.py`, `app/schemas/recognize.py` | `POST /recognize` retourne une sourate et une plage `start_verse` / `end_verse`. Le résultat n'est pas persisté. |
| Résultat côté navigateur | `ui/app/composables/useRecognition.ts` | Appel `$fetch` vers FastAPI et résultat dans une `ref` Vue. |
| Tajwid | `api/assets/quran_tajwid.json`, `app/services/tajwid_service.py` | Snapshot local, puis URL de sauvegarde, puis AlQuran Cloud ; cache mémoire. |
| Détails d'un passage | `ui/app/components/VerseDetailsSheet.vue`, `ui/app/composables/useTajwid.ts`, `ui/app/composables/useQuranContent.ts` | Passage arabe reconnu, tajwid et contenu français chargés à la demande. |
| Contenu français public | `app/routes/quran_content.py` | `GET /quran/content` : traduction locale et tafsirs validés, regroupés par verset. |
| Stockage modifiable | `app/services/feedback_store.py` | Retours utilisateurs dans Supabase via REST ; aucune couche ORM ou connexion SQL. |
| Accès interne | `app/routes/tafsir_review.py`, `ui/app/components/TafsirReviewScreen.vue` | Mot de passe dédié côté backend ; écran `/internal/tafsir`. |

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

La lecture publique est décrite à l'étape 7 ; l'affichage Nuxt reste la
prochaine étape d'intégration.
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

## Étape 4 : import local des brouillons français

Le script `api/scripts/import_tafsir_fr.py` importe un **lot local déjà préparé**.
Il accepte du français existant ou un brouillon traduit à partir du passage
original fourni. Il ne télécharge rien et ne génère aucun texte. La génération
DeepL, ajoutée à l'étape 10, est une commande manuelle séparée ; aucun accès
Content Sync n'est actuellement configuré.

Cette étape prépare l'entrée du pipeline avant la persistance Supabase :

```text
lot local avec passages originaux et provenance
    → contrôle des références et du statut
    → nouveau snapshot interne need_review
    → persistance et review manuelle (étapes suivantes)
```

### Format du lot

Le modèle vide `api/examples/tafsir_import.example.json` indique les champs à
renseigner. Il est volontairement refusé tant que les textes et la provenance
sont vides ; il ne contient aucun faux tafsir.

Un lot contient une seule source (`ibn_kathir` ou `as_saadi`) et une seule
édition/version. Ses champs sont :

| Champ | Rôle |
| --- | --- |
| `schema_version` | Version du format : `1`. |
| `source` | Ouvrage du lot. |
| `source_language` | Langue du passage original : `ar` ou `fr`. |
| `source_edition` | Identification de l'édition, avec éditeur/traducteur lorsque disponibles. |
| `version` | Version réelle du contenu utilisé, identique dans les entrées. |
| `reuse_reference` | Référence aux conditions ou à l'autorisation de réutilisation du corpus local. |
| `entries` | Entre une et treize entrées, toutes dans le pilote. |
| `imported_at` | Date UTC écrite par le script ; une date fournie en entrée est remplacée. |

Chaque entrée reprend `TafsirImportEntry`, donc conserve `surah_id`, `ayah`,
`source`, `text_fr`, `source_reference`, `version`, `status` et `reviewed_at`.
`TafsirDraftImportEntry` ajoute :

- `source_text` : le passage original complet, sans nettoyage ni réécriture ;
- `source_surah_id`, `source_start_ayah`, `source_end_ayah` : sa sourate et sa plage.

Le passage doit couvrir le verset du brouillon et exister dans le catalogue
local. Les couples autorisés sont 1:1–7, 2:1–5 et 2:255. Un même verset ne peut
figurer deux fois dans le lot. Les champs inconnus et les sources/versions
mélangées sont refusés.

Lorsque `source_language = fr`, `text_fr` doit être identique à `source_text`.
Cet import préserve un texte français existant ; les corrections se feront
pendant la review. Lorsque la source est arabe, le français est fourni dans
le lot, avec le passage original utilisé pour sa préparation. Ces contrôles
vérifient les références et la provenance déclarées ; la fidélité de la
traduction reste à vérifier manuellement.

Si un commentaire original couvre plusieurs versets, répéter son passage
**complet** et ses bornes pour les entrées concernées. Le script ne découpe
pas ce commentaire et ne crée pas les entrées manquantes. Deux entrées déclarant
le même groupe doivent conserver la même référence et le même texte original.
Une sortie générée en dehors de Sawt-AI suit le même format, sans ajout issu
de connaissances externes et sans mélange entre ouvrages.

Le statut et la date de review sont contrôlés par `TafsirImportEntry` :
un statut absent devient `need_review`, un statut `verified` ou une date de
review non nulle font échouer l'import. Une source française existante reste
elle aussi obligatoirement en attente de relecture.

### Utilisation

Préparer un fichier par ouvrage dans `api/data/tafsir/inputs/`, à partir du
modèle, avec un contenu réel et réutilisable. Puis, depuis la racine :

```bash
api/.venv/bin/python api/scripts/import_tafsir_fr.py --input api/data/tafsir/inputs/ibn_kathir.json
api/.venv/bin/python api/scripts/import_tafsir_fr.py --input api/data/tafsir/inputs/as_saadi.json
```

Par défaut, chaque commande crée un fichier horodaté distinct dans
`api/data/tafsir/<source>/drafts-<date>.json`. Les versets sont ordonnés par
sourate/verset. Tous les textes, références et métadonnées sont conservés,
avec `need_review`, `reviewed_at = null` et la date d'import.

Pour choisir un nouveau fichier de sortie :

```bash
api/.venv/bin/python api/scripts/import_tafsir_fr.py --input api/data/tafsir/inputs/ibn_kathir.json --output /tmp/ibn_kathir-drafts.json
```

L'écriture publie atomiquement le fichier complet et refuse une destination
existante, même en cas d'import concurrent. Il n'y a pas d'option d'écrasement.
Un échec de validation ou d'écriture ne remplace donc jamais un fichier relu.
Les fichiers créés ont des permissions `0600`. Le répertoire interne est exclu
de Git et des images Docker ; en développement il peut rester accessible au
backend via le volume, mais aucune route ne le sert. Ne pas placer ces lots
dans les assets publics Nuxt.

Pour relancer un import, utiliser la sortie horodatée par défaut ou choisir un
nouveau fichier. Cela crée un autre lot, pas une mise à jour du contenu relu.
L'insertion en Supabase, ajoutée à l'étape 5, refuse également le remplacement
silencieux d'un verset déjà présent/validé.

Le script charge uniquement le catalogue coranique, sans Whisper ni modèles
de reconnaissance. Il ne change aucune route FastAPI, aucun résultat de
reconnaissance et aucun affichage public. Il ne dispense pas de respecter les
conditions de la source : en particulier, il ne transforme pas une réponse
Quran Foundation ordinaire en corpus local conservable durablement.

### Vérification

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/services/test_tafsir_import_service.py api/tests/services/test_tafsir_sources.py api/tests/schemas/test_quran_content.py api/tests/services/test_quran_translation_service.py
```

Les tests du pipeline utilisent uniquement des textes explicitement fictifs.
Ils vérifient les deux sources sur les mêmes références, la conservation des
passages groupés et de la provenance, les imports obligatoirement en attente,
le respect du pilote, l'absence d'écrasement et les échecs d'écriture.
Le corpus réel et les éditions françaises de review restent à fournir.

Commit proposé : `feat: add local French tafsir pilot import pipeline`.

## Étape 5 : stockage Supabase et validation manuelle

`supabase/tafsir_entries.sql` crée la table et ses protections.
`app/services/tafsir_store.py` reprend l'accès REST et les fonctions de
configuration Supabase déjà utilisées par le feedback, sans ORM ni client
générique supplémentaire. Les services ne sont pas encore reliés à des routes
FastAPI ou à une interface ; cette étape prépare leur stockage.

### Installer la table

Exécuter `supabase/tafsir_entries.sql` dans le SQL Editor du projet Supabase.
La migration a été testée sur PostgreSQL 14 isolé, y compris deux applications
successives. Elle n'a pas été exécutée sur le Supabase du projet par l'agent.

La clé primaire est `(surah_id, ayah, source)`. Chaque ligne conserve les champs
de `TafsirEntry`, plus :

- `provenance` : passage original complet, bornes, langue, édition, référence de
  réutilisation et date du snapshot ;
- `updated_at` : date de la dernière écriture, attribuée par PostgreSQL.

Le backend contrôle les références exactes contre le catalogue local ; la table
borne les identifiants et restreint les sources et statuts. La provenance et les
identifiants restent immuables après insertion. Le rôle serveur peut insérer,
lire, puis modifier seulement `text_fr` et `status`. La fonction déclenchée par
PostgreSQL assure ces règles :

- toute insertion commence avec `need_review` et sans date de review ;
- une validation sans changement de texte passe à `verified`, avec une date
  écrite par la base ;
- tout changement de texte remet l'entrée en `need_review` et efface sa date de
  review, même si la requête revendiquait aussi `verified`.

Les rôles `anon` et `authenticated` n'ont aucun droit sur la table. RLS est
activé sans politique publique. Il faut passer par le backend ; être connecté
à Supabase Auth ne donne pas, à lui seul, accès aux brouillons ou à leur validation.
Voir [les règles Supabase sur les droits et RLS](https://supabase.com/docs/guides/database/postgres/row-level-security).

### Charger un snapshot pilote

Utiliser un fichier produit par `import_tafsir_fr.py`, avec `imported_at`,
`need_review` et `reviewed_at = null`. Le nouveau script le revalide, conserve
la provenance de chaque entrée puis insère tout le lot en une seule requête
REST. Un couple sourate/verset/source déjà présent fait échouer **tout le lot**,
sans mise à jour ni écrasement du texte relu.

Avec Docker Compose, la configuration Supabase de `api/.env` est déjà chargée
et le répertoire interne du dépôt est monté dans `/app` :

```bash
docker compose exec api python scripts/store_tafsir_fr.py --input "/app/data/tafsir/ibn_kathir/drafts-<date>.json"
docker compose exec api python scripts/store_tafsir_fr.py --input "/app/data/tafsir/as_saadi/drafts-<date>.json"
```

Remplacer `<date>` par le nom du snapshot réel. Hors Docker, exporter
`SUPABASE_URL` et `SUPABASE_API_KEY` dans l'environnement, puis utiliser :

```bash
api/.venv/bin/python api/scripts/store_tafsir_fr.py --input "api/data/tafsir/ibn_kathir/drafts-<date>.json"
```

La table est fixée à `tafsir_entries`. Le service accepte une clé serveur
`sb_secret_...` ou un JWT `service_role`, et refuse une clé de client.
Les nouvelles clés opaques sont envoyées dans `apikey` uniquement ; les anciens
JWT serveur utilisent également `Authorization: Bearer`. Le décodage du rôle
d'un JWT contrôle son type, pas sa signature : Supabase authentifie la requête.
Voir [la documentation officielle des clés API](https://supabase.com/docs/guides/getting-started/api-keys).
Aucune clé ne doit être ajoutée à la configuration publique Nuxt.

### Opérations pour la future interface interne

`TafsirReviewEntry` contient le texte français, sa provenance et `updated_at`.
Les opérations sont :

| Fonction | Comportement |
| --- | --- |
| `list_tafsirs_for_review(...)` | Liste interne paginée, en attente par défaut ; filtres sourate, source et statut. |
| `update_tafsir_text(...)` | Sauvegarde le texte corrigé et impose une nouvelle review. |
| `verify_tafsir(...)` | Valide une entrée encore en attente ; la base attribue `reviewed_at`. |
| `fetch_verified_tafsirs(surah_id, start_ayah, end_ayah)` | Lecture destinée au public : seulement les entrées validées du passage. |

La correction et la validation demandent `expected_updated_at`, la date de
l'entrée affichée au relecteur. La requête modifie uniquement cette version.
Si un autre changement est intervenu, elle échoue avec `TafsirStoreConflict` :
recharger le contenu avant de relire/valider. Cela évite de valider un texte
différent de celui consulté, sans ajouter de nouveaux statuts au workflow.

La lecture destinée au public filtre `status=verified` dans la requête REST,
puis vérifie à nouveau le statut et les références des lignes reçues. Elle ne
retourne ni provenance originale ni date de dernière modification. Elle ne met
rien en cache : une correction qui annule la validation prend effet dès la
prochaine lecture. Les futures routes publiques devront appeler cette fonction
et les opérations internes devront être protégées côté serveur.

### Vérifier cette étape

Tests Python hors réseau (REST simulé) :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/services/test_tafsir_store.py api/tests/services/test_tafsir_import_service.py api/tests/services/test_tafsir_sources.py api/tests/schemas/test_quran_content.py api/tests/services/test_quran_translation_service.py api/tests/services/test_feedback_store.py
```

Les contrôles SQL, à exécuter uniquement dans une base de test isolée où la
migration est déjà installée et les rôles Supabase existent :

```bash
psql "$TAFSIR_TEST_DATABASE_URL" -X -v ON_ERROR_STOP=1 -f supabase/tests/tafsir_entries.sql
```

Ce fichier utilise des textes explicitement fictifs dans une transaction
annulée à la fin. Il vérifie la séparation des sources, les imports en attente,
la validation, les corrections, les reviews obsolètes, le refus des doublons
avec annulation du lot, l'immuabilité de la provenance et les droits/RLS.
Les tests Python vérifient aussi que la lecture destinée au public écarte les
brouillons et refuse un tafsir validé associé à un autre verset.

Commit proposé : `feat: add tafsir persistence and manual review operations`.

## Étape 6 : interface interne de review

Les routes privées réutilisent `tafsir_store.py` et son contrôle de version.
L'écran Nuxt est disponible à **`/internal/tafsir`** ; il ne figure pas dans
la navigation publique. Comme le projet utilise `app.vue` sans pages Nuxt,
ce composant choisit l'écran interne selon le chemin, sans refondre le
parcours de reconnaissance. Le filtre de sourates réutilise `useSurahOptions`
et `GET /surahs`.

### Activer l'accès

1. Installer la migration de l'étape 5 dans Supabase si elle ne l'est pas encore.
2. Choisir un mot de passe interne **long et aléatoire**, distinct de toute clé
   Supabase. Par exemple, générer une valeur avec `openssl rand -hex 32`.
3. Renseigner `TAFSIR_REVIEW_PASSWORD` dans `api/.env`. Cette variable appartient
   uniquement au backend ; ne pas la placer dans `runtimeConfig.public`, une
   variable `NUXT_PUBLIC_*`, les fichiers versionnés ou la commande d'une URL.
4. Recréer le conteneur API pour prendre en compte l'environnement :

   ```bash
   docker compose up -d --force-recreate api
   ```

5. Ouvrir `http://localhost:3000/internal/tafsir` et saisir ce mot de passe.
   En production, servir le frontend et l'API en HTTPS pour protéger son envoi.

Sans mot de passe configuré, **toutes** les routes de review sont désactivées
avec une réponse `503`. Un mot de passe absent ou incorrect renvoie `401`
avant tout appel au stockage. Le contrôle de connexion ne dépend pas de
Supabase : une table absente sera signalée ensuite lors du chargement de la liste.
La clé serveur Supabase reste exclusivement utilisée par `tafsir_store.py`.

Le mot de passe saisi est envoyé dans `Authorization: Bearer ...`. Le navigateur
le garde dans la mémoire de l'écran, sans `useState`, cookie, localStorage ni
sessionStorage. Recharger la page impose une nouvelle connexion ; se déconnecter
efface les textes chargés et annule une requête en cours. Un `401` après connexion
efface également l'accès local. Pour révoquer le mot de passe, le remplacer dans
l'environnement puis recréer/redémarrer le backend.

Les réponses de l'API interne portent `Cache-Control: no-store` et
`X-Robots-Tag: noindex, nofollow`. Les routes sont exclues du schéma OpenAPI
public. L'écran porte aussi `noindex, nofollow` et ne charge pas le script
Umami, qui reste actif dans le parcours public. Ces mesures accompagnent
l'authentification ; c'est le contrôle serveur qui protège les brouillons.

### Relire, corriger et valider

L'écran liste par défaut les entrées `need_review`, par pages de 50. Les filtres
portent sur la sourate, la source et le statut. Chaque entrée montre son couple
sourate/verset, son ouvrage, le français et, dans un volet dépliable, le passage
original et sa provenance. Le passage original est affiché comme du texte
échappé, même si la source contient des balises HTML.

- **Enregistrer** conserve la correction et remet l'entrée en `need_review`,
  y compris lorsqu'elle était déjà validée. Son ancienne date de review disparaît.
- **Annuler les corrections** restaure le texte chargé sans écrire en base.
- **Valider** passe une entrée en attente à `verified`, avec `reviewed_at`
  attribué par PostgreSQL. Le bouton reste désactivé tant que sa correction
  n'est pas enregistrée. La relecture avec l'édition physique reste manuelle.

Changer les filtres ou la page est bloqué tant qu'une correction n'est pas
enregistrée ou annulée. La validation d'un ouvrage ne valide pas l'autre.
Une entrée qui quitte le statut filtré disparaît de la liste, avec un message
de réussite. Pour la retrouver après validation, filtrer sur **Validé**.

Chaque écriture envoie `expected_updated_at`. Si la version a changé, l'API
renvoie `409` et l'écran bloque les écritures. **Recharger la liste** remplace
alors les corrections non enregistrées par la version stockée : relire cette
version avant de valider. Il n'y a ni écrasement automatique ni validation
implicite après correction.

| Route privée | Action |
| --- | --- |
| `GET /internal/tafsir/access` | Vérifie le mot de passe, réponse `204`. |
| `GET /internal/tafsir` | Liste ; paramètres `surah_id`, `source`, `status`, `limit`, `offset`. |
| `PATCH /internal/tafsir/{surah_id}/{ayah}/{source}` | Corps `text_fr`, `expected_updated_at`. |
| `POST /internal/tafsir/{surah_id}/{ayah}/{source}/verify` | Corps `expected_updated_at`. |

Les entrées invalides ou les champs inattendus sont rejetés avec `422`.
Le client ne peut pas choisir `status` ou `reviewed_at` dans une correction.
Une erreur de configuration de stockage renvoie `503`, une indisponibilité
de stockage `502`, sans renvoyer les détails privés du fournisseur.

### Vérification

Depuis la racine :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/routes/test_tafsir_review.py api/tests/services/test_tafsir_store.py api/tests/test_main.py
```

Depuis `ui/` :

```bash
npm test
npm run test:e2e -- tests/e2e/tafsir-review.spec.ts tests/e2e/recognition.spec.ts tests/e2e/navigation.spec.ts
npm run build
```

Les tests d'API utilisent un stockage simulé et ceux du navigateur une API
simulée, avec uniquement des textes explicitement fictifs. Ils vérifient les
accès refusés, la correction et la validation de la bonne version/source,
le refus d'une review obsolète, l'affichage échappé, la reconnexion après
rechargement et la conservation du parcours public de reconnaissance.
Ils ne modifient pas la base Supabase du projet.

Pour un essai réel, charger des snapshots pilotes réutilisables selon l'étape 5,
puis relire une entrée, enregistrer une correction et la valider. Sans import
réel, la liste vide est attendue. Cette étape n'ajoute pas encore la route
publique de lecture ni l'affichage du français dans les résultats.

Commit proposé : `feat: add protected tafsir review interface`.

## Étape 7 : lecture publique du contenu français

`GET /quran/content` réutilise la traduction locale et
`fetch_verified_tafsirs`. Ses paramètres sont identiques à ceux du tajwid :
`surah_id`, `start_verse`, `end_verse`. La route est publique et en lecture seule.
Elle ne nécessite pas le mot de passe de review. Les écritures et les brouillons
restent accessibles uniquement par les routes internes protégées.

### Réponse par verset

`QuranContentResponse` retourne la sourate, les bornes de la plage,
`tafsir_status` et une liste `ayahs` ordonnée. Chaque élément comporte :

| Champ | Contenu |
| --- | --- |
| `ayah` | Numéro du verset demandé. |
| `translation` | `QuranTranslation` complète : texte, notes, source, traducteur, version et URL ; `null` si ce verset manque dans le pilote. |
| `tafsirs` | Liste des tafsirs validés de ce verset, avec leur source, référence, version et date de review ; `[]` si aucun n'est disponible. |

La plage contient un élément pour chaque verset valide demandé, même si son
contenu français est absent du pilote. Une sourate ou une plage inexistante
est rejetée avant de charger la traduction ou de contacter Supabase : `422`
pour les paramètres absents ou hors bornes, `400` pour une plage inversée ou
dépassant le nombre réel de versets de la sourate.

`VerifiedTafsirEntry` reprend les champs publics de `TafsirEntry` et impose
`status = verified` ainsi qu'une date de review avec fuseau horaire. Le service
filtre déjà `verified` dans la requête Supabase et contrôle les lignes reçues ;
le contrat public refuse également une entrée en attente. Les commentaires
Ibn Kathir et As-Sa‘di gardent leurs identifiants distincts, même sur le même verset.
La provenance originale, `source_text` et `updated_at` ne sont pas exposés.

### Disponibilité et cache

`tafsir_status` décrit la lecture du stockage, pas la présence d'un commentaire :

- `available` : la lecture a réussi ; la liste peut être vide si aucun tafsir
  n'est validé pour les versets demandés ;
- `unavailable` : configuration absente, table inaccessible, panne réseau ou
  données de stockage incohérentes. Tous les tafsirs sont alors omis ; la
  traduction locale reste retournée avec une réponse `200`.

Une incohérence de source, de statut, de date de review ou de référence ne
produit jamais de contenu tafsir public. Le backend journalise un avertissement
avec le type d'erreur et la plage demandée, sans y ajouter le texte privé ni
les détails du fournisseur. Le frontend pourra distinguer l'indisponibilité
du stockage d'une absence normale de tafsir validé.

Le stockage utilise son délai existant de 15 secondes : en cas de panne réseau,
la réponse peut attendre ce délai avant de retourner la traduction seule.
Si le snapshot de traduction est illisible ou invalide, la route renvoie `503`
avec un message générique, sans exposer son chemin local.

Les réponses de contenu portent `Cache-Control: no-store`. Aucun cache tafsir
n'est ajouté : une validation, puis une correction qui annule cette validation,
prennent effet à la lecture suivante. Le cache mémoire de la traduction reste
celui de son service local. Les appels de stockage et de lecture du snapshot
sont exécutés via `run_in_threadpool`, selon le mécanisme déjà utilisé par les
routes du projet.

La route ne lance ni import, ni génération, ni téléchargement QuranEnc/Quran
Foundation. Elle charge seulement les contenus déjà préparés. Elle est appelée
séparément de la reconnaissance audio ; son raccordement à `VerseDetailsSheet`
est décrit à l'étape 8.

### Tester

Avec le backend démarré, pour le verset 2:255 :

```bash
curl -L --fail --silent --show-error 'http://localhost:8000/quran/content?surah_id=2&start_verse=255&end_verse=255'
```

Pour vérifier une plage partiellement couverte par le pilote, demander
`start_verse=254&end_verse=255`. Le verset 254 aura `translation: null`.
Sans tafsir relu/importé, les listes `tafsirs` vides sont attendues. Si le
stockage n'est pas configuré ou la table n'est pas installée, la traduction
est tout de même visible et `tafsir_status` vaut `unavailable`.

Depuis la racine, les tests HTTP et les services utilisés :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/routes/test_quran_content_route.py api/tests/services/test_tafsir_store.py api/tests/services/test_quran_translation_service.py api/tests/schemas/test_quran_content.py api/tests/test_main.py
```

Ces tests utilisent le vrai catalogue et le snapshot QuranEnc livré, avec
Supabase REST simulé et des tafsirs explicitement fictifs. Ils inspectent toute
la réponse HTTP : aucun brouillon ni passage original interne, références
correctes, deux ouvrages distincts, attribution/notes de traduction préservées,
indisponibilité de stockage sans perte de traduction et changements de statut
visibles à la lecture suivante. Ils ne modifient pas le Supabase du projet.

Commit proposé : `feat: expose verified tafsir and French translations through API`.

## Étape 8 : affichage dans les résultats de reconnaissance

`VerseDetailsSheet.vue` réutilise le texte arabe du passage reconnu, disponible
sans nouvel appel API. Le français est chargé uniquement quand l'utilisateur
ouvre **Voir le verset** ou **Vérifier le passage**, via `useQuranContent.ts`
et `GET /quran/content`. Le survol du bouton peut précharger le composant, mais
ne charge pas les textes. Aucun appel à la route de review ni secret interne
n'est ajouté au parcours public.

Le texte arabe est présenté pour le passage entier. `QuranAyahContent.vue`
affiche ensuite le contenu français par numéro de verset : traduction, notes
dépliables, traducteur, source liée et version. Les notes et tous les textes
sont échappés par Vue, sans `v-html`. Une traduction absente reste signalée
comme indisponible, sans remplacement ni génération de texte.

Chaque verset ayant un tafsir validé propose **Ibn Kathir** et **As-Sa‘di**.
La source sans contenu validé est désactivée ; la première source disponible
est sélectionnée. Changer de source change seulement le commentaire affiché
pour ce verset, avec sa référence et sa version. Si aucun commentaire n'est
validé, toute la section tafsir du verset est absente. Les choix de sources
utilisent des boutons avec `aria-pressed` ; les notes et liens restent
accessibles au clavier dans la fenêtre de détails.

Le composable contrôle la plage reçue, les couples sourate/verset de chaque
texte et les identifiants des deux ouvrages. En complément du filtrage serveur,
il exclut tout tafsir qui n'est pas `verified` ou qui n'a pas de date de review
valide. Une réponse `tafsir_status: unavailable` conserve la traduction mais
supprime tout commentaire, avec un message d'indisponibilité.

Les appels utilisent `cache: no-store`, sans cache mémoire ni retry automatique.
Fermer les détails, changer de passage ou détruire le composant annule la
requête et efface le contenu chargé ; une réponse tardive ne peut pas remplacer
celui d'un nouveau passage. Chaque réouverture relit les données publiques,
y compris si une correction a retiré une précédente validation. Un écran déjà
ouvert ne reçoit pas les changements de review en temps réel.

Une erreur de chargement affiche un message générique et **Réessayer le
chargement du contenu français**. Le texte arabe, le résultat reconnu, la copie
et le chargement du tajwid restent indépendants de cet appel. Le frontend
attend au maximum 20 secondes, ce qui couvre le délai de stockage actuel
de 15 secondes côté API.

### Tester

Depuis `ui/` :

```bash
npm test
npm run test:e2e -- tests/e2e/quran-content.spec.ts tests/e2e/verse-details-and-feedback.spec.ts tests/e2e/recognition.spec.ts
npm run lint
npm run build
```

Les tests du composable et de la fiche contrôlent les références, les brouillons
exclus du rendu, la séparation des ouvrages, l'échappement des notes, le rejet
d'une réponse obsolète et les erreurs sans perte du résultat. Chromium vérifie
le parcours reconnaissance → détails → changement de source, la réouverture
après retrait de validation et l'accès au tajwid malgré une panne du français.
Leurs textes français sont explicitement fictifs, avec des réponses API
simulées ; ils n'insèrent rien dans Supabase.

Pour un essai réel, reconnaître Al-Fatiha ou l'un des versets pilotes
d'Al-Baqara puis ouvrir les détails. La traduction QuranEnc locale doit
apparaître. L'absence de tafsir est attendue tant qu'aucun contenu réel n'est
importé puis relu et validé. Pour vérifier le pipeline de review jusqu'au
frontend, valider manuellement une entrée réelle du pilote, ouvrir les détails,
puis corriger cette entrée en interne et rouvrir les détails : le commentaire
doit disparaître jusqu'à sa prochaine validation. Vérifier chaque ouvrage
séparément. Une indisponibilité Supabase sera signalée sans masquer la traduction.

Commit proposé : `feat: display French translation and verified tafsir in recognition results`.

## Étape 9 : relier import, review et publication dans les tests

Deux scénarios complémentaires relient les étapes déjà implémentées, sans
ajouter de fonctionnalité au parcours utilisateur ni modifier un stockage réel.

### Import et routes HTTP sur le même stockage simulé

`api/tests/integration/test_tafsir_pilot_workflow.py` importe trois références
pilotes — 1:1, 2:1 et 2:255 — pour chacun des deux ouvrages. Il appelle le vrai
importeur local, relit ses snapshots, les insère via `insert_tafsir_import`,
puis utilise les vraies routes FastAPI privées et publiques. Seuls le transport
REST Supabase et l'exécution en thread sont simulés. Le catalogue et la
traduction QuranEnc sont les fichiers réels du dépôt ; Whisper n'est pas lancé.

Le scénario vérifie successivement :

1. Les six entrées importées commencent en `need_review` et restent absentes
   de la réponse publique, tandis que la traduction correspond au bon verset.
2. Une validation sans authentification est refusée avant de toucher au stockage.
3. Une correction reste privée et conserve sa provenance originale.
4. Valider l'ancienne révision échoue avec `409` ; valider la révision corrigée
   rend uniquement le tafsir Ibn Kathir de 2:255 disponible.
5. Les autres versets et As-Sa‘di restent en attente. Sa propre validation
   rend ensuite les deux ouvrages disponibles séparément.
6. Une nouvelle correction d'Ibn Kathir retire ce commentaire dès la lecture
   publique suivante, tandis qu'As-Sa‘di reste disponible.

Les champs de provenance privés et le mot de passe de test ne doivent
pas apparaître dans les réponses publiques. La simulation REST sert à vérifier
l'enchaînement des services et des routes ; elle ne remplace pas les tests des
contraintes et du déclencheur PostgreSQL dans `supabase/tests/tafsir_entries.sql`.

### Review et affichage public dans Chromium

Un scénario ajouté à `ui/tests/e2e/tafsir-review.spec.ts` ouvre deux pages :
l'interface interne de review et un résultat public de reconnaissance de 2:255.
Leurs réponses API simulées partagent les mêmes entrées, au lieu de préparer
indépendamment une réponse déjà validée pour l'écran public.

Les corrections et validations effectuées dans l'interface interne gouvernent
ainsi le contenu retourné à chaque réouverture des détails. Le test vérifie
l'absence des brouillons, la publication distincte des deux ouvrages, le
changement de source et le retrait du seul tafsir corrigé. Il contrôle aussi
que le passage original interne n'apparaît pas dans les détails publics.

Les textes de tafsir sont explicitement fictifs dans les deux scénarios. Aucun
texte religieux n'est généré et aucun test ne publie de données dans Supabase.
Ces vérifications ne démontrent pas l'installation de la table distante ni
la fidélité d'une future traduction : l'essai avec un lot réel et la relecture
avec les éditions physiques restent nécessaires.

### Tester

Depuis la racine :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/integration/test_tafsir_pilot_workflow.py
```

Depuis `ui/` :

```bash
npm run test:e2e -- tests/e2e/tafsir-review.spec.ts tests/e2e/quran-content.spec.ts
npm run lint
```

Commit proposé : `test: cover tafsir pilot review and public publication workflow`.

## Étape 10 : génération française du pilote avec DeepL

`api/scripts/generate_tafsir_fr.py` traduit des **passages arabes fournis
localement** en brouillons français. Il ne télécharge aucun corpus. Il réutilise
le catalogue coranique, les contrats d'import, l'écriture atomique privée et le
stockage REST existants. Aucun nouveau service n'est appelé au démarrage, dans
`POST /recognize` ou dans l'API publique.

```text
passages originaux d'un seul ouvrage
    → traduction DeepL de chaque passage entier
    → snapshot interne need_review
    → store_tafsir_fr.py → Supabase
    → relecture/correction manuelle → verified
    → API publique et détails Nuxt
```

### Coût et configuration

Pour le pilote, utiliser une offre gratuite DeepL API. Au 7 octobre 2026,
[DeepL API Developer](https://support.deepl.com/hc/en-us/articles/360021200939-DeepL-API-plans)
offre **1 million de caractères au total**, sans renouvellement de ce quota.
L'ancienne offre API Free, fermée aux nouvelles souscriptions, conserve un
quota de 500 000 caractères par mois pour les comptes existants. Le quota réel
restant dépend du compte. Aucun abonnement payant n'est nécessaire pour un
pilote qui tient dans le quota gratuit disponible.

Créer une clé depuis [le portail développeur DeepL](https://www.deepl.com/en/developers),
puis configurer uniquement l'environnement backend :

```dotenv
DEEPL_API_KEY=<clé du compte API>
DEEPL_API_URL=https://api-free.deepl.com
```

La clé ne doit être placée ni dans Nuxt, ni dans les lots, ni dans Git.
L'[API gratuite utilise le domaine `api-free.deepl.com`](https://developers.deepl.com/docs/getting-started/quickstart).
Le script utilise ce domaine par défaut et ne bascule jamais automatiquement
vers une offre payante. `https://api.deepl.com` reste accepté pour un compte
payant explicitement configuré ; tout autre domaine est refusé.

Dans Docker, ajouter les variables dans `api/.env`, puis recréer le conteneur :

```bash
docker compose up -d --force-recreate api
```

Hors Docker, les variables doivent être exportées dans l'environnement du
processus ; les scripts ne chargent pas implicitement `api/.env`.

La traduction est faite au lancement manuel de la commande. Supabase conserve
ensuite le résultat, le passage original et la provenance. Les consultations
publiques lisent le texte enregistré : elles ne consomment aucun quota DeepL.
Le stockage et les lectures restent soumis aux limites de l'offre Supabase
du projet. Pour le corpus complet, mesurer d'abord les caractères réels et la
qualité du pilote, puis choisir l'offre de traduction adaptée.

### Préparer les passages sources

Copier `api/examples/tafsir_generation.example.json` dans le répertoire interne
`api/data/tafsir/inputs/`, puis renseigner les originaux et leur provenance.
Le modèle vide est volontairement invalide et ne contient aucun tafsir fictif.
Préparer **un fichier distinct par ouvrage**, avec `source = ibn_kathir` ou
`source = as_saadi`. La langue acceptée pour cette commande est `ar` ; un
français déjà existant continue de passer par l'import de l'étape 4.

Le lot conserve `schema_version`, `source`, `source_language`, `source_edition`,
`version` et `reuse_reference`. Sa liste `passages` contient, pour chaque passage :

| Champ | Contenu |
| --- | --- |
| `source_surah_id` | Sourate originale. |
| `source_start_ayah`, `source_end_ayah` | Bornes complètes du commentaire original. |
| `source_reference` | Référence précise de ce passage dans l'édition utilisée. |
| `source_text` | Texte original entier, en texte brut UTF-8. |
| `ayahs` | Versets du pilote auxquels associer ce commentaire entier. |

Les versets demandés doivent appartenir à 1:1–7, 2:1–5 ou 2:255 et être couverts
par le passage. Un original peut couvrir une plage plus large, par exemple
2:1–6, tout en ciblant seulement les versets du pilote : **ne pas tronquer son
texte**. Un verset cible ne peut figurer deux fois et un même passage ne peut
être répété ; regrouper ses versets dans `ayahs`. Les champs inconnus, les
sources mélangées et les références absentes du catalogue sont refusés.

Chaque passage est envoyé entier **une seule fois** à DeepL avec la langue
source `AR`, la langue cible `FR` et la conservation de mise en forme demandée.
Le moteur reçoit uniquement ce passage, sans contexte religieux ajouté,
glossaire externe, synthèse ou instructions d'enrichissement. Les détails du
contrat HTTP suivent [la documentation de traduction DeepL](https://developers.deepl.com/api-reference/translate/request-translation).
La traduction complète est associée à chacun des versets demandés, avec ses
bornes originales ; aucune explication particulière à un verset n'est inventée.

Le script valide **tout le lot**, y compris la limite de 128 Kio par requête,
avant le premier appel. Un passage trop grand est refusé sans découpage ni
troncature automatique. La fidélité de la traduction automatique reste à
contrôler manuellement avec les éditions physiques françaises.

### Vérifier, traduire puis stocker

Valider les fichiers et compter les caractères sources restants, sans clé, appel réseau
ou création de fichier :

```bash
api/.venv/bin/python api/scripts/generate_tafsir_fr.py --input api/data/tafsir/inputs/ibn_kathir-source.json --dry-run
api/.venv/bin/python api/scripts/generate_tafsir_fr.py --input api/data/tafsir/inputs/as_saadi-source.json --dry-run
```

Le comptage porte sur chaque passage unique qui n'a pas encore été sauvegardé
dans le fichier de progression ; partager un commentaire entre plusieurs
versets n'augmente pas ce total. DeepL compte les caractères du texte
source, y compris les espaces et retours à la ligne, selon ses
[règles de décompte](https://support.deepl.com/hc/en-us/articles/360020685720-Usage-count-and-billing-in-DeepL-API).

Une fois la clé configurée et les fichiers réels prêts :

```bash
docker compose exec api python scripts/generate_tafsir_fr.py --input /app/data/tafsir/inputs/ibn_kathir-source.json --output /app/data/tafsir/ibn_kathir/pilot-drafts.json
docker compose exec api python scripts/generate_tafsir_fr.py --input /app/data/tafsir/inputs/as_saadi-source.json --output /app/data/tafsir/as_saadi/pilot-drafts.json
```

Chaque sortie respecte `TafsirFrenchImportBatch` : toutes les entrées sont
`need_review`, sans date de review, avec le texte original intact. Le bloc
privé `generation` conserve le fournisseur, la version des paramètres de
requête, le domaine API, la langue cible et la date UTC de génération. Chaque
entrée générée conserve aussi la trace de son passage : elle est préservée lors
d'une reprise, même si la clé ou l'offre DeepL change. La trace au niveau du lot
correspond au passage traduit le plus récemment. DeepL
choisit son modèle par défaut ; ce bloc n'invente pas d'identifiant de modèle.
Les imports précédents sans ce bloc restent acceptés et gardent leur format.

Les fichiers complets sont publiés atomiquement avec les permissions `0600`,
dans le répertoire exclu de Git et des images Docker. Une destination existante
est refusée **avant** l'appel DeepL. Une réponse invalide ou un échec interrompt
le lot sans fichier partiel, sans nouvel essai automatique et sans import en
base. Depuis l'étape 11, les passages sauvegardés avant un échec sont conservés
et ne sont pas envoyés de nouveau lors de la reprise.

La sortie est directement utilisable par le script de stockage de l'étape 5 ;
il n'est pas nécessaire de repasser par `import_tafsir_fr.py` :

```bash
docker compose exec api python scripts/store_tafsir_fr.py --input /app/data/tafsir/ibn_kathir/pilot-drafts.json
docker compose exec api python scripts/store_tafsir_fr.py --input /app/data/tafsir/as_saadi/pilot-drafts.json
```

Le stockage conserve aussi `generation` dans la provenance privée. L'insertion
refuse toujours les doublons et ne remplace aucun contenu déjà relu.
Utiliser ensuite `/internal/tafsir` pour corriger et valider chaque ouvrage.
**Aucune sortie de génération ne valide automatiquement un tafsir.**

### Vérification de l'étape

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/services/test_tafsir_generation_service.py api/tests/integration/test_tafsir_pilot_workflow.py api/tests/services/test_tafsir_import_service.py api/tests/services/test_tafsir_store.py api/tests/routes/test_tafsir_review.py api/tests/routes/test_quran_content_route.py api/tests/schemas/test_quran_content.py
```

DeepL et Supabase sont simulés ; les originaux et traductions des tests sont
explicitement fictifs. Les contrôles couvrent la traduction unique des groupes,
la séparation des ouvrages, les références, la conservation de la provenance,
les statuts en attente, leur exclusion des lectures publiques, le comptage sans
appel, le refus des écrasements et les échecs sans sortie partielle.
Aucun appel de traduction réel n'a été exécuté pour cette étape. La clé, les
lots sources réels et les références des éditions françaises restent à préparer.

Commit proposé : `feat: add DeepL French tafsir pilot generation pipeline`.

## Étape 11 : reprise après interruption ou quota épuisé

Chaque passage traduit est sauvegardé **avant l'appel DeepL suivant** dans un
fichier de progression privé. Celui-ci conserve le français, l'original,
les références des versets et les métadonnées de génération. Les entrées y
restent `need_review`, avec `reviewed_at = null`.

La commande reprend automatiquement le même lot : elle charge les passages
enregistrés et traduit seulement ceux qui manquent. Une réponse
[DeepL HTTP 456](https://developers.deepl.com/docs/best-practices/error-handling)
arrête les appels et signale le quota épuisé. Une interruption avec `Ctrl+C`
conserve également les passages déjà sauvegardés. Relancer la commande quand
le quota est disponible ; le quota gratuit Developer ne se renouvelle pas
automatiquement, comme indiqué à l'étape 10.

### Fichier de progression

Par défaut : `api/data/tafsir/<source>/progress-<empreinte>.json`. L'empreinte
identifie le contenu exact du lot : auteur, édition, version, provenance,
passages, ordre et versets demandés. Le chemin est affiché au lancement.
Déplacer le fichier source sans changer son contenu ne perd donc pas la reprise.
Un lot modifié utilise une autre empreinte et un autre fichier de progression.

Pour choisir un repère lisible, utiliser `--checkpoint`. Exemple pour Ibn Kathir :

```bash
docker compose exec api python scripts/generate_tafsir_fr.py --input /app/data/tafsir/inputs/ibn_kathir-source.json --output /app/data/tafsir/ibn_kathir/pilot-drafts.json --checkpoint /app/data/tafsir/ibn_kathir/pilot-progress.json
```

**Relancer exactement cette commande** pour reprendre. Choisir un fichier
distinct pour As-Sa‘di. Si un repère explicite existe mais correspond à un
autre lot, ou s'il est corrompu/incohérent, la commande s'arrête avant l'appel
DeepL et conserve ce fichier. Elle ne le supprime pas pour retraduire en silence.

Pour connaître le travail restant avec ce même repère :

```bash
docker compose exec api python scripts/generate_tafsir_fr.py --input /app/data/tafsir/inputs/ibn_kathir-source.json --checkpoint /app/data/tafsir/ibn_kathir/pilot-progress.json --dry-run
```

Le résultat indique le nombre de passages enregistrés, le nombre restant et
les caractères restants. Cette lecture ne demande pas de clé et n'écrit rien.
Utiliser le même environnement ou utilisateur que pour la génération : les
fichiers privés ont les permissions `0600`.

### Conservation et sortie finale

Les mises à jour du repère sont atomiques : le fichier précédent reste complet
si une écriture échoue. Un verrou empêche deux commandes utilisant le même
repère de traduire simultanément. Son fichier `.lock` reste présent ; le verrou
est automatiquement libéré à l'arrêt du processus. Ces fichiers sont exclus
de Git et des images Docker avec le répertoire `data/tafsir`.

Le repère est un fichier interne de travail ; il ne peut pas être importé
directement comme snapshot Supabase et aucune route ne le sert. Le snapshot
final n'est produit qu'une fois tous les passages du lot terminés. Il peut
ensuite être stocké et relu selon le workflow existant. Si l'écriture finale
échoue, la reprise reconstruit le snapshot avec les traductions sauvegardées,
sans appel DeepL supplémentaire ni clé nécessaire quand tout est terminé.
Un snapshot final existant reste protégé contre l'écrasement.

Conserver le repère pour les reprises. Après changement de clé ou passage
manuel de l'offre gratuite à une offre payante, les passages sauvegardés
restent utilisables et gardent leurs dates/domaines API d'origine. La trace
de chaque passage est conservée dans la provenance privée Supabase ; aucune
clé API n'est enregistrée dans les fichiers de progression ou les snapshots.

La reprise garantit la réutilisation des passages **effectivement sauvegardés**.
Si une coupure intervient pendant une requête ou avant la sauvegarde de sa
réponse, ce seul passage peut être renvoyé et avoir consommé du quota. La commande
ne peut pas récupérer une réponse DeepL qu'elle n'a pas reçue/enregistrée.

Vérification hors réseau :

```bash
api/.venv/bin/pytest -c api/pytest.ini api/tests/services/test_tafsir_generation_service.py api/tests/services/test_tafsir_import_service.py api/tests/services/test_tafsir_store.py api/tests/integration/test_tafsir_pilot_workflow.py api/tests/routes/test_quran_content_route.py api/tests/routes/test_tafsir_review.py
```

Les tests simulent quota épuisé, erreur réseau et interruption, puis vérifient
la reprise au passage restant, le comptage sans appel, la conservation des
textes/dates et des auteurs, le refus des repères d'un autre lot ou corrompus,
le verrou et les échecs d'écriture. Les brouillons restent exclus du public.
Aucune traduction réelle n'a été lancée pour cette étape.

Commit proposé : `feat: resume tafsir translation from saved progress`.

## Sources et génération restantes

Le projet sait importer un français déjà préparé, conserver son texte source,
le stocker en `need_review`, le faire relire et afficher uniquement le résultat
validé. La commande manuelle de l'étape 10 ajoute la traduction DeepL des
originaux arabes ; aucun corpus réel de tafsir n'est encore fourni.
Le format d'import actuel accepte une langue source
`ar` ou `fr` ; une source anglaise nécessitera l'ajout explicite de `en` dans
le contrat Pydantic d'import. La colonne JSON de provenance peut déjà conserver
cette langue sans changer la structure de la table.

Le workflow cible reste identique quelle que soit la langue originale :
texte exact d'un ouvrage → traduction française de ce seul texte →
`need_review` → relecture manuelle → `verified`. Le moteur ne devra ni résumer,
ni compléter une partie manquante, ni mélanger les ouvrages. La traduction
française du Coran reste celle de QuranEnc, indépendante de ce processus.

Vérification du [catalogue officiel Quran.com](https://api.quran.com/api/v4/resources/tafsirs?language=en)
le 7 octobre 2026 :

| Source | Ressource référencée actuellement | Variante anglaise du catalogue |
| --- | --- | --- |
| Ibn Kathir | `14`, `ar-tafsir-ibn-kathir`, arabe. | `169`, `en-tafisr-ibn-kathir`, **Ibn Kathir (Abridged)** : édition abrégée distincte à identifier comme telle. |
| As-Sa‘di | `91`, `ar-tafseer-al-saddi`, arabe. | Aucune entrée anglaise As-Sa‘di dans le catalogue consulté. |

Le `slug` de la ressource anglaise ci-dessus est celui retourné par le catalogue
consulté. Les exemples documentaires peuvent employer une autre orthographe :
un futur import doit contrôler la réponse effective du fournisseur.
Le paramètre `language=en` traduit les libellés des ressources, pas leurs
textes, comme le précise [la documentation du catalogue](https://api-docs.quran.com/docs/content_apis_versioned/4.0.0/tafsirs/).

La commande de génération part des deux originaux arabes déjà identifiés pour
traduire directement en français. Une approche anglais → français reste possible avec une
source anglaise As-Sa‘di distincte, identifiée et réutilisable ; l'anglais
Ibn Kathir du catalogue ne doit pas être présenté comme une édition intégrale.
La langue, l'édition, la référence, la version et les conditions de réutilisation
du texte effectivement fourni doivent être conservées. La génération est
raccordée à l'import existant ; il reste à fournir les lots sources réels.
Les règles de stockage Quran Foundation
mentionnées à l'étape 3 s'appliquent également aux ressources anglaises.

## Plan d'intégration

1. **Traduction pilote — réalisée.** Ajouter `api/scripts/import_quran_translation.py`,
   `app/services/quran_translation_service.py` et le snapshot local
   `api/assets/quran_translation_fr.json`. Importer Al-Fatiha et un petit
   échantillon d'Al-Baqara, par exemple 2:1–5 et 2:255. Vérifier les couples
   sourate/verset contre le catalogue, les doublons et les métadonnées. Garder
   les réponses source et la version obtenue au moment de l'import.
2. **Sources identifiées ; pipeline d'import local réalisé.** Les deux ressources
   arabes Quran Foundation sont référencées. L'import de lots locaux conserve
   les passages originaux et impose `need_review`. Fournir un corpus local
   réutilisable ou connecter Content Sync pour préparer les lots réels, et
   préciser les éditions françaises de review. Conserver le texte source
   et sa provenance dans des fichiers internes séparés. Commencer par l'import
   de français existant ou utiliser la génération DeepL de l'étape 10, qui traduit
   uniquement le texte source fourni, sans enrichissement ni mélange entre ouvrages.
   Chaque sortie passe par `TafsirImportEntry`. Une réimportation ne doit jamais
   remplacer silencieusement un texte déjà relu.
3. **Persistance et validation — réalisées dans le dépôt.** La migration
   `supabase/tafsir_entries.sql`, le script de stockage et les opérations de
   review sont ajoutés. Installer la migration dans le projet Supabase avant
   l'import réel. Les contraintes, droits et déclencheur préservent la
   provenance, empêchent les réimports d'écraser les textes et annulent une
   validation après correction. Le backend cible la version effectivement relue.
4. **Interface interne — réalisée.** Les routes FastAPI et l'écran Nuxt de
   review réutilisent le stockage et le catalogue existants. Le mot de passe
   dédié est configuré côté backend ; les filtres sourate/source/statut,
   corrections et validations sont disponibles. Aucun secret serveur dans
   `runtimeConfig.public` ; aucun accès aux brouillons sans authentification.
5. **Lecture publique — réalisée.** `GET /quran/content` retourne la traduction
   locale et les tafsirs validés par verset, avec un contrat public dédié.
   Le filtrage `verified`, les références et les sources sont contrôlés par
   les services existants. Le stockage indisponible ne prive pas la réponse
   de traduction. Le contenu tafsir n'est pas mis en cache.
6. **Affichage — réalisé.** `VerseDetailsSheet.vue` montre le passage arabe
   reconnu ; `useQuranContent.ts` charge le français à l'ouverture des détails.
   `QuranAyahContent.vue` présente la traduction et les choix Ibn Kathir /
   As-Sa‘di par verset. Aucun contenu tafsir lorsqu'une entrée validée manque.
   Une erreur du français ne masque pas le résultat reconnu ni le tajwid.

Ce plan réutilise les schémas Pydantic, le catalogue, les snapshots locaux,
Supabase REST et le composant de détails déjà présents. La reconnaissance audio
ne dépend pas du chargement des traductions ou tafsirs.

## Source de traduction retenue pour le pilote

[La documentation officielle QuranEnc](https://quranenc.com/en/home/api)
décrit la liste des traductions, les lectures par sourate et par verset, ainsi
que les champs `sura`, `aya`, `translation` et `footnotes`. Elle référence la
traduction française de Rachid Maach sous la clé `french_rashid`.
La version sera lue depuis les métadonnées du fournisseur, sans valeur inventée.

Les ressources arabes des tafsirs sont référencées à l'étape 3. L'import local
et la persistance sont prêts ; les corpus réellement réutilisables, l'éventuelle
connexion Content Sync et les éditions physiques françaises de review restent
à préciser avant l'import religieux. Aucun contenu religieux fictif n'est
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
