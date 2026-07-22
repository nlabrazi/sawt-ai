# Benchmark audio de reconnaissance

Ce benchmark mesure la chaîne réellement utilisée par l'application :

`WAV → faster-whisper → matching du passage`

Il complète `verse_detection_corpus.json`, qui mesure uniquement le matching à
partir d'un texte déjà transcrit.

## Ce qui est versionné

- `audio_corpus.public.json` décrit uniquement des sons synthétiques
  reproductibles : silence, bruits blanc et rose, fond domestique non vocal,
  tonalité et mélodie instrumentale.
- `audio_corpus.private.example.json` documente le format des récitations et
  paroles réelles, ainsi que les variantes bruitées à plusieurs SNR.
- aucun WAV, rapport d'exécution, chemin privé ou manifeste privé réel n'est
  versionné.

Les dossiers `evaluation/generated/`, `evaluation/reports/`,
`evaluation/private/` et les fichiers `*.private.json` sont ignorés par Git.
Lorsqu'un manifeste privé participe au build, le dossier construit passe en
`0700` avant toute génération. Chaque WAV et manifeste est d'abord écrit dans
un temporaire `0600`, puis remplacé atomiquement une fois complet ; une erreur
ne laisse donc ni fichier partiel ni fenêtre de lecture en `0644`.

## Limites honnêtes du corpus public

Le corpus public ne contient aucune vraie voix. Une mélodie synthétique permet
de tester un faux positif sur de la musique, mais ne remplace pas une chanson
avec voix. Le fond domestique est non vocal et ne prétend pas reproduire des
enfants qui parlent.

Un cas de TTS français hors ligne est prévu. Il est généré seulement si
`espeak-ng`, `espeak` avec une voix française, ou `pico2wave` est présent. Dans
l'environnement actuel, aucun de ces moteurs n'est installé : ce cas est marqué
`french_tts_unavailable`, jamais compté comme testé.

Des Fatiha sont présentes localement dans
`training/artifacts/normalized_audio_strict/`, mais leur provenance et leur
licence ne sont pas documentées. Elles ne doivent pas être intégrées avant de
confirmer le droit d'usage et le consentement des personnes enregistrées.

## Construire le corpus public

Depuis la racine du dépôt :

```bash
api/.venv/bin/python api/scripts/build_audio_evaluation_corpus.py
```

La commande produit des WAV mono PCM 16 bits à 16 kHz. La génération audio
utilise la bibliothèque standard Python ; `ffmpeg`, déjà présent dans les images
API, sert seulement à normaliser une source locale dans un autre format.
`numpy`, `soundfile` et `librosa` sont déjà des dépendances de production, mais
le générateur ne les impose pas à la venv de tests minimale.

Le manifeste construit contient un SHA-256 par WAV. À seed et source identiques,
les générateurs numériques produisent les mêmes fichiers. Le TTS français reste
dépendant du moteur et de sa version ; son SHA-256 rend toute variation visible
dans le rapport.
Au chargement, le runner vérifie également l'en-tête PCM16 mono, la fréquence et
la durée réelle du WAV. Les collisions entre un identifiant de cas et de
variante sont refusées avant la première écriture.

## Injecter les récitations et négatifs réels

Le protocole de collecte, les objectifs de diversité et la séparation des jeux
sont détaillés dans [`REAL_AUDIO_COLLECTION.md`](REAL_AUDIO_COLLECTION.md).

1. Copier le modèle sans changer son nom versionné :

   ```bash
   cp api/evaluation/audio_corpus.private.example.json \
     api/evaluation/audio_corpus.private.json
   ```

2. Placer les audios dans `api/evaluation/private/sources/`, ou renseigner un
   chemin absolu local dans le manifeste privé.
3. Utiliser des identifiants pseudonymes dans `case.id` et `tags`, puis
   renseigner le passage attendu exact pour chaque récitation.
4. Le modèle laisse volontairement `consent_confirmed` et
   `usage_rights_confirmed` à `false`. Passer le premier à `true` uniquement
   lorsqu'un consentement réel a été obtenu — celui du représentant légal pour
   un enfant — et le second uniquement après vérification du droit d'usage. Le
   build refuse toute source locale tant que les deux attestations ne sont pas
   explicites.
5. Ajouter au minimum de vrais cas négatifs : texte français, conversation,
   chanson avec voix, prose et chanson arabes, silence micro et bruit ambiant.
6. Construire les manifestes ensemble :

   ```bash
   api/.venv/bin/python api/scripts/build_audio_evaluation_corpus.py \
     --manifest api/evaluation/audio_corpus.public.json \
     --manifest api/evaluation/audio_corpus.private.json
   ```

Le manifeste construit ne reprend jamais le chemin du fichier source. Chaque
récitation du modèle produit ces variantes :

| Variante | Condition |
|---|---:|
| `clean` | source normalisée |
| `white_snr20` | bruit blanc à 20 dB |
| `white_snr10` | bruit blanc à 10 dB |
| `pink_snr10` | bruit rose à 10 dB |
| `background_snr5` | fond domestique synthétique à 5 dB |
| `children_snr5` | vraie voix de fond locale et consentie à 5 dB |

Le SNR est calculé sur le RMS du signal et du bruit. Une valeur plus faible
correspond à un cas plus difficile. Un bruit vocal local utilise lui aussi
`consent_confirmed` et `usage_rights_confirmed`; son chemin n'est jamais recopié
dans le manifeste construit. Les sources sont limitées à 90 secondes, comme
`POST /recognize` en production.

## Lancer le pipeline complet sans téléchargement

Le runner force `HF_HUB_OFFLINE=1`, même si l'environnement appelant indiquait
le contraire. Le modèle faster-whisper doit donc déjà
être présent en cache ou être référencé par un chemin local. Avec le conteneur
API déjà démarré :

```bash
docker compose exec api python scripts/evaluate_audio_recognition.py \
  --output evaluation/reports/latest.json
```

En exécution locale, installer `api/requirements.txt`, puis définir
`WHISPER_MODEL_NAME=/chemin/local/vers/le-modele-ctranslate2` avant d'appeler le
même script avec Python.

Pour comparer `turbo` et `large-v3` dans deux processus isolés, sans changer la
configuration de production :

```bash
docker compose exec api python scripts/compare_whisper_models.py \
  --corpus evaluation/generated/audio/manifest.json \
  --output-dir evaluation/reports/model-comparison
```

Le comparateur conserve les rapports détaillés par modèle et produit un
`comparison.json` limité aux métriques de qualité et de performance. Les deux
modèles doivent déjà être présents dans le cache local : aucun téléchargement
implicite n'est autorisé.

Par défaut, le rapport conserve seulement la longueur de la transcription : un
SHA-256 non salé d'une phrase courte serait lui-même vérifiable par dictionnaire.
`--include-transcriptions` exige `--output`, écrit le contenu uniquement dans ce
rapport privé en `0600` et ne le réaffiche pas sur stdout. Le fichier ne doit pas
être publié. Tout rapport issu d'un corpus marqué privé est également écrit en
`0600`. Par défense en profondeur, le runner neutralise le logger INFO du service
de transcription pendant ces exécutions.
Un corpus privé exige toujours `--output`. Son rapport complet, hashes compris,
est écrit atomiquement en `0600`; stdout ne reçoit qu'un résumé expurgé sans
`cases`, transcription ni empreinte audio.

Le runner désactive la prédiction d'imam, sans rapport avec la qualité du
passage, et appelle `run_inference_pipeline` avec la durée réelle. Il conserve
par défaut `allow_ambiguous_result=true`, comme l'API actuelle ; l'option
`--no-allow-ambiguous-result` permet de mesurer séparément une politique stricte.
Le pipeline de production utilise le VAD uniquement pour isoler la parole et
échantillonner jusqu'à trois fenêtres de langue réparties entre début, milieu et
fin. Si l'arabe reste plausible, il décode ensuite le signal complet en arabe,
sans VAD : les modulations longues d'une récitation ne sont ainsi pas coupées.
Un dither blanc déterministe et imperceptible à 40 dB est appliqué uniquement à
ce second décodage. Il évite qu'un signal propre et très modulé soit tronqué par
Whisper ; le filtrage parole/langue a déjà eu lieu sur le signal original.

Il s'agit d'un E2E **backend**. Les permissions du microphone, l'arrêt explicite
de l'enregistrement et les écrans de résultat restent couverts par les tests
frontend dédiés.

## Métriques et garde-fous

Les résultats positifs distinguent :

- `exact_match` : sourate et plage exactes ;
- `correct_surah_wrong_range` : bonne sourate, mauvais versets ;
- `wrong_surah` ;
- `false_negative` : aucun passage proposé.

Une proposition n'est comptée dans les métriques `confident_*` que si le
pipeline renvoie simultanément un passage et `detection.status=confident`. Les
taux `positive_confident_exact_accuracy` et
`positive_confident_surah_accuracy` gardent **toutes** les récitations positives
au dénominateur : les hypothèses `ambiguous` ne sont donc pas présentées comme
des succès confiants. Leurs équivalents macro par source empêchent aussi les
variantes d'un même enregistrement de gonfler ces résultats.

Les négatifs distinguent `true_negative` et `false_positive`. Le rapport expose
notamment `positive_exact_accuracy`, `positive_surah_accuracy`,
`negative_rejection_rate` et `false_positive_rate`, globalement et par catégorie.
Il expose aussi la distribution des raisons de rejet, les latences moyenne,
p50/p95 et le facteur temps réel. Une erreur technique n'est jamais comptée
comme un vrai négatif. Le rapport embarque également la version de
faster-whisper, le nom non sensible du modèle et les seuils de matching afin de
comparer deux exécutions avec la bonne configuration.

Chaque variante conserve son `source_case_id`. Le rapport donne le nombre de
sources uniques, une macro-moyenne par source et une ventilation par variante :
cinq bruits dérivés d'une seule récitation ne sont donc pas présentés comme cinq
voix indépendantes. À chaque chargement, le SHA-256 de chaque WAV est revérifié ;
le rapport contient aussi l'empreinte du manifeste de corpus.

Le mode par défaut est un **smoke test**. Avec le seul manifeste public,
`quality_gate.evaluated` reste à `false` : six sons non vocaux correctement
rejetés ne prouvent ni le rappel coranique ni le rejet de conversations. Le mode
qualité exige par défaut au moins six enregistrements positifs couvrant quatre
sourates et quatre sources négatives marquées `vocal`. Les six enregistrements
positifs doivent chacun posséder au moins une variante bruitée. Les catégories négatives `french_speech`,
`french_conversation`, `vocal_music` et `arabic_non_quran` doivent toutes être
représentées par au moins une source. `--required-negative-category` permet
d'ajouter une catégorie obligatoire.

Les seuils bloquants portent sur les **macro-métriques par source**, pas sur le
nombre de variantes : `macro_positive_exact_accuracy`,
`macro_positive_surah_accuracy`,
`macro_positive_confident_exact_accuracy`,
`macro_positive_confident_surah_accuracy`,
`macro_negative_rejection_rate` et `macro_false_positive_rate`. Une récitation
déclinée sous six bruits ne peut donc pas masquer l'échec d'un autre locuteur.

Le rapport expose également `distinct_positive_surahs`. Le minimum par défaut
est désormais quatre : il empêche une régression vers un benchmark limité à
Al-Fatiha, sans prétendre représenter les 114 sourates.

Le mode qualité applique par défaut les planchers de non-régression mesurés sur
le corpus de référence : `0.65` pour la plage exacte, `0.85` pour la bonne
sourate, `0.33` pour une plage exacte **confidente**, `0.38` pour une bonne
sourate **confidente**, `1.0` pour le rejet des négatifs et `0.0` faux positif.
Les faibles seuils confiants reflètent honnêtement le point de départ ; ils ne
sont pas une cible produit. Tous ces seuils doivent monter à mesure que le
corpus gagne en diversité et que le pipeline progresse.

Exemple de garde qualité pour une itération avancée :

```bash
docker compose exec api python scripts/evaluate_audio_recognition.py \
  --mode quality \
  --min-macro-positive-exact-accuracy 0.90 \
  --min-macro-positive-surah-accuracy 0.95 \
  --min-macro-positive-confident-exact-accuracy 0.80 \
  --min-macro-positive-confident-surah-accuracy 0.90 \
  --min-macro-negative-rejection-rate 1.0 \
  --max-macro-false-positive-rate 0.0 \
  --min-positive-sources 6 \
  --min-distinct-positive-surahs 4 \
  --min-noisy-positive-sources 6 \
  --min-vocal-negative-sources 4 \
  --max-errors 0
```

La commande termine avec un code non nul si un seuil échoue ou si un ensemble
requis est vide. Cela permet la boucle de travail : construire le corpus,
mesurer, corriger le pipeline dans un commit séparé, puis rejouer exactement le
même corpus avant toute nouvelle modification.

## Baseline mesurée le 22 juillet 2026

La baseline structurée et expurgée est versionnée dans
`audio_quality_baseline.json`. C'est un **snapshot d'un run local manuel**. Le
test automatisé associé vérifie uniquement la cohérence interne du JSON et de
ses seuils ; il ne recharge pas Whisper et ne rejoue aucun WAV. Pour revalider
la chaîne réelle, il faut relancer explicitement le runner E2E puis comparer et
mettre à jour le snapshot. Le rapport détaillé et les audios restent locaux.
Le run de référence utilise faster-whisper `1.1.1`, le modèle `turbo` sur CPU
`int8`, six enregistrements couvrant Al-Fatiha, Al-Ikhlas, Al-Falaq et An-Nas,
avec six variantes par source : propre, bruits blanc 20 dB et 10 dB, bruit rose
10 dB, fond domestique 5 dB et voix française synthétique de type enfant à
5 dB.

| Mesure | Résultat |
|---|---:|
| Plage exacte proposée, ambigu inclus | 25/36 — 69,4 % |
| Bonne sourate proposée, ambigu inclus | 32/36 — 88,9 % |
| Passage proposé avec statut `confident` | 15/36 — 41,7 % |
| Plage exacte avec statut `confident` | 13/36 — 36,1 % |
| Bonne sourate avec statut `confident` | 15/36 — 41,7 % |
| Hypothèses avec statut `ambiguous` | 17/36 — 47,2 % |
| Mauvaise sourate | 0/36 |
| Faux négatifs conservateurs | 4/36 |
| Négatifs correctement rejetés | 15/15 — 100 % |
| Faux positifs négatifs | 0/15 |
| Erreurs techniques | 0/51 |
| Latence moyenne / p50 / p95 | 16,04 s / 17,87 s / 32,65 s |

Les quatre faux négatifs correspondent aux mélanges où la voix française
synthétique domine suffisamment la récitation. Deux conservent une hypothèse
explicitement `probable`, sans être présentées comme certaines. Le pipeline
refuse les deux autres au lieu d'inventer une sourate. Les six enregistrements
ne représentent que trois identités de réciteurs et quatre sourates : ces
chiffres constituent un garde-fou de non-régression, pas une estimation générale
sur les 114 sourates ni sur de vraies voix d'enfants.

Les enregistrements publics du run local proviennent de Wikimedia Commons :
[001Fatiha (CC0)](https://commons.wikimedia.org/wiki/File:001Fatiha.ogg),
[AlFātiḥatulKitāb (CC0)](https://commons.wikimedia.org/wiki/File:AlF%C4%81tihatulKit%C4%81b.ogg),
[Al-Fatiha Mujawwad (CC BY-SA 4.0)](https://commons.wikimedia.org/wiki/File:Chapter_1,_Al-Fatiha_(Mujawwad)_-_Recitation_of_the_Holy_Qur%27an.mp3),
[Al-Ikhlas Murattal (CC BY-SA 4.0)](https://commons.wikimedia.org/wiki/File:Chapter_112,_Al-Ikhlas_(Murattal)_-_Recitation_of_the_Holy_Qur%27an.mp3),
[Al-Falaq Murattal (CC BY-SA 4.0)](https://commons.wikimedia.org/wiki/File:Chapter_113,_Al-Falaq_(Murattal)_v2_-_Recitation_of_the_Holy_Qur%27an.mp3),
[An-Nas Murattal (CC BY-SA 4.0)](https://commons.wikimedia.org/wiki/File:Chapter_114,_Al-Nas_(Murattal)_v2_-_Recitation_of_the_Holy_Qur%27an.mp3),
[Twinkle Twinkle vocal (CC BY-SA 3.0)](https://commons.wikimedia.org/wiki/File:Twinkle_twinkle_little_star_(vocal).ogg) et
[article Wikipédia arabe parlé (CC BY-SA 3.0/GFDL)](https://commons.wikimedia.org/wiki/File:University_Article_Spoken_Arabic_Wikipedia.ogg).
Le manifeste privé local conserve l'URL, l'auteur et la licence de chaque
source ; aucune copie audio n'est versionnée.
