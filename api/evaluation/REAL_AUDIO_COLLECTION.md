# Protocole de collecte du corpus audio réel

Ce corpus sert à calibrer la reconnaissance, pas à entraîner Whisper dans un
premier temps. Aucun enregistrement réel ne doit être ajouté au dépôt Git.

## Consentement et stockage

- obtenir un consentement explicite et vérifier le droit d'usage ;
- utiliser un identifiant pseudonyme, jamais un nom de personne ;
- stocker les sources dans `evaluation/private/sources/` ;
- ne jamais inclure une transcription ou un chemin privé dans les logs publics ;
- conserver séparément la preuve de consentement et la table d'identités.

Le manifeste privé doit conserver `consent_confirmed=true` et
`usage_rights_confirmed=true` pour chaque source. Pour un mineur, le consentement
du représentant légal est obligatoire.

## Première cible de couverture

La première phase vise au minimum :

- 200 récitations issues de 20 à 30 réciteurs distincts ;
- plusieurs téléphones, navigateurs, distances et pièces ;
- au moins 20 sourates, avec passages courts et longs ;
- 300 négatifs difficiles : arabe non coranique, français, conversation,
  musique vocale, télévision, silence et bruit ambiant ;
- des fonds vocaux réels consentis, notamment conversation et enfants.

Les variantes synthétiques à plusieurs SNR complètent chaque source propre,
mais ne comptent jamais comme une nouvelle voix.

## Étiquetage

Chaque positif indique la sourate et la plage exacte attendue. Chaque négatif
précise sa catégorie et sa nature vocale ou non vocale. Les erreurs observées
doivent être classées entre transcription, langue, matching, bornes du passage
et politique de décision.

## Séparation des jeux

La séparation développement/test s'effectue par réciteur et source avant toute
augmentation. Deux variantes du même enregistrement ne doivent jamais se
retrouver de part et d'autre de cette séparation.

Les seuils sont calibrés sur le jeu de développement. Le jeu de test reste
figé et sert uniquement à vérifier la précision des résultats confirmés, le
rappel des propositions et le taux de faux positifs sur les négatifs.
