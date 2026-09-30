# Retrait des amorces de recherche

Labels provisoires, non validés humainement. Le corpus, E5-base, ses passages et le classement restent identiques.

Les vingt classements sans nettoyage reproduisent les IDs et scores du benchmark initial.

| Jeu | Requêtes | Top-1 avant | Top-1 après | Top-3 avant | Top-3 après |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benchmark initial | 20 | 65% | 70% | 95% | 95% |
| Amorces synthétiques | 60 | 65% | 70% | 75% | 95% |

Les 60 variations sont construites à partir des mêmes 20 sujets : ce ne sont pas 60 nouveaux cas indépendants. Elles mesurent la résistance aux trois amorces, pas la qualité générale.

## Requêtes modifiées du benchmark initial

- Original : Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère
  - Recherche utilisée : le Prophète conseille à quelqu'un de ne pas se mettre en colère
  - Top 3 avant : 4709, 3396, 4181.
  - Top 3 après : 4709, 5866, 4928.
- Original : Un hadith qui dit que nos actes comptent selon l'intention qu'on avait
  - Recherche utilisée : nos actes comptent selon l'intention qu'on avait
  - Top 3 avant : 4181, 8314, 66511.
  - Top 3 après : 66511, 58226, 4181.
- Original : Le hadith qui explique que la religion est la sincérité
  - Recherche utilisée : la religion est la sincérité
  - Top 3 avant : 66516, 4181, 4309.
  - Top 3 après : 4309, 66516, 4560.

## Échecs Top-3 restant sur le benchmark initial

- Le conseil de parler seulement pour dire du bien sinon se taire — candidats attendus : 5437.

## Essais libres sans réponse unique validée

Les sujets courts peuvent rester ambigus. Le nettoyage n'est pas une garantie d'amélioration pour chaque recherche.

### Je cherche le hadith sur la colère

Recherche utilisée : **la colère**.

Avant :

1. [4181](https://hadeethenc.com/fr/browse/hadith/4181) — Celui qui met à disposition un cheval dans le sentier d’Allah, parce qu’il a foi en Allah et croit en Sa promesse, trouvera dans sa balance, au Jour de la Résurrection, la nourriture de son cheval ainsi que sa boisson, son urine et son crottin.
2. [8266](https://hadeethenc.com/fr/browse/hadith/8266) — Quiconque aime rencontrer Allah, Allah aime le rencontrer ; et quiconque déteste rencontrer Allah, Allah déteste le rencontrer !
3. [3287](https://hadeethenc.com/fr/browse/hadith/3287) — Celui qui contient sa colère alors qu’il pourrait la laisser éclater, Allah - Gloire et Pureté à Lui et qu'Il soit Exalté - l’appellera au Jour de la Résurrection devant tout le monde afin qu’il puisse choisir la Houri de son choix.

Après :

1. [4709](https://hadeethenc.com/fr/browse/hadith/4709) — Ne te mets pas colère !
2. [3743](https://hadeethenc.com/fr/browse/hadith/3743) — Qu’Allah fasse miséricorde à Moïse ! Il fut davantage offensé et pourtant, il se montra patient.
3. [3287](https://hadeethenc.com/fr/browse/hadith/3287) — Celui qui contient sa colère alors qu’il pourrait la laisser éclater, Allah - Gloire et Pureté à Lui et qu'Il soit Exalté - l’appellera au Jour de la Résurrection devant tout le monde afin qu’il puisse choisir la Houri de son choix.

### Donnez moi hadith qui parle de la colère

Recherche utilisée : **la colère**.

Avant :

1. [4181](https://hadeethenc.com/fr/browse/hadith/4181) — Celui qui met à disposition un cheval dans le sentier d’Allah, parce qu’il a foi en Allah et croit en Sa promesse, trouvera dans sa balance, au Jour de la Résurrection, la nourriture de son cheval ainsi que sa boisson, son urine et son crottin.
2. [8266](https://hadeethenc.com/fr/browse/hadith/8266) — Quiconque aime rencontrer Allah, Allah aime le rencontrer ; et quiconque déteste rencontrer Allah, Allah déteste le rencontrer !
3. [3287](https://hadeethenc.com/fr/browse/hadith/3287) — Celui qui contient sa colère alors qu’il pourrait la laisser éclater, Allah - Gloire et Pureté à Lui et qu'Il soit Exalté - l’appellera au Jour de la Résurrection devant tout le monde afin qu’il puisse choisir la Houri de son choix.

Après :

1. [4709](https://hadeethenc.com/fr/browse/hadith/4709) — Ne te mets pas colère !
2. [3743](https://hadeethenc.com/fr/browse/hadith/3743) — Qu’Allah fasse miséricorde à Moïse ! Il fut davantage offensé et pourtant, il se montra patient.
3. [3287](https://hadeethenc.com/fr/browse/hadith/3287) — Celui qui contient sa colère alors qu’il pourrait la laisser éclater, Allah - Gloire et Pureté à Lui et qu'Il soit Exalté - l’appellera au Jour de la Résurrection devant tout le monde afin qu’il puisse choisir la Houri de son choix.

### Je cherche le hadith sur les intentions

Recherche utilisée : **les intentions**.

Avant :

1. [4181](https://hadeethenc.com/fr/browse/hadith/4181) — Celui qui met à disposition un cheval dans le sentier d’Allah, parce qu’il a foi en Allah et croit en Sa promesse, trouvera dans sa balance, au Jour de la Résurrection, la nourriture de son cheval ainsi que sa boisson, son urine et son crottin.
2. [6181](https://hadeethenc.com/fr/browse/hadith/6181) — Lorsqu'un tiers de la nuit était passé, le Prophète (sur lui la paix et le salut) avait l’habitude de se lever et de dire : " Ô gens ! Évoquez Allah !..."
3. [3517](https://hadeethenc.com/fr/browse/hadith/3517) — Par Allah ! Jamais nous ne confions cette tâche à quelqu'un qui la demande ou qui y aspire !

Après :

1. [65047](https://hadeethenc.com/fr/browse/hadith/65047) — N'apprenez pas la science pour vous en pavaner auprès des savants, ni pour vous disputer avec les ignorants
2. [8953](https://hadeethenc.com/fr/browse/hadith/8953) — Ô vous, les gens ! Vous mangez deux plantes que je ne considère pas autrement que mauvaises : l’oignon et l’ail !
3. [58226](https://hadeethenc.com/fr/browse/hadith/58226) — Nous n’utilisons pas pour notre œuvre celui qui la veut [c’est-à-dire : Nous ne confions pas le commandement à celui qui le demande]. Toutefois, toi - Ô Abâ Mûsâ - va au Yémen ! ou, toi - Ô 'Abdallah ibn Qays - va au Yémen !

### Donnez moi un hadith qui parle de la mère

Recherche utilisée : **la mère**.

Avant :

1. [4531](https://hadeethenc.com/fr/browse/hadith/4531) — Ma mère est décédée, mais elle était redevable du jeûne d'un mois. Puis-je m'en acquitter pour elle ? Le Prophète (sur lui la paix et le salut) demanda : Vois-tu si ta mère avait une dette, la règlerais-tu pour elle ? - Certainement, répondit l'homme. - La dette envers Allah est plus en droit d'être acquittée. " conclut le Prophète (sur lui la paix et le salut).
2. [4182](https://hadeethenc.com/fr/browse/hadith/4182) — Ô Messager d’Allah ! Quelle est la personne qui mérite le plus que je lui tienne bonne compagnie ? Ta mère, puis ta mère, puis ta mère, ensuite ton père et enfin du plus proche au plus proche.
3. [4181](https://hadeethenc.com/fr/browse/hadith/4181) — Celui qui met à disposition un cheval dans le sentier d’Allah, parce qu’il a foi en Allah et croit en Sa promesse, trouvera dans sa balance, au Jour de la Résurrection, la nourriture de son cheval ainsi que sa boisson, son urine et son crottin.

Après :

1. [58156](https://hadeethenc.com/fr/browse/hadith/58156) — Surveillez-la ! Si elle accouche d'un enfant blanc aux cheveux lisses et aux yeux teintés de rouge, ce sera l'enfant de Hilâl ibn Umayyah. Mais s'il a les yeux noirs, les cheveux crépus et les jambes fluettes, il sera à Sharîk Ibn Saḥmâ' !
2. [8414](https://hadeethenc.com/fr/browse/hadith/8414) — Un homme a dit : "Je vais certes faire une aumône ! " Il sortit avec son aumône et en fit don à un voleur. Au matin, les gens disaient : " On a fait une aumône à un voleur ! "
3. [6328](https://hadeethenc.com/fr/browse/hadith/6328) — Allah a confié l'utérus à un Ange, qui dit : " Ô Seigneur ! Voici une goutte ! Ô Seigneur ! Voici un caillot ! Ô Seigneur ! Voici un morceau de chair ! " Puis, quand Allah veut achever sa création, l'Ange dit : " Ô Seigneur ! Est-ce un mâle ou une femelle ? Heureux ou malheureux ? Quelle subsistance ? Quel délai de vie ? " Et c'est ainsi que tout sera écrit dans le ventre de sa mère.

## Reproduction

Dans l'environnement Python du lanceur Docker :

```bash
.cache/hadith-cli-venv/bin/python scripts/evaluate_hadith_query_cleanup.py
```

`results.json` contient les requêtes exactes avant/après, tous les Top 5, scores, latences et empreintes de la comparaison.
