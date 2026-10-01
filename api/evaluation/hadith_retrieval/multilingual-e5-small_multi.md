# Retrieval — intfloat/multilingual-e5-small / multi

Labels : **provisional_candidates_not_human_validated**. Les taux sont provisoires si les labels ne sont pas relus humainement.

Top-1 : 65.0% · Top-3 : 80.0% · Latence moyenne : 10.61 ms.

Latence à chaud : encodage + cosinus + regroupement par ID, hors HTTP.

1790 hadiths, 4803 embeddings ; 0 documents tronqués à 512 tokens.

Sections entièrement perdues : `{}`.

## Échecs Top-3

### Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère

IDs attendus provisoires : 4709

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 3517 | 0.866171 | Par Allah ! Jamais nous ne confions cette tâche à quelqu'un qui la demande ou qui y aspire ! |
| 2 | 5804 | 0.865898 | Je garantis une maison à la périphérie du Paradis à quiconque délaisse la polémique même s’il a raison. |
| 3 | 5934 | 0.865404 | Ne souhaiterais-tu pas que je t'envoie avec ce que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) m'a envoyé ? Ne laisse pas une représentation sans l'effacer, ni une tombe surélevée sans l'aplanir. |
| 4 | 7201 | 0.864942 | Qu’aucun d’entre vous ne prie dans un seul habit, sans rien sur ses épaules. |
| 5 | 11241 | 0.864628 | Je demandai au Messager d'Allah (sur lui la paix et le salut) : "Existe-t-il deux prosternations dans la Sourate : 'Le Pèlerinage' ? - Il dit : Oui, et celui qui ne veut pas les faire ne doit pas les réciter !" |

#### Source attendue : [4709](https://hadeethenc.com/fr/browse/hadith/4709) — rang 7, score 0.863969

Document `title_categories` : **18 tokens**, 18 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 8 | 8 | retained |
| categories | 6 | 6 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Ne te mets pas colère !

Les caractères louables
```

Document `hints` : **117 tokens**, 117 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hints | 36 | 36 | retained |
| hints | 32 | 32 | retained |
| hints | 31 | 31 | retained |
| hints | 14 | 14 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La mise en garde contre la colère et ses causes car elle est l'ensemble du mal et le fait de s'en prémunir est l'ensemble du bien.

La colère pour Allah, comme la colère lors de la violation des interdits sacrés d'Allah fait partie de la colère louable.

La répétition des paroles si besoin est jusqu'à ce que l'interlocuteur les retienne et comprenne leur importance.

Le mérite de la demande de recommandation au savant.
```

Document `hadith_explanation` : **344 tokens**, 344 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hadith | 134 | 134 | retained |
| explanation | 206 | 206 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: D'après  (qu'Allah l'agrée) : Un homme a dit au Prophète (qu'Allah le couvre d'éloges et le préserve) : « Fais-moi une recommandation ! » Il (qu'Allah le couvre d'éloges et le préserve) a dit : «  Ne te mets pas colère !  » L’homme répéta à plusieurs reprises [sa demande] et [à chaque fois] il (qu'Allah le couvre d'éloges et le préserve) a dit : « Ne te mets pas en colère ! »

Un des Compagnons (qu'Allah les agrée) a demandé au Prophète (qu'Allah le couvre d'éloges et le préserve) de lui indiquer une chose qui lui serait bénéfique. Alors, il lui a ordonné de ne pas se mettre en colère. Et la signification de cela est d'éviter les causes qui amènent à la colère et de contrôler sa personne si la colère survient de sorte que cette colère ne soit pas suivie d'un meurtre, ou d'une frappe, ou d'une insulte, ou ce qui ressemble à cela.

Et l'homme a réitéré plusieurs fois sa demande de recommandation mais le Prophète (qu'Allah le couvre d'éloges et le préserve) ne lui a rien rajouté comme recommandation si ce n'est : " Ne te mets pas en colère. "
```

### Le conseil de parler seulement pour dire du bien sinon se taire

IDs attendus provisoires : 5437

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 3055 | 0.861937 | La pudeur n'apporte que du bien. |
| 2 | 66511 | 0.861746 | Les actes ne valent que par les intentions |
| 3 | 7201 | 0.850937 | Qu’aucun d’entre vous ne prie dans un seul habit, sans rien sur ses épaules. |
| 4 | 6988 | 0.849897 | Une chamelle maudite ne doit pas nous accompagner ! |
| 5 | 5804 | 0.847806 | Je garantis une maison à la périphérie du Paradis à quiconque délaisse la polémique même s’il a raison. |

#### Source attendue : [5437](https://hadeethenc.com/fr/browse/hadith/5437) — rang 451, score 0.817142

Document `title_categories` : **37 tokens**, 37 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 27 | 27 | retained |
| categories | 6 | 6 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise

Les caractères louables
```

Document `hints` : **125 tokens**, 125 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hints | 29 | 29 | retained |
| hints | 13 | 13 | retained |
| hints | 19 | 19 | retained |
| hints | 22 | 22 | retained |
| hints | 38 | 38 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi en Allah et au Jour Dernier est la base de tout bien et cela pousse à l'accomplissement du bien.

La mise en garde contre les dégâts de la langue.

La religion de l'islam est une religion de cordialité et de générosité.

Ces bribes font partie des branches de la foi et sont parmi les bonnes manières louables.

La profusion de paroles peut amener à des choses répugnées et interdites ; et la sécurité est dans l'absence de paroles, excepté dans le bien.
```

Document `hadith_explanation` : **346 tokens**, 346 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hadith | 118 | 118 | retained |
| explanation | 224 | 224 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah  (qu'Allah le couvre d'éloges et le préserve) a dit : « Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise. Quiconque croit en Allah et au Jour Dernier, qu'il honore son voisin. Et quiconque croit en Allah et au Jour Dernier, qu'il honore son invité. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que le serviteur qui croit en Allah et au Jour Dernier, dont le retour est vers Lui pour y être rétribué pour ses oeuvres, alors sa foi l'incite à accomplir les bribes suivantes :

La première : Dire de belles paroles : parmi la proclamation de la gloire d'Allah, Son unicité, l'ordonnance du convenable, l'interdiction du blâmable, la réforme entre les gens, etc. Et s'il ne le fait pas, alors qu'il s'attache au silence, qu'il s'abstienne de causer du tort et qu'il préserve sa langue.

La seconde : Honorer le voisin en étant bienfaisant envers lui et en ne lui causant aucun tort.

La troisième  : Honorer l'invité qui vient te visiter en lui parlant agréablement, en lui donnant à manger, et ce qui ressemble à cela.
```

### La récompense des parents qui ont perdu trois enfants avant la puberté

IDs attendus provisoires : 8875

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 65053 | 0.850172 | L'un de vous aimerait-il, en rentrant auprès de sa famille, y trouver trois énormes chamelles pleines et grasses ? |
| 2 | 3260 | 0.847050 | Retourne auprès de tes parents et tiens leur compagnie de la plus belle des manières ! |
| 3 | 8871 | 0.840519 | Il n'est pas une femme parmi vous qui perde trois de ses enfants [littéralement : qui les avance (pour l'au-delà)] sans qu'ils ne soient pour elle une protection contre l'Enfer ! |
| 4 | 8877 | 0.837251 | Lorsque l'homme dit : « Les gens sont perdus ! », il est le plus perdu d'entre eux ! |
| 5 | 8873 | 0.836667 | Il n'est pas un musulman ayant perdu trois de ses enfants qui sera touché par l'Enfer, sauf dans la mesure de l'accomplissement de la promesse. |

#### Source attendue : [8875](https://hadeethenc.com/fr/browse/hadith/8875) — rang 10, score 0.832608

Document `title_categories` : **65 tokens**, 65 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 49 | 49 | retained |
| categories | 12 | 12 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Il n'est pas un musulman qui perde trois de ses enfants, avant que ceux-ci n'aient atteint la puberté, sans qu'Allah ne l'introduise au Paradis par Sa miséricorde envers eux.

Les caractéristiques du Paradis et de l'Enfer
```

Document `hadith_explanation` : **132 tokens**, 132 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hadith | 83 | 83 | retained |
| explanation | 45 | 45 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Anas (qu'Allah l'agrée) relate que le Messager d'Allah (sur lui la paix et le salut) a dit : « Il n'est pas un musulman qui perde trois de ses enfants, avant que ceux-ci n'aient atteint la puberté, sans qu'Allah ne l'introduise au Paradis par Sa miséricorde envers eux. »

Aucun musulman ne voit mourir trois de ses enfants, garçons ou filles, qui n'ont pas atteint l'âge de puberté, sans que cela ne soit pour lui une cause d'entrée au Paradis.
```

### Enlever un obstacle de la route fait partie des branches de la foi

IDs attendus provisoires : 3276, 6468

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 5478 | 0.878546 | La pudeur fait partie de la foi |
| 2 | 66511 | 0.857964 | Les actes ne valent que par les intentions |
| 3 | 6387 | 0.855303 | Revenir du combat équivaut à combattre. |
| 4 | 66516 | 0.854668 | La religion, c’est la sincérité |
| 5 | 4188 | 0.854331 | Quiconque effectue une dépense dans le sentier d’Allah, elle lui sera inscrite sept cents fois ! |

#### Source attendue : [3276](https://hadeethenc.com/fr/browse/hadith/3276) — rang 33, score 0.841633

Document `title_categories` : **97 tokens**, 97 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 83 | 83 | retained |
| categories | 10 | 10 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi.

Les branches / ramifications de la foi
```

Document `hadith_explanation` : **304 tokens**, 304 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hadith | 120 | 120 | retained |
| explanation | 180 | 180 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Abû Hurayrah (qu’Allah l’agrée) relate que le Messager d’Allah (sur lui la paix et le salut) a dit : « La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi. »

La foi ne se résume pas à une seule caractéristique ou une seule branche. Elle est composée de plusieurs branches : un peu plus de soixante ou soixante-dix. La meilleure de ces branches est la parole qui atteste qu’il n’y a aucune divinité qui mérite l’adoration en dehors d’Allah et la plus infime consiste à ôter du chemin ce qui est nuisible pour les passants comme une pierre, une branche épineuse, ou autre. Être pudique est également une branche de la foi. Ainsi, les actes font partie de foi selon les gens de la tradition et du groupe (« Ahl as-sunnah wa-l-jamâ'ah ») ; et c'est la vérité qu'indiquent les textes et celle-ci en fait partie.
```


#### Source attendue : [6468](https://hadeethenc.com/fr/browse/hadith/6468) — rang 158, score 0.832577

Document `title_categories` : **83 tokens**, 83 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 68 | 68 | retained |
| categories | 11 | 11 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin

L'augmentation de la Foi et sa diminution
```

Document `hints` : **158 tokens**, 158 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hints | 18 | 18 | retained |
| hints | 11 | 11 | retained |
| hints | 60 | 60 | retained |
| hints | 65 | 65 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi a des niveaux, certains d'entre eux sont meilleurs que d'autres.

La foi est parole, acte et croyance.

La pudeur à l'égard d'Allah, Exalté soit-Il, implique qu'Il ne te voit pas là où Il t'a interdit d'être, et qu'Il ne te trouve pas absent là où Il t'a ordonné d'être.

La mention du nombre [des branches de la foi] ne signifie pas qu'elle y soit limitée. Cela indique plutôt la multitude des œuvres de celle-ci. En effet, les Arabes mentionnaient parfois un nombre pour une chose, sans pour autant  vouloir infirmer autre que lui.
```

Document `hadith_explanation` : **362 tokens**, 362 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| hadith | 125 | 125 | retained |
| explanation | 233 | 233 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) a dit : « La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin, et la pudeur est une branche de la foi. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) informe que la foi possède de nombreuses branches et caractéristiques qui englobent des œuvres, des croyances et des paroles.

Que la plus haute et la meilleure des caractéristiques de la foi est de dire : " Il n'est de divinité [digne d'adoration] qu'Allah " en connaissant sa signification et en œuvrant selon ses implications, à savoir qu'Allah est le Seul et Unique Dieu qui soit digne d'adoration, Lui Seul, et sans qui ou quoi que ce soit d'autre.

Et que la moindre des œuvres de la foi consiste à ôter ce qui nuit aux gens sur leurs chemins.

Ensuite, il a informé (qu'Allah le couvre d'éloges et le préserve) que la pudeur fait partie des caractéristiques de la foi. C'est un comportement qui pousse à l'accomplissement de ce qui est beau et au délaissement de ce qui est laid.
```
