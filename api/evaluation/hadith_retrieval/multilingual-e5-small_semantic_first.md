# Retrieval — intfloat/multilingual-e5-small / semantic_first

Labels : **provisional_candidates_not_human_validated**. Les taux sont provisoires si les labels ne sont pas relus humainement.

Top-1 : 50.0% · Top-3 : 60.0% · Latence moyenne : 9.66 ms.

Latence à chaud : encodage + cosinus + regroupement par ID, hors HTTP.

1790 hadiths, 1790 embeddings ; 796 documents tronqués à 512 tokens.

Sections entièrement perdues : `{"explanation": 183, "hadith": 15, "hints": 13}`.

## Échecs Top-3

### Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère

IDs attendus provisoires : 4709

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 6108 | 0.863738 | Un homme demanda : " Ô, Messager d'Allah ! Quand quelqu'un parmi nous rencontre son frère ou son ami, peut-il s'incliner devant lui ? - Non, dit-il. - Il dit : Peut-il le serrer contre lui et l'embrasser ? - Non, dit-il. - Il dit : Alors, il le prend par la main et lui serre la main ? - Oui, dit-il. " |
| 2 | 8929 | 0.863605 | Parmi les bonnes œuvres que le messager d'Allah (sur lui la paix et le salut) nous a engagées à accomplir, il y a le fait de ne pas lui désobéir, de ne pas nous griffer le visage, de ne pas invoquer le malheur contre nous, de ne pas déchirer nos vêtements et de ne pas nous tirer les cheveux. |
| 3 | 11227 | 0.862887 | Par Celui qui détient mon âme dans Sa main ! Une époque viendra à l'homme où le tueur ne saura pas pourquoi il tue, et où celui qui est tué ne saura pas pourquoi il a été tué ! |
| 4 | 10971 | 0.861370 | Le Prophète (sur lui la paix et le salut) est venu me rendre visite sans monter une mule ou un cheval. |
| 5 | 64691 | 0.860011 | Voulez-vous que je vous informe du meilleur des témoins ? C'est celui qui apporte son témoignage avant qu'on l'interroge ! |

#### Source attendue : [4709](https://hadeethenc.com/fr/browse/hadith/4709) — rang 22, score 0.855219

Document `semantic_first` : **471 tokens**, 471 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 8 | 8 | retained |
| categories | 6 | 6 | retained |
| hints | 36 | 36 | retained |
| hints | 32 | 32 | retained |
| hints | 31 | 31 | retained |
| hints | 14 | 14 | retained |
| hadith | 134 | 134 | retained |
| explanation | 206 | 206 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Ne te mets pas colère !

Les caractères louables

La mise en garde contre la colère et ses causes car elle est l'ensemble du mal et le fait de s'en prémunir est l'ensemble du bien.

La colère pour Allah, comme la colère lors de la violation des interdits sacrés d'Allah fait partie de la colère louable.

La répétition des paroles si besoin est jusqu'à ce que l'interlocuteur les retienne et comprenne leur importance.

Le mérite de la demande de recommandation au savant.

D'après  (qu'Allah l'agrée) : Un homme a dit au Prophète (qu'Allah le couvre d'éloges et le préserve) : « Fais-moi une recommandation ! » Il (qu'Allah le couvre d'éloges et le préserve) a dit : «  Ne te mets pas colère !  » L’homme répéta à plusieurs reprises [sa demande] et [à chaque fois] il (qu'Allah le couvre d'éloges et le préserve) a dit : « Ne te mets pas en colère ! »

Un des Compagnons (qu'Allah les agrée) a demandé au Prophète (qu'Allah le couvre d'éloges et le préserve) de lui indiquer une chose qui lui serait bénéfique. Alors, il lui a ordonné de ne pas se mettre en colère. Et la signification de cela est d'éviter les causes qui amènent à la colère et de contrôler sa personne si la colère survient de sorte que cette colère ne soit pas suivie d'un meurtre, ou d'une frappe, ou d'une insulte, ou ce qui ressemble à cela.

Et l'homme a réitéré plusieurs fois sa demande de recommandation mais le Prophète (qu'Allah le couvre d'éloges et le préserve) ne lui a rien rajouté comme recommandation si ce n'est : " Ne te mets pas en colère. "
```

### Un hadith qui dit que nos actes comptent selon l'intention qu'on avait

IDs attendus provisoires : 4560, 66511

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 3753 | 0.859357 | Un homme avait l'habitude de prêter de l'argent aux gens. Il disait à son servant : ' Si tu rencontres une personne en difficulté, sois conciliant ! Peut-être qu’Allah le sera avec nous |
| 2 | 10104 | 0.858623 | Le temps a tourné comme le jour où Allah a créé les cieux et la terre. L'année compte douze mois, dont quatre sacrés, trois qui se suivent : Dhul Qa'dah, Dhul Ḥijja, Al-Muḥarram et Rajab, le mois de Muḍar. |
| 3 | 8929 | 0.858390 | Parmi les bonnes œuvres que le messager d'Allah (sur lui la paix et le salut) nous a engagées à accomplir, il y a le fait de ne pas lui désobéir, de ne pas nous griffer le visage, de ne pas invoquer le malheur contre nous, de ne pas déchirer nos vêtements et de ne pas nous tirer les cheveux. |
| 4 | 4810 | 0.858379 | Ô Mes serviteurs ! Certes, Je Me suis interdit l’injustice et Je l’ai rendue interdite entre vous. Ne soyez donc pas injustes les uns envers les autres ! |
| 5 | 3757 | 0.857523 | Nous apportions au Prophète (sur lui la paix et le salut) sa part de lait. Il revenait au cours de la nuit et nous saluait de sorte à se faire entendre de celui qui était éveillé sans pour autant réveiller celui qui dormait. |

#### Source attendue : [4560](https://hadeethenc.com/fr/browse/hadith/4560) — rang 1125, score 0.824337

Document `semantic_first` : **727 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 27 | 27 | retained |
| categories | 6 | 6 | retained |
| hints | 30 | 30 | retained |
| hints | 81 | 81 | retained |
| hadith | 205 | 205 | retained |
| explanation | 374 | 159 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Certes, les œuvres ne valent que par les intentions et chaque individu sera rétribué en fonction de son intention

Les oeuvres du coeur

L'incitation à la sincérité. En effet, Allah n'accepte comme oeuvre que celle dont on recherche Son Visage.

Lorsque la personne responsable accomplit des oeuvres par lesquelles on peut se rapprocher d'Allah, Exalté soit-Il, mais elle les fait de manière habituelle, alors elle n'obtiendra pas de rétribution pour celles-ci jusqu'à ce qu'elle ait l'intention de se rapprocher d'Allah à travers ces oeuvres.

'Oumar ibn Al-Khaṭṭâb (qu'Allah l'agrée) relate que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) a dit : « Certes, les œuvres ne valent que par les intentions et l'individu sera rétribué en fonction de son intention. Ainsi donc, quiconque dont l'émigration est vers Allah et Son , alors son émigration sera vers Allah et Son  ; et quiconque dont l'émigration est pour un intérêt mondain qu'il veut acquérir ou pour une femme qu'il veut épouser, alors son émigration sera pour ce vers quoi il a émigré. » Et dans l'expression d'Al Bukhârî : «  Certes, les œuvres ne valent que par les intentions et chaque individu sera rétribué en fonction de son intention. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que toutes les œuvres sont considérées en fonction de l'intention. Ce décret est général et concerne l'ensemble des œuvres, qu'il s'agisse des adorations ou des transactions externes. Ainsi donc, quiconque vise par son œuvre l'acquisition d'un profit n'obtiendra que cet avantage et n'aura pas de rétribution ; et quiconque œuvre dans le but de se rapprocher d'Allah, Exalté soit-Il, obtiendra la rétribution et la récompense de son œuvre, même s'il s'agit d'une œuvre habituelle et normale, comme le fait de manger et boire.

Ensuite, il (qu'Allah le couvre d'éloges et le préserve) a cité un exemple pour montrer l'effet de l'intention sur les oeuvres même si elles sont similaires dans la forme apparente. Il a expliqué que quiconque vise la satisfaction de son Seigneur à travers son émigration et le délaissement de son pays, alors son émigration est une émigration religieuse légale acceptée pour laquelle la personne sera rétribuée en raison de la véracité de son intention. Mais quiconque vise un avantage mondain à travers son émigration, que ce soit de l'argent ou un bien, une position, un commerce, une épouse, etc. Alors, la personne n'obtiendra de son émigration que cet avantage dont il a eu comme intention et il n'aura aucune part de la récompense et de la rétribution [pour cette émigration].
```


#### Source attendue : [66511](https://hadeethenc.com/fr/browse/hadith/66511) — rang 270, score 0.838594

Document `semantic_first` : **697 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 10 | 10 | retained |
| categories | 6 | 6 | retained |
| hints | 30 | 30 | retained |
| hints | 81 | 81 | retained |
| hints | 27 | 27 | retained |
| hadith | 165 | 165 | retained |
| explanation | 374 | 189 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Les actes ne valent que par les intentions

Les oeuvres du coeur

L'incitation à la sincérité. En effet, Allah n'accepte comme oeuvre que celle dont on recherche Son Visage.

Lorsque la personne responsable accomplit des oeuvres par lesquelles on peut se rapprocher d'Allah, Exalté soit-Il, mais elle les fait de manière habituelle, alors elle n'obtiendra pas de rétribution pour celles-ci jusqu'à ce qu'elle ait l'intention de se rapprocher d'Allah à travers ces oeuvres.

L’intention permet de distinguer les cultes les uns des autres, et de distinguer les cultes des habitudes.

D’après l’Émir des Croyants, Abû Hafs ‘Umar b. al‑Khattâb — qu’Allah l’agrée — qui a dit: j’ai entendu le Messager d’Allah ﷺ dire : « Les actes ne valent que par les intentions, et chacun n’aura que ce qu’il a eu l’intention de faire. Celui dont l’émigration fut pour Allah et Son Messager, son émigration est pour Allah et Son Messager ; et celui dont l’émigration fut pour un intérêt mondain qu’il convoitait ou une femme qu’il voulait épouser, son émigration n’est que vers ce pour quoi il a émigré. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que toutes les œuvres sont considérées en fonction de l'intention. Ce décret est général et concerne l'ensemble des œuvres, qu'il s'agisse des adorations ou des transactions externes. Ainsi donc, quiconque vise par son œuvre l'acquisition d'un profit n'obtiendra que cet avantage et n'aura pas de rétribution ; et quiconque œuvre dans le but de se rapprocher d'Allah, Exalté soit-Il, obtiendra la rétribution et la récompense de son œuvre, même s'il s'agit d'une œuvre habituelle et normale, comme le fait de manger et boire.

Ensuite, il (qu'Allah le couvre d'éloges et le préserve) a cité un exemple pour montrer l'effet de l'intention sur les oeuvres même si elles sont similaires dans la forme apparente. Il a expliqué que quiconque vise la satisfaction de son Seigneur à travers son émigration et le délaissement de son pays, alors son émigration est une émigration religieuse légale acceptée pour laquelle la personne sera rétribuée en raison de la véracité de son intention. Mais quiconque vise un avantage mondain à travers son émigration, que ce soit de l'argent ou un bien, une position, un commerce, une épouse, etc. Alors, la personne n'obtiendra de son émigration que cet avantage dont il a eu comme intention et il n'aura aucune part de la récompense et de la rétribution [pour cette émigration].
```

### Il faut souhaiter pour son frère ce qu'on souhaite pour soi

IDs attendus provisoires : 4717, 66520

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 6108 | 0.857282 | Un homme demanda : " Ô, Messager d'Allah ! Quand quelqu'un parmi nous rencontre son frère ou son ami, peut-il s'incliner devant lui ? - Non, dit-il. - Il dit : Peut-il le serrer contre lui et l'embrasser ? - Non, dit-il. - Il dit : Alors, il le prend par la main et lui serre la main ? - Oui, dit-il. " |
| 2 | 6460 | 0.848652 | Lorsque l'homme dépense pour sa famille et en escompte la récompense, c'est pour lui une aumône. |
| 3 | 5348 | 0.844236 | Ne méprise rien du bien, ne serait-ce que de rencontrer ton frère avec un visage souriant. " |
| 4 | 4717 | 0.840921 | Aucun de vous ne croira vraiment jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même. |
| 5 | 5926 | 0.840693 | Il était trois hommes parmi les fils d'Israël : un lépreux, un chauve et un aveugle. Allah voulut alors les éprouver et leur envoya un Ange. |

#### Source attendue : [4717](https://hadeethenc.com/fr/browse/hadith/4717) — rang 4, score 0.840921

Document `semantic_first` : **531 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 28 | 28 | retained |
| categories | 6 | 6 | retained |
| hints | 68 | 68 | retained |
| hints | 29 | 29 | retained |
| hints | 59 | 59 | retained |
| hints | 26 | 26 | retained |
| hints | 97 | 97 | retained |
| hadith | 56 | 56 | retained |
| explanation | 158 | 139 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Aucun de vous ne croira vraiment jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même.

Les caractères louables

L'obligation de l'amour de l'individu pour son frère tout comme il l'aime pour lui-même ; cela parce que l'infirmation de la foi envers quiconque n'aime pas pour son frère ce qu'il aime pour lui-même indique l'obligation de cela.

La fraternité en Allah est au-dessus de la fraternité de filiation. En effet, son droit est encore plus obligatoire.

L'interdiction de tout ce qui infirme cet amour parmi des paroles et des actes, comme : la tricherie, la médisance, l'envie et l'inimitié contre la personne du musulman, son argent, ou son honneur.

L'emploi de certaines expressions stimulantes à l'action, en raison de sa parole : " pour son frère. "

Al Karmânî (qu'Allah lui fasse miséricorde) a dit : " Et parmi la foi, il y a aussi le fait de détester pour son frère ce que l'on déteste pour soi comme mal et qu'il n'a pas mentionné ; cela parce que l'amour d'une chose implique nécessairement certains de ses opposés. Ainsi, le fait de ne pas l'avoir mentionné textuellement suffit largement. "

D'après Anas (qu'Allah l'agrée) : Le Prophète ﷺ a dit : " Aucun de vous ne croira vraiment jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même. "

Le Prophète ﷺ a expliqué que la foi complète de quiconque parmi les musulmans ne se concrétisera pas jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même comme obéissances et types de biens dans la religion et la vie d'ici-bas ; et il répugne pour lui ce qu'il répugne pour lui-même. Ainsi, s'il voit chez son frère musulman un manque dans sa religion, alors il s'efforce de le réformer ; et s'il voit un bien en lui, alors il l'encourage, l'aide et le conseille sincèrement concernant son affaire religieuse ou son affaire mondaine.
```


#### Source attendue : [66520](https://hadeethenc.com/fr/browse/hadith/66520) — rang 7, score 0.839806

Document `semantic_first` : **569 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 28 | 28 | retained |
| categories | 6 | 6 | retained |
| hints | 68 | 68 | retained |
| hints | 29 | 29 | retained |
| hints | 59 | 59 | retained |
| hints | 26 | 26 | retained |
| hints | 97 | 97 | retained |
| hadith | 94 | 94 | retained |
| explanation | 158 | 101 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Aucun de vous ne croira vraiment jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même.

Les caractères louables

L'obligation de l'amour de l'individu pour son frère tout comme il l'aime pour lui-même ; cela parce que l'infirmation de la foi envers quiconque n'aime pas pour son frère ce qu'il aime pour lui-même indique l'obligation de cela.

La fraternité en Allah est au-dessus de la fraternité de filiation. En effet, son droit est encore plus obligatoire.

L'interdiction de tout ce qui infirme cet amour parmi des paroles et des actes, comme : la tricherie, la médisance, l'envie et l'inimitié contre la personne du musulman, son argent, ou son honneur.

L'emploi de certaines expressions stimulantes à l'action, en raison de sa parole : " pour son frère. "

Al Karmânî (qu'Allah lui fasse miséricorde) a dit : " Et parmi la foi, il y a aussi le fait de détester pour son frère ce que l'on déteste pour soi comme mal et qu'il n'a pas mentionné ; cela parce que l'amour d'une chose implique nécessairement certains de ses opposés. Ainsi, le fait de ne pas l'avoir mentionné textuellement suffit largement. "

D’après Abû Ḥamza Anas b. Mâlik — qu’Allah l’agrée —, et d’après Abû Ya‘la Shaddâd b. Aws — qu’Allah l’agrée —, le Prophète ﷺ a dit : " Aucun de vous ne croira vraiment jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même. "

Le Prophète ﷺ a expliqué que la foi complète de quiconque parmi les musulmans ne se concrétisera pas jusqu'à ce qu'il aime pour son frère ce qu'il aime pour lui-même comme obéissances et types de biens dans la religion et la vie d'ici-bas ; et il répugne pour lui ce qu'il répugne pour lui-même. Ainsi, s'il voit chez son frère musulman un manque dans sa religion, alors il s'efforce de le réformer ; et s'il voit un bien en lui, alors il l'encourage, l'aide et le conseille sincèrement concernant son affaire religieuse ou son affaire mondaine.
```

### Le conseil de parler seulement pour dire du bien sinon se taire

IDs attendus provisoires : 5437

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 3055 | 0.837416 | La pudeur n'apporte que du bien. |
| 2 | 3311 | 0.835855 | Ô Abâ Dharr ! Je pense que tu es faible, et j’aime pour toi ce que j’aime pour moi-même. Ne prends pas le commandement de deux personnes, et ne te charge pas de la tutelle des biens de l’orphelin ! |
| 3 | 4952 | 0.835673 | Ne dis pas : « Sur toi la paix ! » car c’est ainsi que l’on salue les morts. Dis plutôt : « Que la paix soit sur toi ! » |
| 4 | 5807 | 0.835670 | Ils ne m’ont laissé d’autres choix que de me solliciter avec rudesse ou de me traiter d’avare. Or, je ne suis pas avare ! |
| 5 | 6094 | 0.833833 | « Qu’un voisin n’empêche pas son voisin de faire passer sa poutre dans son mur. » Ensuite, Abû Hurayrah (qu’Allah l’agrée) dit : " Pourquoi vous vois-je vous en détourner ? Par Allah ! Je vais vous les jeter sur le dos ! " |

#### Source attendue : [5437](https://hadeethenc.com/fr/browse/hadith/5437) — rang 151, score 0.820395

Document `semantic_first` : **500 tokens**, 500 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 27 | 27 | retained |
| categories | 6 | 6 | retained |
| hints | 29 | 29 | retained |
| hints | 13 | 13 | retained |
| hints | 19 | 19 | retained |
| hints | 22 | 22 | retained |
| hints | 38 | 38 | retained |
| hadith | 118 | 118 | retained |
| explanation | 224 | 224 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise

Les caractères louables

La foi en Allah et au Jour Dernier est la base de tout bien et cela pousse à l'accomplissement du bien.

La mise en garde contre les dégâts de la langue.

La religion de l'islam est une religion de cordialité et de générosité.

Ces bribes font partie des branches de la foi et sont parmi les bonnes manières louables.

La profusion de paroles peut amener à des choses répugnées et interdites ; et la sécurité est dans l'absence de paroles, excepté dans le bien.

Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah  (qu'Allah le couvre d'éloges et le préserve) a dit : « Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise. Quiconque croit en Allah et au Jour Dernier, qu'il honore son voisin. Et quiconque croit en Allah et au Jour Dernier, qu'il honore son invité. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que le serviteur qui croit en Allah et au Jour Dernier, dont le retour est vers Lui pour y être rétribué pour ses oeuvres, alors sa foi l'incite à accomplir les bribes suivantes :

La première : Dire de belles paroles : parmi la proclamation de la gloire d'Allah, Son unicité, l'ordonnance du convenable, l'interdiction du blâmable, la réforme entre les gens, etc. Et s'il ne le fait pas, alors qu'il s'attache au silence, qu'il s'abstienne de causer du tort et qu'il préserve sa langue.

La seconde : Honorer le voisin en étant bienfaisant envers lui et en ne lui causant aucun tort.

La troisième  : Honorer l'invité qui vient te visiter en lui parlant agréablement, en lui donnant à manger, et ce qui ressemble à cela.
```

### Je cherche le conseil de faciliter les choses et de ne pas faire fuir les gens

IDs attendus provisoires : 5866

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 10563 | 0.844947 | Cherche-moi des pierres pour me nettoyer et ne m'apporte ni os, ni crottin ! |
| 2 | 6368 | 0.838939 | Tu te dois d’écouter et d’obéir, dans la difficulté comme dans l’aisance, dans ce qui te plaît comme dans ce qui te déplaît, et même si tu bénéficies de privilèges. |
| 3 | 5948 | 0.838747 | Cette façon de vous disperser dans les sentiers et les oueds ne vous est inspirée que par le diable ! |
| 4 | 4295 | 0.838374 | Laissez-moi ! Je ne vous ai pas délaissés. En fait, ce qui a anéanti quiconque était avant vous fut leurs [incessantes] questions et leurs divergences avec leurs Prophètes |
| 5 | 4813 | 0.838116 | On m'a présenté les actes de ma communauté, les bons comme les mauvais. J’ai constaté que l’une de leurs belles œuvres était le fait d’ôter du chemin les choses nuisibles et que l’une de leurs mauvaises œuvres était la glaire laissée dans la mosquée sans être enfouie. |

#### Source attendue : [5866](https://hadeethenc.com/fr/browse/hadith/5866) — rang 14, score 0.834461

Document `semantic_first` : **416 tokens**, 416 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 22 | 22 | retained |
| categories | 6 | 6 | retained |
| hints | 23 | 23 | retained |
| hints | 30 | 30 | retained |
| hints | 42 | 42 | retained |
| hints | 37 | 37 | retained |
| hints | 52 | 52 | retained |
| hints | 17 | 17 | retained |
| hadith | 66 | 66 | retained |
| explanation | 117 | 117 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Facilitez et ne rendez pas difficile ! Annoncez de bonnes nouvelles et ne faites pas fuir !

Les caractères louables

Le devoir du croyant est de faire aimer Allah aux gens et de les encourager dans le bien.

Il convient à celui qui invite les gens à Allah d'observer avec sagesse comment transmettre l'appel de l'Islam aux gens.

Le fait d'annoncer les bonnes nouvelles fait naître la joie, l'acceptation et l'apaisement du coeur quant au prédicateur et à ce qu'il présente aux gens.

Le fait de rendre compliqué fait naître l'envie de fuir, de se détourner et le doute vis-à-vis des paroles du prédicateur.

L'étendue de la miséricorde d'Allah envers Ses serviteurs et le fait qu'Il leur a agréé une religion bienveillante et une Charî'ah (Législation) facilitée.

La facilité ordonnée est ce avec quoi la Charî'ah est venue.

Anas ibn Mâlik (qu'Allah l'agrée) relate que le Prophète (qu'Allah le couvre d'éloges et le préserve) a dit : « Facilitez et ne rendez pas difficile ! Annoncez de bonnes nouvelles et ne faites pas fuir ! »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) ordonne d'alléger et de faciliter les choses pour les gens et de ne pas leur rendre compliqué les choses, qu'il s'agisse de leurs affaires religieuses et mondaines. Et ceci, dans les limites de ce qu'Allah a autorisé et prescrit.

Il encourage aussi (qu'Allah le couvre d'éloges et le préserve) à leur faire la bonne annonce du bien et à ne pas les en faire fuir.
```

### La récompense des parents qui ont perdu trois enfants avant la puberté

IDs attendus provisoires : 8875

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 10417 | 0.838209 | Trois hommes qui étaient de sortie furent surpris par la pluie alors qu'ils marchaient. Ils cherchèrent aussitôt refuge à l'intérieur d'une grotte qui se trouvait dans la montagne, un rocher tomba soudainement et vint obstruer la sortie de la grotte. |
| 2 | 65053 | 0.835140 | L'un de vous aimerait-il, en rentrant auprès de sa famille, y trouver trois énormes chamelles pleines et grasses ? |
| 3 | 8871 | 0.834455 | Il n'est pas une femme parmi vous qui perde trois de ses enfants [littéralement : qui les avance (pour l'au-delà)] sans qu'ils ne soient pour elle une protection contre l'Enfer ! |
| 4 | 8875 | 0.834084 | Il n'est pas un musulman qui perde trois de ses enfants, avant que ceux-ci n'aient atteint la puberté, sans qu'Allah ne l'introduise au Paradis par Sa miséricorde envers eux. |
| 5 | 4959 | 0.832210 | Ô Messager d’Allah ! Aurais-je une récompense par rapport aux enfants d’Abû Salamah du fait que je dépense pour eux et que je ne les abandonne pas à leur propre sort, bien qu'ils soient mes enfants ? Il répondit : « Oui, tu auras la récompense de ce que tu auras dépensé pour eux ! |

#### Source attendue : [8875](https://hadeethenc.com/fr/browse/hadith/8875) — rang 4, score 0.834084

Document `semantic_first` : **193 tokens**, 193 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 49 | 49 | retained |
| categories | 12 | 12 | retained |
| hadith | 83 | 83 | retained |
| explanation | 45 | 45 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Il n'est pas un musulman qui perde trois de ses enfants, avant que ceux-ci n'aient atteint la puberté, sans qu'Allah ne l'introduise au Paradis par Sa miséricorde envers eux.

Les caractéristiques du Paradis et de l'Enfer

Anas (qu'Allah l'agrée) relate que le Messager d'Allah (sur lui la paix et le salut) a dit : « Il n'est pas un musulman qui perde trois de ses enfants, avant que ceux-ci n'aient atteint la puberté, sans qu'Allah ne l'introduise au Paradis par Sa miséricorde envers eux. »

Aucun musulman ne voit mourir trois de ses enfants, garçons ou filles, qui n'ont pas atteint l'âge de puberté, sans que cela ne soit pour lui une cause d'entrée au Paradis.
```

### Enlever un obstacle de la route fait partie des branches de la foi

IDs attendus provisoires : 3276, 6468

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 5434 | 0.849955 | Celui qui a peur part en voyage aux premières heures de la nuit, et celui qui part en voyage aux premières heures de la nuit arrive à bon port. Notez-bien que la marchandise d’Allah est précieuse, Notez bien que la marchandise d’Allah est le Paradis ! |
| 2 | 4813 | 0.849072 | On m'a présenté les actes de ma communauté, les bons comme les mauvais. J’ai constaté que l’une de leurs belles œuvres était le fait d’ôter du chemin les choses nuisibles et que l’une de leurs mauvaises œuvres était la glaire laissée dans la mosquée sans être enfouie. |
| 3 | 8309 | 0.848207 | Prenez garde aux invocations de celui qui subit l’injustice, car elles montent au ciel comme si elles étaient des flammes ! |
| 4 | 10869 | 0.846474 | Lorsque que l'un d’entre vous fait la prière, qu'il prie devant un obstacle, ne serait-ce qu'une flèche ! |
| 5 | 5948 | 0.845660 | Cette façon de vous disperser dans les sentiers et les oueds ne vous est inspirée que par le diable ! |

#### Source attendue : [3276](https://hadeethenc.com/fr/browse/hadith/3276) — rang 6, score 0.845280

Document `semantic_first` : **397 tokens**, 397 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 83 | 83 | retained |
| categories | 10 | 10 | retained |
| hadith | 120 | 120 | retained |
| explanation | 180 | 180 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi.

Les branches / ramifications de la foi

Abû Hurayrah (qu’Allah l’agrée) relate que le Messager d’Allah (sur lui la paix et le salut) a dit : « La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi. »

La foi ne se résume pas à une seule caractéristique ou une seule branche. Elle est composée de plusieurs branches : un peu plus de soixante ou soixante-dix. La meilleure de ces branches est la parole qui atteste qu’il n’y a aucune divinité qui mérite l’adoration en dehors d’Allah et la plus infime consiste à ôter du chemin ce qui est nuisible pour les passants comme une pierre, une branche épineuse, ou autre. Être pudique est également une branche de la foi. Ainsi, les actes font partie de foi selon les gens de la tradition et du groupe (« Ahl as-sunnah wa-l-jamâ'ah ») ; et c'est la vérité qu'indiquent les textes et celle-ci en fait partie.
```


#### Source attendue : [6468](https://hadeethenc.com/fr/browse/hadith/6468) — rang 213, score 0.827429

Document `semantic_first` : **595 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 68 | 68 | retained |
| categories | 11 | 11 | retained |
| hints | 18 | 18 | retained |
| hints | 11 | 11 | retained |
| hints | 60 | 60 | retained |
| hints | 65 | 65 | retained |
| hadith | 125 | 125 | retained |
| explanation | 233 | 150 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin

L'augmentation de la Foi et sa diminution

La foi a des niveaux, certains d'entre eux sont meilleurs que d'autres.

La foi est parole, acte et croyance.

La pudeur à l'égard d'Allah, Exalté soit-Il, implique qu'Il ne te voit pas là où Il t'a interdit d'être, et qu'Il ne te trouve pas absent là où Il t'a ordonné d'être.

La mention du nombre [des branches de la foi] ne signifie pas qu'elle y soit limitée. Cela indique plutôt la multitude des œuvres de celle-ci. En effet, les Arabes mentionnaient parfois un nombre pour une chose, sans pour autant  vouloir infirmer autre que lui.

Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) a dit : « La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin, et la pudeur est une branche de la foi. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) informe que la foi possède de nombreuses branches et caractéristiques qui englobent des œuvres, des croyances et des paroles.

Que la plus haute et la meilleure des caractéristiques de la foi est de dire : " Il n'est de divinité [digne d'adoration] qu'Allah " en connaissant sa signification et en œuvrant selon ses implications, à savoir qu'Allah est le Seul et Unique Dieu qui soit digne d'adoration, Lui Seul, et sans qui ou quoi que ce soit d'autre.

Et que la moindre des œuvres de la foi consiste à ôter ce qui nuit aux gens sur leurs chemins.

Ensuite, il a informé (qu'Allah le couvre d'éloges et le préserve) que la pudeur fait partie des caractéristiques de la foi. C'est un comportement qui pousse à l'accomplissement de ce qui est beau et au délaissement de ce qui est laid.
```

### Allah accorde sa miséricorde à ceux qui sont miséricordieux

IDs attendus provisoires : 6405

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 7185 | 0.884699 | Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il dit alors : Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il dit à nouveau : Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il répondit : Et à ceux qui se les coupent ! |
| 2 | 3556 | 0.883685 | Quand Allah créa Adam (sur lui la paix et le salut), Il [lui] dit : " Va saluer ce groupe - il s'agissait d'un groupe d'Anges assis - et écoute quelle sera leur réponse ! Ce sera ta salutation et celle de ta descendance. - Adam dit : " Que le salut soit sur vous !" Ils répondirent : " Que le salut et la miséricorde d'Allah soient sur toi !" Ils ajoutèrent donc : " et la miséricorde d'Allah". |
| 3 | 3716 | 0.881652 | Qu'Allah fasse miséricorde à un homme bienveillant lorsqu’il vend, conciliant lorsqu’il achète et lorsqu’il réclame son dû ! |
| 4 | 5022 | 0.880786 | Ô Allah ! Voilà untel fils d’untel sous Ta protection et dans Ton voisinage. Préserve-le donc de l’épreuve de la tombe et du châtiment de l’Enfer. Certes, Tu es Digne de loyauté et de louanges. Ô Allah ! Pardonne-lui et fais-lui miséricorde, Tu es certes Celui qui pardonne et Le Très Miséricordieux. |
| 5 | 6334 | 0.876380 | Le Paradis et l’Enfer ont polémiqué. Le Paradis a dit : " Ce sont les faibles et les pauvres qui entrent en mon sein ! " et l’Enfer a dit : " Ce sont les tyrans et les orgueilleux qui entrent en mon sein ! " Allah a alors dit à l’Enfer : " Tu es Mon châtiment ! Par toi, Je Me venge de qui je veux ! " Et Il a dit au Paradis : " Tu es Ma miséricorde ! Par toi, Je fais miséricorde à qui Je veux ! " |

#### Source attendue : [6405](https://hadeethenc.com/fr/browse/hadith/6405) — rang 36, score 0.868568

Document `semantic_first` : **1391 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 44 | 44 | retained |
| categories | 6 | 6 | retained |
| hadith | 175 | 175 | retained |
| explanation | 1162 | 283 | truncated |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs.

Les caractères louables

Usâmah ibn Zayd (qu'Allah l'agrée, lui et son père) dit : « On apporta au Messager d'Allah (sur lui la paix et le salut) le fils de sa fille dans un état d'agonie, les yeux du Messager d'Allah (sur lui la paix et le salut) s'emplirent alors de larmes. Sa'd (qu'Allah l'agréé) demanda : « Qu'est- ce donc, Ô Messager d'Allah ?! » Le Prophète (sur lui la paix et le salut) répondit : « C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs. »

Usâmah ibn Zayd (qu'Allah l'agrée) surnommé le bien aimé, fils du bien-aimé du Prophète (sur lui la paix et le salut), évoqua que l'une des filles du Messager d'Allah (sur lui la paix et le salut) envoya un émissaire pour l'informer que son fils était à l'agonie, c'est à dire aux portes de la mort et qu'elle voulait qu'il soit présent. Quand l'émissaire vint au Prophète (sur lui la paix et le salut) et l'en informa, celui-ci lui répliqua : « Ordonne-lui de patienter et d'escompter la récompense auprès d'Allah ! C'est à Allah qu'appartient ce qu'Il a pris et ce qu'Il a donné. Toute chose auprès de Lui a un terme fixé ! » Le Prophète (sur lui la paix et le salut) ordonna à l'émissaire que sa fille avait envoyé, la mère de l'enfant, de lui transmettre ses paroles : « C'est à Allah qu'appartient ce qu'Il a pris » est une phrase immense, car si toute chose appartient à Allah et qu'Il te reprend ce qu'Il t'a donné, alors c'est Son bien, et s'Il te donne quelque chose, c'est aussi Son bien. Alors pourquoi se mettre en colère si Allah récupère ce qu'il ta donné et qui Lui appartient ? Par conséquent, si Allah nous prend une chose que l'on aime, nous devons dire : « Ceci appartient à Allah, Il peut récupérer ce qu'Il veut et Il peut donner ce qu'Il veut. » Ainsi, l'une des traditions prophétiques pour l'individu est de dire, lorsqu'un malheur le touche : « Nous sommes à Allah et c'est à Allah que nous retournerons », c'est-à-dire que nous appartenons à Allah qui peut faire de nous ce qu'Il veut, et de même pour les choses que l'on aime, s'Il les récupère c'est Son bien et c'est à Allah qu'appartient ce qu'Il a pris et ce qu'Il a donné. Donc, ce qu'Il t'a donné ne t'appartient pas, mais appartient à Allah; c'est [d'ailleurs] pourquoi, tu ne peux utiliser ce qu'Allah t'a donné que d'une manière conforme à ce qu'Il t'a permis. Cela prouve [aussi] que ce qu'Allah nous a donné, constitue bien notre propriété. La parole du Messager d'Allah (sur lui la paix et le salut) : « Toute chose a auprès de Lui un terme fixé ! », c'est-à-dire un terme bien déterminé. Si tu as la certitude en cela, tu seras absolument contenté. Cette dernière phrase signifie que l'individu ne peut modifier le destin qui a été prescrit, ni en l'avançant ni en le retardant comme Allah, Exalté soit-Il, a dit : {(A chaque communauté un terme. Quand leur terme arrive, ils ne peuvent ni le retarder d’une heure, ni l’avancer.)} [Coran : 10/49]. Donc si la chose destinée ne peut être avancée ni retardée, à quoi bon de s'angoisser et s'énerver ? En effet, le fait que tu t'angoisses ou que tu t'énerves ne changera rien au destin. Ensuite, l'émissaire prévint la fille du Prophète (sur lui la paix et le salut) des propos de son père, mais celle-ci insista pour qu'il vienne à sa rencontre en le renvoyant de nouveau auprès de lui. Le Prophète (sur lui la paix et le salut) et un groupe de ses Compagnons se levèrent et arrivèrent auprès de sa fille. L'enfant qui s'agitait, c'est-à-dire se convulsait, fut porté au Prophète (sur lui la paix et le salut) dont les yeux s’emplirent de larmes qui se mirent à couler. Sa'd Ibn 'Ubâdah (qu'Allah l'agréé), le chef des Khazraj, qui était en compagnie du Prophète (sur lui la paix et le salut) demanda alors : « Qu'est- ce donc ? » Il pensait que le Messager d'Allah (sur lui la paix et le salut) pleurait de colère. Le Prophète (sur lui la paix et le salut) répondit : « C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. », c'est-à-dire, j'ai pleuré par miséricorde envers cet enfant et non par colère envers le destin. Puis, le Prophète (sur lui la paix et le salut) dit : « Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs. » On a donc la preuve qu'il est autorisé de pleurer par miséricorde envers une personne atteinte d'un malheur.
```
