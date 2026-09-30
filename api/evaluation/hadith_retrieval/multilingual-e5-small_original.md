# Retrieval — intfloat/multilingual-e5-small / original

Labels : **provisional_candidates_not_human_validated**. Les taux sont provisoires si les labels ne sont pas relus humainement.

Top-1 : 50.0% · Top-3 : 65.0% · Latence moyenne : 10.12 ms.

Latence à chaud : encodage + cosinus + regroupement par ID, hors HTTP.

1790 hadiths, 1790 embeddings ; 796 documents tronqués à 512 tokens.

Sections entièrement perdues : `{"categories": 1096, "hints": 529, "explanation": 117}`.

## Échecs Top-3

### Je cherche le hadith où le Prophète conseille à quelqu'un de ne pas se mettre en colère

IDs attendus provisoires : 4709

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 5456 | 0.862197 | Ô fils d'Adam ! Tant que tu M'invoqueras et espéreras en Moi, alors Je te pardonnerai quoique tu fasses et Je ne M'en soucierai pas. Ô fils d'Adam ! Si tes péchés atteignaient la cime du ciel et que tu implorais Mon pardon, alors Je te pardonnerais et Je ne m'en soucierais pas |
| 2 | 11227 | 0.861482 | Par Celui qui détient mon âme dans Sa main ! Une époque viendra à l'homme où le tueur ne saura pas pourquoi il tue, et où celui qui est tué ne saura pas pourquoi il a été tué ! |
| 3 | 64691 | 0.861222 | Voulez-vous que je vous informe du meilleur des témoins ? C'est celui qui apporte son témoignage avant qu'on l'interroge ! |
| 4 | 3686 | 0.859966 | Transmettez de moi ne serait-ce qu'un verset et rapportez des Fils d'Israël, il n'y a pas de mal. Quant à celui qui ment volontairement à mon sujet, qu'il prépare sa place en Enfer ! |
| 5 | 65105 | 0.859226 | C'est un démon que l'on appelle : " Khinzab. " Lorsque tu le ressens, alors réfugie-toi auprès d'Allah contre lui et crachote trois fois sur ta gauche |

#### Source attendue : [4709](https://hadeethenc.com/fr/browse/hadith/4709) — rang 7, score 0.858727

Document `original` : **471 tokens**, 471 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 8 | 8 | retained |
| hadith | 134 | 134 | retained |
| explanation | 206 | 206 | retained |
| hints | 36 | 36 | retained |
| hints | 32 | 32 | retained |
| hints | 31 | 31 | retained |
| hints | 14 | 14 | retained |
| categories | 6 | 6 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Ne te mets pas colère !

D'après  (qu'Allah l'agrée) : Un homme a dit au Prophète (qu'Allah le couvre d'éloges et le préserve) : « Fais-moi une recommandation ! » Il (qu'Allah le couvre d'éloges et le préserve) a dit : «  Ne te mets pas colère !  » L’homme répéta à plusieurs reprises [sa demande] et [à chaque fois] il (qu'Allah le couvre d'éloges et le préserve) a dit : « Ne te mets pas en colère ! »

Un des Compagnons (qu'Allah les agrée) a demandé au Prophète (qu'Allah le couvre d'éloges et le préserve) de lui indiquer une chose qui lui serait bénéfique. Alors, il lui a ordonné de ne pas se mettre en colère. Et la signification de cela est d'éviter les causes qui amènent à la colère et de contrôler sa personne si la colère survient de sorte que cette colère ne soit pas suivie d'un meurtre, ou d'une frappe, ou d'une insulte, ou ce qui ressemble à cela.

Et l'homme a réitéré plusieurs fois sa demande de recommandation mais le Prophète (qu'Allah le couvre d'éloges et le préserve) ne lui a rien rajouté comme recommandation si ce n'est : " Ne te mets pas en colère. "

La mise en garde contre la colère et ses causes car elle est l'ensemble du mal et le fait de s'en prémunir est l'ensemble du bien.

La colère pour Allah, comme la colère lors de la violation des interdits sacrés d'Allah fait partie de la colère louable.

La répétition des paroles si besoin est jusqu'à ce que l'interlocuteur les retienne et comprenne leur importance.

Le mérite de la demande de recommandation au savant.

Les caractères louables
```

### Un hadith qui dit que nos actes comptent selon l'intention qu'on avait

IDs attendus provisoires : 4560, 66511

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 10104 | 0.861824 | Le temps a tourné comme le jour où Allah a créé les cieux et la terre. L'année compte douze mois, dont quatre sacrés, trois qui se suivent : Dhul Qa'dah, Dhul Ḥijja, Al-Muḥarram et Rajab, le mois de Muḍar. |
| 2 | 5981 | 0.856357 | N’est pas des nôtres celui qui consulte les augures ou pour qui on les consulte ; ni celui qui pratique la voyance ou consulte un voyant ; ni celui qui pratique la sorcellerie ou pour qui on la pratique |
| 3 | 5846 | 0.855775 | Par Celui qui détient mon âme dans Sa Main, si vous demeuriez dans le même état qu'en ma compagnie et durant le rappel, les Anges vous serreraient la main dans vos lits et sur vos routes. Cependant, ô Ḥanẓalah ! Il y a un temps pour chaque chose. |
| 4 | 8302 | 0.855142 | Allah crée quiconque fait, ainsi que ce qu'il fait ! |
| 5 | 65105 | 0.855107 | C'est un démon que l'on appelle : " Khinzab. " Lorsque tu le ressens, alors réfugie-toi auprès d'Allah contre lui et crachote trois fois sur ta gauche |

#### Source attendue : [4560](https://hadeethenc.com/fr/browse/hadith/4560) — rang 1646, score 0.810266

Document `original` : **727 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 27 | 27 | retained |
| hadith | 205 | 205 | retained |
| explanation | 374 | 276 | truncated |
| hints | 30 | 0 | lost |
| hints | 81 | 0 | lost |
| categories | 6 | 0 | lost |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Certes, les œuvres ne valent que par les intentions et chaque individu sera rétribué en fonction de son intention

'Oumar ibn Al-Khaṭṭâb (qu'Allah l'agrée) relate que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) a dit : « Certes, les œuvres ne valent que par les intentions et l'individu sera rétribué en fonction de son intention. Ainsi donc, quiconque dont l'émigration est vers Allah et Son , alors son émigration sera vers Allah et Son  ; et quiconque dont l'émigration est pour un intérêt mondain qu'il veut acquérir ou pour une femme qu'il veut épouser, alors son émigration sera pour ce vers quoi il a émigré. » Et dans l'expression d'Al Bukhârî : «  Certes, les œuvres ne valent que par les intentions et chaque individu sera rétribué en fonction de son intention. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que toutes les œuvres sont considérées en fonction de l'intention. Ce décret est général et concerne l'ensemble des œuvres, qu'il s'agisse des adorations ou des transactions externes. Ainsi donc, quiconque vise par son œuvre l'acquisition d'un profit n'obtiendra que cet avantage et n'aura pas de rétribution ; et quiconque œuvre dans le but de se rapprocher d'Allah, Exalté soit-Il, obtiendra la rétribution et la récompense de son œuvre, même s'il s'agit d'une œuvre habituelle et normale, comme le fait de manger et boire.

Ensuite, il (qu'Allah le couvre d'éloges et le préserve) a cité un exemple pour montrer l'effet de l'intention sur les oeuvres même si elles sont similaires dans la forme apparente. Il a expliqué que quiconque vise la satisfaction de son Seigneur à travers son émigration et le délaissement de son pays, alors son émigration est une émigration religieuse légale acceptée pour laquelle la personne sera rétribuée en raison de la véracité de son intention. Mais quiconque vise un avantage mondain à travers son émigration, que ce soit de l'argent ou un bien, une position, un commerce, une épouse, etc. Alors, la personne n'obtiendra de son émigration que cet avantage dont il a eu comme intention et il n'aura aucune part de la récompense et de la rétribution [pour cette émigration].

L'incitation à la sincérité. En effet, Allah n'accepte comme oeuvre que celle dont on recherche Son Visage.

Lorsque la personne responsable accomplit des oeuvres par lesquelles on peut se rapprocher d'Allah, Exalté soit-Il, mais elle les fait de manière habituelle, alors elle n'obtiendra pas de rétribution pour celles-ci jusqu'à ce qu'elle ait l'intention de se rapprocher d'Allah à travers ces oeuvres.

Les oeuvres du coeur
```


#### Source attendue : [66511](https://hadeethenc.com/fr/browse/hadith/66511) — rang 943, score 0.825113

Document `original` : **697 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 10 | 10 | retained |
| hadith | 165 | 165 | retained |
| explanation | 374 | 333 | truncated |
| hints | 30 | 0 | lost |
| hints | 81 | 0 | lost |
| hints | 27 | 0 | lost |
| categories | 6 | 0 | lost |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Les actes ne valent que par les intentions

D’après l’Émir des Croyants, Abû Hafs ‘Umar b. al‑Khattâb — qu’Allah l’agrée — qui a dit: j’ai entendu le Messager d’Allah ﷺ dire : « Les actes ne valent que par les intentions, et chacun n’aura que ce qu’il a eu l’intention de faire. Celui dont l’émigration fut pour Allah et Son Messager, son émigration est pour Allah et Son Messager ; et celui dont l’émigration fut pour un intérêt mondain qu’il convoitait ou une femme qu’il voulait épouser, son émigration n’est que vers ce pour quoi il a émigré. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que toutes les œuvres sont considérées en fonction de l'intention. Ce décret est général et concerne l'ensemble des œuvres, qu'il s'agisse des adorations ou des transactions externes. Ainsi donc, quiconque vise par son œuvre l'acquisition d'un profit n'obtiendra que cet avantage et n'aura pas de rétribution ; et quiconque œuvre dans le but de se rapprocher d'Allah, Exalté soit-Il, obtiendra la rétribution et la récompense de son œuvre, même s'il s'agit d'une œuvre habituelle et normale, comme le fait de manger et boire.

Ensuite, il (qu'Allah le couvre d'éloges et le préserve) a cité un exemple pour montrer l'effet de l'intention sur les oeuvres même si elles sont similaires dans la forme apparente. Il a expliqué que quiconque vise la satisfaction de son Seigneur à travers son émigration et le délaissement de son pays, alors son émigration est une émigration religieuse légale acceptée pour laquelle la personne sera rétribuée en raison de la véracité de son intention. Mais quiconque vise un avantage mondain à travers son émigration, que ce soit de l'argent ou un bien, une position, un commerce, une épouse, etc. Alors, la personne n'obtiendra de son émigration que cet avantage dont il a eu comme intention et il n'aura aucune part de la récompense et de la rétribution [pour cette émigration].

L'incitation à la sincérité. En effet, Allah n'accepte comme oeuvre que celle dont on recherche Son Visage.

Lorsque la personne responsable accomplit des oeuvres par lesquelles on peut se rapprocher d'Allah, Exalté soit-Il, mais elle les fait de manière habituelle, alors elle n'obtiendra pas de rétribution pour celles-ci jusqu'à ce qu'elle ait l'intention de se rapprocher d'Allah à travers ces oeuvres.

L’intention permet de distinguer les cultes les uns des autres, et de distinguer les cultes des habitudes.

Les oeuvres du coeur
```

### Le conseil de parler seulement pour dire du bien sinon se taire

IDs attendus provisoires : 5437

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 3311 | 0.838374 | Ô Abâ Dharr ! Je pense que tu es faible, et j’aime pour toi ce que j’aime pour moi-même. Ne prends pas le commandement de deux personnes, et ne te charge pas de la tutelle des biens de l’orphelin ! |
| 2 | 3055 | 0.836281 | La pudeur n'apporte que du bien. |
| 3 | 5500 | 0.834705 | Vos biens ne suffiront jamais à contenter les gens, alors contentez-les plutôt avec un visage souriant et un bon comportement. |
| 4 | 4952 | 0.834591 | Ne dis pas : « Sur toi la paix ! » car c’est ainsi que l’on salue les morts. Dis plutôt : « Que la paix soit sur toi ! » |
| 5 | 6094 | 0.834585 | « Qu’un voisin n’empêche pas son voisin de faire passer sa poutre dans son mur. » Ensuite, Abû Hurayrah (qu’Allah l’agrée) dit : " Pourquoi vous vois-je vous en détourner ? Par Allah ! Je vais vous les jeter sur le dos ! " |

#### Source attendue : [5437](https://hadeethenc.com/fr/browse/hadith/5437) — rang 232, score 0.817120

Document `original` : **500 tokens**, 500 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 27 | 27 | retained |
| hadith | 118 | 118 | retained |
| explanation | 224 | 224 | retained |
| hints | 29 | 29 | retained |
| hints | 13 | 13 | retained |
| hints | 19 | 19 | retained |
| hints | 22 | 22 | retained |
| hints | 38 | 38 | retained |
| categories | 6 | 6 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise

Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah  (qu'Allah le couvre d'éloges et le préserve) a dit : « Quiconque croit en Allah et au Jour Dernier, qu'il dise du bien ou qu'il se taise. Quiconque croit en Allah et au Jour Dernier, qu'il honore son voisin. Et quiconque croit en Allah et au Jour Dernier, qu'il honore son invité. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) explique que le serviteur qui croit en Allah et au Jour Dernier, dont le retour est vers Lui pour y être rétribué pour ses oeuvres, alors sa foi l'incite à accomplir les bribes suivantes :

La première : Dire de belles paroles : parmi la proclamation de la gloire d'Allah, Son unicité, l'ordonnance du convenable, l'interdiction du blâmable, la réforme entre les gens, etc. Et s'il ne le fait pas, alors qu'il s'attache au silence, qu'il s'abstienne de causer du tort et qu'il préserve sa langue.

La seconde : Honorer le voisin en étant bienfaisant envers lui et en ne lui causant aucun tort.

La troisième  : Honorer l'invité qui vient te visiter en lui parlant agréablement, en lui donnant à manger, et ce qui ressemble à cela.

La foi en Allah et au Jour Dernier est la base de tout bien et cela pousse à l'accomplissement du bien.

La mise en garde contre les dégâts de la langue.

La religion de l'islam est une religion de cordialité et de générosité.

Ces bribes font partie des branches de la foi et sont parmi les bonnes manières louables.

La profusion de paroles peut amener à des choses répugnées et interdites ; et la sécurité est dans l'absence de paroles, excepté dans le bien.

Les caractères louables
```

### Planter un arbre dont les animaux mangent rapporte une récompense

IDs attendus provisoires : 3911

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 10100 | 0.849344 | Un chien qui était sur le point de mourir de soif tournait autour d’un point d’eau. C’est alors que l'une des prostituées du peuple des enfants d’Israël le vit, elle décida de prendre son chausson afin de le remplir d’eau puis elle l’abreuva. Allah lui pardonna ses péchés grâce à son geste. |
| 2 | 3583 | 0.847006 | Le bien est attaché au toupet des chevaux jusqu’au Jour de la Résurrection. |
| 3 | 4181 | 0.843219 | Celui qui met à disposition un cheval dans le sentier d’Allah, parce qu’il a foi en Allah et croit en Sa promesse, trouvera dans sa balance, au Jour de la Résurrection, la nourriture de son cheval ainsi que sa boisson, son urine et son crottin. |
| 4 | 5856 | 0.843017 | « C’est une subsistance qu’Allah a fait sortir pour vous. Reste-t-il encore de sa viande afin que vous nous en donniez à manger ? » Nous en envoyâmes alors une part au Messager d’Allah (sur lui la paix et le salut) qui la mangea. |
| 5 | 4721 | 0.842209 | Si vous placiez votre confiance en Allah d'une véritable confiance, certainement Il vous pourvoirait comme Il pourvoit aux oiseaux. Tôt le matin, ils quittent leur nid l’estomac vide et le soir ils reviennent rassasiés. |

#### Source attendue : [3911](https://hadeethenc.com/fr/browse/hadith/3911) — rang 7, score 0.841850

Document `original` : **593 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 41 | 41 | retained |
| hadith | 179 | 179 | retained |
| explanation | 359 | 288 | truncated |
| categories | 10 | 0 | lost |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Il n’est pas un musulman qui plante un arbre sans que tout ce qui en est mangé, volé ou prélevé ne soit considéré comme une aumône de sa part.

Jâbir (qu’Allah l’agrée) relate que le Messager d’Allah (sur lui la paix et le salut) a dit : « Il n’est pas un musulman qui plante un arbre sans que tout ce qui en est mangé, volé ou prélevé ne soit considéré comme une aumône de sa part. » Dans une autre version : « Il n’est pas un musulman qui plante un arbre dont se nourrit un homme, un animal ou toute autre chose, sans que cela ne soit considéré comme une aumône de sa part. » Et dans une autre version : « Il n’est pas un musulman qui plante un arbre ou cultive une terre dont se nourrit un humain, un animal ou toute autre chose, sans que cela ne soit considéré comme une aumône de sa part !

Signification du hadith : « Il n’est pas un musulman qui plante un arbre… » ni ne cultive de terre dont un être vivant parmi les créatures se nourrit, sans qu’il ne soit récompensé pour cela. Et même après sa mort, son œuvre continue de perdurer et d’être rétribuée tant que la terre qu'il a cultivée et l'arbre qu'il a planté existent. Dans le hadith de ce chapitre, il y a une incitation à cultiver la terre et planter des arbres car ces deux actions procurent un profit dans la religion. En effet, la personne obtiendra une aumône pour tout ce qui sera mangé de cette terre ou de cet arbre. Et plus étonnant encore, même ce qu’on en vole sera aussi considéré comme une aumône de sa part. Par exemple, si une personne venait à voler des dattes d’un palmier, son propriétaire en obtiendrait une récompense, et cela même s’il venait à porter plainte au tribunal contre le voleur. Le Jour de la Résurrection, Allah, Exalté soit-Il, lui comptabilisera une aumône pour le vol de son bien. De même, si des bêtes ou des insectes venaient à manger des terres cultivées, leur propriétaire obtiendrait encore une fois la récompense d’une aumône pour cela. Ce hadith a précisément mentionné le musulman car c’est le seul qui bénéficie de la récompense de l’aumône dans cette vie d’ici-bas et dans l’au-delà.

Le mérite de l'Islam et ses qualités
```

### Je cherche le conseil de faciliter les choses et de ne pas faire fuir les gens

IDs attendus provisoires : 5866

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 6368 | 0.841370 | Tu te dois d’écouter et d’obéir, dans la difficulté comme dans l’aisance, dans ce qui te plaît comme dans ce qui te déplaît, et même si tu bénéficies de privilèges. |
| 2 | 10563 | 0.840680 | Cherche-moi des pierres pour me nettoyer et ne m'apporte ni os, ni crottin ! |
| 3 | 4706 | 0.836542 | Ne vous enviez pas, ne vous espionnez pas, ne vous haïssez pas, ne vous tournez pas le dos et ne surenchérissez pas les uns les autres sur une vente ! Soyez des serviteurs d’Allah, des frères ! |
| 4 | 3311 | 0.835516 | Ô Abâ Dharr ! Je pense que tu es faible, et j’aime pour toi ce que j’aime pour moi-même. Ne prends pas le commandement de deux personnes, et ne te charge pas de la tutelle des biens de l’orphelin ! |
| 5 | 5500 | 0.835431 | Vos biens ne suffiront jamais à contenter les gens, alors contentez-les plutôt avec un visage souriant et un bon comportement. |

#### Source attendue : [5866](https://hadeethenc.com/fr/browse/hadith/5866) — rang 13, score 0.832130

Document `original` : **416 tokens**, 416 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 22 | 22 | retained |
| hadith | 66 | 66 | retained |
| explanation | 117 | 117 | retained |
| hints | 23 | 23 | retained |
| hints | 30 | 30 | retained |
| hints | 42 | 42 | retained |
| hints | 37 | 37 | retained |
| hints | 52 | 52 | retained |
| hints | 17 | 17 | retained |
| categories | 6 | 6 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: Facilitez et ne rendez pas difficile ! Annoncez de bonnes nouvelles et ne faites pas fuir !

Anas ibn Mâlik (qu'Allah l'agrée) relate que le Prophète (qu'Allah le couvre d'éloges et le préserve) a dit : « Facilitez et ne rendez pas difficile ! Annoncez de bonnes nouvelles et ne faites pas fuir ! »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) ordonne d'alléger et de faciliter les choses pour les gens et de ne pas leur rendre compliqué les choses, qu'il s'agisse de leurs affaires religieuses et mondaines. Et ceci, dans les limites de ce qu'Allah a autorisé et prescrit.

Il encourage aussi (qu'Allah le couvre d'éloges et le préserve) à leur faire la bonne annonce du bien et à ne pas les en faire fuir.

Le devoir du croyant est de faire aimer Allah aux gens et de les encourager dans le bien.

Il convient à celui qui invite les gens à Allah d'observer avec sagesse comment transmettre l'appel de l'Islam aux gens.

Le fait d'annoncer les bonnes nouvelles fait naître la joie, l'acceptation et l'apaisement du coeur quant au prédicateur et à ce qu'il présente aux gens.

Le fait de rendre compliqué fait naître l'envie de fuir, de se détourner et le doute vis-à-vis des paroles du prédicateur.

L'étendue de la miséricorde d'Allah envers Ses serviteurs et le fait qu'Il leur a agréé une religion bienveillante et une Charî'ah (Législation) facilitée.

La facilité ordonnée est ce avec quoi la Charî'ah est venue.

Les caractères louables
```

### Enlever un obstacle de la route fait partie des branches de la foi

IDs attendus provisoires : 3276, 6468

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 10869 | 0.850385 | Lorsque que l'un d’entre vous fait la prière, qu'il prie devant un obstacle, ne serait-ce qu'une flèche ! |
| 2 | 5434 | 0.846106 | Celui qui a peur part en voyage aux premières heures de la nuit, et celui qui part en voyage aux premières heures de la nuit arrive à bon port. Notez-bien que la marchandise d’Allah est précieuse, Notez bien que la marchandise d’Allah est le Paradis ! |
| 3 | 10417 | 0.845778 | Trois hommes qui étaient de sortie furent surpris par la pluie alors qu'ils marchaient. Ils cherchèrent aussitôt refuge à l'intérieur d'une grotte qui se trouvait dans la montagne, un rocher tomba soudainement et vint obstruer la sortie de la grotte. |
| 4 | 4505 | 0.845366 | Nous sommes sortis avec le Prophète (sur lui la paix et le salut) pendant le mois de Ramadan. Il faisait chaud au point que l’un d'entre-nous posait sa main sur sa propre tête afin de se protéger de la chaleur. Parmi nous, personne ne jeûnait à part le Prophète (sur lui la paix et le salut) et ‘Abdullah ibn Rawâḥah. |
| 5 | 4813 | 0.842482 | On m'a présenté les actes de ma communauté, les bons comme les mauvais. J’ai constaté que l’une de leurs belles œuvres était le fait d’ôter du chemin les choses nuisibles et que l’une de leurs mauvaises œuvres était la glaire laissée dans la mosquée sans être enfouie. |

#### Source attendue : [3276](https://hadeethenc.com/fr/browse/hadith/3276) — rang 28, score 0.836046

Document `original` : **397 tokens**, 397 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 83 | 83 | retained |
| hadith | 120 | 120 | retained |
| explanation | 180 | 180 | retained |
| categories | 10 | 10 | retained |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi.

Abû Hurayrah (qu’Allah l’agrée) relate que le Messager d’Allah (sur lui la paix et le salut) a dit : « La foi comporte un peu plus de soixante ou soixante-dix branches. La meilleure d’entre elle est l’attestation qu’il n’y a aucune divinité digne d’être adorée en dehors d’Allah et la plus infime consiste à ôter ce qui est nuisible du chemin. La pudeur est également une branche de la foi. »

La foi ne se résume pas à une seule caractéristique ou une seule branche. Elle est composée de plusieurs branches : un peu plus de soixante ou soixante-dix. La meilleure de ces branches est la parole qui atteste qu’il n’y a aucune divinité qui mérite l’adoration en dehors d’Allah et la plus infime consiste à ôter du chemin ce qui est nuisible pour les passants comme une pierre, une branche épineuse, ou autre. Être pudique est également une branche de la foi. Ainsi, les actes font partie de foi selon les gens de la tradition et du groupe (« Ahl as-sunnah wa-l-jamâ'ah ») ; et c'est la vérité qu'indiquent les textes et celle-ci en fait partie.

Les branches / ramifications de la foi
```


#### Source attendue : [6468](https://hadeethenc.com/fr/browse/hadith/6468) — rang 42, score 0.833118

Document `original` : **595 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 68 | 68 | retained |
| hadith | 125 | 125 | retained |
| explanation | 233 | 233 | retained |
| hints | 18 | 18 | retained |
| hints | 11 | 11 | retained |
| hints | 60 | 53 | truncated |
| hints | 65 | 0 | lost |
| categories | 11 | 0 | lost |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin

Aboû Hourayrah (qu'Allah l'agrée) relate que le Messager d'Allah (qu'Allah le couvre d'éloges et le préserve) a dit : « La foi comporte soixante-dix et quelques - ou soixante et quelques - branches. La meilleure d’entre elles est la parole : " Il n’est de divinité [digne d'adoration] qu'Allah ", et la moindre  consiste à ôter ce qui est nuisible du chemin, et la pudeur est une branche de la foi. »

Le Prophète (qu'Allah le couvre d'éloges et le préserve) informe que la foi possède de nombreuses branches et caractéristiques qui englobent des œuvres, des croyances et des paroles.

Que la plus haute et la meilleure des caractéristiques de la foi est de dire : " Il n'est de divinité [digne d'adoration] qu'Allah " en connaissant sa signification et en œuvrant selon ses implications, à savoir qu'Allah est le Seul et Unique Dieu qui soit digne d'adoration, Lui Seul, et sans qui ou quoi que ce soit d'autre.

Et que la moindre des œuvres de la foi consiste à ôter ce qui nuit aux gens sur leurs chemins.

Ensuite, il a informé (qu'Allah le couvre d'éloges et le préserve) que la pudeur fait partie des caractéristiques de la foi. C'est un comportement qui pousse à l'accomplissement de ce qui est beau et au délaissement de ce qui est laid.

La foi a des niveaux, certains d'entre eux sont meilleurs que d'autres.

La foi est parole, acte et croyance.

La pudeur à l'égard d'Allah, Exalté soit-Il, implique qu'Il ne te voit pas là où Il t'a interdit d'être, et qu'Il ne te trouve pas absent là où Il t'a ordonné d'être.

La mention du nombre [des branches de la foi] ne signifie pas qu'elle y soit limitée. Cela indique plutôt la multitude des œuvres de celle-ci. En effet, les Arabes mentionnaient parfois un nombre pour une chose, sans pour autant  vouloir infirmer autre que lui.

L'augmentation de la Foi et sa diminution
```

### Allah accorde sa miséricorde à ceux qui sont miséricordieux

IDs attendus provisoires : 6405

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 7185 | 0.883617 | Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il dit alors : Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il dit à nouveau : Ô Allah, fais miséricorde à ceux qui se rasent les cheveux ! - Ils demandèrent : Et ceux qui se les coupent, ô envoyé d’Allah ? - Il répondit : Et à ceux qui se les coupent ! |
| 2 | 3556 | 0.874244 | Quand Allah créa Adam (sur lui la paix et le salut), Il [lui] dit : " Va saluer ce groupe - il s'agissait d'un groupe d'Anges assis - et écoute quelle sera leur réponse ! Ce sera ta salutation et celle de ta descendance. - Adam dit : " Que le salut soit sur vous !" Ils répondirent : " Que le salut et la miséricorde d'Allah soient sur toi !" Ils ajoutèrent donc : " et la miséricorde d'Allah". |
| 3 | 65031 | 0.874137 | Je serai certes près du Bassin afin de regarder qui de vous passe auprès de moi, des gens seront saisis à mon niveau et je dirai alors : "Ô Seigneur ! Ils sont de moi et de ma communauté ! |
| 4 | 65105 | 0.873441 | C'est un démon que l'on appelle : " Khinzab. " Lorsque tu le ressens, alors réfugie-toi auprès d'Allah contre lui et crachote trois fois sur ta gauche |
| 5 | 5021 | 0.873110 | Ô Allah ! Pardonne à ceux parmi nous qui sont encore vivants ainsi qu'à nos morts, à nos jeunes ainsi qu'à nos personnes âgées, à nos hommes ainsi qu'à nos femmes, aux personnes présentes ainsi qu'à celles qui sont absentes. Ô Allah ! Celui d’entre nous que Tu maintiens en vie, alors fais-le vivre conformément à l’Islam, et celui d’entre nous dont Tu reprends l’âme, alors fais-le mourir dans la foi. Ô Allah ! Ne nous prive pas de sa récompense et ne nous tente pas après lui ! |

#### Source attendue : [6405](https://hadeethenc.com/fr/browse/hadith/6405) — rang 55, score 0.864646

Document `original` : **1391 tokens**, 512 conservés.

| Section | Tokens | Conservés | État |
| --- | --- | --- | --- |
| title | 44 | 44 | retained |
| hadith | 175 | 175 | retained |
| explanation | 1162 | 289 | truncated |
| categories | 6 | 0 | lost |

`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :

```text
passage: C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs.

Usâmah ibn Zayd (qu'Allah l'agrée, lui et son père) dit : « On apporta au Messager d'Allah (sur lui la paix et le salut) le fils de sa fille dans un état d'agonie, les yeux du Messager d'Allah (sur lui la paix et le salut) s'emplirent alors de larmes. Sa'd (qu'Allah l'agréé) demanda : « Qu'est- ce donc, Ô Messager d'Allah ?! » Le Prophète (sur lui la paix et le salut) répondit : « C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs. »

Usâmah ibn Zayd (qu'Allah l'agrée) surnommé le bien aimé, fils du bien-aimé du Prophète (sur lui la paix et le salut), évoqua que l'une des filles du Messager d'Allah (sur lui la paix et le salut) envoya un émissaire pour l'informer que son fils était à l'agonie, c'est à dire aux portes de la mort et qu'elle voulait qu'il soit présent. Quand l'émissaire vint au Prophète (sur lui la paix et le salut) et l'en informa, celui-ci lui répliqua : « Ordonne-lui de patienter et d'escompter la récompense auprès d'Allah ! C'est à Allah qu'appartient ce qu'Il a pris et ce qu'Il a donné. Toute chose auprès de Lui a un terme fixé ! » Le Prophète (sur lui la paix et le salut) ordonna à l'émissaire que sa fille avait envoyé, la mère de l'enfant, de lui transmettre ses paroles : « C'est à Allah qu'appartient ce qu'Il a pris » est une phrase immense, car si toute chose appartient à Allah et qu'Il te reprend ce qu'Il t'a donné, alors c'est Son bien, et s'Il te donne quelque chose, c'est aussi Son bien. Alors pourquoi se mettre en colère si Allah récupère ce qu'il ta donné et qui Lui appartient ? Par conséquent, si Allah nous prend une chose que l'on aime, nous devons dire : « Ceci appartient à Allah, Il peut récupérer ce qu'Il veut et Il peut donner ce qu'Il veut. » Ainsi, l'une des traditions prophétiques pour l'individu est de dire, lorsqu'un malheur le touche : « Nous sommes à Allah et c'est à Allah que nous retournerons », c'est-à-dire que nous appartenons à Allah qui peut faire de nous ce qu'Il veut, et de même pour les choses que l'on aime, s'Il les récupère c'est Son bien et c'est à Allah qu'appartient ce qu'Il a pris et ce qu'Il a donné. Donc, ce qu'Il t'a donné ne t'appartient pas, mais appartient à Allah; c'est [d'ailleurs] pourquoi, tu ne peux utiliser ce qu'Allah t'a donné que d'une manière conforme à ce qu'Il t'a permis. Cela prouve [aussi] que ce qu'Allah nous a donné, constitue bien notre propriété. La parole du Messager d'Allah (sur lui la paix et le salut) : « Toute chose a auprès de Lui un terme fixé ! », c'est-à-dire un terme bien déterminé. Si tu as la certitude en cela, tu seras absolument contenté. Cette dernière phrase signifie que l'individu ne peut modifier le destin qui a été prescrit, ni en l'avançant ni en le retardant comme Allah, Exalté soit-Il, a dit : {(A chaque communauté un terme. Quand leur terme arrive, ils ne peuvent ni le retarder d’une heure, ni l’avancer.)} [Coran : 10/49]. Donc si la chose destinée ne peut être avancée ni retardée, à quoi bon de s'angoisser et s'énerver ? En effet, le fait que tu t'angoisses ou que tu t'énerves ne changera rien au destin. Ensuite, l'émissaire prévint la fille du Prophète (sur lui la paix et le salut) des propos de son père, mais celle-ci insista pour qu'il vienne à sa rencontre en le renvoyant de nouveau auprès de lui. Le Prophète (sur lui la paix et le salut) et un groupe de ses Compagnons se levèrent et arrivèrent auprès de sa fille. L'enfant qui s'agitait, c'est-à-dire se convulsait, fut porté au Prophète (sur lui la paix et le salut) dont les yeux s’emplirent de larmes qui se mirent à couler. Sa'd Ibn 'Ubâdah (qu'Allah l'agréé), le chef des Khazraj, qui était en compagnie du Prophète (sur lui la paix et le salut) demanda alors : « Qu'est- ce donc ? » Il pensait que le Messager d'Allah (sur lui la paix et le salut) pleurait de colère. Le Prophète (sur lui la paix et le salut) répondit : « C'est une miséricorde qu'Allah a placée dans le cœur de Ses serviteurs. », c'est-à-dire, j'ai pleuré par miséricorde envers cet enfant et non par colère envers le destin. Puis, le Prophète (sur lui la paix et le salut) dit : « Allah n'accorde Sa miséricorde qu'aux miséricordieux d'entre Ses serviteurs. » On a donc la preuve qu'il est autorisé de pleurer par miséricorde envers une personne atteinte d'un malheur.

Les caractères louables
```
