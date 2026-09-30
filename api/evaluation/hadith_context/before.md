# Retrieval — intfloat/multilingual-e5-base / multi

Labels : **provisional_candidates_not_human_validated**. Les taux sont provisoires si les labels ne sont pas relus humainement.

Top-1 : 70.0% · Top-3 : 95.0% · Latence moyenne : 25.89 ms.

Latence à chaud : encodage + cosinus + regroupement par ID, hors HTTP.

1790 hadiths, 4803 embeddings ; 0 documents tronqués à 512 tokens.

Sections entièrement perdues : `{}`.

## Échecs Top-3

### Le conseil de parler seulement pour dire du bien sinon se taire

IDs attendus provisoires : 5437

| Rang | ID | Cosinus | Titre officiel |
| --- | --- | --- | --- |
| 1 | 6988 | 0.830918 | Une chamelle maudite ne doit pas nous accompagner ! |
| 2 | 5835 | 0.826042 | Ordonnez-lui de parler, de se mettre à l’ombre, de s’asseoir mais qu’il poursuive son jeûne. |
| 3 | 65101 | 0.825603 | Lorsque le Messager d'Allah ﷺ relevait son dos de l'inclinaison, il disait : " Allah a entendu quiconque L'a loué |
| 4 | 58226 | 0.818382 | Nous n’utilisons pas pour notre œuvre celui qui la veut [c’est-à-dire : Nous ne confions pas le commandement à celui qui le demande]. Toutefois, toi - Ô Abâ Mûsâ - va au Yémen ! ou, toi - Ô 'Abdallah ibn Qays - va au Yémen ! |
| 5 | 5501 | 0.817751 | Si vous ne pouvez faire autrement que de vous rassembler, donnez au passage son droit ! - Les Compagnons demandèrent : Quel est son droit ? - Il dit : Baisser le regard, s’abstenir de toute nuisance, répondre au salut, ordonner le convenable et interdire le blâmable. |

#### Source attendue : [5437](https://hadeethenc.com/fr/browse/hadith/5437) — rang 91, score 0.803625

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
