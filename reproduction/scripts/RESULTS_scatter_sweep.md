# Où l'arbre reprend l'avantage sur la chaîne

`scatter_sweep.py`. La géométrie des tâches reste identique, seul l'écartement
entre grappes varie de 0.5 à 20 à dispersion intra-grappe fixée à 1, et le budget
varie indépendamment de 200 à 2000. $T=200$, $K=5$ grappes, $d=5$, dix
répétitions, racine au médoïde. Trois bras séparent la non-convexité de
l'optimiseur de la non-linéarité de la cible, et les deux allocations sont
balayées.

## Le point de croisement existe et il est net

Écart relatif $(\mathrm{CTL\text{-}MST} - \mathrm{Chain})/\mathrm{Chain}$ en
pourcentage, négatif quand l'arbre gagne, allocation SLE.

| écartement | B=200 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|
| 0.5 | -9.6 | -4.6 | +4.0 | +15.6 |
| 1 | -14.9 | -9.6 | +1.3 | +17.0 |
| 2 | -22.2 | -19.8 | -5.0 | +17.0 |
| 5 | -20.3 | -32.9 | -18.2 | +19.6 |
| 10 | -21.3 | -38.8 | -19.7 | +27.1 |
| 20 | -24.6 | -46.2 | -31.9 | +22.9 |

C'est exactement la structure attendue, une frontière diagonale. À budget large
la chaîne l'emporte quel que soit l'écartement, à budget serré l'arbre l'emporte,
et la frontière recule vers les grands budgets à mesure que les tâches
s'écartent. À $B=1000$ le signe bascule entre les écartements 1 et 2.

Le bras `lin/mlp` reproduit la même carte à un point près, donc la non-convexité
de l'optimiseur seule ne change rien. Le bras `relu/mlp`, à cible non linéaire,
décale la frontière vers les grands budgets, l'arbre restant devant à $B=2000$
dès l'écartement 5, ce que ni l'un ni l'autre des bras linéaires ne fait.

## Ce qui produit l'avantage est le couple arbre plus SLE

La même grille sous allocation uniforme s'inverse aux grands écartements.

| `lin/lin`, B=500, valeurs absolues | Chain | CTL-MST |
|---|---|---|
| écartement 0.5, SLE | 1.55 | 1.48 |
| écartement 0.5, uniforme | 1.59 | 1.53 |
| écartement 20, SLE | 8.57 | **4.61** |
| écartement 20, uniforme | 9.75 | 16.02 |

À écartement 20 l'arbre passe de 4.61 sous SLE à 16.02 sous partage plat, un
facteur 3.5, quand la chaîne ne bouge que de 8.57 à 9.75. Le partage par
sous-arbre et logarithme d'arête est donc ce qui rend l'arbre gagnant dans ce
régime, et non la topologie seule.

Le mécanisme se lit dans la formule. Le poids $|s_v|\log(1+d(\mathrm{pa}(v),v))$
finance les arêtes longues qui portent de gros sous-arbres. Un arbre à grappes
bien séparées possède exactement cela, quelques ponts inter-grappes longs et
lourds, et SLE les finance. Une chaîne étale sa longueur sur tout son parcours,
il n'y a rien à concentrer, et le terme de taille de sous-arbre y déverse le
budget du côté de la racine en affamant l'extrémité.

Cela réconcilie un résultat qui paraissait contradictoire. Sur des collections à
géométrie homogène, où aucune arête ne se distingue, SLE coûte plutôt qu'il ne
rapporte. Sur des collections à grappes séparées il est ce qui fait la
différence. L'allocation et la géométrie ne sont pas deux réglages indépendants.

## Lecture pour le régime non convexe

Le bras `relu/mlp` sous SLE garde l'arbre devant jusqu'à $B=2000$ dès
l'écartement 5, alors que les deux bras linéaires ont déjà basculé à ce budget.
L'avantage de l'arbre n'est donc pas moins visible hors du cadre linéaire, il est
visible plus longtemps, ce qui va contre l'intuition qu'il faudrait davantage
d'investigation dans le cas non linéaire pour retrouver l'effet.

Sous allocation uniforme, en revanche, le bras `relu/mlp` est le seul où la
chaîne domine à tous les budgets dès l'écartement 5, et de vingt à vingt-cinq
pour cent. La sélection de bassin y joue à plein, une arête longue coûtant un
mauvais bassin sans compensation, et seul le financement ciblé de ces arêtes par
SLE renverse le compte.

## La construction dérivée gagne ici et seulement ici

`hybrid_eval.py`. CTL-MSTq applique les trois corrections que le surrogate impose
et réintroduit le gain de portée avec un poids $\lambda$ balayé de 0 à 50. Rang
moyen sur dix constructions, un étant le meilleur.

| construction | 27 cellules de la grille | 18 cellules à budget serré | 6 cellules réelles |
|---|---|---|---|
| MSTq $\lambda=1.5$ | **3.89** | 3.33 | 6.67 |
| MSTq $\lambda=15$ | 4.26 | **2.39** | 7.50 |
| CTL-MST | 4.52 | 4.94 | 2.67 |
| Chain | 6.74 | 8.89 | **2.33** |
| MSTq $\lambda=0$ | 7.11 | 7.89 | 3.83 |
| CTL-MSTc | 9.33 | 10.00 | 4.50 |

Les deux premières colonnes et la dernière sont presque inversées. Sur la famille
contrôlée la construction dérivée règle complètement la défaillance de CTL-MSTc,
dernier dans 23 des 27 cellules et dans les 18 cellules serrées. Sur les
collections réelles elle est avant-dernière, et le terme de portée y nuit d'autant
plus que son poids monte.

Conséquence pour le papier. La famille synthétique à grappes n'est pas un
substitut fiable des collections réelles pour départager des constructions
d'arbre, le classement s'y inversant presque entièrement. Cela vaut aussi pour
Syn-10, qui appartient à cette famille.

## Réserves

Le pas est fixé à 0.05 pour tous les bras et tous les écartements, sans
recalibration, ce qui est conservateur mais peut désavantager les configurations
à grande échelle de sortie.

L'écartement modifie l'échelle des cibles, donc les RMSE ne se comparent
qu'à écartement fixé, ce que fait le rapport à Chain.

$T=200$ et $K=5$ dans tous les cas, l'effet du nombre de grappes et du nombre de
tâches n'est pas exploré.
