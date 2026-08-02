# Enracinement au nœud central en profondeur, table non convexe

`root_depthcentral.py`, cinq jeux, MLP à une couche cachée de largeur 32 au pas
recalibré par jeu, vingt graines, budgets de la table principale. Les deux
enracinements tournent dans la même boucle sur les mêmes graines et les mêmes
tirages de tâches, donc la comparaison ne porte aucune dérive entre exécutions.

Seul le MST est invariant par enracinement, si bien que la règle en deux passes
s'y applique littéralement. Pour les autres constructions le médoïde reste la
graine de construction et seule l'orientation change, la topologie non orientée
étant identique dans les deux bras. Le centre est obtenu par épluchage des
feuilles, avec départage par le critère du médoïde quand il en reste deux.

## La profondeur baisse comme prévu

| jeu | CTL-MST | CTL-MSTc | Chain | RandTree | CTL-kNN | Star |
|---|---|---|---|---|---|---|
| Syn-10 | 54.0 → 34.7 | 199 → 100 | 199 → 100 | 13.3 → 9.2 | 5.0 → 5.0 | 1 → 1 |
| WEAVE-UK | 52.0 → 32.0 | 499 → 250 | 499 → 250 | 16.6 → 11.4 | 6.0 → 6.0 | 1 → 1 |
| FMNIST | 37.0 → 24.2 | 199 → 100 | 199 → 100 | 13.6 → 9.2 | 9.4 → 6.4 | 1 → 1 |
| CIFAR | 27.7 → 18.8 | 199 → 100 | 199 → 100 | 13.4 → 9.2 | 5.2 → 4.9 | 1 → 1 |
| CelebA | 28.1 → 20.2 | 375 → 188 | 375 → 188 | 14.8 → 10.8 | 5.6 → 5.2 | 1 → 1 |

Les deux arbres en chemin voient leur profondeur exactement divisée par deux,
leur centre étant le milieu du chemin, le MST perd un tiers, et Star est
inchangée puisque son centre est le moyeu.

## L'effet sur la performance est petit et de signe instable

Sur les soixante-douze cellules méthode par jeu par budget, hors Individual,
dix-sept s'améliorent significativement, seize se dégradent et trente-neuf ne
bougent pas. Par méthode, en comptant améliorations, dégradations et cellules
inchangées.

| méthode | mieux | moins bien | inchangé |
|---|---|---|---|
| CTL-MST | 5 | 1 | 6 |
| Chain | 5 | 3 | 4 |
| CTL-MSTc | 3 | 7 | 2 |
| RandTree | 2 | 5 | 5 |
| CTL-kNN | 2 | 0 | 10 |
| Star | 0 | 0 | 12 |

CTL-MST, la seule méthode où la règle en deux passes s'applique au sens strict,
est celle qui en profite le plus, et son meilleur cas est WEAVE-UK à B=2000, où
la RMSE passe de 1202.9 à 1177.1, soit -2.1 % avec p<0.001. Sur Syn-10 le gain
est systématique pour Chain, de -1.0 à -2.3 %, et pour CTL-MST, de -0.7 à
-1.5 %. Sur Fashion-MNIST Chain, CTL-kNN et CTL-MSTc gagnent de 0.2 à 1.6 point
d'exactitude. En sens inverse, CTL-MSTc se dégrade sur CIFAR, sur CelebA et sur
WEAVE-UK, et RandTree se dégrade partout sauf sur Fashion-MNIST.

La meilleure cellule de chaque jeu ne bouge presque pas. Sur WEAVE-UK à B=500
elle passe de 1116.9 avec CTL-MSTc au médoïde à 1126.0 avec Chain au centre,
donc l'enracimenent topologique fait légèrement reculer l'optimum. Sur Syn-10 et
Fashion-MNIST il le fait légèrement avancer.

## Lecture

Diviser la profondeur par deux ne rapporte presque rien. C'est le fait saillant,
et il ne va pas dans le sens de la borne, qui fait de la profondeur une quantité
gouvernante. Le résultat rejoint la mesure de longueur de chemin de
`path_geometry.py`, où le chemin glouton accumule 84.5 de distance contre 7.7
pour le pire chemin du MST tout en le devançant sous le modèle non convexe. Les
deux mesures disent la même chose, la géométrie de l'arbre pilote la borne mais
pas le résultat empirique dès que l'hypothèse de contraction tombe.

Le dispositif d'enracinement antérieur, `expD_root_*`, portait déjà le même
avertissement sans qu'il ait été relevé. Sur WEAVE-UK en régime linéaire il
donne le nœud central en profondeur à 1187.2, le médoïde à 1165.0, une racine
aléatoire à 1152.1 et la pire feuille à 1135.7, cette dernière étant donc la
meilleure des quatre. Le classement favorable au nœud central ne vaut que sur le
jeu synthétique.

## Réserves

Le budget n'est pas recalibré par enracinement, ce qui est le choix conservateur
puisque les deux bras partagent le pas du dispositif publié.

Pour Chain et CTL-MSTc, dont la construction dépend de la graine, l'expérience
mesure l'orientation à topologie fixée et non la construction complète depuis le
nœud central, laquelle donnerait un autre arbre.

Sur WEAVE-UK les vingt répétitions ne font varier que l'initialisation, le
panneau de sous-stations étant fixe.

# Enracinement optimal et profondeur optimale

`depth_sweep.py`. En déroulant la récurrence d'arête le long d'un chemin, la
contribution de l'arête $j$ à l'erreur de la feuille est
$\delta_j \exp(-\lambda \sum_{i=j}^{k} b_i)$, donc une arête n'est corrigée que
par le budget dépensé en aval d'elle. À allocation uniforme cela donne un
escompte géométrique de raison $q = \rho^{B/T}$ par saut, compté à rebours depuis
la feuille, et la quantité gouvernante est la longueur de chemin escomptée
$\Phi(r,v) = \sum_j \delta_j q^{\text{sauts de } j \text{ à } v}$.

Deux prédictions en découlent, toutes deux testées ici.

## L'enracinement, quatre règles comparées

Minimiser $\max_v \Phi$ ou $\frac{1}{T}\sum_v \Phi$ dégénère, quand $q \to 1$,
vers le 1-centre pondéré et la 1-médiane pondérée de l'arbre, la pondération
étant $\delta$ et la distance celle mesurée le long de l'arbre. Ni l'un ni
l'autre n'est le médoïde, qui minimise une distance ambiante en ignorant la
topologie, ni le centre en sauts, qui compte les arêtes sans les peser.

Rang moyen sur toutes les topologies, tous les budgets et les deux jeux, 1 étant
le meilleur des quatre.

| enracinement | rang moyen |
|---|---|
| 1-médiane pondérée | 2.04 |
| 1-centre pondéré | 2.14 |
| centre en sauts | 2.57 |
| médoïde | 3.26 |

Le médoïde est donc le plus mauvais des quatre et les deux règles issues de la
borne les deux meilleures. Sur l'arbre couvrant minimal non contraint, écart au
médoïde, apparié sur cinq répétitions.

| jeu | B | centre en sauts | 1-centre pondéré | 1-médiane pondérée |
|---|---|---|---|---|
| Syn-10 | 500 | -0.84 % | -1.80 % | -0.01 % |
| Syn-10 | 1000 | -1.25 % | -1.37 % | -1.08 % |
| Syn-10 | 2000 | -1.07 % | -1.83 % | -1.44 % |
| WEAVE-UK | 500 | +0.09 % | -0.20 % | +0.09 % |
| WEAVE-UK | 1000 | -1.21 % | -1.50 % | -1.21 % |
| WEAVE-UK | 2000 | -2.03 % | -2.69 % | -2.03 % |

Les valeurs de $p$ plafonnent à 0.062 sur cinq répétitions appariées, cette
valeur étant le minimum atteignable, si bien qu'elle signale un balayage complet
des cinq répétitions dans le même sens. L'ampleur reste modeste, de un à trois
pour cent, mais la direction est celle que la borne prédit et elle est stable.

Cela explique aussi pourquoi le remplacement du médoïde par le centre en sauts,
mesuré séparément plus haut, ne rapportait presque rien. Le centre en sauts est
le troisième des quatre, à peine devant le médoïde.

## La profondeur, pas d'optimum intermédiaire

La courbe est non monotone, mais son point singulier est un maximum et non un
minimum. L'étoile à profondeur 1 se tient correctement, la région de profondeur 3
à 8 est la pire, et l'extrémité profonde gagne partout.

| WEAVE-UK, B=2000, médoïde | profondeur | longueur cumulée | RMSE |
|---|---|---|---|
| profondeur ≤ 1, étoile | 1.0 | 0.661 | 1311 |
| profondeur ≤ 4 | 5.0 | 0.957 | 1414 |
| profondeur ≤ 12 | 12.8 | 1.502 | 1249 |
| profondeur ≤ 32 | 30.3 | 1.831 | 1210 |
| sans contrainte | 39.3 | 1.667 | 1203 |

Syn-10 montre la même forme à B=1000 et B=2000, un creux de performance vers la
profondeur 4 à 8 puis une amélioration continue jusqu'à l'extrémité. À B=500 la
décroissance est monotone dès la profondeur 1.

La raison se lit dans la colonne des longueurs cumulées. Sur WEAVE la profondeur
passe de 1 à 49 quand la longueur cumulée racine-feuille ne passe que de 0.661 à
2.156, soit un facteur 3.3 contre 49. Les sauts raccourcissent presque aussi vite
que leur nombre augmente, donc les intermédiaires restent quasi colinéaires et le
modèle colinéaire $\Phi(\ell) = \frac{d}{\ell}\frac{q(1-q^\ell)}{1-q}$,
décroissant en $\ell$, s'applique sur toute l'étendue balayée. Le compromis que
la borne autorise ne se déclenche jamais dans ce domaine.

L'aplatissement commence vers la profondeur 20 à 25 sur les deux jeux, ordre de
grandeur de l'horizon $1/\kappa$ avec $\kappa = \lambda B/T$, au-delà duquel les
arêtes amont sont exponentiellement atténuées et cessent de compter.

## Réserve

Le balayage plafonne à une profondeur de 49 sur $T=500$, la contrainte portant
sur le nombre d'enfants, si bien que le cap 2 donne un arbre binaire et non un
chemin. L'extrémité à profondeur 499 est la chaîne gloutonne de la table
principale, qui reste la meilleure méthode sous modèle non convexe, ce qui
prolonge la tendance observée ici sans la démontrer dans la même famille.
