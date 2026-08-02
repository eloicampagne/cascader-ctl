# Table 1 refaite avec un pas d'apprentissage propre à chaque méthode

`table1_per_method_eta.py`. Quatre collections, les sept méthodes du papier, les
deux classes de modèle local, les budgets publiés, **20 graines d'évaluation**,
soit les mêmes que la Table 1. Tout le reste vient de `table1_nonconvex.py` :
mêmes chargeurs, même `run_method`, mêmes allocations.

## Protocole de sélection

Grille log de neuf valeurs, $3\cdot10^{-4}$ à $0.5$, augmentée du pas publié pour
chaque couple (collection, classe de modèle), de sorte que la sélection ne peut
pas faire pire que la configuration du papier. Le pas est retenu par
(collection, modèle, budget, méthode) sur **cinq graines disjointes**, indices
décalés de 500, donc sur d'autres tirages de tâches pour Syn-10 et pour les
images. Les graines rapportées ne servent jamais au réglage.

Vérification du bord : sur Syn-10 en linéaire toutes les méthodes retiennent
$\eta=0.5$, la plus grande valeur de la grille. Un balayage jusqu'à $\eta=8$
montre que c'est un vrai optimum, Individual passant de 5.15 à 11.69 puis 35.69.
Ce n'est donc pas une troncature.

Réserve : sur WEAVE-UK les tâches sont fixes, donc les graines ne diffèrent que
par l'initialisation et le jeu de test est partagé entre réglage et évaluation.

## Résultat principal

**La cascade bat l'entraînement indépendant dans 18 des 20 cellules**
(collection, modèle, budget), toutes à $p<10^{-4}$.

Écart de la meilleure cascade à Individual, chacun à son propre pas.

| | linéaire | MLP |
|---|---|---|
| Syn-10, $B=500$ | $-37.2$ % | $-45.3$ % |
| Syn-10, $B=1000$ | $+1.3$ % | $-19.0$ % |
| Syn-10, $B=2000$ | $-0.0$ % (p=0.54) | $-9.1$ % |
| WEAVE-UK, $B=500$ | $-36.2$ % | $-31.3$ % |
| WEAVE-UK, $B=1000$ | $-28.9$ % | $-25.4$ % |
| WEAVE-UK, $B=2000$ | $-22.3$ % | $-20.5$ % |
| CIFAR-10, $B=500$ | $+10.8$ % | $+9.5$ % |
| CIFAR-10, $B=2000$ | $+4.3$ % | $+2.6$ % |
| Fashion-MNIST, $B=500$ | $+3.1$ % | $+2.7$ % |
| Fashion-MNIST, $B=2000$ | $+0.5$ % | $+0.8$ % |

Les deux exceptions sont Syn-10 avec un modèle linéaire à $B=1000$ et $B=2000$,
soit 5 et 10 pas par tâche sur une collection dont la cible est quasi linéaire.
C'est le régime où le budget cesse de contraindre, et l'avantage y disparaît,
comme la carte à deux régimes le prévoit.

## Le réglage n'est pas cosmétique

Sur WEAVE-UK toutes les cascades retiennent $\eta=0.03$ en linéaire et $0.1$ en
MLP, alors qu'Individual retient 0.1 à 0.3 puis 0.2 à 0.5. Le protocole à pas
partagé pénalisait donc la baseline sur cette collection. Individual passe de
1902 à 1622 en MLP à $B=500$, et la cascade gagne encore de 31.3 %.

Sur Syn-10 en linéaire le pas publié vaut $1.27\cdot10^{-3}$ alors que toutes les
méthodes retiennent $0.5$, quatre cents fois plus. C'est ce qui explique la
disparition de l'avantage aux grands budgets : bien réglé, l'entraînement
indépendant y atteint déjà l'optimum linéaire.

## Quelle construction gagne

| collection | linéaire | MLP |
|---|---|---|
| Syn-10, budget serré | CTL-MST | CTL-MST |
| Syn-10, budget large | RandTree, à égalité | Chain |
| WEAVE-UK | CTL-MSTc, Chain à $B=2000$ | CTL-MSTc, Chain à $B=2000$ |
| Fashion-MNIST | CTL-MSTc | Chain |
| CIFAR-10 | Chain | Chain |

CTL-MSTc reste la meilleure sur WEAVE-UK, la seule collection réelle, dans les
deux classes de modèle. Chain gagne sur les images et aux budgets larges, ce qui
prolonge exactement le constat déjà fait dans le fil du reviewer fdRt. Star et
CTL-kNN sont partout les plus faibles des cascades.

## Fichiers

`t1pm_reg_raw.csv` et `t1pm_cls_raw.csv` pour les valeurs par graine et par pas,
`t1pm_reg_summary.csv` et `t1pm_cls_summary.csv` pour les moyennes au pas retenu.
Journaux dans `t1pm_reg.log` et `t1pm_cls.log`.
