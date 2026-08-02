# Parallélisation de la cascade, comptage puis mesure

Une cascade ne se parallélise qu'aux branchements. Les tâches d'une même profondeur
ne dépendent que de parents déjà entraînés, donc elles s'exécutent ensemble, avec une
barrière entre niveaux. Une chaîne n'a que des niveaux de largeur un.

## Comptage structurel

`critical_path.py`. Chemin critique en pas de gradient, nombre de rondes
séquentielles, et parallélisme moyen requis pour réaliser le chemin critique.

| collection | méthode | rondes | largeur max | cp plat | parallélisme moyen |
|---|---|---|---|---|---|
| WEAVE, $T=500$ | Chain | 500 | 1 | 1802 | 1.11 |
| WEAVE | CTL-MST | 53 | 29 | 461 | 4.34 |
| WEAVE | CTL-MSTc | 500 | 1 | 1802 | 1.11 |
| CIFAR, $T=200$ | Chain | 200 | 1 | 1993 | 1.00 |
| CIFAR | CTL-MST | 32 | 12 | 481 | 4.16 |
| Syn-10, $T=200$ | CTL-MST | 53 | 10 | 704 | 2.84 |

Sous SLE le parallélisme moyen tombe à 1.2 à 1.4 pour CTL-MST, l'allocation
concentrant le budget sur les nœuds à gros sous-arbre, qui sont proches de la racine
et donc sur le chemin critique.

CTL-MSTc dégénère en chemin sur les trois collections, largeur maximale un. L'argument
de parallélisation soutient CTL-MST et non CTL-MSTc.

## Mesure

`parallel_wallclock.py`. Tâches synthétiques en grappes, $T=200$, $K=5$, écartement 5,
$d=100$, 2048 points par tâche, modèle local à 256 unités cachées, $B=2000$. Threads
plutôt que processus, de sorte que les modèles restent en mémoire partagée, et
`torch.set_num_threads(1)` pour que le parallélisme de tâches ne concurrence pas celui
des BLAS. Le pas coûte 1.1 ms contre une vingtaine de microsecondes par soumission,
donc le calcul domine d'un facteur cinquante. Huit répétitions, une exécution de
chauffe, et l'ordre des seize configurations tiré au hasard à chaque répétition.

Médiane et écart interquartile, en secondes.

| | P=1 | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| Chain, plat | 2.08 [1.93, 2.22] | 1.93 | 1.90 | 1.89 [1.73, 2.11] |
| Chain, SLE | 1.89 [1.79, 2.01] | 2.12 | 1.85 | 1.87 [1.72, 2.17] |
| CTL-MST, plat | 1.92 [1.83, 1.98] | 1.34 | 1.11 | **0.93** [0.92, 1.00] |
| CTL-MST, SLE | 1.91 [1.86, 2.06] | 1.68 | 1.63 | 1.56 [1.48, 1.79] |

Accélérations contre $P=1$.

| | P=2 | P=4 | P=8 | plafond |
|---|---|---|---|---|
| CTL-MST, plat | ×1.44 | ×1.73 | **×2.06** | ×5.00 |
| CTL-MST, SLE | ×1.14 | ×1.18 | ×1.23 | ×1.51 |
| Chain, plat | ×1.07 | ×1.09 | ×1.10 | ×1.00 |
| Chain, SLE | ×0.89 | ×1.02 | ×1.01 | ×1.00 |

Les 1.07 à 1.10 de la chaîne restent dans l'écart interquartile, dont les intervalles
se chevauchent entièrement, donc elle ne gagne rien.

## L'ordre d'exécution était un confondant

La première série, à trois répétitions et ordre fixe, donnait la chaîne à 1.66 s et
l'arbre à 1.98 s à un seul travailleur, alors que les deux exécutent exactement 1991
pas sous allocation plate. La chaîne passait toujours en premier, sur une machine
froide. Avec l'ordre randomisé l'écart disparaît.

| $P=1$, budget identique | Chain | CTL-MST | ratio | p (Mann-Whitney) |
|---|---|---|---|---|
| plat, 1991 pas | 2.08 s | 1.92 s | 0.93 | 0.195 |
| SLE | 1.89 s | 1.91 s | 1.01 | 0.798 |

Deux chiffres de la première série sont donc à écarter. L'accélération de l'arbre
valait 2.44 et vaut 2.06. Et sous SLE l'arbre semblait atteindre son plafond, alors
qu'il n'en tire que 1.23 sur 1.51.

## Écart au plafond

Le facteur mesuré à budget plat, 2.06, reste loin du plafond 5.0. La borne
$\max(\text{cp}, \text{travail}/P)$ suppose un rééquilibrage du travail entre niveaux
que la barrière interdit. La largeur moyenne vaut cinq, mais beaucoup de niveaux sont
de largeur un ou deux, donc huit travailleurs restent oisifs une bonne part du temps.
La borne réalisable est $\sum_{\ell} \lceil w_\ell / P \rceil$ fois le coût d'une
tâche, à budgets égaux.

## Réserves

Le budget est identique pour toutes les configurations, donc la comparaison est
appariée, mais elle ne dit rien du temps d'horloge à performance égale, qui
demanderait de fixer l'erreur cible plutôt que le nombre de pas.

Huit travailleurs sur une machine à 8 cœurs performants, sans communication réseau.
Un déploiement réparti ajouterait un transfert de vecteur de paramètres par arête, que
la chaîne paie aussi.
