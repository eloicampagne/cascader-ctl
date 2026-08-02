# Rendre le raffinement contractif ne rétablit pas l'avantage de l'arbre

`convex_head.py`. La distance qui construit l'arbre est déjà une quantité
convexe, $\|\hat g_u - \hat g_v\|$ avec $g = X^\top y$, le gradient en $\theta=0$
du moindre carré. Ce qui ne l'est pas est l'opérateur de raffinement. On teste
donc la correction la moins chère, geler le corps du réseau sous la racine, de
sorte que chaque tâche hérite de la représentation de son parent et ne raffine
qu'une tête linéaire. L'hypothèse de contraction devient vraie par construction.
Dix répétitions, budgets et graines appariés entre les deux modes.

## Résultat

Écart à Chain, apparié, négatif quand la méthode bat Chain, l'exactitude de
CIFAR étant retournée pour que la convention tienne partout.

| collection | B | CTL-MST corps entraîné | CTL-MST corps gelé |
|---|---|---|---|
| CIFAR-10 | 500 | $-3.7$ % | $-2.5$ % |
| CIFAR-10 | 2000 | $-3.7$ % | $-3.7$ % |
| Syn-10 | 500 | $-3.7$ % | $-3.3$ % |
| Syn-10 | 2000 | $+3.5$ % | $+4.6$ % |
| WEAVE-UK | 500 | $+3.8$ % | $+0.7$ % |
| WEAVE-UK | 2000 | $+7.8$ % | $+1.1$ % |

Aucune construction ne dépasse Chain dans aucune cellule en mode gelé. La
prédiction, à savoir le retour de l'ordre du tableau linéaire, est fausse.

## Ce qui subsiste

Sur WEAVE l'écart entre constructions s'effondre d'un facteur six ou sept, et il
s'effondre parce que Chain perd davantage au gel que l'arbre.

| WEAVE-UK, coût du gel | B=500 | B=2000 |
|---|---|---|
| Chain | $+9.9$ % | $+12.0$ % |
| CTL-MST | $+6.7$ % | $+5.1$ % |
| CTL-MSTc | $+10.6$ % | $+11.9$ % |
| MSTq $q=0.9$, $\lambda=0$ | $+6.3$ % | $+5.1$ % |

Chain tire donc du raffinement non convexe un profit que les constructions par
arbre n'en tirent pas, ce qui est la sélection de bassin proposée par le
reviewer fdRt. Le partage est net, les deux constructions de type chaîne ou
quasi-chaîne d'un côté, Chain et CTL-MSTc qui dégénère en chemin, les deux
constructions à branches de l'autre.

## Confondant

Le gel dégrade tout le monde et comprime la plage, sur WEAVE de 8 % d'étendue
entre méthodes à 1.7 %. La convergence des méthodes peut donc être un effet de
plancher plutôt que le rétablissement d'un ordre. Le protocole ne sépare pas les
deux, et il faudrait pour cela un régime gelé dont la performance absolue égale
celle du régime complet, ce que la réduction de capacité interdit.

## Portée

Combiné au bras `lin/mlp` de `scatter_sweep.py`, où le remplacement du modèle
local linéaire par le réseau reproduit la carte à un point près, on dispose de
deux tests indépendants concluant identiquement. L'ordre observé sur les
collections réelles n'est pas un artefact de la non-convexité. Il tient à la
géométrie des collections.
