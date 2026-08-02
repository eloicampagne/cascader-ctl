# Distance NTK et arbre dynamique

`dynamic_ntk.py`. Dix répétitions, allocation plate pour toutes les méthodes, la
racine recevant `b_proto_frac * B` et le reste étant partagé également. Cette
allocation commune est imposée par le protocole, SLE exigeant de connaître
l'arbre avant de le construire, ce que ni le régime `ntk-root` ni le régime
dynamique ne permettent. Les valeurs ne sont donc pas comparables à la Table 1.

## La distance NTK ne rapporte presque rien

On remplace $g_v = X_v^\top y_v$ par $\nabla_\theta L_v(\theta_0)$, le gradient
de la perte du réseau en un point partagé. Écart au régime statique, même arbre,
apparié, le signe étant orienté pour que le négatif soit un gain.

| | Chain | CTL-MST | CTL-MSTc | MSTq |
|---|---|---|---|---|
| CIFAR B=500 | $-0.8$ % | $-0.7$ % | $-0.2$ % | $-0.5$ % |
| CIFAR B=2000 | $-0.3$ % | $-0.6$ % | $-0.1$ % | $-0.2$ % |
| Syn-10 B=500 | $+1.4$ % | $+1.6$ % | $+0.3$ % | $+1.1$ % |
| WEAVE B=500 | $-0.0$ % | $+1.5$ % | $+0.0$ % | $+1.5$ % |
| WEAVE B=2000 | $+0.1$ % | $+0.6$ % | $+0.0$ % | $+0.7$ % |

Gain inférieur au point sur CIFAR, significatif dans deux cellules. Perte
significative sur Syn-10 et WEAVE pour la famille MST. La lecture est cohérente
avec le diagnostic de linéarité déjà établi : les cibles de WEAVE et de Syn-10
sont quasi linéaires, donc $X^\top y$ y est déjà le bon proxy et le gradient du
réseau n'y ajoute que le bruit de l'initialisation. Sur CIFAR, où la cible ne
l'est pas, le remplacement paie faiblement.

Évaluer les gradients au point de la racine entraînée plutôt qu'à
l'initialisation dégrade partout, de 1 à 3 points selon la cellule. En ce point
les gradients sont dominés par la structure commune que la racine n'a pas encore
absorbée, ce qui réduit le pouvoir discriminant entre tâches.

## L'arbre dynamique gagne à budget serré

La construction et la cascade avancent ensemble. Chaque parent fraîchement
entraîné score les tâches encore détachées par
$\|\nabla L_v(\theta_u)\| / \|\nabla L_v(\theta_0)\|$, la meilleure est attachée
puis entraînée. Le rapport au démarrage à froid rend la quantité sans échelle.
Aucun pas de gradient hors de la cascade, donc le budget est identique.

Écart au meilleur arbre statique de la cellule, apparié.

| collection | B | meilleur statique | dynamique | écart | p |
|---|---|---|---|---|---|
| Syn-10 | 500 | CTL-MST 6.087 | **5.964** | $-2.0$ % | 0.0023 |
| CIFAR | 500 | Chain 0.7265 | **0.7317** | $-0.7$ % | 0.0033 |
| Syn-10 | 2000 | Chain 5.487 | 5.529 | $+0.8$ % | 0.085 |
| CIFAR | 2000 | Chain 0.7329 | 0.7334 | $-0.1$ % | 0.44 |
| WEAVE | 500 | CTL-MSTc 1121.8 | 1153.8 | $+2.9$ % | $<10^{-4}$ |
| WEAVE | 2000 | Chain 1115.8 | 1151.6 | $+3.2$ % | $<10^{-4}$ |

À budget serré, le régime qui intéresse CTL, l'arbre dynamique bat le meilleur
arbre statique sur deux collections sur trois, significativement, et sans
surcoût. À budget large il est neutre. Sur WEAVE il perd nettement, ce qui
prolonge l'anomalie déjà relevée ailleurs dans ce dossier.

La profondeur de l'arbre dynamique vaut 43 à 56 selon la collection, entre celle
du MST, 21 à 52, et celle de la chaîne, 199 ou 499. Il se place donc dans la
plage de profondeur intermédiaire que le balayage `depth_sweep.py` désignait
comme la pire pour les arbres statiques, et il y gagne quand même. Ce qui compte
est la difficulté de transfert mesurée au point du parent, non la distance
ambiante.

## Ce que cela dit du diagnostic

La limitation ne se trouve pas dans le score de construction. Trois scores
différents, testés à géométrie et budget fixés, se tiennent à quelques points les
uns des autres, alors que le passage du graphe figé au graphe recalculé pendant
la cascade rapporte 2 % à budget serré. C'est l'hypothèse de graphe statique qui
coûte.

## Réserves

L'allocation plate désavantage les constructions à grappes, comme
`scatter_sweep.py` l'a montré. Un protocole permettant SLE au régime dynamique
demanderait de réserver un budget de sondage, ce qui romprait l'égalité de budget.

Le coût de calcul de l'arbre dynamique est $O(T^2)$ évaluations de gradient, soit
12 s pour $T=500$ contre une fraction de seconde pour le MST statique. Négligeable
devant l'entraînement ici, pas nécessairement à plus grande échelle.
