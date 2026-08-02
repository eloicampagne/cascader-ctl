# CTL quand les tâches arrivent une à une

`online_arrival.py`. Cinq répétitions, budget par tâche $b = B/T$ identique pour
tous les bras, ordre d'arrivée tiré au hasard à chaque répétition. Les six bras
se rangent par connaissance décroissante du futur.

| bras | arbre préalable | ordre choisi | proxy de distance |
|---|---|---|---|
| Offline-MSTc | oui, plus allocation SLE | oui | oui |
| Dynamic | non, construit pendant la cascade | oui | non, gradient au point du parent |
| Online-dyn | non | non | non, gradient au point du parent |
| Online-NN | non | non | oui, $X^\top y$ contre les tâches déjà vues |
| Online-prev | non | non | aucun, parent = tâche précédente |
| Individual | sans objet | sans objet | sans objet |

Online-dyn est la version en ligne de l'arbre dynamique. À l'arrivée de $v$, on
évalue $\|\nabla L_v(\theta_u)\| / \|\nabla L_v(\theta_0)\|$ pour chaque tâche $u$
déjà entraînée et on attache $v$ au meilleur. Aucun $\lambda$, aucune allocation
globale, aucune matrice de distances, aucune connaissance des tâches futures.

## Valeurs moyennes

| | Individual | Online-prev | Online-NN | Online-dyn | Dynamic | Offline-MSTc |
|---|---|---|---|---|---|---|
| CIFAR $B{=}500$ (acc) | 0.6376 | 0.6504 | 0.6939 | 0.7162 | **0.7212** | 0.7136 |
| CIFAR $B{=}2000$ | 0.7109 | 0.6907 | 0.7259 | 0.7283 | **0.7308** | 0.7188 |
| Syn-10 $B{=}500$ (MSE) | 14.84 | 16.85 | 7.32 | 6.85 | **6.38** | 8.24 |
| Syn-10 $B{=}2000$ | 6.29 | 7.87 | 6.22 | 6.13 | **5.72** | 6.47 |
| WEAVE $B{=}500$ (RMSE) | 1904.5 | 1129.5 | 1375.1 | 1196.5 | 1180.0 | **1115.3** |
| WEAVE $B{=}2000$ | 1395.0 | **1114.5** | 1217.5 | 1153.2 | 1144.6 | 1120.8 |

## Écart à Individual, apparié

| | Online-prev | Online-NN | Online-dyn | Offline-MSTc |
|---|---|---|---|---|
| CIFAR 500 | $+2.0$ % (0.079) | $+8.8$ % | $+12.3$ % | $+11.9$ % |
| CIFAR 2000 | $-2.8$ % | $+2.1$ % | $+2.5$ % | $+1.1$ % |
| Syn-10 500 | $+13.6$ % | $-50.6$ % | $-53.8$ % | $-44.4$ % |
| Syn-10 2000 | $+25.1$ % | $-1.1$ % (0.85) | $-2.6$ % (0.64) | $+2.8$ % (0.20) |
| WEAVE 500 | $-40.7$ % | $-27.8$ % | $-37.2$ % | $-41.4$ % |
| WEAVE 2000 | $-20.1$ % | $-12.7$ % | $-17.3$ % | $-19.7$ % |

Les $p$ non écrits valent moins de $0.002$. Online-dyn bat l'entraînement
indépendant dans cinq cellules sur six, la sixième étant Syn-10 à budget large,
c'est-à-dire le régime où le budget cesse de contraindre.

## Ce que coûte l'ignorance du futur

Écart de chaque bras à Offline-MSTc, négatif quand le bras en ligne gagne.

| | Online-NN | Online-dyn | Dynamic |
|---|---|---|---|
| CIFAR 500 | $-2.8$ % | $+0.4$ % | $+1.1$ % |
| CIFAR 2000 | $+1.0$ % | $+1.3$ % | $+1.7$ % |
| Syn-10 500 | $-11.2$ % | $-16.9$ % | $-22.6$ % |
| Syn-10 2000 | $-3.8$ % | $-5.2$ % | $-11.5$ % |
| WEAVE 500 | $+23.3$ % | $+7.3$ % | $+5.8$ % |
| WEAVE 2000 | $+8.6$ % | $+2.9$ % | $+2.1$ % |

Online-dyn se tient à moins de $7.3$ % de la référence hors ligne partout, et la
dépasse sur Syn-10 et CIFAR à budget serré. La construction hors ligne, avec sa
matrice $O(T^2)$, son $\lambda$ et son allocation SLE, n'achète donc presque rien
que le régime en ligne ne retrouve.

## Le score compte plus que la distance ambiante

Online-dyn bat Online-NN dans les six cellules. Les deux règles diffèrent
uniquement par ce qu'elles mesurent, $X^\top y$ à l'initialisation contre la norme
du gradient au point réel du parent candidat. Le second est meilleur partout, et
sa longueur d'arête moyenne est plus grande, $0.901$ contre $0.584$ sur CIFAR,
$0.313$ contre $0.214$ sur Syn-10, $0.261$ contre $0.160$ sur WEAVE. Il ne
minimise donc pas la distance ambiante, il minimise la difficulté du transfert
effectivement mesurée. C'est la conclusion déjà tirée dans
`RESULTS_dynamic_ntk.md`, retrouvée ici sans aucune connaissance du futur.

## Le contrôle, et la réserve WEAVE

Online-prev ignore les distances et attache dans l'ordre d'arrivée. Sur Syn-10 il
fait nettement pire qu'Individual, $+13.6$ % et $+25.1$ %, et sur CIFAR à budget
large $-2.8$ %. Le gain vient donc bien du choix du parent, non du fait de copier
des paramètres.

Sur WEAVE, en revanche, Online-prev est le meilleur bras en ligne, $-40.7$ % et
$-20.1$ %, devant Online-dyn, et il égale la référence hors ligne. Sa longueur
d'arête moyenne vaut $0.459$ contre $0.233$ pour l'arbre dynamique. WEAVE
récompense donc les arêtes longues, ce qui prolonge exactement l'anomalie relevée
ailleurs dans ce dossier, où le terme de portée de CTL-MSTc n'y gagne que là et où
l'arbre dynamique y perd. La lecture propre selon laquelle le gain vient de la
distance tient sur deux collections sur trois, et il faut le dire ainsi.

## Réserves

Cinq répétitions, contre vingt dans la Table 1.

Le coût de Online-dyn est de $O(T^2/2)$ évaluations de gradient sans mise à jour,
soit le même ordre que la matrice de distances hors ligne. Il n'y a pas
d'économie de calcul, seulement la suppression de la connaissance du futur.

Le pas $\eta$ reste fixé par collection. Le régime en ligne supprime la
dépendance à $\lambda$ et à l'allocation, pas celle au pas.
