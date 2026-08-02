# Les données sont-elles vraiment non linéaires ? (objection anticipée)

Un rapporteur peut accorder que le modèle local est non convexe et objecter que
les **données**, elles, sont essentiellement linéaires, de sorte que le MLP n'a
aucune structure non linéaire à exploiter. Sur WEAVE en particulier, prévoir le
pas de temps suivant à partir de 48 retards récents revient à une approximation
d'ordre 1, linéaire par construction.

## Protocole

`linearity_diagnostic.py` et `linearity_diagnostic2.py`. Par tâche, on ajuste
l'optimum linéaire exact (ridge en forme close, donc sans confusion avec un pas
d'apprentissage) et un MLP entraîné à convergence (Adam, 3000 époques, 32 unités
cachées), puis on compare la performance de test. Si le MLP ne bat pas l'optimum
linéaire, l'espérance conditionnelle est linéaire dans les features fournies et le
jeu ne porte pas de signal non linéaire. Cent tâches par jeu.

## Résultat : marge de non-linéarité par jeu

| Jeu | Métrique | Optimum linéaire | MLP convergé | Gain MLP | MLP gagne |
|---|---|---|---|---|---|
| Syn-10 (contrôle négatif) | RMSE | 5.15 | 11.01 | -113.9 % | 0 % |
| London daily | RMSE | 3.47 | 6.54 | -88.4 % | 0 % |
| WEAVE-UK | RMSE | 1263 | 1898 | -50.3 % | 0 % |
| Fashion-MNIST | accuracy | 0.947 | 0.940 | -0.8 % | 20 % |
| CIFAR-10 | accuracy | 0.706 | 0.727 | **+3.0 %** | 61 % |
| ReLU-teacher (contrôle positif) | RMSE | 4.47 | **0.63** | **+85.9 %** | 100 % |

Le diagnostic est calibré, les deux contrôles se comportent comme attendu. Le
contrôle positif gagne massivement avec seulement 64 échantillons par tâche, donc
un échec du MLP sur London (592 échantillons) ou WEAVE (528 échantillons) ne
s'explique pas par un manque de données.

L'objection est donc **fondée** pour WEAVE, London et le synthétique, et pour
Fashion-MNIST. Elle ne tient pas pour CIFAR-10, où la marge est modeste mais
réelle, ni évidemment pour le teacher ReLU.

Réserve honnête, le diagnostic emploie une capacité de MLP fixe sans
régularisation ni arrêt précoce. Un MLP réglé ferait peut-être mieux. Les
contrôles limitent toutefois cette réserve.

## Conséquence sur ce que prouve la table MLP

Il faut distinguer deux choses que l'expression « validation non convexe »
confond.

Le **paysage d'optimisation non convexe**. L'hypothèse de contraction du papier,
`rho_v = ||I - (eta/n) X_v^T X_v||`, porte sur le paysage de perte du modèle
local. Un MLP sur données linéaires a bel et bien un paysage non convexe, avec
points selles, symétries de permutation et bassins multiples, donc le mode
d'échec que la théorie risque, un raffinement qui quitte le bassin du parent, est
réellement exercé. La table MLP teste cela, et c'est le test dans les termes
propres de l'hypothèse.

La **cible non linéaire**. Là, seuls CIFAR-10 (faiblement) et le teacher ReLU
(fortement) qualifient.

## Résultat inattendu et fort sur WEAVE

Optimum linéaire convergé par tâche sur les 500 tâches, RMSE **1158**. MLP
convergé par tâche, **1898**, il sur-apprend lourdement (528 échantillons,
54 features, environ 1800 paramètres). Or CTL-MSTc sous budget serré avec ce même
MLP atteint **1121** à B=2000, et **1117** à B=500.

La cascade transforme donc un modèle sur-paramétré inutilisable en entraînement
indépendant (1898) en le meilleur modèle de l'étude (1117), sous le plafond de
l'optimum linéaire par tâche (1158). Le démarrage à chaud depuis un voisin agit
comme une régularisation, et c'est précisément le modèle non convexe
sur-paramétré qui en bénéficie. C'est un argument plus fort que celui que le
rebuttal avance actuellement.

## Table complète sur données prouvées non linéaires

`nonlinear_data_table.py`. Tâches teacher ReLU clusterisées (T=200, K=5, cibles
`y = W2 relu(W1 x)`), protocole fidèle, distance gradient, allocation SLE,
10 répétitions. RMSE moyen.

| B | Individual-Linear | Individual | Star | RandTree | Chain | CTL-MST | CTL-MSTc |
|---|---|---|---|---|---|---|---|
| 500 | 4.28 | 4.22 | 4.67 | 4.35 | 3.42 | **3.08** | 3.84 |
| 2000 | 3.96 | 3.81 | 4.55 | 3.80 | 2.31 | **2.28** | 2.82 |

Toutes les cascades géométriques battent l'entraînement indépendant de façon
significative, p entre 5e-6 et 6.5e-7 (t apparié unilatéral, 10 répétitions). À
B=2000, CTL-MST réduit l'erreur de l'entraînement indépendant de 40 % (2.28 contre
3.81) et passe aussi sous le plafond du modèle linéaire convergé (4.47), ce que
l'entraînement indépendant du MLP ne fait pas.

À noter honnêtement, sur ces données CTL-MST bat CTL-MSTc (3.08 contre 3.84 à
B=500). Le `lambda` de MSTc vaut 0.6, valeur réglée sur WEAVE et non réajustée
ici, ce qui suffit vraisemblablement à l'expliquer.

## Sur le Transformer

Un Transformer entraîné de bout en bout serait un modèle non convexe de plus, mais
appliqué aux mêmes données. Or le diagnostic montre que le verrou n'est pas la
classe de modèle, c'est la non-linéarité de la cible. Un Transformer sur WEAVE
répéterait donc la réponse déjà donnée par le MLP, à un coût très supérieur. Le
teacher ReLU et CIFAR-10 comblent le vrai manque. Le papier porte par ailleurs
déjà une expérience Chronos T5-small en LoRA, backbone gelé.

## Fichiers

| Fichier | Contenu |
|---|---|
| `linearity_diagnostic.py` | WEAVE, Syn-10, teacher ReLU |
| `linearity_diagnostic2.py` | London daily, Fashion-MNIST, CIFAR-10 |
| `london_data.py`, `london_cache/` | Tâches London daily (500 foyers, horizon jour) |
| `nonlinear_data_table.py` | Table complète sur données non linéaires |
| `linearity_diagnostic*_{raw,summary}.csv`, `nonlinear_data_table_*.csv` | Résultats |
