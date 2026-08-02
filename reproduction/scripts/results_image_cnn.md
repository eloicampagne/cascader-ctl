# Un modèle adapté aux images plutôt qu'un MLP générique

Le MLP sur PCA-100 est un modèle non convexe générique, pas un modèle adapté aux
images. La projection en composantes principales détruit précisément la structure
spatiale qu'une convolution exploite, de sorte qu'un rapporteur pouvait objecter
que l'absence de marge non linéaire mesurée sur Fashion-MNIST et la faible marge
sur CIFAR-10 tenaient au choix du modèle et non aux données.

## Protocole

`image_cnn.py`. Le modèle local devient un petit réseau convolutif appliqué à
l'image brute standardisée, deux blocs convolution-ReLU-maxpool en 8 puis
16 canaux suivis d'une lecture linéaire, soit 2033 paramètres sur Fashion-MNIST
et 2417 sur CIFAR-10, à comparer aux 3265 du MLP. La taille est donc comparable
et la comparaison porte sur le biais inductif.

Tout le reste est inchangé. Les tâches binaires clusterisées sont identiques à
celles de la table publiée, l'échantillonneur ne tirant que des indices à partir
des étiquettes et du générateur aléatoire, un même germe sélectionne donc les
mêmes images quelle que soit la représentation. La distance gradient et la
racine médoïde restent calculées sur les features PCA-100, l'allocation reste
SLE, le budget reste un nombre de pas de gradient, et le pas d'apprentissage est
recalibré sur la ligne de base Individual comme pour le MLP, ce qui joue en
faveur de la ligne de base.

## Marge de non-linéarité avec un modèle adapté

Par tâche, optimum linéaire contre CNN entraîné à convergence (Adam, 800 époques,
perte d'entraînement nulle dès 500 époques), 100 tâches par jeu. La référence
linéaire est prise comme le maximum par tâche entre une régression logistique sur
PCA-100 et une régression logistique sur pixels bruts, ce qui la renforce
artificiellement et rend la comparaison conservatrice.

| Jeu | Logistique PCA-100 | Logistique pixels | Meilleur linéaire | CNN convergé | Gain CNN | CNN gagne |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.9474 | 0.9498 | 0.9523 | 0.9536 | +0.13 % | 38 % |
| CIFAR-10 | 0.7060 | 0.7155 | 0.7222 | 0.7760 | **+7.45 %** | **84 %** |

Le MLP mesurait +3.0 % sur CIFAR-10 avec 61 % de victoires. Le prior
convolutionnel double donc la marge non linéaire détectable, et contre une
référence linéaire renforcée. Sur Fashion-MNIST la conclusion antérieure tient,
le jeu ne porte pas de structure non linéaire exploitable même par un modèle
taillé pour les images, ce qui écarte l'explication par le choix du MLP.

## Table complète avec le CNN comme modèle local

10 répétitions, exactitude moyenne. Pas d'apprentissage retenu par calibration
sur Individual, 0.1 sur les deux jeux.

| Jeu | B | Individual | Star | RandTree | Chain | CTL-MST | CTL-MSTc |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 500 | 0.7620 | 0.6952 | 0.8242 | **0.9297** | 0.9171 | 0.9231 |
| Fashion-MNIST | 2000 | 0.9400 | 0.9162 | 0.9205 | **0.9484** | 0.9271 | 0.9454 |
| CIFAR-10 | 500 | 0.6155 | 0.5442 | 0.6164 | **0.7344** | 0.7233 | 0.7236 |
| CIFAR-10 | 2000 | 0.6974 | 0.6325 | 0.7065 | **0.7667** | 0.7281 | 0.7598 |

Écart à Individual et t apparié unilatéral sur les 10 répétitions.

| Jeu | B | Chain | CTL-MST | CTL-MSTc |
|---|---|---|---|---|
| Fashion-MNIST | 500 | +0.168 (3.6e-9) | +0.155 (2.1e-9) | +0.161 (6.5e-9) |
| Fashion-MNIST | 2000 | +0.008 (2.3e-3) | -0.013 (n.s.) | +0.005 (4.7e-3) |
| CIFAR-10 | 500 | +0.119 (2.9e-9) | +0.108 (9.1e-9) | +0.108 (1.3e-8) |
| CIFAR-10 | 2000 | +0.069 (4.0e-8) | +0.031 (4.4e-5) | +0.062 (1.6e-8) |

Star dégrade partout, RandTree n'aide qu'à budget serré sur Fashion-MNIST.

## Lecture

Le prior convolutionnel et la cascade se cumulent au lieu de se substituer, ce
qui est l'inverse de l'objection attendue. Sur CIFAR-10 à B=2000, l'avantage de
CTL-MSTc sur Individual passe de 1.2 point avec le MLP à 6.2 points avec le CNN,
et l'exactitude absolue de la cascade monte de 0.723 à 0.760. La cascade atteint
donc un niveau plus élevé avec une architecture adaptée, et pas seulement un
avantage relatif plus grand face à une ligne de base affaiblie. Cette distinction
importe, car le CNN part effectivement de plus loin à budget serré, Individual
tombant de 0.911 à 0.762 sur Fashion-MNIST à B=500.

Sur Fashion-MNIST, la cascade avec CNN n'atteint pas le niveau de la cascade avec
MLP à B=2000, 0.9454 contre 0.951. C'est cohérent avec l'absence de marge non
linéaire mesurée sur ce jeu, un modèle plus riche n'ayant rien de plus à y
apprendre.

À noter honnêtement, CTL-MST reste en retrait de Chain sur CIFAR-10, 0.728 contre
0.767 à B=2000, comme sur les données teacher ReLU. Le classement entre
constructions géométriques dépend du régime.

## CelebA, tenté puis abandonné

Le cas CelebA aurait été le plus fort du lot, le CNN y étant entraîné depuis une
initialisation aléatoire sans backbone gelé, ce qui aurait levé la réserve que le
rebuttal formule lui-même en écrivant que CelebA repose sur des features ResNet-18
gelées et ne compte donc pas parmi les preuves bout-en-bout. Le coût l'a rendu
impraticable, environ 3.25 millions de multiplications-accumulations par image
contre 0.2 sur CIFAR-10, soit de l'ordre de neuf heures par répétition sur les
deux budgets publiés. L'expérience a été arrêtée et **aucun de ses résultats
n'est rapporté**.

Ce qui reste utilisable si l'on veut la reprendre.

`celeba_raw_extract.py` reconstruit le cache d'images brutes 64x64 dans l'ordre
exact du cache de features, et vérifie par assertion que les étiquettes
récupérées coïncident avec `celeba_cache/y_*.npy`. Cette vérification est passée.
Les tableaux `celeba_cache/Ximg_{train,test}.npy` sont sur disque, 185 Mo.
`image_cnn.py` gère CelebA de bout en bout, avec un CNN à trois blocs de
8081 paramètres, contre 16449 pour la tête MLP sur features, et les options
`--rep0` et `--eta` permettent de répartir les répétitions entre processus.

La seule mesure complète obtenue est la calibration du pas d'apprentissage sur
Individual à B=2000, `image_cnn_celeba_eta_calibration.log`, exactitude 0.524 /
0.585 / 0.630 / 0.639 / 0.622 pour eta valant 0.01 / 0.05 / 0.1 / 0.2 / 0.5,
optimum intérieur à 0.2. Elle situe le problème, Individual atteignant 0.639
depuis zéro contre 0.685 avec les features gelées au même budget, donc un régime
sensiblement plus difficile. Les quelques valeurs de méthodes qui figurent en fin
de ce journal proviennent d'une répétition unique et ne doivent pas être citées.

## Fichiers

| Fichier | Contenu |
|---|---|
| `image_cnn.py` | CNN local, modes `diag` et `table`, option `--dataset` |
| `image_cnn_diag_{fmnist,cifar}_{raw,summary}.csv` | Marge de non-linéarité |
| `image_cnn_table_{fmnist,cifar}_{raw,summary}.csv`, `image_cnn_table_raw.csv` | Table complète |
| `image_cnn_{diag,table}_{FMNIST,CIFAR}.log` | Journaux d'exécution |
| `celeba_raw_extract.py`, `celeba_cache/Ximg_*.npy` | Cache d'images brutes CelebA, vérifié, inutilisé |
| `image_cnn_celeba_eta_calibration.log` | Calibration CelebA, seule mesure complète de la tentative |
