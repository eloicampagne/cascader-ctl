# CTL sur la hiérarchie M5, avec LightGBM et WRMSSE

`m5_hierarchy.py`. Gardé pour plus tard, hors de la réponse au reviewer.

## Pourquoi

M5 note en WRMSSE sur 42840 séries réparties en douze niveaux d'agrégation,
chaque niveau pesant $1/12$, chaque série pesant sa part de chiffre d'affaires
dans son niveau, et le dénominateur de RMSSE étant la MSE de la marche aléatoire
en échantillon. La hiérarchie **est** un arbre donné par le domaine, et elle a la
propriété dont CTL vit : les agrégats sont denses et bien estimés, les feuilles
sont à 68 % de zéros. La question qui met CTL en valeur est donc : l'arbre appris
par le proxy $X^\top y$ retrouve-t-il, ou dépasse-t-il, l'arbre du domaine ?

## Protocole

300 séries du magasin CA_1 sur quatre niveaux : le total du magasin (Level3), ses
3 catégories (Level8), ses 7 rayons (Level9), et 289 articles tirés au hasard
(Level12). Poids officiels renormalisés sur les niveaux représentés.

Prévision directe à 28 jours, tous les retards remontant d'au moins 28 jours,
donc sans fuite ni récursion. Entraînement jusqu'à d_1913, validation d_1914 à
d_1941, test d_1942 à d_1969 ouvert une seule fois. Dix-neuf variables, retards
28/35/42/56/364, moyennes glissantes 7/28/56 s'arrêtant à $t-28$, jour de
semaine, mois, événement, SNAP.

Modèle local LightGBM. Le transfert est littéral : l'enfant reprend l'ensemble
boosté du parent par `init_model` et lui ajoute $b_v$ arbres. Le budget compte les
itérations ajoutées, comme le papier compte les pas de gradient. Trois graines.

## Résultat

WRMSSE sur le test.

| $B$ | $b$ | Individual | Star | Hierarchy | CTL-MST |
|---|---|---|---|---|---|
| 600 | 2 | 0.8562 | 0.7882 ($-7.9$ %) | 0.7628 ($-10.9$ %) | **0.7455 ($-12.9$ %)** |
| 1500 | 5 | 0.7183 | 0.6391 ($-11.0$ %) | **0.6300 ($-12.3$ %)** | 0.6306 ($-12.2$ %) |
| 3000 | 10 | 0.5951 | **0.5646 ($-5.1$ %)** | 0.5713 ($-4.0$ %) | 0.5908 ($-0.7$ %) |

Tous les $p \le 0.04$ sauf CTL-MST à $b=10$. Le classement sur validation
reproduit celui du test aux deux budgets serrés, CTL-MST 0.8479 contre Hierarchy
0.8814 et Star 0.9085 à $B=600$.

À budget serré l'arbre appris bat la hiérarchie du domaine, donc le proxy du
papier retrouve et dépasse une structure fournie gratuitement par M5. Quand le
budget se desserre l'ordre s'inverse vers Star : même carte à deux régimes que
partout ailleurs dans le dossier.

## Réserve principale, à traiter avant toute publication

Avec `init_model`, un enfant hérite des arbres de tous ses ancêtres. À $B=600$ un
modèle CTL-MST porte 33 arbres en moyenne contre 2 pour Individual et 4 pour
Star, et la profondeur de l'arbre appris vaut 24 contre 3 pour la hiérarchie. Le
budget d'entraînement est égal, le coût d'inférence ne l'est pas. C'est une
différence de fond avec le cas MLP, où l'héritage est gratuit. À coût d'inférence
égal, c'est-à-dire 33 itérations pour Individual, la tendance de la troisième
ligne suggère qu'Individual repasserait devant, et il faut le vérifier.

## Autres réserves

Un seul magasin, quatre niveaux sur douze, 289 articles échantillonnés sur 3049,
donc les poids sont renormalisés et la WRMSSE n'est pas celle de la compétition.

Les gagnants de M5 utilisent des ensembles LightGBM avec variables de prix et de
promotion, agrégations croisées et combinaison de modèles. Notre configuration
est volontairement simple pour rester comparable au reste du papier.

## Ce qui a été essayé avant et qui échoue

`m5_tasks.py`, MLP local sur les 30490 séries du niveau 12 uniquement, sans
hiérarchie : la cascade perd partout, de $+36.7$ % à $T=500$ à $+9.6$ % à
$T=30490$. L'écart se réduit dans la direction prédite mais ne change jamais de
signe. Ce script souffre en outre de deux défauts corrigés dans
`m5_hierarchy.py` : réglage du pas sur le test faute de validation, et prévision
à un pas au lieu de l'horizon à 28 jours. Ne pas réutiliser tel quel.
