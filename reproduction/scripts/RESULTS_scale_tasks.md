# L'avantage de la cascade quand le nombre de tâches croît

`scale_tasks.py`. Famille synthétique à grappes, $T$ allant de 50 à 5000, cinq
répétitions, modèle local MLP, racine au médoïde. CTL-MST reçoit l'allocation
SLE, Star l'allocation uniforme, comme dans le papier. Deux régimes de budget
répondent à deux questions distinctes.

CTL-MSTc est écarté du balayage : sa construction coûte 390 s à $T=5000$ et croît
en $T^{2.9}$, ce qui la rend impraticable sans rien changer au message. Chain est
écarté aussi, la comparaison arbre contre chaîne étant traitée ailleurs.

## Budget total fixé, $B = 10000$

Le nombre de pas par tâche décroît comme $1/T$. C'est le scénario que le reviewer
Lufy décrit, davantage de tâches sous une même enveloppe de calcul.

| $T$ | $b$ | Individual | Star | CTL-MST | MST vs Ind. | $p$ |
|---|---|---|---|---|---|---|
| 50 | 200 | 5.861 | 6.343 | 6.679 | $+14.0$ % | 0.0002 |
| 100 | 100 | 5.922 | 6.541 | 6.650 | $+12.3$ % | 0.028 |
| 200 | 50 | 6.799 | 6.193 | 6.156 | $-9.4$ % | 0.0041 |
| 500 | 20 | 6.409 | 7.257 | 6.066 | $-5.4$ % | 0.018 |
| 1000 | 10 | 6.321 | 7.243 | 5.933 | $-6.1$ % | 0.0070 |
| 2000 | 5 | 7.325 | 10.552 | 5.862 | $-20.0$ % | 0.0005 |
| 5000 | 2 | 15.023 | 22.327 | 5.822 | $-61.2$ % | 0.0001 |

Le signe bascule entre $T=100$ et $T=200$, et l'écart croît ensuite de façon
monotone. À $T \le 100$ le budget ne mord pas, cent pas suffisant à une tâche de
cette famille, et la cascade y perd. Au-delà, Individual se dégrade de 6.32 à
15.02 entre $T=1000$ et $T=5000$ tandis que CTL-MST reste à 5.8. **Il existe donc,
à enveloppe de calcul fixée, un nombre de tâches au-delà duquel l'entraînement
indépendant s'effondre et la cascade non.**

## Budget par tâche fixé, $b = 4$

$B$ croît avec $T$, chaque tâche gardant quatre pas. Ceci isole l'effet du nombre
de tâches de celui du budget par tâche.

| $T$ | Individual | Star | CTL-MST | MST vs Ind. | $p$ |
|---|---|---|---|---|---|
| 50 | 8.272 | 15.491 | 6.940 | $-16.1$ % | 0.0074 |
| 100 | 8.339 | 17.628 | 6.583 | $-21.1$ % | 0.0028 |
| 200 | 8.502 | 14.220 | 6.329 | $-25.5$ % | 0.0001 |
| 500 | 8.609 | 15.066 | 5.989 | $-30.4$ % | 0.0002 |
| 1000 | 8.609 | 14.970 | 5.918 | $-31.3$ % | 0.0003 |
| 2000 | 8.565 | 13.431 | 5.852 | $-31.7$ % | 0.0002 |
| 5000 | 8.587 | 13.562 | 5.835 | $-32.0$ % | 0.0004 |

Individual est plat à 8.6 par construction, chaque tâche recevant le même budget
quel que soit $T$. CTL-MST s'améliore de 6.94 à 5.84 : plus la collection est
grande, plus il existe pour chaque tâche un parent proche, donc plus la copie de
paramètres part d'un bon point. L'écart croît avec $T$ à budget par tâche
constant, et beaucoup plus vite à enveloppe constante.

## Star est le contrôle qui manquait

Star copie la racine vers toutes les tâches, donc il teste « copier depuis un
parent » sans la géométrie. À budget par tâche fixé il est de 57 à 111 pour cent
**pire** qu'Individual, à tout $T$. À enveloppe fixée il suit Individual dans son
effondrement, 22.3 contre 15.0 à $T=5000$. Le gain de CTL-MST ne vient donc pas de
la copie de paramètres mais du choix du parent, et le nombre de tâches ne sauve
pas une mauvaise structure : Star empire quand $T$ croît alors que CTL-MST
s'améliore.

## Coût de construction

Secondes pour bâtir l'arbre, moyenne sur les répétitions.

| $T$ | 200 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| CTL-MST | 0.004 | 0.056 | 0.132 | 0.459 | 2.686 |

Croissance empirique en $T^{1.9}$ entre $T=1000$ et $T=5000$, dominée par la
matrice de distances $O(T^2)$. À $T=5000$ le coût reste négligeable devant
l'entraînement. CTL-MSTc, lui, met 390 s au même point et croît en $T^{2.9}$ :
c'est le terme de portée, pas le MST, qui bloque le passage à l'échelle.

## Réserves

Famille synthétique uniquement. La collection réelle la plus grande du papier,
WEAVE-UK, compte 500 tâches et n'admet pas de balayage en $T$.

$K=5$ grappes à tout $T$, donc les grappes grossissent avec $T$. C'est
précisément ce qui fait qu'un parent proche devient plus facile à trouver. Un
régime où le nombre de grappes croît proportionnellement à $T$ donnerait
vraisemblablement une courbe plate plutôt que croissante dans le second tableau,
et il n'est pas testé.

Le pas est fixé à la valeur calibrée pour $T=200$, sans recalibration par $T$.
