# Expériences officielles de rebuttal NeurIPS 2026 — Résultats

Générés par `rebuttal_official_experiments.py` (python3.12, CPU, seeds fixés).
Fichiers bruts et résumés au format CSV dans ce dossier. Le journal complet de
l'exécution est dans `full_run.log`. Chaque expérience est indexée sur les items
de `../revision_roadmap_official.md`.

---

## Correction préalable du code (importante)

Le chemin d'entraînement MLP de `CTL/trainer.py` contenait un bug silencieux.
`SimpleMLP` et `ResidualMLP` renvoient une sortie de forme `(n,)` alors que
`train_model_sgd` remettait la cible en forme `(n,1)`. `MSELoss` diffusait alors
sur `(n,n)` paires, ce qui corrompait le gradient et faisait converger tout MLP
vers la moyenne globale (RMSE ≈ 2100 sur un nœud WEAVE quel que soit le nombre
de pas, contre ≈ 1350 après correction). Le chemin linéaire n'était pas affecté
car `LinearModel` renvoie `(n,1)`. La correction aligne les formes avant la
perte (`loss = criterion(pred.reshape(-1), y_t.reshape(-1))`), sans effet sur
les résultats linéaires publiés. Ce bug explique vraisemblablement pourquoi
aucune expérience MLP antérieure n'était concluante.

---

## EXP-A1 — CTL non convexe de bout en bout sur WEAVE-UK (item 1, les 3 rapporteurs)

Un `SimpleMLP` (d → 32 → ReLU → 1) est entraîné de bout en bout à chaque nœud,
tous poids libres, aucun backbone gelé. Même protocole que la Table 6 du papier
(distance gradient, MST médoïde, allocation SLE, T = 500 sous-stations, 3
répétitions). RMSE moyen en Wh.

| B | η | Individual-MLP | Star-MLP | CTL-MST-MLP | CTL-MST-Linear | Réduction CTL vs Indiv |
|---|---|---|---|---|---|---|
| 500 | 0.05 | 2168 | 2008 | **1195** | 1133 | 44.9 % |
| 1000 | 0.05 | 2072 | 1846 | **1158** | 1132 | 44.1 % |
| 2000 | 0.05 | 1887 | 1631 | **1151** | 1134 | 39.0 % |
| 1000 | 0.2 | 1616 | 1444 | **1154** | 1378 | 28.6 % |

Constats. Les gains prédits survivent au régime non convexe à toutes les
combinaisons (B, η) testées (réductions de 15.6 % à 44.9 %, 9 combinaisons,
CTL premier partout). Le cascade MLP atteint le niveau du cascade linéaire
(1151 vs 1134 au meilleur budget), et le dépasse nettement à η = 0.2
(1154 vs 1378), régime où le modèle linéaire se dégrade. L'ordre
Individual > Star > CTL est préservé partout.

Fichiers `expA1_nonconvex_weave_raw.csv`, `expA1_nonconvex_weave_summary.csv`.

---

## EXP-A2 — Tâches synthétiques authentiquement non linéaires (items 1 et 10)

Générateur à enseignants MLP à deux couches (d = 5, 16 unités cachées, 5
clusters de poids, T = 100, n_train = 32, B = 500, 10 graines). L'élève est un
`SimpleMLP` à 32 unités entraîné de bout en bout. Le processus générateur est
non linéaire, l'optimisation est non convexe, et la donnée ne peut pas être
ajustée par un modèle linéaire.

| Méthode | RMSE moyen | Écart-type |
|---|---|---|
| Individual-Linear | 9.97 | 1.47 |
| Individual-MLP | 8.99 | 1.38 |
| Star-MLP | 8.14 | 1.22 |
| CTL-RandomTree-MLP | 8.60 | 1.16 |
| CTL-MST-MLP (proxy gradient) | **4.50** | 0.76 |
| CTL-MST-MLP (distance oracle enseignants) | 4.24 | 0.62 |

Constats. La cascade divise l'erreur par deux en régime non convexe
(réduction de 50.0 % vs Individual-MLP, t apparié = 15.65, p = 7.8e-8).
Le proxy gradient, pourtant défini à partir du modèle linéaire
(g_v = X_vᵀ y_v), recueille presque tout le gain de la distance oracle entre
poids des enseignants (4.50 vs 4.24, écart de 6 %), ce qui répond à n7Am W3
sur la robustesse du proxy hors linéarité. Un arbre aléatoire annule le gain
(8.60), preuve que la géométrie des tâches, non la structure en cascade seule,
porte l'amélioration.

Fichiers `expA2_nonconvex_synth_raw.csv`, `expA2_nonconvex_synth_summary.csv`.

---

## EXP-B — Frontière CTL vs TL direct et looseness de la borne (item 3)

Géométrie contrôlée reproduisant la comptabilité du théorème 2.3. Optimum
source en 0, optimum cible à distance ρ le long d'une direction u, k
intermédiaires équirépartis sur le segment et déplacés orthogonalement de γ·ρ.
Chaque nœud reçoit le même budget b (les intermédiaires sont des tâches à
résoudre de toute façon). Le TL direct raffine la cible b pas depuis θ*_s, CTL
raffine b pas depuis le parent cascadé. Le mode `cond_het` donne aux
intermédiaires des données mal conditionnées (n = d + 4) et à la cible des
données bien conditionnées (n = 4d), produisant ρ_v ≪ ρ_max. Taux de
contraction exacts ρ_v = ‖I − (η/n) X_vᵀX_v‖₂. 20 graines, d = 20,
b ∈ {5, 20, 80}, ρ ∈ {0.5, …, 8}, γ ∈ {0, …, 2}, k ∈ {1, 3, 7}.

Fraction de configurations où CTL bat le TL direct (b = 20).

| γ | homogène | ρ_v ≪ ρ_max |
|---|---|---|
| 0 | 0.95 | 0.71 |
| 0.25 | 0.90 | 0.70 |
| 0.5 | 0.79 | 0.65 |
| 1 | 0.49 | 0.52 |
| 2 | 0.18 | 0.33 |

Par ρ (homogène, b = 20, toutes valeurs de γ confondues) : 0.57 à ρ = 0.5
(source et cible déjà similaires), 0.67 à 0.72 pour ρ ∈ [1, 8].

Looseness des bornes (médiane du rapport borne / erreur réalisée) : 12.9 pour
la borne cascade (ρ_max, δ_max), 2.1 pour la borne directe (ρ_v). L'accord
entre le classement des bornes relâchées et le vainqueur empirique vaut 34 % à
54 % selon (b, cond_het), proche du hasard.

Constats. Les trois régimes de défaillance nommés par Lufy sont confirmés et
quantifiés. La borne relâchée du théorème 2.3 est une condition suffisante
qualitative, environ six fois plus lâche côté cascade que côté direct, ce qui
confirme empiriquement la concession de la remarque 2.4. La carte empirique
fournit la règle pratique demandée : CTL domine dès que les intermédiaires
restent proches du chemin géodésique (γ ≤ 0.5) et que la cible n'est pas déjà
voisine de la source.

Fichiers `expB_boundary_raw.csv`, `expB_boundary_summary.csv`,
`expB_bound_agreement.csv`.

---

## EXP-C — Sensibilité η et τ (item 2)

Balayage η sur WEAVE-UK (T = 500, B = 1000, 3 répétitions) et synthétique
(T = 100, B = 500, 10 graines), modèle linéaire.

| η | WEAVE Individual | WEAVE CTL-MST | Synthétique Individual | Synthétique CTL-MST |
|---|---|---|---|---|
| 0.001 | 2520 | 1700 | 20.2 | 18.6 |
| 0.003 | 2487 | 1286 | 19.9 | 16.2 |
| 0.01 | 2401 | **1163** | 19.4 | 11.1 |
| 0.03 | 2243 | **1133** | 17.7 | 5.9 |
| 0.1 | 1729 | **1188** | 12.1 | 3.3 |
| 0.3 | 1858 | 1632 | 2.4 | 1.7 |

Constats. CTL-MST bat Individual à chaque η sur les deux jeux, le classement
des méthodes est donc invariant au réglage. Sur WEAVE le plateau
η ∈ [0.01, 0.1] tient dans 5 % du RMSE. L'optimum exact diffère entre jeux de
données comme l'anticipe la théorie (η dépend de la courbure locale via
ρ_v = ‖I − ηX_vᵀX_v‖), ce qui explique les valeurs distinctes rapportées pour
FMNIST et CIFAR-10 sans réglage fin par jeu.

Balayage τ pour CTL-MSTc (WEAVE, B = 1000, λ = 0.75, τ = c × médiane des
distances) : RMSE de 1130 (c = 4) à 1154 (c = 0.25), amplitude 2.2 %. Combiné
au balayage λ de mai (EXP4, plateau λ ∈ [0.1, 5] sous 1 %), les trois
hyperparamètres de CTL-MSTc sont sur des plateaux larges.

Fichiers `expC_eta_raw.csv`, `expC_eta_summary.csv`, `expC_tau_raw.csv`,
`expC_tau_summary.csv`.

---

## EXP-D — Sensibilité au choix de racine (item 6)

MST gradient fixé, la racine varie : médoïde (défaut du papier), centre de
profondeur (excentricité minimale), feuille d'excentricité maximale, 5 racines
aléatoires. WEAVE B = 1000 (3 répétitions), synthétique T = 100 (10 graines).

| Racine | WEAVE-UK | Synthétique |
|---|---|---|
| médoïde | 1165 | 11.31 |
| centre de profondeur | 1187 | **10.58** |
| aléatoire (moy.) | 1152 | 11.24 |
| feuille excentrée | **1136** | 11.07 |

Constats. L'amplitude totale entre le meilleur et le pire choix vaut 4.5 % sur
WEAVE et 6.9 % sur le synthétique, à comparer à l'écart de ~50 % entre CTL et
Individual. Aucune règle d'enracinement ne domine sur les deux jeux (le centre
de profondeur gagne sur le synthétique, conformément à l'appendice H, et perd
sur WEAVE). Le choix de racine est donc un effet de second ordre. Sur le
schéma itératif demandé par fdRt, le MST non orienté est indépendant de la
racine, donc le ré-enracinement au centre de profondeur après construction est
exact et gratuit (un parcours BFS, O(T) par nœud testé), sans étape
approximative.

Fichiers `expD_root_raw.csv`, `expD_root_summary.csv`.

---

## EXP-E — Perturbation de la solution source (item 13)

Cascade CTL-MST synthétique (T = 100, B = 500, 10 graines), bruit gaussien
relatif d'amplitude σ_p·‖θ̂_racine‖ injecté dans la racine après son
raffinement.

| σ_p | 0 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| RMSE moyen | 11.44 | 11.44 | 11.46 | 11.47 | 11.62 | 11.71 |

Constats. Une perturbation de 20 % dégrade le RMSE de 0.32 %, une perturbation
de 100 % de 2.42 %. Les raffinements aval contractent l'erreur injectée en
racine, exactement le mécanisme que la théorie décrit. L'hypothèse
θ̂_s ≈ θ*_s est donc une commodité d'analyse et non une condition de validité
pratique.

Fichiers `expE_perturbation_raw.csv`, `expE_perturbation_summary.csv`.

---

## EXP-F — Sonde d'hétérogénéité inter-clusters (item 4)

Cinq clusters de tâches progressivement découplés (τ_between de 2 à 64,
τ_within = 2, T = 100, 10 graines). Mesure du gain CTL vs Individual et de la
fraction d'arêtes du MST traversant une frontière de cluster.

| τ_between | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Gain CTL (%) | 34.1 | 40.3 | 43.4 | 47.0 | 48.0 | 53.0 |
| Arêtes inter-clusters | 0.45 | 0.19 | 0.08 | 0.06 | 0.05 | 0.05 |

Constats. Face à des clusters partiellement incompatibles, le MST fondé sur la
distance gradient isole les clusters de lui-même (la fraction d'arêtes
transversales tombe à 5 %, proche du minimum de 4/99 imposé par la
connexité) et le gain ne s'effondre jamais, aucun transfert négatif n'est
observé. La cascade se comporte comme une forêt de cascades quasi
indépendantes, ce qui délimite empiriquement la frontière de l'hypothèse
d'homogénéité à paramétrisation partagée constante.

Fichiers `expF_heterogeneity_raw.csv`, `expF_heterogeneity_summary.csv`.

---

## Index des fichiers

| Fichier | Contenu |
|---|---|
| `rebuttal_official_experiments.py` | Script complet (EXP-A1 à EXP-F) |
| `full_run.log` | Journal de l'exécution complète |
| `expA1_*` | Non convexe WEAVE (raw, summary) |
| `expA2_*` | Non convexe synthétique (raw, summary) |
| `expB_*` | Frontière CTL vs TL, accord des bornes |
| `expC_*` | Sensibilité η et τ |
| `expD_*` | Sensibilité racine |
| `expE_*` | Perturbation source |
| `expF_*` | Hétérogénéité |
