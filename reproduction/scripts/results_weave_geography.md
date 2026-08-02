# Arbre géographique sur WEAVE-UK (Lufy W4)

Généré par `weave_geography.py`. Teste la géométrie de tâche apprise (proxy
gradient) contre un prior de domaine informé, un arbre couvrant construit à
partir des coordonnées physiques des postes.

## Données géographiques

Les coordonnées viennent du parquet WEAVE `s3://weave.energy/smart-meter`,
colonne `geometry` (struct x, y en longitude/latitude) par `lv_feeder_unique_id`,
qui est l'identifiant `id_unique` des tâches, exactement la source de
`generate_data.ipynb`. Une seule partition mensuelle (2024-02) suffit, la
géométrie étant statique par poste. Les 500 postes sont appariés sans exception,
tous dans le bbox d'Oxford (lon -1.30 à -1.19, lat 51.71 à 51.79). Conversion en
mètres locaux par projection équirectangulaire autour de la latitude moyenne,
erreur relative bien sous le mètre sur une zone de dix kilomètres. Cache dans
`weave_coords.csv`.

## Protocole

Geo-MST reprend exactement la cascade fidèle de `table1_nonconvex.py`. Seuls la
matrice de distance et la racine changent, distance euclidienne géographique et
racine au médoïde géographique, même construction MST et même allocation SLE. Les
méthodes gradient (Individual, Star, CTL-MST, CTL-MSTc) gardent leur racine
d'origine, le médoïde de features brutes de la Table 1. Linéaire et MLP, budgets
500, 1000, 2000, 20 répétitions.

## Résultat : RMSE moyen (20 répétitions)

| Régime | B | Individual | Star | Geo-MST | CTL-MST | CTL-MSTc |
|---|---|---|---|---|---|---|
| linéaire | 500 | 2520 | 2094 | 1521 | 1456 | **1350** |
| linéaire | 1000 | 2521 | 1719 | 1330 | 1294 | **1227** |
| linéaire | 2000 | 2492 | 1401 | 1215 | 1212 | **1163** |
| MLP | 500 | 1901 | 1317 | 1137 | 1167 | **1117** |
| MLP | 1000 | 1617 | 1340 | 1147 | 1179 | **1114** |
| MLP | 2000 | 1392 | 1354 | 1168 | 1202 | **1121** |

## Significativité (t apparié sur les 20 graines)

| Régime | B | CTL-MSTc vs Geo-MST | Geo-MST vs CTL-MST |
|---|---|---|---|
| linéaire | 500 | p=2.7e-5 | p=0.12 (n.s.) |
| linéaire | 1000 | p=4.3e-9 | p=5.5e-3 (Geo pire) |
| linéaire | 2000 | p=1.4e-10 | p=0.56 (n.s.) |
| MLP | 500 | p=1.8e-16 | p=2.6e-16 (Geo meilleur) |
| MLP | 1000 | p=2.6e-22 | p=4.2e-15 (Geo meilleur) |
| MLP | 2000 | p=5.2e-22 | p=1.0e-15 (Geo meilleur) |

## Lecture

La géographie est une baseline forte, bien plus dure que les arbres aléatoires ou
en chaîne. Elle bat largement Star et Individual, égale le MST du proxy gradient
en linéaire (1215 contre 1212 à B=2000, non significatif) et le dépasse en MLP
(1168 contre 1202 à B=2000, p=1e-15). La proximité physique est donc un vrai prior
de similarité de tâche pour la prévision de charge. Cependant CTL-MSTc, la
construction complète avec terme de couverture, bat le prior spatial à tous les
budgets et dans les deux régimes, avec une réduction significative partout
(p de 2.7e-5 à 5.2e-22). La géométrie apprise sensible à la couverture apporte
donc une valeur au-delà à la fois du prior spatial informé et du MST gradient nu.

## Ablation structure contre allocation (plan 2x2)

Généré par `weave_geo_ablation.py`. La bonne tenue de Geo-MST peut venir de deux
sources confondues, la structure d'arbre qui suit la proximité géographique ou
l'allocation SLE que tout arbre reçoit. On croise structure {MST géographique,
arbre aléatoire} et allocation {SLE, uniforme}, même racine au médoïde
géographique, mêmes distances géographiques pour le SLE, seule change l'adéquation
des arêtes à la géographie. RMSE moyen, 20 répétitions.

| Régime | B | geo+SLE | geo+unif | rand+SLE | rand+unif |
|---|---|---|---|---|---|
| linéaire | 500 | 1526 | 1832 | 1462 | 2004 |
| linéaire | 2000 | 1221 | 1382 | 1214 | 1534 |
| MLP | 500 | 1137 | 1162 | 1210 | 1207 |
| MLP | 2000 | 1168 | 1148 | 1294 | 1185 |

La dissociation est nette et dépend du régime.

En **linéaire**, l'allocation domine. À structure fixée, geo+SLE égale rand+SLE
(1221 contre 1214 à B=2000, p=0.20 n.s.), la géographie n'apporte rien une fois le
SLE en place. À structure fixée, SLE bat massivement l'uniforme (rand 1214 contre
1534, p=8e-9). La bonne tenue de l'arbre géographique en linéaire vient donc du
SLE, pas du prior spatial.

En **MLP**, la structure domine. À allocation fixée, geo bat rand fortement
(1168 contre 1294 à B=2000, p=6e-15), et geo+uniforme (1148) bat même rand+SLE
(1294). L'allocation, elle, n'aide plus, l'uniforme égale ou dépasse le SLE. Le
prior spatial porte donc réellement le gain non convexe.

Implication pour le papier. La Table 1 publiée compare CTL-MST (SLE) à RandTree
(uniforme), ce qui confond structure et allocation. En linéaire, un arbre
aléatoire avec SLE rivalise avec l'arbre géométrique, l'avantage y est donc en
grande partie un effet d'allocation. La géométrie apprise ne se démarque
clairement de l'aléatoire que dans le régime non convexe. Une vérification directe
serait CTL-MST (proxy gradient) + SLE contre RandTree + SLE.

## Ablation sur le proxy gradient (claim centrale du papier)

Généré par `weave_grad_ablation.py`. Le même plan 2x2 mais sur la géométrie apprise
du papier, le proxy gradient, croisant {MST gradient, arbre aléatoire} x {SLE,
uniforme}, raciné au médoïde de features brutes de la Table 1. gradMST+SLE est
exactement le CTL-MST publié et reproduit sa valeur (1212 à B=2000 linéaire), ce
qui valide le protocole. RMSE moyen, 20 répétitions.

| Régime | B | gradMST+SLE | gradMST+unif | rand+SLE | rand+unif |
|---|---|---|---|---|---|
| linéaire | 500 | 1467 | 1822 | 1472 | 1983 |
| linéaire | 1000 | 1300 | 1546 | 1295 | 1658 |
| linéaire | 2000 | 1212 | 1341 | 1213 | 1384 |
| MLP | 500 | 1167 | 1198 | 1209 | 1234 |
| MLP | 1000 | 1177 | 1198 | 1236 | 1248 |
| MLP | 2000 | 1202 | 1162 | 1274 | 1194 |

Structure à allocation fixée (gradMST+SLE contre rand+SLE), test apparié.

| Régime | B | gradMST+SLE | rand+SLE | p | verdict |
|---|---|---|---|---|---|
| linéaire | 500 | 1467 | 1472 | 0.89 | égalité |
| linéaire | 1000 | 1300 | 1295 | 0.77 | égalité |
| linéaire | 2000 | 1212 | 1213 | 0.85 | égalité |
| MLP | 500 | 1167 | 1209 | 1.3e-7 | gradMST meilleur |
| MLP | 1000 | 1177 | 1236 | 4.0e-11 | gradMST meilleur |
| MLP | 2000 | 1202 | 1274 | 6.1e-12 | gradMST meilleur |

Conclusion nette. En **linéaire**, la géométrie apprise du gradient n'apporte
rien sur un arbre aléatoire une fois le SLE fixé, l'égalité est parfaite à tous
les budgets. Tout l'avantage de CTL-MST sur RandTree dans la Table 1 publiée est
un effet d'allocation, SLE contre uniforme, pas de géométrie. En **MLP**, la
géométrie du gradient bat l'aléatoire significativement partout, elle porte alors
un gain réel. La géométrie apprise ne se démarque donc que dans le régime non
convexe, exactement le régime vers lequel les rapporteurs poussent.

## Fichiers

| Fichier | Contenu |
|---|---|
| `weave_geography.py` | Arbre géo vs gradient, linéaire + MLP |
| `weave_geo_ablation.py` | Plan 2x2 géographie x allocation |
| `weave_grad_ablation.py` | Plan 2x2 proxy gradient x allocation (claim centrale) |
| `weave_geo_raw.csv`, `weave_geo_summary.csv` | Géo vs gradient (brut, résumé) |
| `weave_geo_ablation_raw.csv`, `weave_geo_ablation_summary.csv` | Ablation 2x2 |
| `weave_coords.csv` | Coordonnées des 500 postes (lon, lat, mètres locaux) |
| `weave_geo_run.log`, `weave_geo_ablation_run.log` | Journaux d'exécution |

## Exécution consolidée WEAVE (weave_all.py, weave_chain_add.py)

Tous les nombres WEAVE du rebuttal proviennent d'une exécution unique
(`weave_all_raw.csv`, 20 répétitions, un seul seeding) couvrant Individual, Star,
RIT, Geo-MST, CTL-MST, CTL-MSTc, puis Chain ajouté par `weave_chain_add.py` sous
le même protocole. Statistique du rebuttal, t apparié unilatéral sur graines
(chaque graine fixe le même tirage de tâches pour toutes les méthodes), Holm sur
les dix cellules affichées, affichage plafonné à 1e-4, Wilcoxon signé donnant le
même verdict sur les dix cellules (neuf significatives, Syn B=2000 n.s.).
