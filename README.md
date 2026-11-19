# RSI Béton — interface Streamlit + Plotly + PyVista

Première version web du calculateur de thermochimie, mécanique jeune âge et potentiel de RSI, **issue directement de `FINAL_corrige_etape1.py`**.

## Utiliser l'application intégralement dans un navigateur (sans rien installer)

1. Créer ou utiliser un compte **GitHub** : https://github.com/.
2. Créer un **nouveau dépôt** GitHub (public pour des exemples non confidentiels, ou privé si nécessaire). Dans le navigateur : **Add file → Upload files**, puis envoyer **les fichiers et dossiers extraits de l'archive ZIP** (`app.py`, `physics.py`, `requirements.txt`, `packages.txt`, `.streamlit/`, etc.). Le fichier `app.py` doit être à la racine du dépôt.
3. Aller sur **https://share.streamlit.io/**, se connecter avec GitHub, puis choisir **Create app** / **New app**.
4. Sélectionner le dépôt, la branche par défaut et **Main file path = `app.py`**. Dans les paramètres avancés, choisir si possible **Python 3.12**, pour conserver un environnement stable avec les dépendances scientifiques.
5. Cliquer sur **Deploy**. La plateforme installe automatiquement les dépendances Python de `requirements.txt` et les dépendances Linux éventuelles de `packages.txt`. L'interface s'ouvre via une URL `*.streamlit.app`.

Les services cloud ont des limites de mémoire et de CPU ; ce prototype limite donc volontairement les maillages à **1 600 cellules en 2D et 216 en 3D**. Les calculs plus lourds seront à traiter après l'étape d'optimisation des performances.

Le rendu 3D est assuré par **PyVista + stpyvista**. Si le moteur graphique du serveur ne prend pas en charge le rendu VTK, l'application affiche automatiquement des coupes tridimensionnelles **Plotly**, sans interrompre le calcul ni les autres graphiques. Le fichier `.vti` est exporté si PyVista fonctionne.

**Confidentialité :** ne jamais déposer de données confidentielles dans un dépôt public ; même avec un dépôt privé, vérifier les conditions de sécurité et d'accès du service cloud choisi avant de charger des données sensibles.

## Utilisation locale (Windows, macOS, Linux)

Installer Python **3.12** sur https://www.python.org/downloads/ si nécessaire, puis dans le dossier décompressé :

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Sous Windows, selon le terminal, remplacer `python` par `py` si nécessaire. Le navigateur ouvre normalement `http://localhost:8501/`.

`packages.txt` sert uniquement à l'installation des bibliothèques système sur les serveurs Linux, et n'est pas une commande à lancer sur Windows.

## Fonctionnalités de cette version

- Saisie des paramètres de calcul : dimension 2D ou 3D, géométrie, pas de maillage, caractéristiques thermomécaniques, températures, décoffrage, durée et conditions sur chaque face.
- Lancement du **moteur numérique corrigé** sans ouvrir Tkinter ni Matplotlib.
- Tableau de bord : pic thermique, délai du pic, potentiel RSI, pas de temps, courbes Tmax/Tcentre, E, résistance et ε∞.
- Cartes 2D Plotly avec axes métriques, palettes adaptées, zoom et survol.
- Résultats 3D sur maillage volumique PyVista et coupes orthogonales, avec visualisation Plotly de secours.
- Export ZIP : paramètres au format JSON (unités SI), historiques CSV, tableaux NPZ, métadonnées et maillage VTK/VTI lorsqu'il est disponible.

### Référence exacte des champs

La température est stockée **aux centres des cellules** ; les déformations RSI et déplacements sont **aux nœuds** ; les contraintes sont **aux centres des éléments**. Les tableaux sont orientés `(y,x)` en 2D et `(z,y,x)` en 3D. Le nom, l'unité, la position et la clé NPZ (`f00`, `f01`, etc.) de chaque champ sont inscrits dans `champs_metadata.json`.

La carte « Température au pic thermique » correspond **à la date du pic global**, et non à un maximum temporel local de chaque maille. Les cartes de déplacement/contrainte jeune âge correspondent à différents instants où la composante enregistrée atteignait son extrême : **ne pas les interpréter comme un seul état mécanique simultané**.

### Hypothèses conservées

Ce prototype **n'ajoute pas de nouveau modèle de RSI**. Il conserve la loi empirique de gonflement ultime, la cinétique d'hydratation, les propriétés de matériau et les hypothèses mécaniques de l'étape 1.

- La RSI affichée est un **potentiel de gonflement ultime**, pas une cinétique pluriannuelle ni une probabilité de dommage.
- Les conditions de face restent **couplées entre thermique et mécanique** selon le sens attribué à l'option dans le programme source.
- Avant décoffrage, une face coffrée peut être modélisée mécaniquement bloquée ; cette hypothèse peut surestimer des contraintes.
- L'arrêt anticipé selon un seuil de température n'intervient que **4 jours après le début** dans le code existant, sauf atteinte de la durée maximale.
- La géométrie est restreinte à une grille cartésienne rectangulaire, avec des dimensions qui doivent être un nombre entier de mailles. Le formulaire empêche les géométries trop petites ou ne respectant pas cette contrainte.
- Les résultats ne remplacent pas une validation par comparaison à un logiciel de référence, ni un dimensionnement réglementaire.

## Architecture

```text
app.py                         # interface Streamlit, Plotly, PyVista, exports
physics.py                     # solveur issu de FINAL_corrige_etape1.py
requirements.txt               # dépendances Python (cloud et local)
packages.txt                   # dépendances du rendu VTK sous Linux
.streamlit/config.toml         # palette et configuration de l'application
tests/test_physics.py          # tests 2D, 3D, maillage, données
reference/FINAL_corrige_etape1.py  # source conservée pour audit
```

Le moteur est indépendant du navigateur et peut être appelé en Python via `from physics import run_simulation`. Il ne crée aucun objet Tkinter et n'utilise pas de variables globales partagées entre utilisateurs.

## Vérifications et limites de validation

Tester le calcul sur des cas simples :

```bash
python -m unittest discover -s tests -v
```

Les **4 tests du moteur** ont été exécutés dans l'environnement de préparation et réussissent. Une comparaison numérique directe avec `FINAL_corrige_etape1.py` a été faite en 2D et 3D avec égalité des champs de température au pic, de déformation RSI et de contraintes RSI pour les cas courts testés.

**Limite :** l'environnement de préparation n'a pas pu installer les paquets Streamlit/PyVista (accès réseau indisponible). La compilation Python de l'interface a réussi, mais **son rendu Streamlit complet, l'intégration stpyvista et le déploiement Community Cloud n'ont pas été exécutés sur un véritable serveur**. L'affichage Plotly de secours limite les conséquences d'une incompatibilité VTK, mais un premier essai sur Community Cloud est nécessaire.

---

**Évolutions prévues ensuite :** validation graphique, amélioration de l'éditeur de géométrie, profils thermiques et sondes, comparaison des scénarios, puis optimisation du solveur et nouveaux indicateurs physiques.
