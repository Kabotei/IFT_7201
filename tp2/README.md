# Apprentissage par renforcement - TP2

Ceci constitue le dossier principal du code pour le TP2. SVP toujours effectuer vos commandes à partir de ce dossier.

## Setup

Il est fortement recommandé d'utiliser un environnement virtuel afin d'installer et exécuter le code afin d'éviter des problèmes de versionnage. Si les libraries ne sont pas installables sur votre poste de travail, svp vous fier aux messages d'erreur ou les installer une par une selon les libraries manquantes.

Pour créer un environnement virtuel, svp vous fiez à la [documentation Python](https://docs.python.org/3/library/venv.html).

## Question 1

Le meilleur modèle vous est offert. Il a été entraîné sur 500 épisodes. Il est à noté que l'apprentissage a stagnée après 400 épisodes avec un Gain de 260, mais que seul l'épisode 500 est fourni (en raison de l'espace de stockage des modèles sauvegardés).

### Paramètres

Les paramètres de l'entraînement se retrouvent dans le fichier `q1/types.py` dans la classe `TrainingParams` en temps que variables par défaut. Les valeurs en commentaires représentent les valeurs initiales proposées, alors que ceux inscritent actuellement représentent ceux du dernier essais (présent dans les résultats du rapport). Vous pouvez les modifier afin qu'ils soient pris en compte lors du prochain entraînement. 

### Experiments

Chaque entraînement est régulièrement sauvegardé, concept appelé `experiment` dans ce TP. Chaque expérience est créée avec un nom aléatoire et sauvegarde régulièrement l'état complet de l'entraînement (modèle, paramètres, etc.). Ainsi, une expérience peut être reprise ou testée à partir d'un episode sauvegardé. 

> Note: une fois l'expérience créée, les changements de valeurs de paramètres d'entraînement n'ont plus d'effet sur cette expérience. Seules les valeurs initiales seront prises en compte. Ainsi, pour que ces changements fassent effet, vous devrez créer une nouvelle expérience.

### Exécution

```shell
python -m q1.main {train,test} [-n {n_trajectories=300}] [-e {experiment=new one}] [--episode {checkpoint episode}=last saved episode] [--save (if on test mode to save video)]
```

**Exemple 1 : Créer une expérience sur 500 trajectoires**

```shell
python -m q1.main train -n 500
```

Le nom de la nouvelle expérience sera affiché dans le terminal.

**Exemple 2 : Reprendre une expérience en terminant les 500 trajectoires**

```shell
python -m q1.main train -n 500 -e experience_1
```

> Note : utilisez le NOM de l'expérience et non son path

**Exemple 3 : Forcer le réaffichage des derniers graphiques**

```shell
python -m q1.main train -n 0 -e experience_1
```

**Exemple 4 : Reprendre un entraînement à partir de l'épisode 60 jusqu'à 500 épisodes**

```shell
python -m q1.main train -n 500 -e experience_1 --episode 60
```

> Ceci évrasera toutes les sauvegardes des prochains checkpoint au fur et à mesure

**Exemple 5 : Tester 2 fois une expérience sur le dernier episode sauvegardé**

```shell
python -m q1.main test -n 2 -e experience_1
```

**Exemple 6 : Tester 2 fois une expérience sur l'épisode 60**

```shell
python -m q1.main test -n 2 -e experience_1 --episode 60
```

**Exemple 7 : Tester 2 fois une expérience sur le dernier episode sauvegardé et sauvegarder les vidéos**

```shell
python -m q1.main test -n 2 -e experience_1 --save
```

> Note : Les vidéos en live paraîtront plus lent qu'en réalité si le traitement graphique de votre ordinateur est peu performant.

## Question 2

### Exécution

```shell
python -m q2.main {single,compare,plot} [-n {max_trajectories=500}] [-l {lambda=0.9}] [-r {runs=30} (only for compare mode)]
```

**Exemple 1 : Effectuer un entraînement simple et rapide (optimal)**

```shell
python -m q2.main single
```

**Exemple 2 : Effectuer un entraînement sur maximum 500 trajectoires avec lambda = 0.5**

```shell
python -m q2.main single -n 500 -l 0.5
```

**Exemple 3 : Comparer différentes valeurs de lambda sur 500 trajectoires, moyenné sur 20 essais**

```shell
python -m q2.main compare -n 500 -r 20
```

> Note : chaque entraînement de comparaison sauvegarde les résultats dans le fichier `results`. Il sera donc écrasé à la fin de chaqeu comparaison en faveur des nouveaux résultats.

> Note : les valeurs de lambda à comparer sont modifiables dans le fichier `q2/main.py` dans la fonction `main()`

> Note : Cette commande écrasera les fichiers `fig_0.png` et `fig_1.png` en faveurs des nouveaux graphiques

**Exemple 4 : Réafficher les graphiques de la dernière comparaison**

```shell
python -m q2.main plot
```

> Note : Cette commande écrasera les fichiers `fig_0.png` et `fig_1.png` en faveurs des nouveaux graphiques
