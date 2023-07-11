# Asynchronous n-step Q-Learning for computing shortest paths in graphs

## Setup

Utiliser un environnement virtuel est le meilleur moyen d'exécuter le présent code. Voici un exemple de procédure pour Linux :

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

### Entraînement

```shell
python nstepq.py
```

Les paramètres d'entraînement peuvent être modifiés à l'intérieur du script, dans la fonction `main()`.

## Test

```shell
python test.py
```

Vous pouvez choisir le modèle à tester en changeant la valeur de `timestamp` dans les paramètres du script. Le modèle `last_model` vous est offert afin d'être conforme aux résultats de l'article. 
