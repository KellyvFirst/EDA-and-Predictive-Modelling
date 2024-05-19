# EDA and Predictive Modelling

Ce projet est une application de machine learning qui utilise plusieurs modèles de classification pour faire des prédictions sur le jeu de données Iris. Il utilise également des techniques d'analyse exploratoire des données (EDA) pour visualiser et comprendre les données.

## Structure du projet

Le projet est organisé en plusieurs dossiers et fichiers :

- Le dossier principal "Dashboard" contient des fichiers eda_v1.py et app.py ne sont que des versions précédentes du projet sans utilisation de rest Api.
- Il comprend les sous-dossiers: "Client_Streamlit","ML_API"
- Le sous-dossier "Client_Streamlit" contient le fichier "eda.py" et un autre fichier "rest_client.py".
- Le dossier "ML_API" contient plusieurs fichiers pickle qui sont des modèles de machine learning pré-entraînés et des transformateurs de données, ainsi que le fichier rest_Api.py qui ser d'API REST.

## Fonctionnement du projet

L'utilisateur interagit avec l'application Streamlit, qui est définie dans `rest_client.py`. L'utilisateur peut choisir de réaliser une analyse exploratoire des données (EDA), qui est effectuée par la fonction `eda` du fichier `eda.py`. L'utilisateur peut également choisir une méthode de transformation des données et un modèle de machine learning. Ces choix sont envoyés à l'API REST définie dans `rest_Api.py` dans le dossier `ML_API`.

L'API REST reçoit la requête, utilise les modèles de machine learning pré-entraînés pour faire une prédiction, et renvoie la prédiction au client. Les résultats de la prédiction sont ensuite affichés à l'utilisateur via l'interface Streamlit.

## Comment exécuter le projet

1. Clonez ce dépôt sur votre machine locale.
2. Naviguez vers le dossier du dépôt dans un terminal.
3. Installez les dépendances nécessaires avec la commande `pip install -r requirements.txt`.
4. Exécutez premierement l'API REST avec la commande `python rest_Api.py` ensuite lancez l'application streamlit avec la commande `streamlit run rest_client.py`.
