# Librairie numérique du cours MTH2210 de Polytechnique Montréal

Ce site web contient la documentation de la librairie numérique MTH2210 de Polytechnique Montréal. 

## Mise en place

Voici les étapes à suivre afin d'obtenir une installation minimale vous permettant de réaliser les laboratoires:

### Étape 1 - Installation d'Anaconda

1. Installez Anaconda en suivant les étapes sur le site web [https://www.anaconda.com/docs/getting-started/installation](https://www.anaconda.com/docs/getting-started/installation)

### Étape 2 - Création de l'environnement "mth2210"
1. Lancez le programme Anaconda Navigator
2. Cliquez sur l'onglet 'Environments' puis créez un nouvel environnement en cliquant sur le bouton 'Create'. Nommez cet environnement 'mth2210' et choissisez le 'Packages' Python version 3.14.
3. Retournez sur l'onglet 'Home' d'Anaconda Navigator. Une liste défilante des environnments est visible vers le haut de la fenêtre. Cliquez sur cette liste et choisissez l'environnement 'mth2210' précédemment créé. Il est bien important de ne pas choisir l'environnement 'base (root)', car on ne peut garantir qu'il n'y aura pas de conflits entre les différentes librairies.
4. Cherchez l'application Jupyter Notebook et cliquez sur le bouton 'Install'. 

### Installation de la librairie MTH2210.py

1. Créez un dossier sous votre machine

## Réalisation des laboratoires

1. Lancez le programme Anaconda Navigator
2. Assurez vous que l'environnement 'mth2210' est sélectionné et lancez l'application Jupyter notebook.
3. Déplacez vos dans le dossier sur votre ordinateur contenant les gabarits des laboratoires
4. Réalisez le laboratoires en vous assurant d'inscrire vos noms et matricules

```
pip install "MTH2210[notebook] @ git+https://github.com/AntoninPaquette/MTH2210.py.git"
```   

## Guide

Un guide expliquant comment résoudre des problèmes d'interpolation, des problèmes non-linéaires et des EDOs est disponible au lien suivant:

```{toctree}
:maxdepth: 1

guide/index
```

## Références

La documentation détaillée de chaque fonction de la librairie MTH2210 est aussi disponible au lien suivant:

```{toctree}
:maxdepth: 1

api/list_fct
```