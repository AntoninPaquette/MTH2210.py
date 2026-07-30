# Librairie numérique du cours MTH2210 de Polytechnique Montréal

Ce site web contient la documentation de la librairie numérique MTH2210 de Polytechnique Montréal. 

## Installation minimale

Dans le cadre de ce cours, nous vous reccomandons cette installation minimale:

1. Installation de Python soit via Anaconda ou via Miniconda [https://www.anaconda.com/docs/getting-started/installation](https://www.anaconda.com/docs/getting-started/installation)
2. Installation de Visual Studio Code: [https://code.visualstudio.com/](https://code.visualstudio.com/)
3. Installation de l'extension Python de VSCode [https://code.visualstudio.com/docs/python/python-quick-start](https://code.visualstudio.com/docs/python/python-quick-start)
4. Installation de l'extension Jupyter de VSCode [https://code.visualstudio.com/docs/datascience/jupyter-notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
5. Création d'un environnement virtuel dans un dossier approprié sur votre machine. Nous recommandons de créer cet environnement virtuel via VSCode en suivant les étapes décrites au lien suivant: [https://code.visualstudio.com/docs/python/environments](https://code.visualstudio.com/docs/python/environments)
6. Installation de la librairie MTH2210. Pour installer la librairie, assurez vous tout d'abord que l'environnement virtuel est activé, puis installer la librairie avec la commande suivante (dans la console de VSCode) 
```
pip install git+https://github.com/AntoninPaquette/MTH2210.py.git[notebook]
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