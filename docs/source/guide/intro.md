# Introduction à Numpy

**Documentation de Numpy:** [Version stable](https://numpy.org/doc/stable/)



```{eval-rst}
.. jupyter-execute::

    import numpy as np
    x = np.array([[1,2],[3,4],[5,6]])
    x
```


# Introduction à Matplotlib

**Documentation de Matplotlib:** [Version stable](https://matplotlib.org/stable/index.html)

La librairie Matplotlib permet d'afficher des graphiques. Par exemple, on peut afficher un graphique des fonctions $f_1(x) = \sin(x)+x$ et $f_2(x) = \exp(0.3x)$

```{eval-rst}
.. jupyter-execute::
    
    import matplotlib.pyplot as plt

    x = np.linspace(0, 5, 250)
    f1 = np.sin(x) + x
    f2 = np.exp(0.3*x)

    fig, ax = plt.subplots(1)

    ax.plot(x, f1, label="f1")
    ax.plot(x, f2, label="f2")
    
    ax.set_xlabel("x")
    ax.set_title("Fonctions f1 et f2")
    fig.legend()
```

Plusieurs graphiques peuvent aussi être afficher à l'aide de la commande ``plt.subplots``

```{eval-rst}
.. jupyter-execute::
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x, f1, label="f1")
    ax[1].plot(x, f2, label="f2")
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[0].set_title("Fonction f1")
    ax[1].set_title("Fonction f2")
```
