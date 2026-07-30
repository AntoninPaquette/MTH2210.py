# Résolution de problèmes d'interpolation

Cette section est dédiée à la résolution de problèmes d'interpolation. Le
premier type d'interpolation consiste à résoudre le problème suivant:
Connaissant les $n+1$ points d'interpolation $(x_0,y_0),(x_1,y_1),\ldots,
(x_n,y_n)$, on cherche le polynôme $p$ de degré $n$ tel que $p(x_0)=y_0,
p(x_1)=y_1,\ldots,p(x_n)=y_n$. L'algorithme disponible pour résoudre ce
problème est la méthode de Lagrange: {py:func}`MTH2210.lagrange`. L'algorithme employée
est une version autre que celle du cours et est basé sur
*Barycentric Lagrange Interpolation* (Berrut J. et Trefethen L.N.).

Une autre méthode d'interpolation est la spline cubique et l'algorithme
disponible est la méthode des splines cubiques: {py:func}`MTH2210.splinec`.

## Exemple d'interpolation avec la méthode de Lagrange

Soit les points d'interpolation $(-1,2), \ (0,-4), \ (2.5,10), \ (3,5)$. On
veut afficher le polynôme de degré 3 sur l'intervalle $[-1,3]$. On utilise
la fonction  ainsi:

```{eval-rst}
.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt

    from MTH2210 import lagrange, splinec

    xi = [-1, 0, 2.5, 3]
    yi = [2, -4, 10, 5]
    xfin = np.linspace(np.min(xi),np.max(xi),250)
    Lx = lagrange(xi , yi , xfin)

    fig, ax = plt.subplots(1)
    ax.scatter(xi, yi, label="Pt inter")
    ax.plot(xfin, Lx, label="P_3")

    ax.set_xlabel("x")
    ax.set_title("Interpolation de Lagrange") 
    ax.legend()  
```


## Exemple d'interpolation avec la méthode des splines cubiques

On peut aussi employer une spline cubique afin d'interpoler les points
précédents. On impose que $S'(x_0) = 10$ et $S''(x_3) = 0$.

```{eval-rst}
.. jupyter-execute::

    Sx = splinec(xi , yi , xfin, [4,2], [10, 0])

    fig, ax = plt.subplots(1)
    ax.scatter(xi, yi, label="Pt inter")
    ax.plot(xfin, Lx, label="P_3")
    ax.plot(xfin, Sx, label="Spline cubique")

    ax.set_xlabel("x")
    ax.set_title("Interpolation avec spline cubique")
    ax.legend()   
```
