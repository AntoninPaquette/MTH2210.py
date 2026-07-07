from typing import Callable, Tuple, Union

import numpy as np

machine_precision = np.finfo(np.float64).eps

def bissection(fct:Callable, x0:float, x1:float, nb_it_max:int, tol_rel:float):
    """
    Résolution d'une équation non-linéaire de la forme f(r)=0 avec la méthode
    de la bissection.

    Parameters
    ----------
    fct : Callable
        Fonction f pour laquelle on cherche la racine
    x0 : float 
        Première approximation initiale
    x1 : float 
        Deuxième approximation initiale
    nb_it_max : int
       Nombre maximum d'itérations
    tol_rel : float
       Tolérance sur l'approximation de l'erreur relative

    Returns
    -------
    approx : 1D ndarray de taille nb_iter 
        1D array contenant les itérations
    err_abs : 1D ndarray de taille nb_iter 
        1D array contenant les erreurs absolues

    See Also
    --------

    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import bissection
    >>> (approx, err_abs) = bissection(lambda x: x-1, 0.75, 1.5 , 100, 1e-6)
    """

    # Check si problème avec la fonction fct
    try:
        fct(x0)
    except Exception as e:
        raise RuntimeError(f"Problème avec la fonction fct. Voici le message d'erreur: {e}")

    # Promote x0 et x1 en float
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    if fct(x0) * fct(x1) > 0:
        raise Exception("La condition f(x0)*f(x1)<0 n'est pas respectée.\nArrêt de l''algorithme")
    elif fct(x0) == 0:
        approx = np.array([x0])
        err_abs = np.array([0.])
        return approx, err_abs
    elif fct(x1) == 0:
        approx = np.array([x0])
        err_abs = np.array([0.])
        return approx, err_abs

    # Initialisation des vecteurs
    app = np.nan*np.ones(nb_it_max)
    err_rel	=	np.inf*np.ones(nb_it_max)
    arret =	False
    nb_it = 1
    x_gauche = np.min([x0,x1])
    x_droite = np.max([x0,x1])

    for t in range(nb_it_max):

        if t==0:
            x_gauche	=	np.min([x0,x1])
            x_droite	=	np.max([x0,x1])
        else:
            if f_gauche*f_milieu < 0:
                x_droite	=	x_milieu
            elif f_droite*f_milieu < 0:
                x_gauche	=	x_milieu
            else:
                print("Problème avec la fonction f.\nArrêt de l''algorithme\n")
                break

        x_milieu	=	(x_gauche + x_droite)/2
        app[t]		=	x_milieu

        if t==0:
            if fct(app[t]) == 0:
                arret = True
                break
        else:
            err_rel[t-1] = abs(app[t]-app[t-1])/(abs(app[t]) + machine_precision)
            if (err_rel[t-1] <= tol_rel) or (fct(app[t]) == 0):
                arret = True
                break

        f_gauche = fct(x_gauche)
        f_droite = fct(x_droite)
        f_milieu = fct(x_milieu)

    nb_it = t+1
    approx  = app[0:nb_it]
    err_abs = np.inf*np.ones(nb_it)

    if arret:
        err_abs = np.abs(approx[-1] - approx)
    else:
        print("La méthode de la bissection n'a pas convergée")

    return approx, err_abs