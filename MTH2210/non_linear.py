from typing import Callable

import numpy as np
import numpy.typing as npt

machine_precision = np.finfo(np.float64).eps

def _init_non_linear(fct:Callable, x0:float, nb_it_max:int, tol_rel:float, x1=None, dfct=None):

    # Vérification de x0 et potentiellement de x1
    if not isinstance(x0, (int, np.integer, float, np.floating)) or isinstance(x0, bool):
        raise Exception("Le paramètre x0 n'est pas de type int ou float")
    
    # Promote x0 en float
    x0 = np.float64(x0)

    if x1 is not None:
        if not isinstance(x1, (int, np.integer, float, np.floating)) or isinstance(x1, bool):
            raise Exception("Le paramètre x1 n'est pas de type int ou float")
        x1 = np.float64(x1)

       
    # Vérifie le type de nb_it_max
    if not isinstance(nb_it_max, (int, np.integer)) or isinstance(nb_it_max, bool):
        raise Exception("Le paramètre nb_it_max n'est pas de type int")

    # Vérifie le type de nb_it_max
    if not isinstance(tol_rel, (float, np.floating)):
        raise Exception("Le paramètre tol_rel n'est pas de type float")
    
    # Vérifie si problème avec la fonction fct
    try:
        fct(x0)
    except Exception as e:
        raise RuntimeError(f"Problème avec la fonction fct. Voici le message d'erreur: {e}")

    if not isinstance(fct(x0), (float, np.floating)):
        raise Exception("La fonction fct ne retourne pas un float")

    if dfct is not None:
        try:
            dfct(x0)
        except Exception as e:
            raise RuntimeError(f"Problème avec la fonction derivée de fct. Voici le message d'erreur: {e}")

        if not isinstance(dfct(x0), (float, np.floating)):
            raise Exception("La fonction dérivée de fct ne retourne pas un float")

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
    secante : Méthode de la sécante
    ptfixe : Méthode des points-fixes
    newton1d : Méthode de Newton en 1D
    newtonNd : Méthode de Newton en N dimension
    
    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import bissection
    >>> (approx, err_abs) = bissection(lambda x:x**2-2, 0, 2 , 200, 1e-6)
    """

    _init_non_linear(fct, x0, nb_it_max, tol_rel, x1=x1)

    # Promote x0 et x1 en float
    x0 = np.float64(x0)
    x1 = np.float64(x1)

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
    err_rel	= np.inf*np.ones(nb_it_max)
    arret =	False
    nb_it = 1
    x_gauche = np.min([x0,x1])
    x_droite = np.max([x0,x1])

    for t in range(nb_it_max):

        if t==0:
            x_gauche = np.min([x0,x1])
            x_droite = np.max([x0,x1])
        else:
            if f_gauche * f_milieu < 0:
                x_droite = x_milieu
            elif f_droite * f_milieu < 0:
                x_gauche = x_milieu
            else:
                print("Problème avec la fonction f.\nArrêt de l''algorithme\n")
                break

        x_milieu = (x_gauche + x_droite)/2
        app[t] = x_milieu

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


def secante(fct:Callable, x0:float, x1:float, nb_it_max:int, tol_rel:float):
    """
    Résolution d'une équation non-linéaire de forme f(x)=0 avec la méthode
    de la sécante

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
    bissection : Méthode de la bissection
    ptfixe : Méthode des points-fixes
    newton1d : Méthode de Newton en 1D
    newtonNd : Méthode de Newton en N dimension
    
    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import secante
    >>> (approx, err_abs) = secante(lambda x:x**2-2, 0, 2 , 50, 1e-6)
    """

    _init_non_linear(fct, x0, nb_it_max, tol_rel, x1=x1)

    # Promote x0 et x1 en float
    x0 = np.float64(x0)
    x1 = np.float64(x1)

    # Initialisation des vecteurs
    app = np.nan * np.ones(nb_it_max+2)
    app[0] = x0
    app[1] = x1
    err_rel = np.inf * np.ones(nb_it_max+2)
    err_rel[0] = (x1 - x0)/(x1 + machine_precision)
    arret = False

    for t in range(1,nb_it_max+1):
        
        app[t+1] = app[t] - fct(app[t]) * (app[t] - app[t-1]) / (fct(app[t]) - fct(app[t-1]))

        if abs(fct(app[t]) - fct(app[t-1])) == 0:
            print(f"L'approximation de la dérivée de f avec les points x={app[t]:.5e} et x={app[t-1]:.5e} est exactement 0.\nArrêt de l'algorithme\n")
            break

        err_rel[t] = abs(app[t+1]-app[t])/(abs(app[t+1]) + machine_precision)

        if (err_rel[t] <= tol_rel) or (fct(app[t+1]) == 0):
            arret = True
            break

    nb_it = t+2
    approx = app[0:nb_it]
    err_abs = np.inf * np.ones(nb_it)

    if arret:
        err_abs = np.abs(approx[-1] - approx)
    else:
        print("La méthode de la sécante n'a pas convergée")

    return approx, err_abs


def newton1d(fct:Callable, dfct:Callable, x0:float, nb_it_max:int, tol_rel:float):
    """ 
    Methode de Newton pour la resolution f(x) = 0

    Parameters
    ----------
    fct : Callable
        Fonction f pour laquelle on cherche la racine
    dfct : Callable
        Dérivée de la fonction f
    x0 : float 
        Première approximation initiale
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
    bissection : Méthode de la bissection
    secante : Méthode de la sécante
    ptfixe : Méthode des points-fixes
    newtonNd : Méthode de Newton en N dimension
    
    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import newton1d
    >>> (approx, err_abs) = newton1d(lambda x:x**2-2, lambda x: 2*x, 0.1, 50, 1e-6)
    """

    _init_non_linear(fct, x0, nb_it_max, tol_rel, dfct=dfct)

    x0 = np.float64(x0)

    app	= np.nan * np.ones(nb_it_max+1)
    app[0] = x0
    err_rel	= np.inf * np.ones(nb_it_max+1)
    arret = False

    for t in range(nb_it_max):

        app[t+1] = app[t] - fct(app[t])/dfct(app[t])

        if np.abs(dfct(app[t])) == 0:
            print(f"La derivee de f au point x={app[t]:.5e} est exactement 0.\nArret de l'algorithme")
            break

        err_rel[t] = np.abs(app[t+1]-app[t])/(np.abs(app[t+1]) + machine_precision)
        if (err_rel[t] <= tol_rel) or (fct(app[t+1]) == 0):
            arret = True
            break

    nb_it = t+1
    approx = app[0:nb_it]
    err_abs	= np.inf * np.ones(nb_it)

    if arret:
        err_abs	= abs(approx[-1] - approx);
    else:
        print("La methode de Newton n'a pas convergée")

    return approx, err_abs 


def ptfixe(fct:Callable, x0:float, nb_it_max:int, tol_rel:float):
    """ 
    Methode des points-fixes pour la resolution de f(x) = x

    Parameters
    ----------
    fct : Callable
        Fonction f pour laquelle on cherche la racine
    x0 : float 
        Approximation initiale du point-fixe
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
    bissection : Méthode de la bissection
    secante : Méthode de la sécante
    newton1d : Méthode de Newton en 1D
    newtonNd : Méthode de Newton en N dimension

    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import ptfixe
    >>> (approx, err_abs) = ptfixe(lambda x : -x**2/10 + x + 1, 4, 50, 1e-6)
    """

    _init_non_linear(fct, x0, nb_it_max, tol_rel)

    app	= np.nan * np.ones(nb_it_max+1)
    app[0] = x0
    err_rel	= np.inf * np.ones(nb_it_max+1)
    arret = False


    for t in range(nb_it_max):
        app[t+1] = fct(app[t]);
        err_rel[t] = np.abs(app[t+1]-app[t])/(abs(app[t+1]) + machine_precision)
        if err_rel[t] <= tol_rel or fct(app[t+1]) == app[t+1]:
            arret = True
            break


    nb_it = t+1
    approx = app[0:nb_it]
    err_abs	= np.inf * np.ones(nb_it)

    if arret:
        err_abs	= abs(approx[-1] - approx)
    else:
        print("La methode des points-fixes n'a pas convergée")

    return approx, err_abs

def newtonNd(fct:Callable, x0:npt.ArrayLike, nb_it_max:int, tol_rel:float, h:float=None, dfct:Callable=None):
    """ 
    Methode de Newton pour la resolution de F(x) = 0, pour F: R^n -> R^n

    Parameters
    ----------
    fct : Callable
        Fonction f pour laquelle on cherche la racine. Cette fonction doit prendre 
        en entrée un vecteur de taille n et retourner un vecteur de taille n
    x0 : 1D ndarray de taille n 
        Approximation initiale du problème de racine
    nb_it_max : int
        Nombre maximum d'itérations
    tol_rel : float
        Tolérance sur l'approximation de l'erreur relative
    h : float (optionnel)
        Pas h des différences centrées d'ordre 2 employées afin d'approximer la matrice jacobienne de fct
        Soit le paramètre h ou dfct doit être fournis en entrée
    dfct : Callable (optionnel)
        Fonction prenant en entrée un vectuer de dimension n et retournant la matrice jacobienne de fct
        Soit le paramètre h ou dfct doit être fournis en entrée

    Returns
    -------
    approx : 2D ndarray de taille nb_iter x n 
        2D array contenant les itérations. La rangée i correspond à l'approximation obtenue 
        à l'itération i
    err_abs : 1D ndarray de taille nb_iter 
        1D array contenant les erreurs absolues

    See Also
    --------
    bissection : Méthode de la bissection
    secante : Méthode de la sécante
    ptfixe : Méthode des points-fixes
    newton1d : Méthode de Newton en 1D

    Examples
    --------

    Exemple avec le paramètre h

    >>> import numpy as np
    >>> from MTH2210 import newtonNd
    >>> def fct(x):
    >>>     return np.array([x[0]**2 + x[1]**2 -1,-x[0]**2 + x[2]])    
    >>> (approx, err_abs) = newtonNd(fct, np.array([1,1]), 20, 1e-6, h=1e-3)

    Exemple avec la matrice jacobienne

    >>> import numpy as np
    >>> from MTH2210 import newtonNd
    >>> def fct(x):
    >>>     return np.array([x[0]**2 + x[1]**2 -1,-x[0]**2 + x[2]])
    >>> def dfct(x):
    >>>     return np.array([[2*x[0], 2*x[1]],[-2*x[0],1]])
    >>> (approx, err_abs) = newtonNd(fct ,np.array([1,1]), 20, 1e-6, dfct = dfct)
    """

    try:
        x0 = np.asarray(x0,dtype=np.float64)
    except:
        raise Exception("L'approximation initiale x0 ne peut être convertie en np.ndarray de type np.float64")

    if x0.ndim != 1:
        raise Exception("L'approximation initale x0 n'est pas unidimensionnelle")
    
    nb_var = x0.shape[0]
    
    # Vérifie si problème avec la fonction fct
    try:
        f_x0 = fct(x0)
    except Exception as e:
        raise RuntimeError(f"Problème avec la fonction fct. Voici le message d'erreur: {e}")
    
    if f_x0.ndim != 1 or f_x0.shape[0] != nb_var:
        raise Exception(f"La fonction fct ne renvoie pas un array unidimensionnel de taille {nb_var}")

    if not np.issubdtype(f_x0.dtype, np.floating):
        raise Exception("La fonction fct ne renvoie pas un array de type float")

    # Vérifie le type de nb_it_max
    if not isinstance(nb_it_max, (int, np.integer)) or isinstance(nb_it_max, bool):
        raise Exception("Le paramètre nb_it_max n'est pas de type int")

    # Vérifie le type de nb_it_max
    if not isinstance(tol_rel, (float, np.floating)):
        raise Exception("Le paramètre tol_rel n'est pas de type float")
    
    if (h is None and dfct is None) or (h is not None and dfct is not None):
        raise Exception("Le paramètre h ou le paramètre dfct doit être passé en argument")

    if h is not None:
        if not isinstance(h, (float, np.floating)):
            raise Exception("Le paramètre h n'est pas de type float")


    # Vérification de la dérivée de fct
    if dfct is not None:
        try:
            dfct_x0 = dfct(x0)
        except Exception as e:
            raise RuntimeError(f"Problème avec la fonction derivée de fct. Voici le message d'erreur: {e}")

        if dfct_x0.ndim != 2 or dfct_x0.shape != (nb_var,nb_var):
            raise Exception(f"La fonction dérivée de fct ne retourne pas une matrice bidimensionnel de taille {nb_var}x{nb_var}")

        if not np.issubdtype(dfct_x0.dtype, np.floating):
            raise Exception("La fonction dérivée de fct ne retourne pas un array de floats")

    # Initialisation des matrices app et err
    app	    = np.nan * np.ones((nb_it_max+1,nb_var))
    app[0]  = x0
    err_rel	= np.inf * np.ones(nb_it_max+1)
    arret   = False

    # Methode de Newton
    for t in range(nb_it_max):
        
        if h is None:
            jac	= dfct(app[t])
        else:
            jac = _app_jacobienne(fct, app[t], h)
        delta_x     = -np.linalg.solve(jac, fct(app[t]))
        app[t+1]    = app[t] + delta_x
        
        cond_jac = np.linalg.cond(jac)

        if cond_jac > 1e12:
            print(f"La matrice jacobienne de f a l'iteration {t:.d} est très mal conditionnée (cond(jacobienne={cond_jac:.e})\nArret de l'algorithme")
            break
            
        err_rel[t]	=	np.linalg.norm(app[t+1]-app[t])/(np.linalg.norm(app[t+1]) + machine_precision)
        if err_rel[t] <= tol_rel or np.linalg.norm(fct(app[t+1])) == 0:
            arret = True
            break

    nb_it	=	t+1
    approx	=	app[:nb_it+1]

    if arret:
        err_abs = np.array([np.linalg.norm(approx[-1] - app) for app in approx])
    else:
        print("La methode de Newton n'a pas convergée")

    return approx, err_abs

def _app_jacobienne(f,x0,h_init):

    nb_var = x0.shape[0]
    # if np.min(x0) == 0:
    #     h_init	=	1e-6
    # else:
    #     h_init	=	1e-3 * np.min(x0)

    h   = h_init / 2**np.arange(2)
    app	= np.zeros((2, nb_var, nb_var))

    for t in range(2):
        for d in range(nb_var):
            delta_h		=	np.zeros(nb_var)
            delta_h[d]	=	h[t]
            app[t,:,d]	=	(f(x0+delta_h) - f(x0-delta_h))/(2*h[t])

    app_finale = (2**2 * app[1] - app[0])/(2**2 - 1)

    return app_finale