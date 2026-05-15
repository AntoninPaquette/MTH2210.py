#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet

Modified in May 2026

@author: Antonin Paquette-Rufiange
"""



#%%##################################
# Import des bibliothèques requises #
#####################################

from ..Module_coeur import check_type_arguments, check_relative_tolerance, writing_function
import types
import numpy as np



#%%########################################
# Fonction de vérification des paramètres #
###########################################

# Vérifie que le jeu de paramètres reçu par la méthode respecte les types attendus et les hypothèses mathématiques
def check_parameters_consistency(f, x0, t0, tm, m, output):
    # Vérification des types des paramètres reçus
    params_array = [[f,      "f",      types.FunctionType],
                    [x0,     "x0",     [np.ndarray, float]],
                    [t0,     "t0",     float],
                    [tm,     "tm",     float],
                    [m,      "m",      int],
                    [output, "output", str]]
    check_type_arguments.check_parameters(params_array)
    # Vérification de la cohérence des paramètres
    try:    f(x0, t0)
    except: raise ValueError("Fonction f non définie en (x0,t0)")
    try:    f(x0, tm)
    except: raise ValueError("Fonction f non définie en (x0,tm)")
    if type(x0) == float:
        if not(check_type_arguments.check_generic(f(x0,t0), float)[0]): raise ValueError("f(x0,t0) n'est pas un float (type reçu :"+check_type_arguments.get_type(f(x0,t0))+")")
        if not(check_type_arguments.check_generic(f(x0,tm), float)[0]): raise ValueError("f(x0,tm) n'est pas un float (type reçu :"+check_type_arguments.get_type(f(x0,tm))+")")
    else:
        if not(check_type_arguments.check_generic(f(x0,t0), np.ndarray)[0]): raise ValueError("f(x0,t0) n'est pas un vecteur np.ndarray (type reçu :"+check_type_arguments.get_type(f(x0,t0))+")")
        if not(check_type_arguments.check_generic(f(x0,tm), np.ndarray)[0]): raise ValueError("f(x0,tm) n'est pas un vecteur np.ndarray (type reçu :"+check_type_arguments.get_type(f(x0,tm))+")")
        if len(f(x0,t0)) != len(x0): raise ValueError("les dimensions de f(x0,t0) (= "+str(len(f(x0,t0)))+") et x0 (= "+str(len(x0))+") diffèrent")
    if m < 0: raise ValueError("Nombre d'itérations m défini à une valeur négative")



#%%########################################
# Fonctions de mise en page des résultats #
###########################################

# Crée la chaîne de caractères qui sera renvoyée pour chaque itération
def format_iter(k, x_k, t_k):

    if type(x_k) == float:
        if k == 0:
            header  = "{:>4} || {:^11} | {:^9}"
            header  = header.format("k", "x_k", "t_k")
            header += "\n"
            header += "-"*(4+11+8 + 4+3)
            header += "\n"
        else:
            header = ""
        iter_infos = "{:>4} || {:>+11.4e} | {:>+9.4f}"
        iter_infos = iter_infos.format(k, x_k, t_k)

    else:
        if k == 0:
            n = len(x_k)
            len_str_xk = 2+11*n+2*(n-1)
            header  = "{:>4} || " + "{:^"+str(len_str_xk)+"}" + " | " + "{:^9}"
            header  = header.format("k", "x_k", "t_k")
            header += "\n"
            header += "-"*(4+len_str_xk+9 + 4+3)
            header += "\n"
        else:
            header = ""
        iter_infos  = "{:>4} || ".format(k)
        iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in x_k])+"] | "+"{:>+9.4f}".format(t_k)

    return(header+iter_infos)



#%%########################################
# Fonctions de tests des critères d'arrêt #
###########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(k, m):
    if k >= m: return(True, "Nombre maximal d'itérations k_max = {} atteint".format(m))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(x0, t0, tm, m):
    k = 0
    x = x0
    t = t0
    h = (tm-t0)/m
    list_x = [x]
    list_t = [t]
    return(k, x, t, h, list_x, list_t)

# Exécute une itération de la méthode
def iter_algo(f, k, x, t, h, list_x, list_t):
    k += 1
    x = x + h*f(x,t)
    t += h
    list_x.append(x)
    list_t.append(t)
    return(k, x, t, list_x, list_t)



#%%#####################################
# Définition de la fonction principale #
########################################



def euler(f, x0, t0, tm, m, output=""):
    """
    Résout numériquement une équation différentielle ordinaire par la méthode d'Euler explicite.

    Cette fonction approxime la solution de l'EDO :
        dx/dt = f(x, t)
        x(t_0) = x_0
    à l'aide du schéma d'Euler :
        x_{k+1} = x_k + h * f(x_k, t_k)
        t_{k+1} = t_k + h

    où h = (t_m - t_0) / m.

    Parameters
    ----------
    f : callable
        Fonction définissant l'équation différentielle. Elle doit prendre en entrée
        un état `x` et un temps `t`, et retourner une valeur de même nature et dimension que `x`.
    x0 : float ou numpy.ndarray
        Condition initiale à l'instant `t0`. Peut être un scalaire (problème 1D)
        ou un tableau NumPy pour les systèmes de dimension supérieure.
    t0 : float
        Temps initial.
    tm : float
        Temps final.
    m : int
        Nombre de pas de discrétisation de l'intervalle [t0, tm].
        Le pas de temps est donné par h = (tm - t0) / m.
    output : str, optionnel
        Définit la destination des sorties intermédiaires :
        - `"pipe"` : affichage dans la sortie standard,
        - nom de fichier : écriture dans un fichier,
        - `""` ou `"None"` : aucune sortie (valeur par défaut).

    Returns
    -------
    list_x : list
        Liste des approximations de la solution x(t_k).
    list_t : list
        Liste des instants t_k.

    Raises
    ------
    TypeError
        Si les paramètres n'ont pas les types attendus.
    ValueError
        Si `f(x0, t0)` ou `f(x0, tm)` est mal défini ou incompatible avec `x0`.

    Notes
    -----
    - La fonction vérifie que :
      - `f` est définie en `(x0, t0)` et `(x0, tm)`,
      - `f(x0, t0)` renvoie un objet de même type et dimension que `x0`.

    - Cas particuliers pour les problèmes de dimension 1 :
      - Si `x0` est un scalaire, alors `f(x, t)` doit renvoyer un scalaire.
      - Si `x0` est un tableau NumPy de taille 1, des incohérences de type peuvent apparaître.

    - Utilisation recommandée :
      - Si dim(x) > 1 :
        - `x0` sous forme de `numpy.ndarray`,
        - `f(x, t)` retourne un tableau de même dimension.
      - Si dim(x) = 1 :
        - `x0` scalaire (`float` ou `int`),
        - `f(x, t)` retourne un scalaire.

    - Cas non garantis :
      - Variables complexes,
      - Variable 1D définie comme un tableau NumPy de taille 1.

    See Also
    --------
    rk4 : Méthode de Runge-Kutta d'ordre 4.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> list_x, list_t = euler(lambda x, t: np.cos(t), 0.0, 0.0, 2*np.pi, 100)

    >>> f = lambda x, t: np.array([np.cos(t), np.sin(t)])
    >>> x0 = np.array([0.0, 0.0])
    >>> list_x, list_t = euler(f, x0, 0.0, 2*np.pi, 100)

    >>> def f(x, t):
    ...     return np.array([x[0]*(x[1] - 1), x[1]*(1 - x[0])])
    >>> x0 = np.array([2.0, 1.0])
    >>> list_x, list_t = euler(f, x0, 0.0, 10.0, 100)
    """


# def euler(f, x0, t0, tm, m, output=""):
#     """
#     Méthode de résolution numérique d'une équation (dx/dt)(t) = f(x(t),t) par le schéma d'Euler :
#         - x_0 donné, t_0 donné, pas de temps h donné,
#         - x_kp1 = x_k + h*f(x_k,t_k),
#         - t_kp1 = t_k + h.

#     Les arguments attendus sont :
#         - une fonction f, admettant en entrée un vecteur x et un réel t, renvoyant un vecteur f(x,t),
#         - un vecteur  x0, condition initiale de l'équation,
#         - deux réels  t0 et tm, les bornes de l'intervalle de temps sur lequel l'équation est appliquée,
#         - un entier    m, le pas de discrétisation de [t0,tm], définissant donc h = (tm-t0)/m.

#     L'argument optionnel est une chaîne de caractères output (défaut = "") qui renvoie les affichages de la fonction vers :
#         - la sortie standard si output = "pipe",
#         - un fichier ayant output comme nom+extension (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
#         - nulle part (aucune information écrite ni sauvegardée) si output = "" ou output = "None".

#     La méthode vérifie les conditions suivantes :
#         - la fonction f est définie en (x0,t0) et en (x0,tm),
#         - f(x0,t0) renvoie un vecteur de la même dimension et même type que x0,
#         - tous les paramètres reçus ont bien le type attendu.

#     À noter que si x est un vecteur de dim 1, f doit être implémentée avec parcimonie pour ne pas renvoyer un mauvais type. Par exemple :
#         - x = np.array(0) est un np.ndarray, y = np.array([0]) également,
#         - f(x,t) = np.cos(x) est un float,
#         - f(y,t) = np.cos(y) est un np.ndarray,
#         - f(x,t) = np.cos(t) est un float.

#     Ces différences de types peuvent faire échouer la méthode si x est de dimension 1. La méthode est conçue pour fonctionner suivant :
#         - si la dimension de x est > 1 :
#             - x      défini par un np.array([coordonnées]),
#             - f(x,t) renvoyant  un np.ndarray de même dimension que x,
#         - si x est de dimension 1 :
#             - x      défini par un float ou un int,
#             - f(x,t) renvoyant  un float,
#         - cas sans garantie de fonctionnement correct :
#             - x      complexe,
#             - x      de dimension 1 défini par un np.array([valeur]).

#     Returns
#     -------
#     list_x
#         Liste des points x(t_k)
#     list_t
#         Liste des instants t_k.

#     See Also
#     --------
#     MTH2210.EDO.rk4.rk4 : Méthode de Runge-Kutta d'ordre 4

#     Examples
#     --------
#     >>> import matplotlib.pyplot as plt
#     >>> euler(lambda x,t : np.cos(t), 0, 0, 2*np.pi, 100),
#     >>> euler(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 100),
#     >>> def f(x,t):
#     >>>     x0,x1 = 1,1
#     >>>     return(np.array([x[0]*(x[1]-1),x[1]*(1-x[0])]))
#     >>> x = np.array([2,1])
#     >>> list_x, list_t = euler(f, x, 0, 10, 100).
#     >>> plt.plot(list_t, list_t)
#     >>> plt.show()

#     test
    
#     >>> import matplotlib.pyplot as plt
#     >>> N = 8
#     >>> y = np.zeros(N)
#     >>> x1 = np.linspace(0, 10, N, endpoint=True)
#     >>> x2 = np.linspace(0, 10, N, endpoint=False)
#     >>> plt.plot(x1, y, 'o')
#     [<matplotlib.lines.Line2D object at 0x...>]
#     >>> plt.plot(x2, y + 0.5, 'o')
#     [<matplotlib.lines.Line2D object at 0x...>]
#     >>> plt.ylim([-0.5, 1])
#     (-0.5, 1)
#     >>> plt.show()

#     """

    # Test des paramètres et définition de la destination de sortie des itérations
    if check_type_arguments.check_real(x0)[0]: x0 = float(x0)
    check_parameters_consistency(f, x0, t0, tm, m, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)

    # Initialisation de l'algorithme
    k, x, t, h, list_x, list_t = init_algo(x0, t0, tm, m)
    write_iter(k, x, t)

    # Déroulement de l'algorithme
    while not(stopping_criteria(k, m)[0]):
        k, x, t, list_x, list_t = iter_algo(f, k, x, t, h, list_x, list_t)
        write_iter(k, x, t)

    write_stopping(stopping_criteria(k, m)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, list_t)
