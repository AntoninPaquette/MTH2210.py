import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve

def _init_interpolation(xi, yi, x):

    for arg, arg_name in zip([xi, yi, x],["xi", "yi", "x"]):

        try:
            arg = np.atleast_1d(arg)
        except:
            raise Exception(f"Le paramètre {arg_name} ne peut être converti en np.ndarray de type np.float64")
        
        if arg.ndim != 1:
            raise Exception(f"Le paramètre {arg_name} n'est pas unidimensionnelle")
    
    xi = np.atleast_1d(xi)
    yi = np.atleast_1d(yi)
    x = np.atleast_1d(x)

    if xi.shape[0] != yi.shape[0]:
        raise Exception("Les paramètres xi et yi doivent être de même taille")

    if xi.shape[0] == x.shape[0] and np.allclose(xi, x):
        raise Exception("Le polynôme d'interpolation est evalué exactement au points d'interpolation")

    return xi, yi, x

def lagrange(xi:npt.ArrayLike, yi:npt.ArrayLike, x:npt.ArrayLike):
    """
    Polynôme de Lagrange passant par les points xi et yi

    Le calcul du polynome interpolant est base sur la formule barycentrique 
    une variation stable de la formule d'interpolation de Lagrange vue au
    cours. Cette implementation est basee sur celle de Greg von Winckel,
    elle-meme basee sur l'article de Berrut et Trefethen [1]_.


    Parameters
    ----------
    xi : Arraylike de dimension 1
        Abscisses des points d'interpolation
    yi : Arraylike de dimension 1
        Ordonnées des points d'interpolation
    x : Arraylike de dimension 1
        Points où le polynôme de Lagrange sera évalué

    Returns
    -------
    NDArray de dimension 1
        Valeur du polynôme de Lagrange aux points x

    See Also
    --------
    splinec : Spline cubique

    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import lagrange 
    >>> Lx = lagrange([1,2,4,5], [1,9,2,11], np.linspace(1,5))

    References
    ----------
    .. [1] Barycentric interpolation is a variant of Lagrange polynomial interpolation that is fast and stable. It deserves to be known as the standard method of polynomial interpolation." (Berrut and Trefethen, 2004)
    """
    (xi , yi , x) = _init_interpolation(xi, yi, x)
 
    M = xi.shape[0]  
    N = x.shape[0]   

    # Calcul des poids barycentriques
    Xi = np.tile(xi, (M,1)) 
    W = 1/np.prod(Xi - Xi.T + np.eye(M), axis=0)
    W_repeated = np.tile(W, (N,1))

    # Calcul des distances entre les xi et x
    xdist = np.tile(x, (M,1)).T - np.tile(xi, (N,1))

    # Change the null distance by np.nan if x xontains xi, so that the distance is null and the inverse should be identified by np.nan 
    idx_0 = np.argwhere(np.isclose(xdist, 0))
    xdist[idx_0[:,0], idx_0[:,1]] = np.nan

    H = W_repeated/xdist
    Lx = np.linalg.matmul(H, yi) / np.sum(H, axis=1)

    Lx[idx_0[:,0]] = yi[idx_0[:,1]] 

    return Lx

def _diff_div(x,y,n):
    diff_x	=	x[n:] - x[0:-n]
    diff_y	=	np.diff(y)
    f		=	diff_y/diff_x

    return f


def _poly_spline(xi,yi,Spp,h,x):

    Px = -Spp[0] / (6*h) * (x - xi[1])**3 + Spp[1] / (6*h) * (x - xi[0])**3 - \
          (yi[0] / h - Spp[0]*h/6) * (x - xi[1]) + (yi[1]/h - Spp[1]*h/6) * (x - xi[0])
    
    return Px

def splinec(xi:npt.ArrayLike, yi:npt.ArrayLike, x:npt.ArrayLike, type_S:npt.ArrayLike, val_S:npt.ArrayLike):
    """
    Spline cubique passant par les points xi et yi avec differents type de conditions frontieres

    Parameters
    ----------
    xi : Arraylike de dimension 1
        Abscisses des points d'interpolation
    yi : Arraylike de dimension 1
        Ordonnées des points d'interpolation
    x : Arraylike de dimension 1
        Points où le polynôme de Lagrange sera évalué
    type_S : Arraylike de dimension 1 avec 2 éléments
        Vecteur de 2 éléments contenant le type des conditions frontieres imposees en x0 et xn. Les choix possibles sont:

		* [1,1] -> Spline naturelle 
		* [2,2] -> Spline avec courbure prescrite
		* [3,3] -> Spline avec courbure constante
		* [4,4] -> Spline avec pente prescrite
		* [i,j] -> Spline avec condition i imposee en x0 et condition j imposee en xn
    val_S : Arraylike de dimension 1 avec 2 éléments
        Vecteur de 2 éléments contenant les deux conditions limites imposées en x0 et xn. Les choix possibles sont:
        
		* Si type_S(1) = 1 ou 3, alors val_S(1) = nan
		* Si type_S(1) = 2 ou 4, alors val_S(1) = a, où a représente resp. la courbure ou la pente en x0
		* Si type_S(2) = 1 ou 3, alors val_S(1) = nan
		* Si type_S(2) = 2 ou 4, alors val_S(1) = b, où b représente resp. la courbure ou la pente en xn

    Returns
    -------
    NDArray de dimension 1
        Valeur de la spline aux points x

    See Also
    --------
    lagrange : Interpolant de Lagrange

    Examples
    --------
    >>> import numpy as np
    >>> from MTH2210 import splinec 
    >>> Sx = splinec([1,2,4,5], [1,9,2,11], np.linspace(1,5), [1,1] , [np.nan,np.nan])
    >>> Sx = splinec([1,2,4,5], [1,9,2,11], np.linspace(1,5), [2,2] , [5,-6])
    >>> Sx = splinec([1,2,4,5], [1,9,2,11], np.linspace(1,5), [3,3] , [np.nan,np.nan])
    >>> Sx = splinec([1,2,4,5], [1,9,2,11], np.linspace(1,5), [4,4] , [-30,-10])
    >>> Sx = splinec([1,2,4,5], [1,9,2,11], np.linspace(1,5), [3,4] , [np.nan,-10])
    """

    (xi , yi , x) = _init_interpolation(xi, yi, x)

    try:
        type_S = np.asarray(type_S, dtype=int)
    except:
        raise Exception(f"Le paramètre 'type_S' ne peut être converti en np.ndarray de type int")
    
    if type_S.ndim != 1 or type_S.shape[0] !=2:
        raise Exception(f"Le paramètre 'type_S' n'est pas un vecteur de dimension 1 contenant 2 éléments")
    elif not np.all(np.isin(type_S, [1,2,3,4])):
        raise Exception("Le paramètre 'type_S' doit contenir les valeurs 1, 2, 3 ou 4")

    try:
        val_S = np.asarray(val_S, dtype=np.float64)
    except:
        raise Exception(f"Le paramètre 'val_S' ne peut être converti en np.ndarray de type float")
    
    if val_S.ndim != 1 or val_S.shape[0] !=2:
        raise Exception(f"Le paramètre 'val_S' n'est pas un vecteur de dimension 1 contenant 2 éléments")
    
    #  Calcul des coefficients S''

    # Assemblage de la matrice
    nb_f	=	xi.shape[0]
    h		=	np.diff(xi)
    denom	=	h[0:-1] + h[1:]

    diagonals = [np.concatenate([h[0:-1]/denom,[0]]), 
                 np.concatenate([[0], 2*np.ones(nb_f-2), [0]]),
                 np.concatenate([[0], h[1:]/denom])]
    
    # Création de la matrice tridiagonale sparse
    matrice_temp = diags_array(diagonals, offsets=[-1,0,1])
    
    # Changement de l'encodage sparse pour pouvoir effectuer des assignements
    matrice = matrice_temp.tolil()

    # Calcul des 2eme differences divisees
    mat_diff_div1 =	_diff_div(xi, yi, 1)
    mat_diff_div2 =	_diff_div(xi, mat_diff_div1, 2)

    # Terme de droite
    B = np.zeros(nb_f)
    B[1:-1]	= 6*mat_diff_div2

    # Imposition des conditions frontieres
    match type_S[0]:
        case 1:
            matrice[0,0] = 1
            B[0]         = 0
        case 2:
            matrice[0,0] = 1
            B[0]         = val_S[0]
        case 3:
            matrice[0,0:2] = [1,-1]
            B[0]           = 0
        case 4:
            matrice[0,0:2] = [2,1]
            B[0]           = 6/h[0] * ((yi[1] - yi[0])/h[0] - val_S[0])

    match type_S[1]:
        case 1:
            matrice[-1,-1] = 1
            B[-1]          = 0 
        case 2:
            matrice[-1,-1] = 1
            B[-1]		   = val_S[1]
        case 3:
            matrice[-1,-2:] = [-1,1]
            B[-1]   		= 0
        case 4:
            matrice[-1,-2:] = [1,2] 
            B[-1]			= 6/h[-1] * (val_S[1] - (yi[-1] - yi[-2])/h[-1])

    # Resolution du systeme lineaire avec reformattage de la matrice sparse
    Spp	= spsolve(matrice.tocsr(), B)


    # Calcul de la spline aux points x 

    Sx	=	np.nan * np.ones(x.shape[0])

    for t in range(nb_f-1):
        x_inter	    = (x>= xi[t]) & (x <= xi[t+1])
        Sx[x_inter] = _poly_spline(xi[t:t+2], yi[t:t+2],Spp[t:t+2], h[t], x[x_inter])

    return Sx

