from typing import Callable, Tuple, Union
import numpy as np
import numpy.typing as npt

ArrayFloat = npt.NDArray[np.floating]

def edo_init(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
) -> Tuple[int, ArrayFloat, ArrayFloat, float]:
    
    tspan = np.asarray(tspan, dtype=float)
    Y0 = np.atleast_1d(Y0)

    if tspan.shape[0] != 2:
        raise ValueError("Le vecteur tspan doit contenir 2 composantes, [t0 , tf]")

    if not isinstance(nbpas, (int, np.integer)) or nbpas <= 0:
        raise ValueError("L'argument nbpas doit être un entier > 0.")

    try:
        result = fct(float(tspan[0]), Y0)
    except IndexError:
        raise ValueError("Le nombre de composantes de Y0 et f ne concorde pas")
    except Exception as e:
        raise RuntimeError(e)

    result_array = np.atleast_1d(result)

    if Y0.shape[0] != result_array.shape[0]:
        raise ValueError("Le nombre de composantes de Y0 et f ne concorde pas")

    N = Y0.shape[0]

    Y = np.zeros((nbpas + 1, N), dtype=float)

    # Initial condition
    Y[0, :] = Y0

    temps = np.linspace(tspan[0], tspan[1], nbpas + 1, dtype=float)

    h = temps[1] - temps[0]

    return Y, temps, h


def euler(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
):
    """
    Résout numériquement une équation différentielle ordinaire par la méthode d'Euler explicite.

    Parameters
    ----------
    fct : Callable
        Fonction définissant l'équation différentielle. Cette fonction doit prendre comme premier argument le temps `t` et comme deuxième argument la variable `x`. Cette fonction doit retourner de même dimension et type que `x`.
    tspan : 1D ArrayLike de taille 2
        Intervalle de temps [t0, tf]
    Y0 : ArrayLike
        Condition initiale à l'instant `t0`. 
        - Un scalaire pour le cas 1D 
        - Un 1D array de dimension (N,) pour le cas à en ND
    nbpas : int
        Nombre de pas de discrétisation de l'intervalle [t0, tf].
        Le pas de temps est donné par h = (tf - t0) / nbpas.

    Returns
    -------
    temps : ndarray de dimension (nbpas +1,)
        Vecteur des pas de temps
    Y : ndarray de dimension (nbpas +1, N)
        Approximation de la solution de l'EDO au pas de temps spécifié par `temps`

        - Y[0] = Y
        - Y[k] est l'approximation au temps `temps[k]`

    See Also
    --------
    euler_modifie : Méthode d'Euler modifié
    pt_milieu : Méthode du point milieu
    rk4 : Méthode de Runge-Kutta d'ordre 4.
    
    Examples
    --------

    Exemple 1D

    >>> import numpy as np
    >>> from MTH2210 import euler
    >>> (t,y) = euler(lambda t, y: np.cos(t), [0,2], 1, 1000)

    Exemple ND inline

    >>> (t,y) = euler(lambda t, y: np.array([y[1],-y[0]]), [0,10], [1,0], 1000)

    Exemple ND

    >>> def my_edo(t,z):
    >>>     f = zeros_like(z)
    >>>     f[0] = z[1]
    >>>     f[1] = -z[0]
    >>>     return f
    >>> (t,y) = euler(my_edo, [0,10], [1,0], 1000)
    """

    Y, temps, h = edo_init(fct, tspan, Y0, nbpas)

    for t in range(nbpas):

        yt = Y[t] 
        fval = np.atleast_1d(fct(temps[t], yt))

        Y[t+1] = yt + h * fval  

    return temps, Y

def euler_modifie(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    Résout numériquement une équation différentielle ordinaire par la méthode d'Euler modifié (méthode d'ordre 2).

    Parameters
    ----------
    fct : Callable
        Fonction définissant l'équation différentielle. Cette fonction doit prendre comme premier argument le temps `t` et comme deuxième argument la variable `x`. Cette fonction doit retourner de même dimension et type que `x`.
    tspan : 1D ArrayLike de taille 2
        Intervalle de temps [t0, tf]
    Y0 : ArrayLike
        Condition initiale à l'instant `t0`. 
        - Un scalaire pour le cas 1D 
        - Un 1D array de dimension (N,) pour le cas à en ND
    nbpas : int
        Nombre de pas de discrétisation de l'intervalle [t0, tf].
        Le pas de temps est donné par h = (tf - t0) / nbpas.

    Returns
    -------
    temps : ndarray de dimension (nbpas +1,)
        Vecteur des pas de temps
    Y : ndarray de dimension (nbpas +1, N)
        Approximation de la solution de l'EDO au pas de temps spécifié par `temps`
        - Y[0] = Y)
        - Y[k] est l'approximation au temps `temps[k]`

    See Also
    --------
    euler_modifie : Méthode d'Euler modifié
    pt_milieu : Méthode du point milieu
    rk4 : Méthode de Runge-Kutta d'ordre 4.

    Examples
    --------

    Exemple 1D

    >>> import numpy as np
    >>> from MTH2210 import euler_modifie
    >>> (t,y) = euler_modifie(lambda t, y: np.cos(t), [0,2], 1, 1000)

    Exemple ND inline

    >>> (t,y) = euler_modifie(lambda t, y: np.array([y[1],-y[0]]), [0,10], [1,0], 1000)

    Exemple ND

    >>> def my_edo(t,z):
    >>>     f = zeros_like(z)
    >>>     f[0] = z[1]
    >>>     f[1] = -z[0]
    >>>     return f
    >>> (t,y) = euler_modifie(my_edo, [0,10], [1,0], 1000)
    """

    Y, temps, h = edo_init(fct, tspan, Y0, nbpas)

    for t in range(nbpas):

        yt      = Y[t]
        y_tilde = yt + h * fct(temps[t], Y[t])

        Y[t+1]  = Y[t] + h/2 * (fct(temps[t], Y[t]) + fct(temps[t] + h, Y[t] + y_tilde))

    return temps, Y

def euler_modifie(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
):
    """
    Résout numériquement une équation différentielle ordinaire par la méthode d'Euler modifié (méthode d'ordre 2).

    Parameters
    ----------
    fct : Callable
        Fonction définissant l'équation différentielle. Cette fonction doit prendre comme premier argument le temps `t` et comme deuxième argument la variable `x`. Cette fonction doit retourner de même dimension et type que `x`.
    tspan : 1D ArrayLike de taille 2
        Intervalle de temps [t0, tf]
    Y0 : ArrayLike
        Condition initiale à l'instant `t0`. 
        - Un scalaire pour le cas 1D 
        - Un 1D array de dimension (N,) pour le cas à en ND
    nbpas : int
        Nombre de pas de discrétisation de l'intervalle [t0, tf].
        Le pas de temps est donné par h = (tf - t0) / nbpas.

    Returns
    -------
    temps : ndarray de dimension (nbpas +1,)
        Vecteur des pas de temps
    Y : ndarray de dimension (nbpas +1, N)
        Approximation de la solution de l'EDO au pas de temps spécifié par `temps`
        
        - Y[0] = Y
        - Y[k] est l'approximation au temps `temps[k]`

    See Also
    --------
    euler : Méthode d'Euler
    pt_milieu : Méthode du point milieu
    rk4 : Méthode de Runge-Kutta d'ordre 4.

    Examples
    --------

    Exemple 1D

    >>> import numpy as np
    >>> from MTH2210 import euler_modifie
    >>> (t,y) = euler_modifie(lambda t, y: np.cos(t), [0,2], 1, 1000)

    Exemple ND inline

    >>> (t,y) = euler_modifie(lambda t, y: np.array([y[1],-y[0]]), [0,10], [1,0], 1000)

    Exemple ND

    >>> def my_edo(t,z):
    >>>     f = zeros_like(z)
    >>>     f[0] = z[1]
    >>>     f[1] = -z[0]
    >>>     return f
    >>> (t,y) = euler_modifie(my_edo, [0,10], [1,0], 1000)
    """

    Y, temps, h = edo_init(fct, tspan, Y0, nbpas)

    for t in range(nbpas):

        yt      = Y[t]
        y_tilde = yt + h * fct(temps[t], Y[t])

        Y[t+1]  = Y[t] + h/2 * (fct(temps[t], Y[t]) + fct(temps[t] + h, y_tilde))

    return temps, Y

def pt_milieu(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
):
    """
    Résout numériquement une équation différentielle ordinaire par la méthode du point milieu (méthode d'ordre 2).

    Parameters
    ----------
    fct : Callable
        Fonction définissant l'équation différentielle. Cette fonction doit prendre comme premier argument le temps `t` et comme deuxième argument la variable `x`. Cette fonction doit retourner de même dimension et type que `x`.
    tspan : 1D ArrayLike de taille 2
        Intervalle de temps [t0, tf]
    Y0 : ArrayLike
        Condition initiale à l'instant `t0`. 
        - Un scalaire pour le cas 1D 
        - Un 1D array de dimension (N,) pour le cas à en ND
    nbpas : int
        Nombre de pas de discrétisation de l'intervalle [t0, tf].
        Le pas de temps est donné par h = (tf - t0) / nbpas.

    Returns
    -------
    temps : ndarray de dimension (nbpas +1,)
        Vecteur des pas de temps
    Y : ndarray de dimension (nbpas +1, N)
        Approximation de la solution de l'EDO au pas de temps spécifié par `temps`
        
        - Y[0] = Y
        - Y[k] est l'approximation au temps `temps[k]`

    See Also
    --------
    euler : Méthode d'Euler
    euler_modifie : Méthode d'Euler modifié
    rk4 : Méthode de Runge-Kutta d'ordre 4.

    Examples
    --------

    Exemple 1D

    >>> import numpy as np
    >>> from MTH2210 import pt_milieu
    >>> (t,y) = pt_milieu(lambda t, y: np.cos(t), [0,2], 1, 1000)

    Exemple ND inline

    >>> (t,y) = pt_milieu(lambda t, y: np.array([y[1],-y[0]]), [0,10], [1,0], 1000)

    Exemple ND

    >>> def my_edo(t,z):
    >>>     f = zeros_like(z)
    >>>     f[0] = z[1]
    >>>     f[1] = -z[0]
    >>>     return f
    >>> (t,y) = pt_milieu(my_edo, [0,10], [1,0], 1000)
    """

    Y, temps, h = edo_init(fct, tspan, Y0, nbpas)

    for t in range(nbpas):

        yt      = Y[t]
        k1      = h * fct(temps[t], Y[t])

        Y[t+1]  = Y[t] + h * fct(temps[t] + h/2, Y[t] + k1/2)

    return temps, Y


def rk4(
    fct: Callable[[float, ArrayFloat], Union[float, ArrayFloat]],
    tspan: npt.ArrayLike,
    Y0: npt.ArrayLike,
    nbpas: int,
):
    """
    Résout numériquement une équation différentielle ordinaire par la méthode de Runge-Kutta d'ordre 4.


    Parameters
    ----------
    f : Callable
        Fonction définissant l'équation différentielle. Cette fonction doit prendre comme premier argument le temps `t` et comme deuxième argument la variable `x`. Cette fonction doit retourner de même dimension et type que `x`.
    tspan : 1D ArrayLike de taille 2
        Intervalle de temps [t0, tf]
    Y0 : ArrayLike
        Condition initiale à l'instant `t0`. 
        - Un scalaire pour le cas 1D 
        - Un 1D array de dimension (N,) pour le cas à en ND
    nbpas : int
        Nombre de pas de discrétisation de l'intervalle [t0, tf].
        Le pas de temps est donné par h = (tf - t0) / nbpas.

    Returns
    -------
    temps : ndarray de dimension (nbpas +1,)
        Vecteur des pas de temps
    Y : ndarray de dimension (nbpas +1, N)
        Approximation de la solution de l'EDO au pas de temps spécifié par `temps`
        
        - Y[0] = Y
        - Y[k] est l'approximation au temps `temps[k]`

    See Also
    --------
    euler : Méthode d'Euler
    euler_modifie : Méthode d'Euler modifié
    pt_milieu : Méthode du point milieu

    Examples
    --------

    Exemple 1D

    >>> import numpy as np
    >>> from MTH2210 import rk4
    >>> (t,y) = rk4(lambda t, y: np.cos(t), [0,2], 1, 1000)

    Exemple ND inline

    >>> (t,y) = rk4(lambda t, y: np.array([y[1],-y[0]]), [0,10], [1,0], 1000)

    Exemple ND

    >>> def my_edo(t,z):
    >>>     f = zeros_like(z)
    >>>     f[0] = z[1]
    >>>     f[1] = -z[0]
    >>>     return f
    >>> (t,y) = rk4(my_edo, [0,10], [1,0], 1000)
    """

    Y, temps, h = edo_init(fct, tspan, Y0, nbpas)

    for t in range(nbpas):

        k1 = h * fct(temps[t], Y[t])
        k2 = h * fct(temps[t] + h/2 , Y[t] + k1/2)
        k3 = h * fct(temps[t] + h/2 , Y[t] + k2/2)
        k4 = h * fct(temps[t] + h , Y[t] + k3)

        Y[t+1] = Y[t] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return temps, Y