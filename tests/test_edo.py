import pytest

import numpy as np
from numpy.linalg import norm

from MTH2210 import euler, euler_modifie, pt_milieu, rk4
from tests.utils import order_computation

# Intégration exacte pour soution polynomiale de degré 1
@pytest.mark.parametrize("algo", [euler, euler_modifie, pt_milieu, rk4])
def test_exact_degree1(algo):
    fct = lambda t,y: 3
    y0 = 1
    tspan = [0,5]
    y_ex = lambda t: 3*t + 1
    (temps, y) = algo(fct, tspan, y0, 100)

    y_ex_t = y_ex(temps)

    assert norm(y_ex_t - y[:,0], np.inf)/norm(y_ex_t, np.inf) < 1e-12

# Intégration exacte pour soution polynomiale de degré 2
@pytest.mark.parametrize("algo", [euler_modifie, pt_milieu, rk4])
def test_exact_degree2(algo):
    fct = lambda t,y: -4*t
    y0 = 1
    tspan = [0,5]
    y_ex = lambda t: -2*t**2 + 1
    (temps, y) = algo(fct, tspan, y0, 100)

    y_ex_t = y_ex(temps)

    assert norm(y_ex_t - y[:,0], np.inf)/norm(y_ex_t, np.inf) < 1e-12

# Intégration exacte pour soution polynomiale de degré 4
@pytest.mark.parametrize("algo", [rk4])
def test_exact_degree4(algo):
    fct = lambda t,y: 5*t**3
    y0 = 1
    tspan = [0,5]
    y_ex = lambda t: 5/4 * t**4 + 1
    (temps, y) = algo(fct, tspan, y0, 100)

    y_ex_t = y_ex(temps)

    assert norm(y_ex_t - y[:,0], np.inf)/norm(y_ex_t, np.inf) < 1e-12


# Vérification de l'ordre de convergence pour une EDO scalaire
@pytest.mark.parametrize("algo, ordre_expected", [(euler, 1), (euler_modifie, 2), (pt_milieu, 2) , (rk4, 4)])
def test_order_scalar(algo, ordre_expected):

    fct = lambda t,y: 2*y - t + 4
    y0 = 1
    tspan = [0,5]
    y_ex = lambda t: -7/4 + 1/2*t + 11/4*np.exp(2*t)
    
    nb_eval = 8
    nb_pas_init = 100
    nb_pas = nb_pas_init * 2**np.arange(0,nb_eval)
    erreur_abs = np.nan * np.ones(nb_eval)

    for t in range(nb_eval):
        (temps, y) = algo(fct, tspan, y0, nb_pas[t])
        erreur_abs[t] = norm(y_ex(temps) - y[:,0], np.inf)

    tol = 0.2
    (ordre, ordre_app) = order_computation(erreur_abs, 2, tol)

    assert np.abs(ordre - ordre_expected) < tol

# Vérification de l'ordre de convergence pour systèmes d'EDOs
@pytest.mark.parametrize("algo, ordre_expected", [(euler, 1), (euler_modifie, 2), (pt_milieu, 2) , (rk4, 4)])
def test_order_system(algo, ordre_expected):

    fct = lambda t,y: np.array([[-2,1],[1,-2]]) @ y + np.array([2*np.exp(-t),3*t])
    y0 = [2,3]
    tspan = [0,5]
    y_ex = lambda t: -7/6*np.array([1,-1]) * (np.exp(-3*t)[:, np.newaxis]) + 4*np.array([1,1]) * (np.exp(-t)[:, np.newaxis]) + \
						1/2*np.array([1,-1]) * (np.exp(-t)[:, np.newaxis]) + np.array([1,1]) * ((t*np.exp(-t))[:, np.newaxis]) + np.array([1,2])*(t[:, np.newaxis]) -1/3*np.array([4,5])
    
    nb_eval = 8
    nb_pas_init = 100
    nb_pas = nb_pas_init * 2**np.arange(0,nb_eval)
    erreur_abs = np.nan * np.ones(nb_eval)

    for t in range(nb_eval):
        (temps, y) = algo(fct, tspan, y0, nb_pas[t])
        erreur_abs[t] = norm(y_ex(temps) - y, np.inf)


    tol = 0.2
    (ordre, ordre_app) = order_computation(erreur_abs, 2, tol)

    assert np.abs(ordre - ordre_expected) < tol


