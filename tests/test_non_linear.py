import pytest

import numpy as np
from numpy.linalg import norm
from scipy.ndimage import laplace

from MTH2210 import bissection, secante, ptfixe, newton1d, newtonNd
from tests.utils import order_computation


def order_computation_nl(erreur, tol=0.2):

    ordre_app			=	np.log(erreur[1:-1]/erreur[2:])/np.log(erreur[:-2]/erreur[1:-1])
    stable_region		=	(ordre_app>0) & (np.abs(np.gradient(ordre_app))<tol) & (np.abs(laplace(ordre_app, mode='nearest'))<2*tol)
    ind_stable_region	=	np.where(stable_region)

    
	# Sanity check
    if len(ind_stable_region) == 0:
        print("Il n'y a pas de zone asymptotique")
    elif len(ind_stable_region) < 2:
        print("La zone asymptotique n'est pas tres grande")
    elif np.any(np.gradient[ind_stable_region]!=1):
        print("La zone asymptotique est brisee")
	
    ordre = np.mean(ordre_app[ind_stable_region])

    return ordre, ordre_app

def order_computation_bissec(erreur):
    # Least-square fit
    iterations = range(len(erreur))

    # convert necessary because .fit scale the data 
    poly_fit = np.polynomial.polynomial.Polynomial.fit(iterations,np.log2(erreur),deg=1).convert()

    return -poly_fit.coef[-1]

def taux_computation_nl(erreur, ordre, tol=0.2):

    taux_app			=	erreur[1:] / (erreur[:-1]**ordre)	
    stable_region		=	(taux_app>0) & (abs(np.gradient(taux_app))<tol) & (np.abs(laplace(taux_app, mode='nearest'))<2*tol)
    ind_stable_region	=	np.where(stable_region)

	# Sanity check
    if len(ind_stable_region) ==0:
        print("Il n'y a pas de zone asymptotique")
    elif len(ind_stable_region) < 2:
        print("La zone asymptotique n'est pas tres grande")
    elif np.any(np.gradient[ind_stable_region]!=1):
        print("La zone asymptotique est brisee")
	
    taux = np.mean(taux_app[ind_stable_region])

    return taux, taux_app

fcts = [lambda x : x**2 - 10, lambda x: np.exp(x) - x**3  - (np.exp(np.pi) - np.pi**3 )]
dfcts = [lambda x: 2*x, lambda x: np.exp(x) - 3*x**2 ]
xleft = [2, 2.5]
xright = [4.5, 3.75]
racines = [np.sqrt(10), np.pi]

g = [lambda x: -x**2/10 + x + 1, lambda x: -x**2/6 + x + 9/6]
pt_fixe = [np.sqrt(10),3]
x0_g = [1,1]
ordre_g = [1,2]
taux_g = [-2*np.sqrt(10)/10 + 1, 1/6]

@pytest.mark.parametrize("fct, xleft, xright, racine", list(zip(fcts, xleft, xright, racines)))
def test_bissect(fct, xleft, xright, racine):

    [app, err] = bissection(fct, xleft, xright, 200, 1e-12)

    assert np.abs(racine - app[-1])/np.abs(racine) < 1e-11

    tol = 0.2
    ordre = order_computation_bissec(err[:-1])

    assert np.abs(ordre - 1) < tol

@pytest.mark.parametrize("fct, xleft, xright, racine", list(zip(fcts, xleft, xright, racines)))
def test_secante(fct, xleft, xright, racine):

    [app, err] = secante(fct, xleft, xright, 20, 1e-14)

    assert np.abs(racine - app[-1])/np.abs(racine) < 1e-12

    tol = 0.4
    (ordre, ordre_app) = order_computation_nl(err[:-1], tol)
    
    assert np.abs(ordre - (1+np.sqrt(5))/2) < 0.1

@pytest.mark.parametrize("fct, dfct, xleft, racine", list(zip(fcts, dfcts, xleft, racines)))
def test_newton1d(fct, dfct, xleft, racine):

    [app, err] = newton1d(fct, dfct, xleft, 20, 1e-14)

    assert np.abs(racine - app[-1])/np.abs(racine) < 1e-12

    tol = 0.4
    # (ordre, ordre_app) = order_computation_nl(err[:-1], tol)
    
    # assert np.abs(ordre - 2) < 0.1

@pytest.mark.parametrize("g, x0_g, pt_fixe, ordre_g, taux_g", list(zip(g, x0_g, pt_fixe, ordre_g, taux_g)))
def test_ptfixe(g, x0_g, pt_fixe, ordre_g, taux_g):

    [app, err] = ptfixe(g, x0_g, 100, 1e-12)

    assert np.abs(pt_fixe - app[-1]) < 1e-11

    (ordre, ordre_app) = order_computation_nl(err[:-1], 0.4)

    assert np.abs(ordre - ordre_g) < 0.1

    [taux, taux_app] = taux_computation_nl(err[:-1], ordre_g, 0.1)

    assert np.abs(taux-taux_g) < 0.1

def test_newton1d_multiple_roots():

    fct = lambda x: x*np.sin(x)**2
    dfct = lambda x: np.sin(x)**2 + 2*x*np.sin(x)*np.cos(x)

    (app1, err1) = newton1d(fct, dfct, 1, 200, 1e-12)
    (app2, err2) = newton1d(fct, dfct, 3, 200, 1e-12)

    assert np.abs(app1[-1]) < 1e-10
    assert np.abs(app2[-1]-np.pi) < 1e-10

    (ordre1, ordre_app1) = order_computation_nl(err1[:-1])
    (ordre2, ordre_app2) = order_computation_nl(err2[:-1])

    assert np.abs(ordre1 - 1) < 0.1
    assert np.abs(ordre2 - 1) < 0.1

    (taux1, taux_app1) = taux_computation_nl(err1[:-1], 1)
    (taux2, taux_app2) = taux_computation_nl(err2[:-1], 1)

    assert np.abs(taux1 - (3-1)/3) < 0.1
    assert np.abs(taux2 - (2-1)/2) < 0.1


def test_newtonNd_with_derivative():
    fct = lambda x: np.array([5*np.sin(0.1*x[0]*x[1]) - x[2] - (5*np.sin(-0.2)-5),
                              x[0]**2 + x[1]**2 + x[2]**2 - 30,
                              x[0] - x[1] - x[2] + 2])

    jac_fct = lambda x: np.array([[5*0.1*x[1]*np.cos(0.1*x[0]*x[1]) , 5*0.1*x[0]*np.cos(0.1*x[0]*x[1]) , -1 ],
                                  [2*x[0] , 2*x[1] , 2*x[2] ],
                                  [1, -1, -1]])

    racine_syst	= [1, -2, 5]		  
    x0_syst	=	[1.5, -3, 4]

    (app, err) = newtonNd(fct, x0_syst, 20, 1e-12, dfct=jac_fct)

    assert norm(racine_syst - app[-1])/norm(racine_syst) < 1e-11

    (ordre, ordre_app) = order_computation_nl(err[:-1])

    assert np.abs(ordre - 2) < 0.1

def test_newtonNd_without_derivative():
    fct = lambda x: np.array([5*np.sin(0.1*x[0]*x[1]) - x[2] - (5*np.sin(-0.2)-5),
                              x[0]**2 + x[1]**2 + x[2]**2 - 30,
                              x[0] - x[1] - x[2] + 2])

    racine_syst	= [1, -2, 5]		  
    x0_syst	=	[1.5, -3, 4]

    (app, err) = newtonNd(fct, x0_syst, 20, 1e-12, h=1e-3)

    assert norm(racine_syst - app[-1])/norm(racine_syst) < 1e-11

    (ordre, ordre_app) = order_computation_nl(err[:-1])

    assert np.abs(ordre - 2) < 0.1