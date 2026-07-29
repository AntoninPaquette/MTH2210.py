import pytest

import numpy as np
from numpy.linalg import norm

from MTH2210 import lagrange, splinec
from tests.utils import order_computation

@pytest.mark.parametrize("poly, degree", [(lambda x: (x-1)*(x+2.5)*(x-np.pi)*(x+11), 4), (lambda x: (x-2)**2 * (x+5)**3 * (x-np.exp(1))**4, 11)])
def test_lagrange_exact(poly, degree):

    xi = np.linspace(-5, 5, degree+1)
    yi = poly(xi)
    x = np.linspace(np.min(xi), np.max(xi), 1000)
    y_lagrange = lagrange(xi, yi, x)
    y_ex = poly(x)

    err_rel = norm(y_ex - y_lagrange) / norm(y_ex)

    assert err_rel < 1e-14

@pytest.mark.parametrize("degree", np.arange(5)+1)
def test_lagrange_order(degree):
    fct = lambda x: np.exp(2*x)
    nb_pts = degree + 1
    nb_loop = 12

    x_interest = 1/3 ** nb_loop

    erreur = np.nan * np.ones(nb_loop)
    y_ex = fct(x_interest)

    for t in range(nb_loop):
        a = 0 
        b = 0.5**(t-1)
        k = np.arange(1, nb_pts+1)
        xi = 1/2*(a+b) + 1/2*(b-a)*np.cos((2*k-1)*np.pi/(2*nb_pts))
        yi = fct(xi)

        y_lagrange = lagrange(xi, yi, x_interest)

        erreur[t] = np.abs(y_ex - y_lagrange)[0]

    tol = 0.2
    [ordre, ordre_app] = order_computation(erreur, 2, tol)

    assert np.abs(ordre - (degree+1)) < tol

def spline_example(x):
    if np.min(x)<0 or np.max(x)>4:
        raise Exception("Pas dans le domaine de la fonction")

    x = np.asarray(x)

    px = ((x>=0) & (x<1)) * x**2 + \
         ((x>=1) & (x<3)) * (-x**3 + 4*x**2 - 3*x + 1) + \
		 ((x>=3) & (x<=4)) * (-5*x**2 + 24*x - 26)

    return px


spline_combination = [([2,2], [2,-10]), ([2,3], [2,np.nan]), ([2,4], [2,-16]),
                      ([3,2], [np.nan, -10]), ([3,3], [np.nan, np.nan]), ([3,4], [np.nan, -16]),
                      ([4,2], [0, -10]), ([4,3], [0, np.nan]), ([4,4], [0, -16])]

@pytest.mark.parametrize("spline_type, spline_val", spline_combination)
def test_spline_exact(spline_type, spline_val):
    xi = [0,1,3,4]
    yi = spline_example(xi)
    x  = np.linspace(1,4, 1000)
    y_exacte = spline_example(x)
        
    Sx = splinec(xi , yi, x, spline_type, spline_val)
			
    err_rel	=	norm(y_exacte - Sx)/norm(y_exacte)
    assert err_rel < 1e-14

def test_spline_courbure_prescrite():
    fct = lambda x: -4*x**3 + np.pi*x**2 + 11*x - np.exp(1)
    d2fct = lambda x: -24*x + 2*np.pi
    a = np.pi/12
    b = 4

    nb_pts = 10
    xi = np.linspace(a, b, nb_pts)
    yi = fct(xi)

    x = np.linspace(a, b, 1000)

    y_ex = fct(x)

    Sx = splinec(xi, yi, x, [1,2], [np.nan, d2fct(b)])

    err_rel = norm(y_ex - Sx)/norm(y_ex)

    assert err_rel < 1e-14

def test_spline_pente_courbure_prescrite():
    fct = lambda x: -4*x**3 + np.pi*x**2 + 11*x - np.exp(1)
    dfct = lambda x: -12*x**2 + 2*np.pi*x + 11

    a = -10
    b = np.pi/12

    nb_pts = 10
    xi = np.linspace(a, b, nb_pts)
    yi = fct(xi)

    x = np.linspace(a, b, 1000)

    y_ex = fct(x)

    Sx = splinec(xi, yi, x, [4, 1], [dfct(a), np.nan])

    err_rel = norm(y_ex - Sx)/norm(y_ex)

    assert err_rel < 1e-14

