import numpy as np
from numpy.linalg import norm
from scipy.ndimage import laplace

def order_computation(erreur, ratio_h, tol=0.2):

    ordre_app			=	np.log(erreur[:-1]/erreur[1:])/np.log(ratio_h)
    stable_region		=	(ordre_app>0) & (np.abs(np.gradient(ordre_app))<tol) & (np.abs(laplace(ordre_app, mode='nearest'))<2*tol)
    ind_stable_region	=	np.where(stable_region)
	
	# Sanity check
    if len(ind_stable_region) ==0:
        print("Il n'y a pas de zone asymptotique")
    elif len(ind_stable_region) < 2:
        print("La zone asymptotique n'est pas tres grande")
    elif np.any(np.gradient[ind_stable_region]!=1):
        print("La zone asymptotique est brisee")
	
    ordre = np.mean(ordre_app[ind_stable_region])

    return ordre, ordre_app