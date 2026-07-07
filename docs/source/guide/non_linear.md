(non_linear)= 
# Résolution de problèmes non-linéaires

Cette section est dédiée à la résolution de problèmes non-linéaires. Deux types de problèmes non-linéaires sont étudiés
1. Problème de racines 
2. Problème de point-fixe

Le problème de recherche de racine $r \in \mathbb{R}^n$ d'une fonction $F:\mathbb{R}^n \to \mathbb{R}^n$ consiste à trouver $r$ tel que
\begin{align*}
    F(r) = 0.
\end{align*}

Les algorithmes disponibles pour trouver les racines d'une fonction $F$ sont:
1. Bissection pour $n=1$ : {py:func}`MTH2210.bissection`,
2. Sécante pour $n=1$ : {py:func}`MTH2210.secante`,
3. Newton avec dérivée pour $n=1$ : {py:func}`MTH2210.newton_1d`,
4. Newton avec dérivée pour $n\geq 1$ : {py:func}`MTH2210.newton_nd_avec_der`,
5. Newton sans dérivée pour $n\geq 1$ : {py:func}`MTH2210.newton_nd`.

Le deuxième type de problème à résoudre est le problème de recherche d'un point
fixe $z \in \mathbb{R}$ d'une fonction $g:\mathbb{R} \to \mathbb{R}$ tel que
\begin{align*}
    g(z) = z.
\end{align*}
 
L'algorithme disponible pour trouver les points-fixes d'une fonction $g$ est:
1. Point-fixe pour $n=1$ : {py:func}`MTH2210.point_fixe`


## Exemple de résolution d'une équation non-linéaire

On cherche à calculer une approximation de $\sqrt{10}$ en calculant la racine
positive de $f(x) = x^2 - 10$. On définit tout d'abord la fonction $f$:

```{eval-rst}
.. jupyter-execute::
    
    import numpy as np
    import matplotlib.pyplot as plt

    from MTH2210 import bissection, secante, newton_1d, point_fixe

    def my_fct_nl(x):
        f = x**2 - 10
        return f

    def my_dfct_nl(x):
        df = 2*x
        return df
```

 

On appelle ensuite les fonctions {py:func}`MTH2210.bissection`, {py:func}`MTH2210.secante` et
{py:func}`MTH2210.newton_1d` afin de résoudre ce problème. On choisit ``x_0=2.5`` et
``x_1=4`` de sorte que $f(x_0)f(x_1)<0$ et une tolérance sur l'erreur relative
de ``tol=10e-9``.

```{eval-rst}
.. jupyter-execute::
    
    x0 = 2.5
    x1 = 4.
    tol = 1e-9

    (approx_bis , f_bis) = bissection(my_fct_nl , x0 , x1 , 100 , tol)
    (approx_sec , f_sec, df_sec) = secante(my_fct_nl , x0 , x1 , 50 , tol)
    (approx_new , f_new, df_new) = newton_1d(my_fct_nl , my_dfct_nl , x0 , 20 , tol)
```

La méthode des points-fixes peut aussi être employée pour approximer
$\sqrt{10}$. On considère la fonction $g(x) = -\frac{x^2}{10} + x+ 1$ dont
un point-fixe attractif est $\sqrt{10}$

<!-- ```{eval-rst}
.. jupyter-execute::
    
    def fct_g(x):
        g = -x**2/10 + x + 1
        return g

    (approx_fixe , err_fixe) = point_fixe(fct_g , x0 , 50 , 1e-9)
``` -->


On peut ensuite afficher l'évolution des erreurs selon l'itération. 

```{eval-rst}
.. jupyter-execute::

    err_bis = np.abs(approx_bis - np.sqrt(10))
    err_sec = np.abs(approx_sec - np.sqrt(10))
    err_newton = np.abs(approx_new - np.sqrt(10))

    fig, ax = plt.subplots(1)
    ax.plot(err_bis,label="Bissection")
    ax.plot(err_sec,label="Sécante")
    ax.plot(err_newton,label="Newton")

    ax.set_yscale('log')
    ax.set_xlabel("Nb itérations")
    ax.set_ylabel("Erreur absolue")
    fig.legend()
```

Les tableaux des ratios des erreurs peuvent aussi être produits pour les
méthodes des points-fixes, de la sécante et de Newton:
<!--
```@example 1
ratio_fixe_1 = err_fixe[2:end] ./ err_fixe[1:end-1]
ratio_fixe_a = err_fixe[2:end] ./ err_fixe[1:end-1] .^ ((1+sqrt(5))/2)
ratio_fixe_2 = err_fixe[2:end] ./ err_fixe[1:end-1] .^ 2

ratio_sec_1 = err_sec[2:end] ./ err_sec[1:end-1]
ratio_sec_a = err_sec[2:end] ./ err_sec[1:end-1] .^ ((1+sqrt(5))/2)
ratio_sec_2 = err_sec[2:end] ./ err_sec[1:end-1] .^ 2

ratio_new_1 = err_new[2:end] ./ err_new[1:end-1]
ratio_new_a = err_new[2:end] ./ err_new[1:end-1] .^ ((1+sqrt(5))/2)
ratio_new_2 = err_new[2:end] ./ err_new[1:end-1] .^ 2

@printf("Ratio des erreurs pour points-fixes\n")
@printf("e_{n+1}/e_{n}           e_{n+1}/e_{n}^a         e_{n+1}/e_{n}^2\n")
for t=1:length(ratio_fixe_1)
    @printf("%16.15e   %16.15e   %16.15e\n", ratio_fixe_1[t] , ratio_fixe_a[t] , ratio_fixe_2[t])
end
@printf("\n\nRatio des erreurs pour la sécante\n")
@printf("e_{n+1}/e_{n}           e_{n+1}/e_{n}^a         e_{n+1}/e_{n}^2\n")
for t=1:length(ratio_sec_1)
    @printf("%16.15e   %16.15e   %16.15e\n", ratio_sec_1[t] , ratio_sec_a[t] , ratio_sec_2[t])
end
@printf("\n\nRatio des erreurs pour Newton\n")
@printf("e_{n+1}/e_{n}           e_{n+1}/e_{n}^a         e_{n+1}/e_{n}^2\n")
for t=1:length(ratio_new_1)
    @printf("%16.15e   %16.15e   %16.15e\n" , ratio_new_1[t] , ratio_new_a[t] , ratio_new_2[t])
end
```

On constate, tel qu'attendu, que la méthode de la sécante converge au nombre
d'or ``\frac{1+ \sqrt{5}}{2}``, que la méthode de Newton converge à l'ordre 2
et que la méthode des points-fixes converge à l'ordre 1 et à un taux de
convergence de ``-\frac{2\sqrt{10}}{10}+1``. -->
