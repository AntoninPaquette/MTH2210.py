(EDO)=
# Résolution d'équations différentielles ordinaires

Cette section est dédiée à la résolution numérique d'équations ordinaires de forme:

$$
\frac{dY}{dt}(t) = F(t,Y(t)) \quad \text{avec} \quad Y(t_0) = Y_0.
$$

$m=2\,\text{kg}$ et de constante de rappel $k=3\,\text{N/m}$.


Les algorithmes disponibles sont:
1. Euler explicite: {py:func}`MTH2210.EDO.euler.euler`,
2. Runge-Kutta d'ordre 4: {py:func}`MTH2210.EDO.rk4.rk4`.

(EDO-exemple)= 
## Exemple de résolution d'un système d'équations différentielles ordinaires
Soit l'équation différentielle ordinaire d'ordre 2 décrivant un système masse-ressort
de masse $m=2\,\text{kg}$ et de constante de rappel $k=3\,\text{N/m}$:

\begin{align*}
\frac{d^2 y}{dt^2}(t) = -\frac{k}{m} y(t) \quad \text{avec} \quad y(1) = 5 \text{ et } \frac{d y}{dt}(1) =0.
\end{align*}

On s'intéresse à la solution entre $t_0=1\,\text{s}$ et $t_f = 4\,\text{s}$. On doit tout d'abord effectuer un changement de variables:

\begin{align*}
z_1(t) &= y(t)\\ 
z_2(t) &= \frac{d y}{dt}(t)
\end{align*}

afin de transformer l'EDO d'ordre 2 en ce système d'EDOs d'ordre 1:

\begin{align*}
\frac{d z_1}{dt}(t) &= z_2(t)\\ 
\frac{d z_2}{dt}(t) &= -\frac{k}{m}z_1(t)
\end{align*}

On peut appliquer la méthode d'Euler explicite afin de résoudre ce système d'EDOs.
Il faut donc créer la fonction $F:\mathbb{R} \times \mathbb{R}^2 \to \mathbb{R}^2$:
<!--
```@example 1
function my_edo(t,z)
    m = 2.
    k = 3.
    f = zeros(length(z))
    f[1] = z[2]
    f[2] = -k/m*z[1]
    return f
end
nothing # hide
```

Il faut ensuite appeler la fonction [`euler`](@ref):

```@example 1
using MTH2210
using Plots

t0 = 1.
tf = 4.
tspan = [t0,tf]
z0 = [5.,0.]
nbpas = 1000
(temps, z) = euler(my_edo, tspan , z0 , nbpas)
nothing # hide
```

Les résultats peuvent ensuite être affichés dans un graphique.

```@example 1
plot(temps,z[1,:],label="y(t)")
plot!(temps,z[2,:],label="y'(t)",xlabel="temps [s]",title="Approximation avec Euler exp.")
plot!([],[],label="",size=(400,300)); savefig("edo-plot.png"); nothing # hide
```

![Solutions numériques du systèmes d'EDOs](edo-plot.png) -->
