import numpy as np
import pytest


@pytest.fixture
def ode_integrators():
    return [
        algo(euler, 1),
        algo(eulermod, 2),
        algo(ptmilieu, 2),
        algo(rk4, 4),
    ]


@pytest.fixture
def ode_problems():
    tspan = [0.0, 5.0]

    fct_degre1 = ode_prob(lambda t, y: 3.0, 1.0, tspan,
                          lambda t: 3*t + 1, 1)

    fct_degre2 = ode_prob(lambda t, y: -4*t, 1.0, tspan,
                          lambda t: -2*t**2 + 1, 2)

    fct_degre4 = ode_prob(lambda t, y: 5*t**3, 1.0, tspan,
                          lambda t: 5/4*t**4 + 1, 4)

    fct_scalar = ode_prob(
        lambda t, y: 2*y - t + 4,
        1.0,
        tspan,
        lambda t: -7/4 + 0.5*t + 11/4*np.exp(2*t),
        np.inf,
    )

    fct_system = ode_prob(
        lambda t, y: np.array([[-2, 1], [1, -2]]) @ y
                     + np.array([2*np.exp(-t), 3*t]),
        np.array([2.0, 3.0]),
        tspan,
        lambda t: (
            -7/6*np.array([1, -1])*np.exp(-3*t)
            + 4*np.array([1, 1])*np.exp(-t)
            + 0.5*np.array([1, -1])*np.exp(-t)
            + np.array([1, 1])*t*np.exp(-t)
            + np.array([1, 2])*t
            - (1/3)*np.array([4, 5])
        ),
        np.inf,
    )

    return {
        "poly": (fct_degre1, fct_degre2, fct_degre4),
        "scalar": fct_scalar,
        "system": fct_system,
    }


# ------------------------------------------------------------------
# Exact integration
# ------------------------------------------------------------------
@pytest.mark.parametrize("ode_int", lambda ode_integrators: ode_integrators)
def test_exact_integration(ode_integrators, ode_problems):
    for ode_int in ode_integrators:
        for prob in ode_problems["poly"]:
            if prob.degre <= ode_int.order:
                temps, y = ode_int.name(prob.fct, prob.tspan, prob.y0, 100)

                sol_ex = np.array([prob.ex_sol(t) for t in temps])

                # flatten Y if scalar
                y_flat = y.reshape(-1)

                erreur_rel = (
                    np.linalg.norm(y_flat - sol_ex, ord=np.inf)
                    / np.linalg.norm(sol_ex, ord=np.inf)
                )

                assert erreur_rel < 1e-14


# ------------------------------------------------------------------
# Order of convergence (scalar)
# ------------------------------------------------------------------
@pytest.mark.parametrize("ode_int", lambda ode_integrators: ode_integrators)
def test_order_scalar(ode_integrators, ode_problems):
    tol = 0.2
    nb_eval = 6
    nb_pas_init = 100

    for ode_int in ode_integrators:
        nb_pas = (2 ** np.arange(nb_eval)) * nb_pas_init
        erreur = np.full(nb_eval, np.inf)

        for i, n in enumerate(nb_pas):
            temps, y = ode_int.name(
                ode_problems["scalar"].fct,
                ode_problems["scalar"].tspan,
                ode_problems["scalar"].y0,
                int(n),
            )

            sol_ex = np.array([ode_problems["scalar"].ex_sol(t) for t in temps])
            erreur[i] = np.linalg.norm(y.reshape(-1) - sol_ex, ord=np.inf)

        ordre_app, _ = order_computation(erreur, 2, tol)

        assert abs(ordre_app - ode_int.order) < tol


# ------------------------------------------------------------------
# Order of convergence (system)
# ------------------------------------------------------------------
@pytest.mark.parametrize("ode_int", lambda ode_integrators: ode_integrators)
def test_order_system(ode_integrators, ode_problems):
    tol = 0.2
    nb_eval = 6
    nb_pas_init = 100

    for ode_int in ode_integrators:
        nb_pas = (2 ** np.arange(nb_eval)) * nb_pas_init
        erreur = np.full(nb_eval, np.inf)

        for i, n in enumerate(nb_pas):
            temps, y = ode_int.name(
                ode_problems["system"].fct,
                ode_problems["system"].tspan,
                ode_problems["system"].y0,
                int(n),
            )

            sol_ex = np.vstack([ode_problems["system"].ex_sol(t) for t in temps])

            erreur[i] = np.linalg.norm(y - sol_ex, ord=np.inf)

        ordre_app, _ = order_computation(erreur, 2, tol)

        assert abs(ordre_app - ode_int.order) < tol


# ------------------------------------------------------------------
# Shape tests
# ------------------------------------------------------------------
@pytest.mark.parametrize("ode_int", lambda ode_integrators: ode_integrators)
def test_shapes(ode_integrators, ode_problems):
    nb_pas = 17

    for ode_int in ode_integrators:
        # scalar case
        temps, y = ode_int.name(
            ode_problems["scalar"].fct,
            ode_problems["scalar"].tspan,
            ode_problems["scalar"].y0,
            nb_pas,
        )

        assert len(temps) == nb_pas + 1
        assert y.shape == (nb_pas + 1, 1)

        # system case
        temps, y = ode_int.name(
            ode_problems["system"].fct,
            ode_problems["system"].tspan,
            ode_problems["system"].y0,
            nb_pas,
        )

        assert y.shape == (nb_pas + 1, 2)