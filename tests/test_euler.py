import numpy as np
import pytest

from MTH2210.EDO.euler import euler


# ============================================================
# 1. Test scalaire exact : dx/dt = 1  => x(t) = x0 + t
# Euler est exact ici.
# ============================================================
def test_euler_affine_exact():
    f = lambda x, t: 1.0
    x0 = 2.5
    t0, tm = 0.0, 1.0
    m = 20

    list_x, list_t = euler(f, x0, t0, tm, m)

    expected = x0 + np.asarray(list_t)
    assert np.allclose(np.asarray(list_x), expected, atol=1e-12, rtol=0.0)


# ============================================================
# 2. Test scalaire exact : dx/dt = 0  => x(t) = constante
# ============================================================
def test_euler_constant_exact():
    f = lambda x, t: 0.0
    x0 = -3.0

    list_x, list_t = euler(f, x0, 0.0, 10.0, 50)

    assert len(list_x) == len(list_t) == 51
    assert np.allclose(np.asarray(list_x), x0, atol=1e-12, rtol=0.0)


# ============================================================
# 3. Test précision scalaire : dx/dt = x  => x(t) = x0 * exp(t)
# Euler n'est pas exact -> on utilise une tolérance plus large.
# ============================================================
def test_euler_exponential_reasonable_accuracy():
    f = lambda x, t: x
    x0 = 1.0
    t0, tm = 0.0, 1.0
    m = 2000  # grand m pour réduire l'erreur d'Euler

    list_x, list_t = euler(f, x0, t0, tm, m)

    t = np.asarray(list_t)
    x = np.asarray(list_x)
    expected = x0 * np.exp(t)

    # tolérance relativement permissive (Euler ordre 1)
    assert np.allclose(x, expected, rtol=5e-3, atol=5e-3)


# ============================================================
# 4. Test vectoriel : rotation faible sur un petit temps
# x' = [x2, -x1] ; solution exacte : rotation
# Euler peut dériver, donc on reste sur tm petit et m grand.
# ============================================================
def test_euler_vector_rotation_small_time():
    f = lambda x, t: np.array([x[1], -x[0]])
    x0 = np.array([1.0, 0.0])
    t0, tm = 0.0, 0.1
    m = 5000

    list_x, list_t = euler(f, x0, t0, tm, m)

    final = np.asarray(list_x[-1])
    # solution exacte: [cos(tm), -sin(tm)] pour x0=[1,0]
    expected = np.array([np.cos(tm), -np.sin(tm)])

    # Euler reste approximatif -> tolérance modérée
    assert np.allclose(final, expected, rtol=2e-2, atol=2e-2)


# ============================================================
# 5. Cohérence des longueurs et valeurs initiales
# ============================================================
@pytest.mark.parametrize(
    "x0",
    [0.0, 1.2, np.array([0.0, 1.0, -2.0])]
)
def test_euler_output_length_and_initial_value(x0):
    f = lambda x, t: x if not isinstance(x, np.ndarray) else np.ones_like(x)
    t0, tm = 0.0, 2.0
    m = 10

    list_x, list_t = euler(f, x0, t0, tm, m)

    assert len(list_x) == m + 1
    assert len(list_t) == m + 1

    # première valeur = condition initiale
    if isinstance(x0, np.ndarray):
        assert np.allclose(list_x[0], x0)
    else:
        assert list_x[0] == x0

    assert list_t[0] == t0
    assert list_t[-1] == pytest.approx(tm)


# ============================================================
# 6. m négatif doit provoquer une erreur
# ============================================================
def test_euler_negative_steps_raises():
    f = lambda x, t: 1.0
    with pytest.raises(ValueError):
        euler(f, 0.0, 0.0, 1.0, -10)


# ============================================================
# 7. f retourne un mauvais type -> doit échouer
# (selon vos checks, c'est souvent ValueError)
# ============================================================
def test_euler_bad_return_type_raises():
    def f(x, t):
        return "not a number"

    with pytest.raises(Exception):
        euler(f, 0.0, 0.0, 1.0, 10)