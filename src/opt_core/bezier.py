"""Differentiable Bezier curve operations using JAX.

All functions are JIT-compiled and fully differentiable, supporting
automatic differentiation through JAX's transformation system.

A Bezier curve of degree n is defined by (n+1) control points:
    C(t) = sum_{i=0}^{n} B_{i,n}(t) * P_i,  t in [0, 1]
where B_{i,n}(t) is the Bernstein basis polynomial.

In this package, control points have 4 dimensions: (x, y, z, angle),
where the first 3 are spatial coordinates and the 4th encodes an
in-plane rotation angle for the slicing plane.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple


@jit
def _binom(n: int, k: jnp.ndarray) -> jnp.ndarray:
    """Compute binomial coefficient C(n, k) via log-gamma for stability.

    Args:
        n: Total count (scalar).
        k: Selection count (array).

    Returns:
        Binomial coefficients with same shape as k.
    """
    return jnp.exp(
        jax.scipy.special.gammaln(n + 1)
        - jax.scipy.special.gammaln(k + 1)
        - jax.scipy.special.gammaln(n - k + 1)
    )


@jit
def _bernstein_poly(n: int, i: jnp.ndarray, t: float) -> jnp.ndarray:
    """Evaluate Bernstein basis polynomials B_{i,n}(t).

    Args:
        n: Degree of the polynomial.
        i: Indices array of shape (n+1,).
        t: Parameter value in [0, 1].

    Returns:
        Bernstein basis values of shape (n+1,).
    """
    return _binom(n, i) * (t ** i) * ((1 - t) ** (n - i))


@jit
def evaluate(control_points: jnp.ndarray, t_values: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a Bezier curve at given parameter values.

    Constructs the full Bernstein basis matrix and evaluates all points
    via a single matrix multiplication (cuBLAS gemm on GPU).

    Args:
        control_points: Control points of shape (n+1, d) where n is the
            curve degree and d is the dimension.
        t_values: Parameter values of shape (m,) in [0, 1].

    Returns:
        Curve points of shape (m, d).
    """
    n = control_points.shape[0] - 1
    i = jnp.arange(n + 1)
    coeffs = _binom(n, i)                                         # (n+1,)
    t_col = t_values[:, None]                                      # (m, 1)
    basis = coeffs * (t_col ** i) * ((1 - t_col) ** (n - i))      # (m, n+1)
    return basis @ control_points                                  # (m, d)


@jit
def derivative(control_points: jnp.ndarray, t_values: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the first derivative (hodograph) of a Bezier curve.

    The derivative of a degree-n Bezier curve is a degree-(n-1) Bezier curve
    with control points: Q_i = n * (P_{i+1} - P_i).

    Args:
        control_points: Control points of shape (n+1, d).
        t_values: Parameter values of shape (m,) in [0, 1].

    Returns:
        First derivative vectors of shape (m, d).
    """
    n = control_points.shape[0] - 1
    deriv_points = n * (control_points[1:] - control_points[:-1])
    return evaluate(deriv_points, t_values)


@jit
def second_derivative(control_points: jnp.ndarray, t_values: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the second derivative of a Bezier curve.

    Args:
        control_points: Control points of shape (n+1, d).
        t_values: Parameter values of shape (m,) in [0, 1].

    Returns:
        Second derivative vectors of shape (m, d).
    """
    n = control_points.shape[0] - 1
    second_deriv_points = n * (n - 1) * (
        control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
    )
    return evaluate(second_deriv_points, t_values)


@jit
def differential(control_points: jnp.ndarray, t_values: jnp.ndarray) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute curve points and derivatives up to 2nd order.

    This is a convenience function that evaluates the curve value,
    first derivative, and second derivative in one call.

    Args:
        control_points: Control points of shape (n+1, d) for arbitrary
            dimension d.
        t_values: Parameter values of shape (m,) in [0, 1].

    Returns:
        Tuple of:
            - curve_points: Curve values, shape (m, d).
            - first_derivatives: First derivatives, shape (m, d).
            - second_derivatives: Second derivatives, shape (m, d).
    """
    curve_points = evaluate(control_points, t_values)
    first_deriv = derivative(control_points, t_values)
    second_deriv = second_derivative(control_points, t_values)
    return curve_points, first_deriv, second_deriv


@jit
def subdivide(control_points: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Subdivide a Bezier curve at parameter t using de Casteljau's algorithm.

    Splits the curve into two sub-curves [0, t] and [t, 1], each represented
    by their own set of control points with the same degree as the original.

    Args:
        control_points: Control points of shape (n+1, d).
        t: Subdivision parameter in [0, 1].

    Returns:
        Tuple of (left_points, right_points), each of shape (n+1, d).
        left_points defines the curve over [0, t] (reparameterized to [0, 1]).
        right_points defines the curve over [t, 1] (reparameterized to [0, 1]).
    """
    n = control_points.shape[0]
    temp = control_points
    left_points = [temp[0]]
    right_points = [temp[-1]]

    for _ in range(1, n):
        temp = (1 - t) * temp[:-1] + t * temp[1:]
        left_points.append(temp[0])
        right_points.append(temp[-1])

    right_points = right_points[::-1]
    return jnp.stack(left_points), jnp.stack(right_points)
