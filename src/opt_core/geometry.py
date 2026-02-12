"""Differentiable geometric transformations using JAX.

Provides quaternion-to-rotation-matrix conversion and axis-angle
rotation construction. All operations are JIT-compiled and fully
differentiable through JAX's automatic differentiation system.

Quaternion convention: (w, x, y, z) where w is the scalar part.
"""

import jax.numpy as jnp
from jax import jit


@jit
def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    Uses the standard quaternion-to-rotation formula. The input quaternion
    does not need to be pre-normalized (but should be close to unit norm
    for meaningful results in the differentiable pipeline).

    Args:
        q: Quaternion of shape (4,) in (w, x, y, z) convention.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y]
    ])


@jit
def axis_angle_to_rotation_matrix(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Create a rotation matrix from an axis and angle.

    Implements Rodrigues' rotation formula via the quaternion representation
    for numerical stability and consistent differentiability.

    Args:
        axis: Unit rotation axis of shape (3,).
        angle: Rotation angle in radians.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    a = jnp.cos(angle / 2)
    b, c, d = -axis * jnp.sin(angle / 2)
    return jnp.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),         2*(b*d + a*c)],
        [2*(b*c + a*d),         a*a + c*c - b*b - d*d,  2*(c*d - a*b)],
        [2*(b*d - a*c),         2*(c*d + a*b),          a*a + d*d - b*b - c*c]
    ])
