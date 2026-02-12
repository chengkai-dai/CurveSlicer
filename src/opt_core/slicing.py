"""Core slicing computation for curved layer DLP printing.

Provides differentiable functions to:
    1. Apply physical constraints to Bezier control points
    2. Transform mesh samples by rigid body transformation
    3. Compute slicing planes from a Bezier curve
    4. Determine soft layer transitions for each sample point

The layer transition computation is the key differentiable mechanism:
for each sample point and each pair of consecutive slicing planes,
it produces a soft indicator of whether the point crosses from one
side of the plane pair to the other. This enables gradient-based
optimization of the slicing curve.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple

from . import bezier
from .geometry import quaternion_to_rotation_matrix


@jit
def apply_control_point_constraints(control_points: jnp.ndarray) -> jnp.ndarray:
    """Apply physical constraints to Bezier control points.

    Ensures the printing curve starts from the build platform:
        - P_0.y is fixed near zero (y = -0.0001) so the first plane
          sits at the platform surface.
        - P_1.x = P_0.x and P_1.z = P_0.z so the curve begins
          vertically (tangent perpendicular to the platform).

    These constraints are applied in a differentiable manner using
    JAX's `at[].set()` operations.

    Args:
        control_points: Control points of shape (n, 4) with columns
            (x, y, z, angle).

    Returns:
        Constrained control points of shape (n, 4).
    """
    control_points = control_points.at[0, 1].set(-0.0001)
    control_points = control_points.at[1, 0].set(control_points[0, 0])
    control_points = control_points.at[1, 2].set(control_points[0, 2])
    return control_points


@jit
def transform_samples(sample_positions: jnp.ndarray,
                      sample_normals: jnp.ndarray,
                      quaternion: jnp.ndarray,
                      translation: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rigid body transformation to mesh samples.

    Transforms sample positions and normals, then lifts the object so
    its lowest point rests on the build platform (y = 0). This ensures
    the object is always sitting on the platform regardless of the
    orientation parameters.

    The transformation order is:
        1. Normalize the quaternion
        2. Rotate positions and normals
        3. Translate positions
        4. Shift y so that min(y) = 0

    All operations are differentiable with respect to quaternion and
    translation.

    Args:
        sample_positions: Vertex positions of shape (n, 3).
        sample_normals: Vertex normals of shape (n, 3).
        quaternion: Orientation as unit quaternion (w, x, y, z), shape (4,).
        translation: Position offset, shape (3,).

    Returns:
        Tuple of (transformed_positions, transformed_normals).
    """
    quaternion = quaternion / jnp.linalg.norm(quaternion)
    R = quaternion_to_rotation_matrix(quaternion)

    # Rotate and translate positions: (n, 3) @ (3, 3) + (3,)
    positions = sample_positions @ R.T + translation
    # Lift so lowest point is at y=0
    min_y = jnp.min(positions[:, 1])
    positions = positions.at[:, 1].add(-min_y)

    # Rotate normals (no translation for normals)
    normals = sample_normals @ R.T

    return positions, normals


@jit
def compute_slicing_planes(control_points: jnp.ndarray,
                           t_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute slicing planes from Bezier curve control points.

    Each point on the Bezier curve defines a slicing plane:
        - Position: the curve point (x, y, z components)
        - Normal: the normalized tangent vector (first derivative)
        - Angle: the in-plane rotation angle (4th component of curve)

    The plane normal is the direction perpendicular to the slicing plane,
    pointing in the printing direction (along the curve).

    Args:
        control_points: Control points of shape (n, 4).
        t_values: Parameter values of shape (m,) uniformly in [0, 1].

    Returns:
        Tuple of:
            - plane_positions: Centers of slicing planes, shape (m, 3).
            - plane_normals: Unit normals of slicing planes, shape (m, 3).
            - plane_angles: In-plane rotation angles, shape (m,).
    """
    curve_points = bezier.evaluate(control_points, t_values)
    curve_derivatives = bezier.derivative(control_points, t_values)

    plane_positions = curve_points[:, :3]
    plane_angles = curve_points[:, 3]

    plane_normals = curve_derivatives[:, :3]
    plane_normals = plane_normals / (
        jnp.linalg.norm(plane_normals, axis=1, keepdims=True) + 1e-8
    )

    return plane_positions, plane_normals, plane_angles


@jit
def compute_layer_transitions(sample_positions: jnp.ndarray,
                              plane_positions: jnp.ndarray,
                              plane_normals: jnp.ndarray,
                              k: float = 1e4) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute differentiable layer transitions for all sample points.

    For each sample point and each pair of consecutive slicing planes,
    determines a soft indicator of whether the point transitions between
    the layers (i.e., lies between the two planes).

    The computation uses:
        - Signed distance from each point to each plane
        - tanh(distance * k) as a smooth sign function
        - ReLU(-sign_i * sign_{i+1}) to detect transitions: when
          consecutive signs differ, the point lies between those planes

    Args:
        sample_positions: Mesh vertex positions, shape (n_samples, 3).
        plane_positions: Slicing plane centers, shape (n_planes, 3).
        plane_normals: Slicing plane normals, shape (n_planes, 3).
        k: Sharpness parameter. Larger k gives sharper (harder) transitions.
            Default 1e4.

    Returns:
        Tuple of:
            - status_signs: Soft sign of distance to each plane,
              shape (n_samples, n_planes).
            - layer_transitions: Soft layer membership indicators,
              shape (n_samples, n_planes - 1).
    """
    # Signed distances via matmul: d[i,j] = (sample_i - base_j) · n_j
    # Expand: sample_i · n_j - base_j · n_j  =>  matmul - broadcast
    base_dot_normal = jnp.sum(plane_positions * plane_normals, axis=1)  # (n_planes,)
    distances = sample_positions @ plane_normals.T - base_dot_normal    # (n_samples, n_planes)

    # Smooth sign function
    status_signs = jnp.tanh((distances + 1e-8) * k)

    # Detect transitions between consecutive planes
    # When sign_i and sign_{i+1} differ, their product is negative
    layer_transitions = jax.nn.relu(-status_signs[:, :-1] * status_signs[:, 1:])

    return status_signs, layer_transitions
