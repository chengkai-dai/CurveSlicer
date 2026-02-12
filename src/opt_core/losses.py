"""Differentiable loss functions for curved slicing optimization.

Each loss function captures a specific manufacturing constraint or
quality objective for DLP 3D printing with curved layers. All functions
operate on JAX arrays and are fully differentiable.

Loss categories:
    **Constraints** (must be satisfied for a valid print):
        - self_collision_loss: Each point belongs to at most one layer
        - completeness_loss: All points are covered by the slicing planes
        - floating_loss: No disconnected floating regions
        - environment_collision_loss: No collision with the resin tank

    **Objectives** (quality metrics to optimize):
        - support_loss: Minimize unsupported overhangs
        - surface_quality_loss: Minimize staircase artifacts on critical surfaces
        - curvature_smoothness_loss: Encourage smooth printing paths
        - chamfer_distance: Measure curve-to-mesh spatial coverage
"""

import functools

import jax
import jax.numpy as jnp
from jax import jit

from .geometry import axis_angle_to_rotation_matrix
from .collision import check_point_cuboid_collision


@jit
def self_collision_loss(layer_transitions: jnp.ndarray,
                        valid: jnp.ndarray) -> float:
    """Penalize sample points assigned to multiple layers.

    Each sample point should be intersected by at most one pair of
    consecutive slicing planes. Points with total transition weight > 1
    are counted as self-collisions.

    Args:
        layer_transitions: Soft transition indicators,
            shape (n_samples, n_planes - 1).
        valid: Indices of valid sample points, shape (n_valid,).

    Returns:
        Scalar penalty (0 when no self-collisions).
    """
    transition_sum = jnp.sum(layer_transitions, axis=1)
    return jnp.sum(jax.nn.relu(transition_sum - 1.0))


@jit
def support_loss(layer_transitions: jnp.ndarray,
                 sample_normals: jnp.ndarray,
                 plane_normals: jnp.ndarray,
                 sin_support_angle: float,
                 valid: jnp.ndarray) -> float:
    """Penalize unsupported overhangs exceeding the support angle.

    A surface element is considered unsupported when the angle between
    its outward normal and the printing direction (layer normal) exceeds
    the support angle threshold. Specifically, the penalty activates when:
        dot(sample_normal, plane_normal) < -sin(support_angle)

    The penalty is weighted by the layer transition (so only the assigned
    layer contributes).

    Args:
        layer_transitions: Shape (n_samples, n_planes - 1).
        sample_normals: Surface normals at vertices, shape (n_samples, 3).
        plane_normals: Slicing plane normals, shape (n_planes, 3).
        sin_support_angle: Sine of the maximum support angle (radians).
        valid: Valid vertex indices, shape (n_valid,).

    Returns:
        Scalar support penalty.
    """
    # Dot product via matmul: (n_samples, n_planes)
    normal_dots = sample_normals @ plane_normals.T

    # Violation when dot < -sin_alpha (overhang too steep)
    support_violations = jax.nn.relu(-normal_dots - sin_support_angle)

    # Weight by layer assignment
    weighted = jnp.einsum('ij,ij->i', support_violations[:, :-1], layer_transitions)
    return jnp.sum(weighted[valid])


@jit
def surface_quality_loss(layer_transitions: jnp.ndarray,
                         sample_normals: jnp.ndarray,
                         plane_normals: jnp.ndarray,
                         sin_quality_angle: float,
                         surface_valid_idx: jnp.ndarray) -> float:
    """Penalize poor surface quality at critical surface regions.

    Surface quality degrades when the slicing plane is nearly tangent
    to the surface (causing staircase artifacts). This loss penalizes
    when |dot(sample_normal, plane_normal)| exceeds the quality angle
    threshold at user-specified surface-critical regions.

    Args:
        layer_transitions: Shape (n_samples, n_planes - 1).
        sample_normals: Shape (n_samples, 3).
        plane_normals: Shape (n_planes, 3).
        sin_quality_angle: Sine of the quality angle threshold.
        surface_valid_idx: Indices of surface-critical vertices.

    Returns:
        Scalar surface quality penalty (averaged over critical vertices).
    """
    # Dot product via matmul: (n_samples, n_planes)
    normal_dots = sample_normals @ plane_normals.T

    cliff_violations = jax.nn.relu(jnp.abs(normal_dots) - sin_quality_angle)

    weighted = jnp.einsum('ij,ij->i', cliff_violations[:, :-1], layer_transitions)
    return jnp.sum(weighted[surface_valid_idx]) / surface_valid_idx.shape[0]


@jit
def completeness_loss(status_signs: jnp.ndarray,
                      valid: jnp.ndarray) -> float:
    """Ensure all valid sample points are covered by the slicing planes.

    For a complete slicing:
        - The first plane should be below all points (positive sign)
        - The last plane should be above all points (negative sign)

    Violations are measured as squared deviations from the ideal:
        L = (n_valid + sum(last_signs))^2 + (sum(first_signs) - n_valid)^2

    When perfectly satisfied, first_signs are all +1 (sum = n_valid) and
    last_signs are all -1 (sum = -n_valid), giving L = 0.

    Args:
        status_signs: Soft signs of distance to each plane,
            shape (n_samples, n_planes).
        valid: Valid sample indices, shape (n_valid,).

    Returns:
        Scalar completeness penalty (0 when fully covered).
    """
    n_valid = valid.shape[0]
    first_plane_sign = jnp.sum(status_signs[:, 0][valid])
    last_plane_sign = jnp.sum(status_signs[:, -1][valid])
    return (n_valid + last_plane_sign) ** 2 + (first_plane_sign - n_valid) ** 2


@jit
def floating_loss(layer_transitions: jnp.ndarray,
                  sample_positions: jnp.ndarray,
                  sample_normals: jnp.ndarray,
                  plane_normals: jnp.ndarray,
                  connection: jnp.ndarray,
                  valid: jnp.ndarray,
                  k: float = 1e4) -> float:
    """Penalize floating (disconnected) regions in the print.

    Detects sample points that are both:
        1. Unsupported from below (overhang facing against printing direction)
        2. Have no mesh neighbor that could provide structural support
           (all neighbors are "above" in the printing direction)

    This prevents the optimizer from creating layers that would fail
    during printing due to lack of structural support from previously
    printed layers.

    The ``connection`` array stores mesh neighbor indices for each vertex.
    Padding entries (for vertices with fewer neighbors than the maximum)
    point to auxiliary far-away points that don't affect the support check.

    Args:
        layer_transitions: Shape (n_samples, n_planes - 1).
        sample_positions: Shape (n_samples, 3).
        sample_normals: Shape (n_samples, 3).
        plane_normals: Shape (n_planes, 3).
        connection: Mesh neighbor indices, shape (n_samples, max_neighbors).
        valid: Valid sample indices, shape (n_valid,).
        k: Sharpness parameter.

    Returns:
        Scalar floating penalty.
    """
    # Weighted average printing direction per sample via matmul
    sample_dp = layer_transitions @ plane_normals[:-1, :]
    norms = jnp.sqrt(jnp.sum(sample_dp ** 2, axis=1, keepdims=True) + 1e-8)
    sample_dp = sample_dp / norms

    # Far-away fill points for padding (won't affect support check)
    filled_points = sample_dp * 1e5
    stacked_samples = jnp.vstack([sample_positions, filled_points])

    # Look up neighbor positions
    conn_points = stacked_samples[connection]

    # Direction vectors to each neighbor
    difference = conn_points - sample_positions[:, None, :]
    e = difference / (jnp.sqrt(jnp.sum(difference ** 2, axis=2, keepdims=True) + 1e-8))

    # Dot product of neighbor directions with printing direction
    e_dot_dp = jnp.sum(e * sample_dp[:, None, :], axis=2)
    min_e_dot_dp = jnp.min(e_dot_dp, axis=1)

    # Identify unsupported overhangs via matmul
    normal_dots = sample_normals @ plane_normals.T
    unsupported = jnp.einsum('ij,ij->i', normal_dots[:, :-1], layer_transitions)
    unsupported = jax.nn.relu(-unsupported)
    unsupported_mask = jnp.where(unsupported > 0.0, 1.0, 0.0)

    # Floating = unsupported AND no neighbor below (min_e_dot_dp > 0)
    floating = jax.nn.sigmoid(min_e_dot_dp[valid] * 1e4) * unsupported_mask[valid]
    return jnp.sum(floating)


@functools.partial(jit, static_argnames=('step',))
def environment_collision_loss(layer_transitions: jnp.ndarray,
                               sample_positions: jnp.ndarray,
                               plane_positions: jnp.ndarray,
                               plane_normals: jnp.ndarray,
                               plane_angles: jnp.ndarray,
                               platform_samples: jnp.ndarray,
                               cuboid_planes: jnp.ndarray,
                               valid: jnp.ndarray,
                               k: float = 1e4,
                               step: int = 2) -> float:
    """Penalize collisions between the printed object and the DLP tank.

    At each checked layer, the function:
        1. Constructs a local coordinate frame from the plane normal and angle
        2. Transforms the tank cuboid planes into world coordinates
        3. Computes which sample points have been printed (cumulative weight)
        4. Checks for collisions between printed points and the tank

    The cumulative weight approach gives a soft approximation of the set
    of points printed up to each layer. Combined with the soft collision
    check, the entire computation is differentiable.

    Args:
        layer_transitions: Shape (n_samples, n_planes - 1).
        sample_positions: Shape (n_samples, 3).
        plane_positions: Slicing plane centers, shape (n_planes, 3).
        plane_normals: Slicing plane normals, shape (n_planes, 3).
        plane_angles: In-plane rotation angles, shape (n_planes,).
        platform_samples: Build platform sample points, shape (n_platform, 3).
        cuboid_planes: Precomputed tank planes,
            shape (n_cuboids, 6, 6).
        valid: Valid sample indices, shape (n_valid,).
        k: Sharpness parameter for collision detection.
        step: Check every ``step``-th layer for efficiency. Default 2.
            This argument is static (not traced by JAX).

    Returns:
        Scalar collision penalty.
    """
    # Pre-compute cumulative weights once (hoisted out of vmap)
    cum_weights = jnp.clip(jnp.cumsum(layer_transitions, axis=1), 0.0, 1.0001)
    # Pre-index valid sample positions
    sample_pos_valid = sample_positions[valid]

    def layer_collision(i):
        center = plane_positions[i]
        normal = plane_normals[i]
        theta = plane_angles[i]

        # Build local coordinate frame
        u = jnp.cross(normal, jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
        u_norm = jnp.linalg.norm(u)
        u = jnp.where(
            u_norm < 1e-6,
            jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            u / u_norm
        )
        v = jnp.cross(normal, u)
        v = v / (jnp.linalg.norm(v) + 1e-8)

        # Apply in-plane rotation
        rot = axis_angle_to_rotation_matrix(normal, theta)
        u = rot @ u
        v = rot @ v

        rotation_matrix = jnp.stack([u, normal, v], axis=1)

        # Transform cuboid planes from local to world frame
        transformed_centers = jnp.einsum(
            'ijk,kl->ijl', cuboid_planes[:, :, :3], rotation_matrix.T
        ) + center
        transformed_normals = jnp.einsum(
            'ijk,kl->ijl', cuboid_planes[:, :, 3:], rotation_matrix.T
        )
        transformed_planes = jnp.concatenate(
            [transformed_centers, transformed_normals], axis=2
        )

        # Use pre-computed cumulative weights (indexed per layer)
        weights_valid = cum_weights[valid, i - 1]
        weighted_samples = sample_pos_valid * weights_valid[:, None]

        # Combine printed samples with platform samples
        current_samples = jnp.concatenate(
            [weighted_samples, platform_samples]
        )

        # Check for collisions
        collisions = check_point_cuboid_collision(current_samples, transformed_planes, k)
        return jnp.sum(collisions)

    layer_idx = jnp.arange(1, plane_positions.shape[0], step)
    penalties = jax.vmap(layer_collision)(layer_idx)
    return jnp.sum(penalties)


@jit
def curvature_smoothness_loss(control_points: jnp.ndarray,
                              t_values: jnp.ndarray) -> float:
    """Penalize rapid changes in curvature along the Bezier curve.

    Computes the curvature at each sampled point using the cross product
    formula: kappa = |r' x r''| / |r'|^3, then penalizes the mean
    squared difference of consecutive curvature values.

    This encourages smooth printing paths, avoiding sudden directional
    changes that can cause mechanical issues.

    Args:
        control_points: Shape (n_cp, d). Only the first 3 columns (spatial)
            are used for curvature computation.
        t_values: Parameter values, shape (m,).

    Returns:
        Scalar curvature smoothness penalty.
    """
    from . import bezier as _bezier

    _, first_deriv, second_deriv = _bezier.differential(control_points, t_values)

    # Use only spatial components for curvature
    first_deriv = first_deriv[:, :3]
    second_deriv = second_deriv[:, :3]

    velocity_norm = jnp.maximum(jnp.linalg.norm(first_deriv, axis=1), 1e-8)
    cross_product = jnp.cross(first_deriv, second_deriv)
    cross_norm = jnp.linalg.norm(cross_product, axis=1)
    curvature = cross_norm / (velocity_norm ** 3)

    curvature_change = jnp.diff(curvature) ** 2
    return jnp.mean(curvature_change)


@jit
def chamfer_distance(sample_positions: jnp.ndarray,
                     plane_positions: jnp.ndarray) -> float:
    """Compute the symmetric Chamfer distance between samples and planes.

    Measures how well the slicing planes spatially cover the mesh by
    computing the average nearest-neighbor squared distance in both
    directions.

    Args:
        sample_positions: Mesh vertex positions, shape (n_samples, 3).
        plane_positions: Slicing plane centers, shape (n_planes, 3).

    Returns:
        Scalar Chamfer distance.
    """
    dists_to_curve = jax.vmap(
        lambda p: jnp.min(jnp.sum((plane_positions - p[None, :]) ** 2, axis=1))
    )(sample_positions)

    dists_to_samples = jax.vmap(
        lambda p: jnp.min(jnp.sum((sample_positions - p[None, :]) ** 2, axis=1))
    )(plane_positions)

    return (
        jnp.sum(dists_to_curve) / sample_positions.shape[0]
        + jnp.sum(dists_to_samples) / plane_positions.shape[0]
    ) / 2.0
