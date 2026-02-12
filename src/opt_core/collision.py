"""Differentiable collision detection for the DLP resin tank.

The DLP printing tank is modeled as a set of axis-aligned cuboids
(base plate + 4 walls). Each cuboid face is represented by its center
point and outward-pointing normal. Collision detection uses smooth
approximations (tanh for sign, logsumexp for max) to remain fully
differentiable through JAX.

A point is considered *inside* a cuboid if it lies on the negative
(interior) side of all 6 face planes simultaneously.
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def precompute_cuboid_planes(tank_width: float, tank_length: float,
                             tank_height: float, thickness: float) -> jnp.ndarray:
    """Precompute the plane representations of tank wall cuboids.

    The DLP resin tank consists of:
        - A base plate (centered at origin, slight positive y offset)
        - Four side walls extending downward into the resin

    Each cuboid face is stored as a 6-vector: [center_xyz, normal_xyz].

    Args:
        tank_width: Width of the tank (x-direction).
        tank_length: Length of the tank (z-direction).
        tank_height: Depth of the tank (y-direction, into the resin).
        thickness: Wall/plate thickness.

    Returns:
        Array of shape (n_cuboids, 6, 6) where n_cuboids=5, each cuboid
        has 6 face planes, and each plane is [cx, cy, cz, nx, ny, nz].
    """
    ht = thickness / 2  # half thickness

    x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    z = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    cuboids = [
        # Base plate
        {
            "center": jnp.array([0.0, ht / 2, 0.0], dtype=jnp.float32),
            "half_size": jnp.array([tank_width / 2, tank_length / 2, ht / 2], dtype=jnp.float32),
        },
        # Right wall (+x)
        {
            "center": jnp.array([tank_width / 2 + ht, -tank_height / 2, 0.0], dtype=jnp.float32),
            "half_size": jnp.array([thickness / 2, tank_length / 2, tank_height / 2], dtype=jnp.float32),
        },
        # Left wall (-x)
        {
            "center": jnp.array([-tank_width / 2 - ht, -tank_height / 2, 0.0], dtype=jnp.float32),
            "half_size": jnp.array([thickness / 2, tank_length / 2, tank_height / 2], dtype=jnp.float32),
        },
        # Front wall (+z)
        {
            "center": jnp.array([0.0, -tank_height / 2, tank_length / 2 + ht], dtype=jnp.float32),
            "half_size": jnp.array([(tank_width + thickness) / 2, thickness / 2, tank_height / 2],
                                   dtype=jnp.float32),
        },
        # Back wall (-z)
        {
            "center": jnp.array([0.0, -tank_height / 2, -tank_length / 2 - ht], dtype=jnp.float32),
            "half_size": jnp.array([(tank_width + thickness) / 2, thickness / 2, tank_height / 2],
                                   dtype=jnp.float32),
        },
    ]

    all_planes = []
    for cuboid in cuboids:
        center = cuboid["center"]
        hw, hh, hd = cuboid["half_size"]

        # 6 faces of the cuboid: center-normal pairs
        faces = [
            (center + y * hd, y),     # +y face (front)
            (center - y * hd, -y),    # -y face (back)
            (center + z * hh, z),     # +z face (top)
            (center - z * hh, -z),    # -z face (bottom)
            (center + x * hw, x),     # +x face (right)
            (center - x * hw, -x),    # -x face (left)
        ]

        planes = [jnp.concatenate([c, n]) for c, n in faces]
        all_planes.append(jnp.stack(planes))

    return jnp.stack(all_planes)


@jit
def check_point_cuboid_collision(points: jnp.ndarray,
                                 cuboid_planes: jnp.ndarray,
                                 k: float) -> jnp.ndarray:
    """Check if points are inside cuboids using differentiable approximation.

    A point is inside a cuboid if its signed distance to every face plane
    is negative (i.e., it is on the interior side of all faces).

    The function uses:
        - tanh(distance * k) as a smooth sign function
        - logsumexp as a smooth maximum over face signs

    The penalty is non-zero only when a point is inside a cuboid
    (all face-signs negative).

    Args:
        points: Sample points of shape (n_points, 3).
        cuboid_planes: Cuboid face planes of shape (n_cuboids, 6, 6),
            where each plane is [center_xyz, normal_xyz].
        k: Sharpness parameter for soft approximations.

    Returns:
        Collision penalties of shape (n_points, n_cuboids).
        Positive values indicate collision (point inside cuboid).
    """
    plane_centers = cuboid_planes[:, :, :3]   # (n_cuboids, 6, 3)
    plane_normals = cuboid_planes[:, :, 3:]   # (n_cuboids, 6, 3)

    # Signed distance from each point to each face of each cuboid
    # Shape: (n_points, n_cuboids, 6)
    diff = points[:, None, None, :] - plane_centers[None, :, :, :]
    distances = jnp.sum(diff * plane_normals[None, :, :, :], axis=3)

    # Smooth sign of distances
    distance_signs = jax.nn.tanh(distances * k)

    # Smooth max over the 6 faces (axis=2): if max sign < 0, point is inside
    # logsumexp(x * k) / k approximates max(x) as k -> inf
    smooth_max = jax.scipy.special.logsumexp(distance_signs * k, axis=2) / k

    # Penalty is non-zero when smooth_max < 0 (point inside all halfspaces)
    collision_penalty = jax.nn.relu(-smooth_max)

    return collision_penalty
