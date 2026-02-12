"""Curved Slicing DLP - Differentiable curved layer optimization for DLP 3D printing.

This package provides a fully differentiable optimization framework for
computing optimal curved slicing paths for DLP (Digital Light Processing)
3D printing. The optimizer finds Bezier curve parameters that minimize
printing defects (unsupported overhangs, floating regions, collisions)
while satisfying manufacturing constraints.

All core computations use JAX for automatic differentiation, enabling
gradient-based optimization of the slicing curve, object orientation,
and object position simultaneously.

Modules:
    bezier: Differentiable Bezier curve evaluation, derivatives, and subdivision
    geometry: Quaternion and rotation matrix operations
    slicing: Slicing plane computation and layer transition detection
    losses: Manufacturing-aware differentiable loss functions
    collision: DLP tank collision detection
    optimizer: Main optimization engine with multi-restart support
    logging: Structured optimization progress logger
    config: Configuration dataclasses

Quick start::

    import numpy as np
    from curved_slicing import optimize, SlicingConfig, OptimizerConfig

    # Prepare mesh data (positions, normals, connectivity, ...)
    result = optimize(
        initial_control_points=control_points,
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        initial_translation=np.array([0.0, 0.0, 0.0]),
        sample_positions=positions,
        sample_normals=normals,
        platform_samples=platform_pts,
        connection=neighbor_indices,
        valid=valid_indices,
        surface_valid_idx=surface_indices,
        slicing_config=SlicingConfig(n_layers=200, support_angle=0.87),
        optimizer_config=OptimizerConfig(learning_rate=0.1, max_iterations=2000),
    )
    best_cp, best_quat, best_trans, best_loss = result
"""

from .config import SlicingConfig, OptimizerConfig
from .optimizer import optimize, compute_all_losses, normalize_curve, build_optimizer
from . import bezier
from . import geometry
from . import losses
from . import collision
from . import slicing

__version__ = "0.1.0"

__all__ = [
    # Main API
    "optimize",
    "compute_all_losses",
    "normalize_curve",
    "build_optimizer",
    # Configuration
    "SlicingConfig",
    "OptimizerConfig",
    # Submodules
    "bezier",
    "geometry",
    "losses",
    "collision",
    "slicing",
]
