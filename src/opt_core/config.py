"""Configuration dataclasses for curved slicing optimization.

Provides structured configuration for the physical printing setup
and optimization hyperparameters. All angles are in radians.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class SlicingConfig:
    """Physical parameters for the DLP curved slicing setup.

    Attributes:
        n_layers: Number of slicing layers (planes) sampled along the curve.
        support_angle: Maximum overhang angle (radians) before support is needed.
            Default ~50 degrees.
        surface_quality_angle: Angle threshold (radians) for surface quality
            evaluation. Default 0.0 (disabled).
        tank_width: Width of the resin tank in the x-direction.
        tank_length: Length of the resin tank in the z-direction.
        tank_height: Depth of the resin tank in the y-direction (into liquid).
        wall_thickness: Thickness of the tank walls.
    """

    n_layers: int = 200
    support_angle: float = field(default_factory=lambda: float(np.deg2rad(50.0)))
    surface_quality_angle: float = 0.0
    tank_width: float = 1.5
    tank_length: float = 1.5
    tank_height: float = 0.15
    wall_thickness: float = 0.02


@dataclass
class OptimizerConfig:
    """Hyperparameters for the curved slicing optimization process.

    Attributes:
        learning_rate: Base learning rate for the Adam optimizer.
        max_iterations: Maximum gradient descent iterations per restart.
        max_restarts: Maximum number of optimization restarts with curve
            normalization between each restart.
        bad_steps_limit: Number of consecutive unstable steps (NaN or diverging
            loss) before triggering an early restart.
        control_points_lr_scale: Learning rate multiplier for control points
            (relative to base learning rate).
        quaternion_lr_scale: Learning rate multiplier for the orientation
            quaternion.
        translation_lr_scale: Learning rate multiplier for the translation
            vector.
        floating_weight: Weight for the floating loss in the constraint term.
        completeness_weight: Weight for the completeness loss in the constraint
            term.
        collision_weight: Weight for the collision loss in the constraint term.
        support_weight: Weight for the support loss in the objective term.
        surface_quality_weight: Weight for surface quality loss in the
            objective term. Default 0.0 (disabled).
        constraint_penalty: Multiplier for the augmented Lagrangian squared
            constraint penalty.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        k: Sharpness parameter for differentiable soft-sign approximations.
            Larger values give sharper (closer to hard) transitions.
        collision_check_step: Check collision every N-th layer for efficiency.
        setup_opt: When True, jointly optimize quaternion and translation
            alongside control points. When False (default), only control
            points are optimized.
        optimize_params: Tuple of parameter names to optimize. Derived
            automatically from ``setup_opt`` in ``__post_init__`` if left
            at the default value. Can be overridden manually with valid
            entries: 'control_points', 'quaternion', 'translation'.
        log_interval: Print progress every N iterations when verbose is True.
    """

    learning_rate: float = 0.1
    max_iterations: int = 2000
    max_restarts: int = 3
    bad_steps_limit: int = 10

    # Per-parameter learning rate scales
    control_points_lr_scale: float = 0.1
    quaternion_lr_scale: float = 0.5
    translation_lr_scale: float = 0.01

    # Loss weights
    floating_weight: float = 10.0
    completeness_weight: float = 1.0
    collision_weight: float = 1.0
    support_weight: float = 1.0
    surface_quality_weight: float = 0.0
    constraint_penalty: float = 0.5

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Differentiable approximation sharpness
    k: float = 1e4

    # Collision checking
    collision_check_step: int = 2

    # Setup optimization (jointly optimize orientation & position)
    setup_opt: bool = False

    # Which parameters to optimize (auto-derived from setup_opt)
    optimize_params: Tuple[str, ...] = None  # type: ignore[assignment]

    # Logging
    log_interval: int = 50

    def __post_init__(self):
        if self.optimize_params is None:
            if self.setup_opt:
                self.optimize_params = (
                    'control_points', 'quaternion', 'translation',
                )
            else:
                self.optimize_params = ('control_points',)
