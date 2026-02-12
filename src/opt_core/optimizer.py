"""Curved slicing optimizer for DLP 3D printing.

Provides the main optimization engine that jointly optimizes:
    - Bezier curve control points (defining the slicing path)
    - Object orientation (quaternion)
    - Object position (translation)

to minimize printing defects while satisfying manufacturing constraints.

The optimization uses an **augmented Lagrangian** approach:
    total_loss = objective + penalty * constraint^2

where:
    - objective = support_loss + surface_quality_loss (quality metrics)
    - constraint = weighted sum of floating, completeness, and collision
      losses (manufacturing feasibility)

Multiple restarts with curve normalization (trimming unused curve
portions) between restarts help escape local minima.
"""

import functools
import math
import time

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import numpy as np
from typing import Tuple, Dict, Optional, Callable

from .config import SlicingConfig, OptimizerConfig
from .logging import OptimizationLogger
from .slicing import (
    apply_control_point_constraints,
    transform_samples,
    compute_slicing_planes,
    compute_layer_transitions,
)
from .losses import (
    self_collision_loss,
    support_loss,
    floating_loss,
    completeness_loss,
    surface_quality_loss,
    environment_collision_loss,
)
from .collision import precompute_cuboid_planes
from . import bezier


def _no_op_optimizer():
    """Create an optimizer that applies zero updates (freezes parameters)."""
    return optax.chain(optax.set_to_zero())


def build_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    """Build a multi-transform optimizer with per-parameter learning rates.

    Creates an optax optimizer that applies different learning rates and
    gradient clipping to each parameter group. Parameters not listed in
    ``config.optimize_params`` are frozen (zero updates).

    Args:
        config: Optimizer configuration with learning rates and parameter
            selection.

    Returns:
        An optax GradientTransformation.
    """
    lr = config.learning_rate

    scales = {
        'control_points': config.control_points_lr_scale,
        'quaternion': config.quaternion_lr_scale,
        'translation': config.translation_lr_scale,
    }

    def _make_opt(param_name: str):
        if param_name not in config.optimize_params:
            return _no_op_optimizer()
        scale = scales.get(param_name, 1.0)
        return optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(lr * scale),
        )

    optimizers = {name: _make_opt(name) for name in scales}

    return optax.multi_transform(
        optimizers,
        param_labels=lambda params: {k: k for k in params.keys()},
    )


@functools.partial(jit, static_argnames=('collision_check_step',))
def compute_all_losses(
    control_points: jnp.ndarray,
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    t_values: jnp.ndarray,
    sample_positions: jnp.ndarray,
    sample_normals: jnp.ndarray,
    platform_samples: jnp.ndarray,
    connection: jnp.ndarray,
    valid: jnp.ndarray,
    surface_valid_idx: jnp.ndarray,
    sin_support_angle: float,
    sin_surface_quality_angle: float,
    cuboid_planes: jnp.ndarray,
    k: float = 1e4,
    collision_check_step: int = 2,
) -> Tuple[float, float, float, float, float]:
    """Compute all individual loss components through the full pipeline.

    Chains together the complete differentiable computation:
        1. Apply control point constraints
        2. Transform mesh samples by quaternion and translation
        3. Compute slicing planes from the Bezier curve
        4. Compute soft layer transitions
        5. Evaluate all loss terms

    Args:
        control_points: Bezier control points, shape (n_cp, 4).
        quaternion: Object orientation as quaternion (w,x,y,z), shape (4,).
        translation: Object position offset, shape (3,).
        t_values: Curve parameter values, shape (n_layers,).
        sample_positions: Mesh vertex positions, shape (n_samples, 3).
        sample_normals: Mesh vertex normals, shape (n_samples, 3).
        platform_samples: Platform surface points, shape (n_platform, 3).
        connection: Vertex neighbor indices, shape (n_samples, max_neighbors).
        valid: Valid vertex indices, shape (n_valid,).
        surface_valid_idx: Surface-critical vertex indices.
        sin_support_angle: Sine of the support angle threshold.
        sin_surface_quality_angle: Sine of the surface quality threshold.
        cuboid_planes: Precomputed tank planes from ``precompute_cuboid_planes``.
        k: Sharpness parameter for differentiable approximations.
        collision_check_step: Check collision every N-th layer.

    Returns:
        Tuple of scalar losses:
            (collision, floating, support, completeness, surface_quality)
    """
    # 1. Apply constraints
    control_points = apply_control_point_constraints(control_points)
    quaternion = quaternion / jnp.linalg.norm(quaternion)

    # 2. Transform samples
    positions, normals = transform_samples(
        sample_positions, sample_normals, quaternion, translation
    )

    # 3. Compute slicing planes
    plane_pos, plane_nrm, plane_ang = compute_slicing_planes(
        control_points, t_values
    )

    # 4. Compute layer transitions
    status_signs, layer_trans = compute_layer_transitions(
        positions, plane_pos, plane_nrm, k
    )

    # 5. Evaluate losses
    penalty_support = support_loss(
        layer_trans, normals, plane_nrm, sin_support_angle, valid
    )

    penalty_floating = floating_loss(
        layer_trans, positions, normals, plane_nrm, connection, valid, k
    )

    penalty_completeness = completeness_loss(status_signs, valid)

    penalty_self_collision = self_collision_loss(layer_trans, valid)

    penalty_env_collision = environment_collision_loss(
        layer_trans, positions, plane_pos, plane_nrm, plane_ang,
        platform_samples, cuboid_planes, valid, k, step=collision_check_step
    )

    penalty_collision = penalty_env_collision + penalty_self_collision

    penalty_surface_quality = surface_quality_loss(
        layer_trans, normals, plane_nrm, sin_surface_quality_angle,
        surface_valid_idx
    )

    return (
        penalty_collision,
        penalty_floating,
        penalty_support,
        penalty_completeness,
        penalty_surface_quality,
    )


@jit
def normalize_curve(
    control_points: jnp.ndarray,
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    t_values: jnp.ndarray,
    sample_positions: jnp.ndarray,
    valid: jnp.ndarray,
    k: float = 1e4,
) -> jnp.ndarray:
    """Trim the Bezier curve to cover only the printed object.

    After optimization, the curve may extend beyond the actual printed
    layers. This function determines the maximum used layer and subdivides
    the curve to keep only the relevant portion (with a small 2% margin).

    This is used between optimization restarts to refocus the curve
    parameterization on the active printing region.

    Args:
        control_points: Optimized control points, shape (n_cp, 4).
        quaternion: Orientation quaternion, shape (4,).
        translation: Translation vector, shape (3,).
        t_values: Parameter values, shape (n_layers,).
        sample_positions: Mesh vertex positions, shape (n_samples, 3).
        valid: Valid vertex indices.
        k: Sharpness parameter.

    Returns:
        Trimmed control points, shape (n_cp, 4).
    """
    control_points = apply_control_point_constraints(control_points)
    quaternion = quaternion / jnp.linalg.norm(quaternion)

    # Transform samples (normals not needed here)
    positions, _ = transform_samples(
        sample_positions, jnp.zeros_like(sample_positions),
        quaternion, translation
    )

    # Compute planes and layer transitions
    plane_pos, plane_nrm, _ = compute_slicing_planes(control_points, t_values)
    _, layer_trans = compute_layer_transitions(positions, plane_pos, plane_nrm, k)

    # Find the highest used layer and subdivide
    max_layer = jnp.max(jnp.argmax(layer_trans[valid], axis=1)) + 1
    subdivision_time = jnp.minimum(max_layer / t_values.shape[0] * 1.02, 1.0)
    left_cp, _ = bezier.subdivide(control_points, subdivision_time)

    return left_cp


def optimize(
    initial_control_points: np.ndarray,
    initial_quaternion: np.ndarray,
    initial_translation: np.ndarray,
    sample_positions: np.ndarray,
    sample_normals: np.ndarray,
    platform_samples: np.ndarray,
    connection: np.ndarray,
    valid: np.ndarray,
    surface_valid_idx: np.ndarray,
    slicing_config: Optional[SlicingConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    callback: Optional[Callable[[int, int, Dict[str, jnp.ndarray], float], None]] = None,
    verbose: bool = True,
    log_file: Optional[str] = "optimization.log",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the curved slicing optimization.

    Optimizes the Bezier curve control points (and optionally object
    orientation and position) to minimize printing defects while
    satisfying manufacturing constraints. Uses an augmented Lagrangian
    formulation with multiple restarts.

    **Optimization loop:**
        For each restart:
            1. Run gradient descent for ``max_iterations`` steps
            2. Track the best parameters seen so far
            3. If loss becomes unstable, early-stop the current restart
            4. Normalize (trim) the curve for the next restart
            5. Reset optimizer state

    Progress is reported via Python logging (logger name:
    ``curved_slicing.optimizer``). A log file is written by default;
    set *log_file* to ``None`` to disable.

    Args:
        initial_control_points: Starting Bezier control points, shape (n_cp, 4).
        initial_quaternion: Starting orientation (w,x,y,z), shape (4,).
        initial_translation: Starting translation, shape (3,).
        sample_positions: Mesh vertex positions, shape (n_samples, 3).
        sample_normals: Mesh vertex normals, shape (n_samples, 3).
        platform_samples: Platform surface points, shape (n_platform, 3).
        connection: Vertex neighbor indices, shape (n_samples, max_neighbors).
        valid: Valid vertex indices, shape (n_valid,).
        surface_valid_idx: Surface-critical vertex indices.
        slicing_config: Physical setup parameters. Uses defaults if None.
        optimizer_config: Optimization hyperparameters. Uses defaults if None.
        callback: Optional function called each step with
            (restart, iteration, params_dict, loss_value).
        verbose: If True, also print progress to the console.
        log_file: Path to write a detailed log file. Defaults to
            ``"optimization.log"``. Set to ``None`` to disable file
            logging.

    Returns:
        Tuple of:
            - best_control_points: Optimized control points, shape (n_cp, 4).
            - best_quaternion: Optimized quaternion, shape (4,).
            - best_translation: Optimized translation, shape (3,).
            - best_loss: Final best loss value (float).
    """
    sc = slicing_config or SlicingConfig()
    oc = optimizer_config or OptimizerConfig()
    log = OptimizationLogger(verbose, log_file)
    timings = {}

    # Precompute static data
    t0 = time.time()
    t_values = jnp.linspace(0, 1, sc.n_layers)
    sin_support = jnp.sin(sc.support_angle)
    sin_quality = jnp.sin(sc.surface_quality_angle)
    cuboid_planes = precompute_cuboid_planes(
        sc.tank_width, sc.tank_length, sc.tank_height, sc.wall_thickness
    )

    # Convert to JAX arrays and mark as non-differentiable (constants)
    sample_pos_jax = jax.lax.stop_gradient(jnp.array(sample_positions))
    sample_nrm_jax = jax.lax.stop_gradient(jnp.array(sample_normals))
    connection_jax = jax.lax.stop_gradient(jnp.array(connection))
    platform_jax = jax.lax.stop_gradient(jnp.array(platform_samples))
    valid_jax = jax.lax.stop_gradient(jnp.array(valid))
    surface_valid_jax = jax.lax.stop_gradient(jnp.array(surface_valid_idx))
    timings["precompute"] = time.time() - t0

    # Initialize parameters
    params = {
        'control_points': jnp.array(initial_control_points),
        'quaternion': jnp.array(initial_quaternion),
        'translation': jnp.array(initial_translation),
    }

    optimizer = build_optimizer(oc)
    opt_state = optimizer.init(params)

    # ---- Define the total loss function (closed over constants) ----
    @jit
    def total_loss(params: Dict[str, jnp.ndarray]) -> float:
        losses = compute_all_losses(
            params['control_points'],
            params['quaternion'],
            params['translation'],
            t_values,
            sample_pos_jax,
            sample_nrm_jax,
            platform_jax,
            connection_jax,
            valid_jax,
            surface_valid_jax,
            sin_support,
            sin_quality,
            cuboid_planes,
            oc.k,
            oc.collision_check_step,
        )
        collision, floating, support, completeness, surface_quality = losses

        # Constraint violation (must be zero for a valid print)
        constraint = (
            oc.floating_weight * floating
            + oc.completeness_weight * completeness
            + oc.collision_weight * collision
        )

        # Objective (quality to minimize)
        objective = oc.support_weight * support + oc.surface_quality_weight * surface_quality

        # Augmented Lagrangian: objective + penalty * constraint^2
        return objective + oc.constraint_penalty * constraint ** 2

    # ---- Define a single gradient step ----
    @jit
    def update_step(params, opt_state):
        loss, grads = value_and_grad(total_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # Re-normalize quaternion to stay on the unit sphere
        new_params['quaternion'] = (
            new_params['quaternion'] / jnp.linalg.norm(new_params['quaternion'])
        )
        # Gradient diagnostics (on-device scalars, cheap reduce ops)
        grad_stats = {
            key: {
                'grad_norm': jnp.linalg.norm(grads[key]),
                'grad_max': jnp.max(jnp.abs(grads[key])),
                'grad_min': jnp.min(jnp.abs(grads[key])),
                'update_norm': jnp.linalg.norm(updates[key]),
            }
            for key in grads
        }
        return new_params, new_opt_state, loss, grad_stats

    # ---- Helper to evaluate individual losses ----
    def _get_loss_components(p):
        losses = compute_all_losses(
            p['control_points'], p['quaternion'], p['translation'],
            t_values, sample_pos_jax, sample_nrm_jax, platform_jax,
            connection_jax, valid_jax, surface_valid_jax,
            sin_support, sin_quality, cuboid_planes,
            oc.k, oc.collision_check_step,
        )
        return tuple(float(v) for v in losses)

    # ---- Compute initial loss from components (avoids separate total_loss JIT) ----
    def _total_loss_from_components(components):
        """Reconstruct total loss from the 5 individual components on the host."""
        coll, flt, sup, comp, surf = components
        constraint = (oc.floating_weight * flt
                      + oc.completeness_weight * comp
                      + oc.collision_weight * coll)
        objective = oc.support_weight * sup + oc.surface_quality_weight * surf
        return objective + oc.constraint_penalty * constraint ** 2

    best_params = params.copy()

    t0 = time.time()
    init_components = _get_loss_components(params)
    timings["initial_loss"] = time.time() - t0

    best_loss = _total_loss_from_components(init_components)
    best_iteration = 0
    best_restart = 0

    log.header(sample_positions.shape[0], initial_control_points.shape[0], sc, oc)
    log.initial(best_loss, init_components)
    log.timing("precompute + transfer", timings["precompute"])
    log.timing("initial loss eval", timings["initial_loss"])

    total_start = time.time()
    timings["iterations"] = 0.0
    timings["normalize_curve"] = 0.0

    for restart in range(oc.max_restarts):
        bad_steps = 0
        local_best_loss = float('inf')
        early_stopped = False
        restart_start = time.time()

        log.restart_begin(restart, oc.max_restarts)

        for iteration in range(oc.max_iterations):
            current_params = params.copy()
            t_step = time.time()
            params, opt_state, loss, grad_stats = update_step(params, opt_state)
            loss_val = float(loss)
            step_dt = time.time() - t_step

            # Record JIT warmup of update_step (first ever call)
            if "jit_update_step" not in timings:
                timings["jit_update_step"] = step_dt
                log.timing("JIT compile update_step", step_dt)

            timings["iterations"] += step_dt

            if callback is not None:
                callback(restart, iteration, params, loss_val)

            # Stability check
            if math.isnan(loss_val) or loss_val > local_best_loss * 100:
                bad_steps += 1
                if bad_steps >= oc.bad_steps_limit:
                    early_stopped = True
                    log.diverged(iteration)
                    break
            else:
                bad_steps = 0
            local_best_loss = min(local_best_loss, loss_val)

            # Track global best
            improved = False
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = current_params
                best_iteration = iteration
                best_restart = restart
                improved = True

            log.step(iteration, loss_val, best_loss, improved,
                     time.time() - restart_start)

            # Gradient health diagnostics (only at log intervals)
            if log._should_log(iteration):
                host_stats = {
                    k: {sk: float(sv) for sk, sv in v.items()}
                    for k, v in grad_stats.items()
                    if k in oc.optimize_params
                }
                log.grad_health(host_stats)

        # Restart summary
        iters_done = iteration + 1 if early_stopped else oc.max_iterations
        log.restart_end(iters_done, time.time() - restart_start,
                        local_best_loss, best_loss)

        # Normalize curve for next restart
        t_norm = time.time()
        params = best_params.copy()
        params['control_points'] = normalize_curve(
            params['control_points'],
            params['quaternion'],
            params['translation'],
            t_values,
            sample_pos_jax,
            valid_jax,
            oc.k,
        )
        best_params = params.copy()
        opt_state = optimizer.init(params)
        timings["normalize_curve"] += time.time() - t_norm

    total_time = time.time() - total_start
    timing_breakdown = {
        "precompute + transfer": timings["precompute"],
        "initial loss eval": timings["initial_loss"],
        "JIT compile update_step": timings.get("jit_update_step", 0.0),
        "iteration loop (total)": timings["iterations"],
        "normalize_curve (total)": timings["normalize_curve"],
        "wall time (optim only)": total_time,
    }
    log.summary(total_time, best_restart, best_iteration,
                best_loss, _get_loss_components(best_params),
                timing_breakdown)
    log.close()

    return (
        np.array(best_params['control_points']),
        np.array(best_params['quaternion']),
        np.array(best_params['translation']),
        best_loss,
    )
