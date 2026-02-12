"""Run curved slicing optimization on a mesh model.

This script loads a mesh, extracts the required geometric data, and
calls the ``curved_slicing`` optimizer to find an optimal curved
slicing path for DLP printing.

Usage:
    python run_optimization.py                          # bunny (default)
    python run_optimization.py --model hook
    python run_optimization.py --model bunny --lr 0.05 --iters 3000
    python run_optimization.py --help
"""

import argparse
import os
import sys
import time

# Project root (directory containing this script)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add src/ to Python path so the curved_slicing package is importable
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Suppress XLA autotuner C++ noise (buffer_comparator, gemm_fusion_autotuner).
# These go through abseil C++ logging directly to fd 2, bypassing Python.
# Strategy: redirect fd 2 to /dev/null, but keep Python sys.stderr on the
# original terminal so our own logging still works.
_real_stderr_fd = os.dup(2)
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
sys.stderr = os.fdopen(_real_stderr_fd, "w", closefd=False)

import jax
import numpy as np

# Persistent compilation cache — 2nd+ runs skip JIT entirely
jax.config.update("jax_compilation_cache_dir", os.path.join(
    _PROJECT_ROOT, ".jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from opt_core.mesh import Mesh
from opt_core.optimizer import optimize, SlicingConfig, OptimizerConfig


# ---------------------------------------------------------------------------
# Mesh data extraction
# ---------------------------------------------------------------------------

def load_mesh_data(mesh_path: str,
                   valid_threshold: float = 0.05,
                   surface_threshold: float = 0.5,
                   scale: float = 1.0):
    """Load a mesh and extract all arrays needed by :func:`optimize`.

    Args:
        mesh_path: Path to the ``.obj`` file.
        valid_threshold: Minimum y-coordinate for a vertex to be
            considered a valid optimization target (avoids the very
            bottom contact area).
        surface_threshold: Minimum y-coordinate for a vertex to be
            labelled surface-critical (for surface quality loss).
        scale: Optional uniform scale applied after normalization.

    Returns:
        Tuple of ``(sample_positions, sample_normals, connection,
        valid_idx, surface_valid_idx)``, all as numpy arrays.
    """
    mesh = Mesh(filename=mesh_path)
    mesh.normalize()
    mesh.centered()
    if scale != 1.0:
        mesh.scale(scale)

    n_vertices = mesh.topology.n_vertices()

    # --- Batch extract positions and normals (C++ backed, fast) ---
    sample_positions = np.array(mesh.topology.points(), dtype=np.float32)
    sample_normals = np.array(mesh.compute_vertex_normals(), dtype=np.float32)

    # --- Vectorized threshold masks ---
    y_coords = sample_positions[:, 1]
    valid_idx = np.where(y_coords > valid_threshold)[0].astype(np.int32)
    surface_valid_idx = np.where(y_coords > surface_threshold)[0].astype(np.int32)

    # --- Build padded neighbor connectivity matrix ---
    max_neighbors = max(
        len(list(mesh.topology.vv(v))) for v in mesh.topology.vertices()
    )
    connection = np.empty((n_vertices, max_neighbors), dtype=np.int32)
    for v in mesh.topology.vertices():
        i = v.idx()
        nb_idx = [nb.idx() for nb in mesh.topology.vv(v)]
        n_nb = len(nb_idx)
        connection[i, :n_nb] = nb_idx
        # Pad with far-away sentinel indices (n_vertices + vertex_idx)
        connection[i, n_nb:] = n_vertices + i

    return sample_positions, sample_normals, connection, valid_idx, surface_valid_idx


def make_initial_control_points(n_cp: int = 6, height: float = 1.01):
    """Generate a straight vertical initial Bezier curve.

    Args:
        n_cp: Number of control points.
        height: Height of the top-most control point (y).

    Returns:
        Control points array of shape ``(n_cp, 4)``.
    """
    cp = np.zeros((n_cp, 4), dtype=np.float32)
    cp[:, 1] = np.linspace(-0.0001, height, n_cp)
    return cp


# ---------------------------------------------------------------------------
# Per-model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "bunny": dict(
        mesh="data/bunny.obj",
        valid_threshold=0.05,
        surface_threshold=0.5,
        scale=1.0,
        height=1.01,
    ),
    "hook": dict(
        mesh="data/hook.obj",
        valid_threshold=0.05,
        surface_threshold=0.5,
        scale=1.0,
        height=1.01,
        control_points=np.array([
            [0.00295642, -0.001, -0.00147242, 0.0],
            [0.00295642, 0.7014588, 0.00147242, 0.0],
            [-0.09895631, 0.83290154, -0.00900843, 0.0],
            [0.2500953, 1.0621607, 0.00835638, 0.0],
            [0.86034113, 1.0184051, -0.01001998, 0.0],
            [0.71769106, 0.35, -0.00129661, 0.0],
        ], dtype=np.float32),
    ),
    "armadillo": dict(
        mesh="data/armadillo.obj",
        valid_threshold=0.05,
        surface_threshold=0.5,
        scale=1.2,
        height=1.01 * 1.25,
    ),
    "woman-pully": dict(
        mesh="data/woman-pully.obj",
        valid_threshold=0.05,
        surface_threshold=0.5,
        scale=0.85,
        height=1.01,
    ),
    "fertility": dict(
        mesh="data/fertility.obj",
        valid_threshold=0.05,
        surface_threshold=0.5,
        scale=0.75,
        height=1.01,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curved slicing optimization for DLP 3D printing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="bunny",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model name (selects mesh path and presets)")
    parser.add_argument("--mesh", default=None,
                        help="Override mesh path (ignores --model preset)")
    parser.add_argument("--n-layers", type=int, default=200,
                        help="Number of slicing layers")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Base learning rate")
    parser.add_argument("--iters", type=int, default=2000,
                        help="Max iterations per restart")
    parser.add_argument("--restarts", type=int, default=3,
                        help="Number of optimization restarts")
    parser.add_argument("--output", default="results",
                        help="Directory to save results")
    parser.add_argument("--log-file", default="optimization.log",
                        help="Log file path (empty to disable)")
    parser.add_argument("--setup-opt", action="store_true",
                        help="Also optimize quaternion and translation "
                             "(default: only control_points)")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable console output")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]

    # ---- Load mesh ----
    mesh_path = args.mesh or os.path.join(_PROJECT_ROOT, preset["mesh"])
    print(f"Loading mesh: {mesh_path}")
    sample_positions, sample_normals, connection, valid_idx, surface_valid_idx = \
        load_mesh_data(
            mesh_path,
            valid_threshold=preset["valid_threshold"],
            surface_threshold=preset["surface_threshold"],
            scale=preset.get("scale", 1.0),
        )
    print(f"  Vertices: {sample_positions.shape[0]}  |  "
          f"Valid: {valid_idx.shape[0]}  |  "
          f"Surface-critical: {surface_valid_idx.shape[0]}")

    # ---- Load platform samples ----
    platform_samples = np.loadtxt(
        os.path.join(_PROJECT_ROOT, "data", "platform_samples.txt"),
        delimiter=",",
    ).astype(np.float32)
    print(f"  Platform samples: {platform_samples.shape[0]}")

    # ---- Initial parameters ----
    if "control_points" in preset:
        initial_cp = preset["control_points"]
    else:
        initial_cp = make_initial_control_points(height=preset["height"])

    initial_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    initial_translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ---- Configure ----
    slicing_config = SlicingConfig(
        n_layers=args.n_layers,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=args.lr,
        max_iterations=args.iters,
        max_restarts=args.restarts,
        setup_opt=args.setup_opt,
    )

    # ---- Run optimization ----
    start_time = time.time()
    best_cp, best_quat, best_trans, best_loss = optimize(
        initial_control_points=initial_cp,
        initial_quaternion=initial_quaternion,
        initial_translation=initial_translation,
        sample_positions=sample_positions,
        sample_normals=sample_normals,
        platform_samples=platform_samples,
        connection=connection,
        valid=valid_idx,
        surface_valid_idx=surface_valid_idx,
        slicing_config=slicing_config,
        optimizer_config=optimizer_config,
        verbose=not args.quiet,
        log_file=args.log_file or None,
    )
    elapsed = time.time() - start_time

    # ---- Save results ----
    os.makedirs(args.output, exist_ok=True)
    cp_path = os.path.join(args.output, "best_control_points.txt")
    quat_path = os.path.join(args.output, "best_quaternion.txt")
    trans_path = os.path.join(args.output, "best_translation.txt")

    np.savetxt(cp_path, best_cp, fmt="%.6f")
    np.savetxt(quat_path, best_quat.reshape(1, -1), fmt="%.6f")
    np.savetxt(trans_path, best_trans.reshape(1, -1), fmt="%.6f")

    print(f"\nTotal time: {elapsed:.1f}s  |  Best loss: {best_loss:.6f}")
    print(f"Results saved to {args.output}/")
    print(f"  {cp_path}")
    print(f"  {quat_path}")
    print(f"  {trans_path}")

    print("\nBest control points:")
    print(best_cp)
    print(f"Best quaternion: {best_quat}")
    print(f"Best translation: {best_trans}")


if __name__ == "__main__":
    main()
