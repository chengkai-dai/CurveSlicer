"""Logging utilities for curved slicing optimization.

Provides :class:`OptimizationLogger`, a lightweight wrapper around
Python's :mod:`logging` module that handles file and console output
for the optimization loop. This keeps all formatting and I/O concerns
out of the core optimizer code.

Typical usage inside :func:`~curved_slicing.optimizer.optimize`::

    log = OptimizationLogger(verbose=True, log_file="optimization.log")
    log.header(n_samples, n_cp, slicing_config, optimizer_config)
    log.initial(total_loss, loss_components)
    ...
    log.close()
"""

import logging as _logging
from typing import Dict, Optional, Sequence, Tuple

_SEP = "=" * 64

# Type alias for the 5-element loss breakdown returned by compute_all_losses:
#   (collision, floating, support, completeness, surface_quality)
LossComponents = Tuple[float, float, float, float, float]


def _fmt_losses(components: LossComponents) -> str:
    """Format a loss-component tuple into a human-readable string."""
    coll, flt, sup, comp, surf = components
    return (f"coll={coll:<10.4f} float={flt:<10.4f} "
            f"supp={sup:<10.4f} comp={comp:<10.4f} "
            f"surf={surf:<10.4f}")


class OptimizationLogger:
    """Structured logger for the curved-slicing optimisation loop.

    Manages a :class:`logging.Logger` with optional file and console
    handlers.  All formatting details live here so that
    :func:`~curved_slicing.optimizer.optimize` only needs one-line
    calls such as ``log.step(...)`` or ``log.restart_end(...)``.

    Args:
        verbose: Attach a console (stderr) handler at *INFO* level.
        log_file: Path to a log file.  ``None`` disables file output.
    """

    def __init__(
        self,
        verbose: bool = True,
        log_file: Optional[str] = "optimization.log",
    ) -> None:
        self._logger = _logging.getLogger("curved_slicing.optimizer")
        self._handlers: list[_logging.Handler] = []
        self._iter_width = 1
        self._max_iterations = 1
        self._log_interval = 50

        if log_file:
            fh = _logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(_logging.Formatter(
                "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)
            self._handlers.append(fh)

        if verbose:
            ch = _logging.StreamHandler()
            ch.setLevel(_logging.INFO)
            ch.setFormatter(_logging.Formatter("%(message)s"))
            self._logger.addHandler(ch)
            self._handlers.append(ch)

        if self._handlers:
            self._logger.setLevel(_logging.DEBUG)

    # ------------------------------------------------------------------
    # Public API – called from optimizer.optimize()
    # ------------------------------------------------------------------

    def header(self, n_samples: int, n_cp: int,
               slicing_config, optimizer_config) -> None:
        """Log the configuration banner at the start of optimisation."""
        sc = slicing_config
        oc = optimizer_config
        self._max_iterations = oc.max_iterations
        self._iter_width = len(str(oc.max_iterations))
        self._log_interval = max(1, oc.log_interval)

        self._logger.info("")
        self._logger.info(_SEP)
        self._logger.info("  Curved Slicing Optimization")
        self._logger.info(
            "  Samples: %d  |  Layers: %d  |  Control points: %d",
            n_samples, sc.n_layers, n_cp,
        )
        self._logger.info(
            "  Restarts: %d  |  Max iters/restart: %d  |  LR: %s",
            oc.max_restarts, oc.max_iterations, oc.learning_rate,
        )
        self._logger.info("  Optimizing: %s", ", ".join(oc.optimize_params))
        if oc.k_start is not None and oc.k_start != oc.k:
            self._logger.info(
                "  k-annealing: %.0f -> %.0f (log-space, per restart)",
                oc.k_start, oc.k,
            )
        else:
            self._logger.info("  k: %.0f (constant)", oc.k)
        self._logger.info(_SEP)

    def initial(self, total_loss: float,
                components: LossComponents) -> None:
        """Log the initial (pre-optimisation) loss breakdown."""
        self._logger.info("  Initial  loss=%.6f", total_loss)
        self._logger.info("           %s", _fmt_losses(components))

    def restart_begin(self, restart: int, max_restarts: int) -> None:
        """Log the start of a new restart round."""
        self._logger.info("")
        self._logger.info("--- Restart %d/%d ---", restart + 1, max_restarts)

    def step(self, iteration: int, loss_val: float,
             best_loss: float, improved: bool, elapsed: float,
             k_current: float = None) -> None:
        """Log a single iteration (only at ``log_interval`` boundaries).

        Args:
            k_current: If not None, the current k value (shown when
                k-annealing is active).
        """
        if not self._should_log(iteration):
            return
        marker = " *" if improved else ""
        k_str = f"  k={k_current:.0f}" if k_current is not None else ""
        self._logger.info(
            "  [%*d/%d]  loss=%.6f  best=%.6f%s%s  (%.1fs)",
            self._iter_width, iteration + 1, self._max_iterations,
            loss_val, best_loss, marker, k_str, elapsed,
        )

    def grad_health(self, stats: Dict[str, Dict[str, float]]) -> None:
        """Log per-parameter gradient diagnostics.

        Called right after :meth:`step` at ``log_interval`` boundaries.

        Args:
            stats: ``{param_name: {'grad_norm': ..., 'grad_max': ...,
                       'grad_min': ..., 'update_norm': ...}}``
        """
        # Determine column width from the longest parameter name
        max_key = max(len(k) for k in stats) if stats else 15
        for key in sorted(stats):
            s = stats[key]
            frozen = all(s[m] == 0.0 for m in ('grad_norm', 'grad_max'))
            tag = "  (frozen)" if frozen else ""
            self._logger.info(
                "  %*s  grad  %-*s  norm=%.4e  max=%.4e  min=%.4e%s",
                self._iter_width + 5, "",          # indent to align with step
                max_key, key,
                s['grad_norm'], s['grad_max'], s['grad_min'], tag,
            )
            if not frozen:
                self._logger.info(
                    "  %*s  upd   %-*s  norm=%.4e",
                    self._iter_width + 5, "",
                    max_key, key,
                    s['update_norm'],
                )

    def diverged(self, iteration: int) -> None:
        """Log an early-stop event due to loss divergence."""
        self._logger.warning(
            "  ** Diverged at iter %d, early stop", iteration + 1,
        )

    def restart_end(self, iters_done: int, elapsed: float,
                    local_best: float, global_best: float) -> None:
        """Log the summary line at the end of a restart round."""
        speed = iters_done / elapsed if elapsed > 0 else 0
        self._logger.info(
            "  -- %d iters in %.1fs (%.1f it/s) | "
            "local_best=%.6f | global_best=%.6f",
            iters_done, elapsed, speed, local_best, global_best,
        )

    def timing(self, label: str, elapsed: float) -> None:
        """Log a timing measurement for a named phase."""
        self._logger.info("  [time] %-24s %7.2fs", label, elapsed)

    def summary(self, total_time: float, best_restart: int,
                best_iteration: int, best_loss: float,
                components: LossComponents,
                timing_breakdown: Optional[dict] = None) -> None:
        """Log the final optimisation summary with optional timing."""
        self._logger.info("")
        self._logger.info(_SEP)
        self._logger.info("  Optimization complete in %.1fs", total_time)
        self._logger.info(
            "  Best: restart %d, iter %d, loss=%.6f",
            best_restart + 1, best_iteration + 1, best_loss,
        )
        self._logger.info("  %s", _fmt_losses(components))
        if timing_breakdown:
            self._logger.info("")
            self._logger.info("  Timing breakdown:")
            for label, secs in timing_breakdown.items():
                self._logger.info("    %-26s %7.2fs", label, secs)
        self._logger.info(_SEP)

    def close(self) -> None:
        """Remove and close all handlers added by this logger."""
        for h in self._handlers:
            self._logger.removeHandler(h)
            h.close()
        self._handlers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_log(self, iteration: int) -> bool:
        return ((iteration + 1) % self._log_interval == 0
                or iteration == self._max_iterations - 1)
