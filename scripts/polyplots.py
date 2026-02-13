from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from typing import Iterable, Callable


# ============================================================
# Common utilities
# ============================================================

_RUN_RE = re.compile(r"^run(\d+)$", re.IGNORECASE)

def parse_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(xs, float), np.asarray(ys, float)


# ============================================================
# Generic walker for "simple" properties:
# base/C{carbon}/{func}/run{run}/prod/<filename>
# ============================================================

@dataclass(frozen=True)
class SeriesKey:
    carbon: int
    func: int
    run: int

@dataclass(frozen=True)
class Series:
    key: SeriesKey
    path: Path


from pathlib import Path
import numpy as np
import re


def parse_polystat_xvg(path):
    """
    Parse GROMACS polystat-style .xvg files.

    Supports:
      - intdist.xvg   (single y column)
      - persist.xvg   (single y column)
      - polystat.xvg  (multiple y columns with legends)

    Returns:
      x : np.ndarray
      ys : dict[str, np.ndarray]
          Mapping legend -> y-values
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # --- extract legends (only present for multi-column files) ---
    legends = {}
    for ln in lines:
        m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', ln)
        if m:
            legends[int(m.group(1))] = m.group(2)

    # --- numeric data ---
    rows = []
    for ln in lines:
        if not ln or ln[0] in ("#", "@"):
            continue
        parts = ln.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"No numeric XY data found in {path}")

    x = arr[:, 0]
    ycols = arr[:, 1:]

    ys = {}
    for i in range(ycols.shape[1]):
        legend = legends.get(i, f"y{i}")
        ys[legend] = ycols[:, i]

    return x, ys



def index_prod_property(
    base: Path,
    filename: str,
    *,
    carbons: Iterable[int] | None = None,
    funcs: Iterable[int] | None = None,
    runs: Iterable[int] | None = None,
) -> dict[int, dict[int, dict[int, Path]]]:
    """
    Returns nested dict:
      data[carbon][func][run] = path_to_file
    for files matching:
      base/C*/<func>/run*/prod/<filename>
    """
    base = Path(base)
    carbons_set = set(carbons) if carbons is not None else None
    funcs_set = set(funcs) if funcs is not None else None
    runs_set = set(runs) if runs is not None else None

    data: dict[int, dict[int, dict[int, Path]]] = {}

    for p in sorted(base.glob(f"C*/[0-9]*/run*/prod/{filename}")):
        try:
            carbon_dir = p.parents[3].name  # C20
            func_dir   = p.parents[2].name  # 10
            run_dir    = p.parents[1].name  # run2

            carbon = int(carbon_dir.lstrip("C"))
            func = int(func_dir)

            m = _RUN_RE.match(run_dir)
            if not m:
                continue
            run = int(m.group(1))
        except Exception:
            continue

        if carbons_set is not None and carbon not in carbons_set:
            continue
        if funcs_set is not None and func not in funcs_set:
            continue
        if runs_set is not None and run not in runs_set:
            continue

        data.setdefault(carbon, {}).setdefault(func, {})[run] = p

    return data


def load_runs_from_index(
    idx: dict[int, dict[int, dict[int, Path]]],
    *,
    carbon: int,
    func: int,
    runs: list[int],
    quiet_missing: bool = False,
    verbose: bool = True,
    atol_grid: float = 1e-6,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """
    Load runs for a given carbon+func from an index created by index_prod_property().
    Enforces identical x-grid (skips mismatches).
    """
    x_ref = None
    ys: list[np.ndarray] = []

    paths_by_run = idx.get(carbon, {}).get(func, {})

    for run in runs:
        p = paths_by_run.get(run)

        if p is None or not p.exists():
            if not quiet_missing and verbose:
                prod = Path("C")  # just for printing context below
                print(f"[MISS] no file for C{carbon}/{func}/run{run}/prod")
            continue

        x, y = parse_xvg(p)

        if x_ref is None:
            x_ref = x
        else:
            if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=0, atol=atol_grid):
                if verbose:
                    print(f"[SKIP] x-grid mismatch: {p}")
                continue

        ys.append(y)

    return x_ref, ys


# ============================================================
# KDE (dependency-free)
# ============================================================

def _silverman_bw(samples: np.ndarray) -> float:
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 0:
        sigma = std if std > 0 else 1.0
    return 0.9 * sigma * n ** (-1 / 5)


def kde_1d(samples: np.ndarray, grid: np.ndarray, bw: str | float = "scott") -> np.ndarray:
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)

    if isinstance(bw, str):
        h = _silverman_bw(x)
    else:
        h = float(bw)

    if h <= 0:
        h = 1.0

    diffs = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (h * np.sqrt(2 * np.pi))
    return dens


# ============================================================
# Generic "simple property" superplot
# ============================================================

def plot_superplot_simple(
    base: Path,
    *,
    filename: str,
    carbons: list[int],
    funcs: list[int],
    runs: list[int],
    ncols: int = 3,
    thin_runs: bool = True,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    savepath: Path | None = None,
    dpi: int = 300,
    verbose: bool = True,
    atol_grid: float = 1e-6,
    # NEW: choose ONE visualization mode
    mode: str = "time",          # "time" or "kde"
    kde_mode: str = "mean",      # used only if mode="kde": "mean" or "runs"
    kde_bw: str | float = "scott",
    kde_points: int = 256,
) -> tuple[plt.Figure, np.ndarray]:
    """
    mode="time": plot x vs y (thin runs + thick mean per func)
    mode="kde" : plot KDE of y-values (no time/x plot). Uses mean curve or pooled runs.
    """
    if mode not in ("time", "kde"):
        raise ValueError("mode must be 'time' or 'kde'")

    idx = index_prod_property(base, filename, carbons=carbons, funcs=funcs, runs=runs)

    n = len(carbons)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.6 * ncols, 4.0 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    if title is None:
        title = f"{filename} Superplot — {mode}"

    for i, carbon in enumerate(carbons):
        ax = axes[i]
        plotted = False

        funcs_use = [0] if carbon == 6 else funcs
        quiet_missing = (carbon == 6)

        # For KDE mode we collect samples per func, then plot KDE curves
        kde_payload: dict[int, np.ndarray] = {}
        color_map: dict[int, str] = {}

        for func in funcs_use:
            x_ref, ys = load_runs_from_index(
                idx,
                carbon=carbon,
                func=func,
                runs=runs,
                quiet_missing=quiet_missing,
                verbose=verbose,
                atol_grid=atol_grid,
            )
            if x_ref is None or len(ys) == 0:
                continue

            Y = np.vstack(ys)
            mean_y = Y.mean(axis=0)

            if mode == "time":
                if thin_runs:
                    for y in ys:
                        ax.plot(x_ref, y, alpha=0.2, linewidth=1)

                line, = ax.plot(x_ref, mean_y, linewidth=2.5, label=f"{func}% (n={len(ys)})")
                color_map[func] = line.get_color()
                plotted = True

            else:  # mode == "kde"
                # Make a dummy line to get a consistent color per func from mpl cycle
                # (then immediately remove it)
                dummy, = ax.plot([], [], linewidth=2.5, label=f"{func}% (n={len(ys)})")
                color_map[func] = dummy.get_color()
                dummy.remove()

                if kde_mode == "mean":
                    kde_payload[func] = mean_y
                elif kde_mode == "runs":
                    kde_payload[func] = Y.ravel()
                else:
                    raise ValueError("kde_mode must be 'mean' or 'runs'")
                plotted = True

        # ---- axis formatting ----
        ax.set_title(f"C{carbon}")

        if mode == "time":
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if plotted:
                ax.legend(frameon=False, fontsize=9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        else:
            # KDE-only plot: x = density, y = value
            ax.set_xlabel("density")
            ax.set_ylabel(ylabel)

            if plotted and kde_payload:
                # build y grid from pooled samples to set consistent range
                all_samples = np.concatenate([v[np.isfinite(v)] for v in kde_payload.values() if v.size > 0])
                if all_samples.size == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ylo, yhi = np.min(all_samples), np.max(all_samples)
                    if np.isclose(ylo, yhi):
                        ylo -= 1.0
                        yhi += 1.0
                    y_grid = np.linspace(ylo, yhi, kde_points)

                    for func, samples in kde_payload.items():
                        d = kde_1d(samples, y_grid, bw=kde_bw)
                        ax.plot(y_grid, d, linewidth=2.5,
                                color=color_map.get(func),
                                label=f"{func}%")
    
                        
                    ax.legend(frameon=False, fontsize=9)
                    ax.grid(True, axis="x", alpha=0.2)
                    ax.grid(False, axis="y")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[OK] Saved {savepath}")

    return fig, axes

# ============================================================
# RDF superplot (your functional code, wrapped)
# ============================================================


def rdf_path_resolve(
    base: Path,
    solvent: str,
    carbon: int,
    func: int,
    run: int,
    *,
    static_name: str | None = None,
) -> Path | None:
    prod = base / f"C{carbon}" / str(func) / f"run{run}" / "prod"

    candidates: list[Path] = []

    # 1) Static first (if provided)
    if static_name:
        candidates.append(prod / static_name)

    # 2) Dynamic fallback
    candidates.extend([
        prod / f"rdf_Cpoly_PDCmolCOM_{solvent}_C{carbon}_{func}_run{run}.xvg",
        prod / f"rdf_Cpoly_DCBmolCOM_{solvent}_C{carbon}_{func}_run{run}.xvg",
    ])

    for p in candidates:
        if p.exists():
            return p
    return None




def load_runs_rdf(
    base: Path,
    solvent: str,
    carbon: int,
    func: int,
    runs: list[int],
    *,
    static_name: str | None = None,
    quiet_missing: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    x_ref = None
    ys: list[np.ndarray] = []

    for run in runs:
        p = rdf_path_resolve(base, solvent, carbon, func, run, static_name=static_name)

        if p is None:
            if not quiet_missing and verbose:
                prod = base / f"C{carbon}" / str(func) / f"run{run}" / "prod"
                print(f"[MISS] no RDF in {prod}")
            if static_name:
                msg += f" (also tried static_name={static_name})"
            print(msg)
            continue

        x, y = parse_xvg(p)

        if x_ref is None:
            x_ref = x
        else:
            if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=0, atol=5):
                if verbose:
                    print(f"[SKIP] x-grid mismatch: {p}")
                continue

        ys.append(y)

    return x_ref, ys


def plot_superplot_rmsdist(base: Path, **kwargs):
    return plot_superplot_simple(
        base,
        filename="rmsdist.xvg",
        xlabel="time (ps)",
        ylabel="rmsdist",
        **kwargs,
    )

from pathlib import Path
import numpy as np
import re


def compute_overall_mean_std_across_runs(
    base: Path,
    *,
    filename: str,
    carbon: int | list[int],
    func: int | list[int],
    runs: list[int],
    tmin: float | None = None,   # only use data with x >= tmin
    tmax: float | None = None,   # only use data with x <= tmax
    verbose: bool = True,
    # NEW: robustness knobs
    drop_nonfinite: bool = True,          # drop NaN/Inf before averaging
    min_points: int = 1,                  # minimum points required after filtering (and dropping NaN/Inf)
    report_nonfinite: bool = True,        # print how many NaN/Inf were dropped per run
):
    """
    Compute ONE scalar per run (time-average), then mean/std across runs.

    Returns
    -------
    results : dict
        results[(carbon, func)] = {
            "run_means": {run: mean_value_for_that_run, ...},
            "mean": overall_mean_across_runs,
            "std":  std_across_runs (ddof=1, 0 if only 1 run),
            "n":    number_of_runs_used
        }

    Notes
    -----
    - If your .xvg contains NaNs (common in polystat outputs), np.mean(...) becomes NaN.
      Setting drop_nonfinite=True (default) makes the function ignore those frames and still
      compute a valid mean when possible.
    """
    carbons = carbon if isinstance(carbon, (list, tuple)) else [carbon]
    funcs = func if isinstance(func, (list, tuple)) else [func]

    def parse_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in ("#", "@"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
        return np.asarray(xs, float), np.asarray(ys, float)

    results: dict[tuple[int, int], dict] = {}

    for c in carbons:
        for f in funcs:
            run_means: dict[int, float] = {}

            for run in runs:
                p = base / f"C{c}" / str(f) / f"run{run}" / "prod" / filename
                if not p.exists():
                    if verbose:
                        print(f"[MISS] {p}")
                    continue

                x, y = parse_xvg(p)

                # time/window mask
                mask = np.ones_like(x, dtype=bool)
                if tmin is not None:
                    mask &= (x >= tmin)
                if tmax is not None:
                    mask &= (x <= tmax)

                ysel = y[mask]

                if ysel.size == 0:
                    if verbose:
                        print(f"[SKIP] no data after filtering: {p}")
                    continue

                # drop NaN/Inf (common in persist/polystat outputs)
                if drop_nonfinite:
                    finite = np.isfinite(ysel)
                    n_bad = int((~finite).sum())
                    if report_nonfinite and verbose and n_bad > 0:
                        print(f"[WARN] dropped {n_bad} NaN/Inf points: {p}")
                    ysel = ysel[finite]

                if ysel.size < min_points:
                    if verbose:
                        print(f"[SKIP] insufficient valid points (n={ysel.size}): {p}")
                    continue

                # robust mean
                run_means[run] = float(ysel.mean())

            if len(run_means) == 0:
                if verbose:
                    print(f"[WARN] No valid runs for C{c}, func={f}")
                continue

            vals = np.array(list(run_means.values()), dtype=float)

            # (vals should be finite, but guard anyway)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                if verbose:
                    print(f"[WARN] All run means non-finite for C{c}, func={f}")
                continue

            overall_mean = float(vals.mean())
            overall_std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

            results[(c, f)] = {
                "run_means": run_means,
                "mean": overall_mean,
                "std": overall_std,
                "n": int(vals.size),
            }

    return results





def plot_property_vs_chainlength_by_func(
    *,
    base_root: Path,
    solvents: list[str],
    filename: str,
    carbons: list[int],
    funcs: list[int],
    runs: list[int],
    # NEW: for polystat.xvg choose which series to use
    polystat_metric: str | None = None,   # "rg" or "end-to-end" when filename=="polystat.xvg"
    tmin: float | None = None,
    tmax: float | None = None,
    drop_nonfinite: bool = True,
    min_points: int = 1,
    ncols: int = 3,
    show_errorbars: bool = True,
    outpath: Path | None = None,
    dpi: int = 300,
    verbose: bool = True,
):
    """
    Superplot:
      x = carbon chain length
      y = time-averaged property (mean across runs)
      panels = functionalization levels
      lines = solvents (e.g. 1-2-DCB vs 1-4-DCB)

    Directory structure:
      base_root/<solvent>/C{carbon}/{func}/run{run}/prod/<filename>

    Supports single-series XVG files (persist.xvg, intdist.xvg, rmsdist.xvg, etc.)
    and polystat.xvg (multi-series) via polystat_metric="rg" or "end-to-end".
    """

    base_root = Path(base_root)

    # --- parser for both single- and multi-series polystat-style files ---
    def parse_polystat_xvg(path: Path):
        lines = path.read_text(errors="replace").splitlines()

        legends = {}
        for ln in lines:
            m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', ln)
            if m:
                legends[int(m.group(1))] = m.group(2)

        rows = []
        for ln in lines:
            if not ln or ln[0] in ("#", "@"):
                continue
            parts = ln.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"No numeric XY data found in {path}")

        x = arr[:, 0]
        ycols = arr[:, 1:]

        ys = {}
        for i in range(ycols.shape[1]):
            legend = legends.get(i, f"y{i}")
            ys[legend] = ycols[:, i]
        return x, ys

    def parse_simple_xvg(path: Path):
        xs, ys = [], []
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in ("#", "@"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
        return np.asarray(xs, float), np.asarray(ys, float)

    # --- choose series from polystat.xvg ---
    def pick_polystat_series(ys: dict[str, np.ndarray], metric: str) -> tuple[str, np.ndarray]:
        m = metric.strip().lower().replace("_", "-")

        items = [(k, k.lower()) for k in ys.keys()]

        def pick(pred):
            for k, kl in items:
                if pred(kl):
                    return k
            return None

        if m in ("rg", "gyr", "gyration", "radius-of-gyration"):
            # Prefer the non-eigen Rg: <R\sg\N> without "eig"
            key = pick(lambda s: ("r\\sg\\n" in s) and ("eig" not in s))
            if key is None:
                key = pick(lambda s: ("gyr" in s) or ("radius" in s) or ("rg" in s) or ("r\\sg\\n" in s))
            if key is None:
                raise KeyError(f"No Rg-like legend found. Available: {list(ys.keys())}")
            return key, ys[key]

        if m in ("end-to-end", "end2end", "endtoend", "ete"):
            key = pick(lambda s: ("end" in s) and ("eig" not in s))
            if key is None:
                raise KeyError(f"No end-to-end-like legend found. Available: {list(ys.keys())}")
            return key, ys[key]

        raise ValueError("polystat_metric must be 'rg' or 'end-to-end'")

    # --- validate polystat request ---
    is_polystat = (Path(filename).name == "polystat.xvg")
    if is_polystat and not polystat_metric:
        raise ValueError("When filename='polystat.xvg', you must set polystat_metric='rg' or 'end-to-end'")

    # --- compute per-(solvent, carbon, func) mean±std across runs ---
    def overall_mean_std_for_combo(
        base_solvent: Path,
        *,
        carbon: int,
        func: int,
        filename: str,
        runs: list[int],
    ) -> tuple[float, float, int]:
        run_means = []

        for run in runs:
            p = base_solvent / f"C{carbon}" / str(func) / f"run{run}" / "prod" / filename

            if not p.exists():
                if verbose:
                    print(f"[MISS] {p}")
                continue

            if Path(filename).name == "polystat.xvg":
                x, ys_map = parse_polystat_xvg(p)
                _, y = pick_polystat_series(ys_map, polystat_metric)  # type: ignore[arg-type]
            else:
                x, y = parse_simple_xvg(p)

            mask = np.ones_like(x, dtype=bool)
            if tmin is not None:
                mask &= (x >= tmin)
            if tmax is not None:
                mask &= (x <= tmax)

            ysel = y[mask]

            if drop_nonfinite:
                ysel = ysel[np.isfinite(ysel)]

            if ysel.size < min_points:
                continue

            run_means.append(float(ysel.mean()))

        if len(run_means) == 0:
            return np.nan, np.nan, 0

        vals = np.asarray(run_means, float)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
        n = int(np.isfinite(vals).sum())
        return mean, std, n

    # ---- figure layout ----
    n_panels = len(funcs)
    ncols_use = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols_use)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_use,
        figsize=(5.2 * ncols_use, 3.8 * nrows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    # ylabel label
    ylab = filename
    if is_polystat:
        ylab = f"polystat: {polystat_metric}"

    # ---- plot ----
    for i, func in enumerate(funcs):
        ax = axes[i]
        plotted = False

        for solvent in solvents:
            base_solvent = base_root / solvent

            xs, means, stds = [], [], []

            for carbon in carbons:
                mean, std, n = overall_mean_std_for_combo(
                    base_solvent,
                    carbon=carbon,
                    func=func,
                    filename=filename,
                    runs=runs,
                )
                xs.append(carbon)
                means.append(mean)
                stds.append(std)

            xs = np.asarray(xs, int)
            means = np.asarray(means, float)
            stds = np.asarray(stds, float)

            ok = np.isfinite(means)
            if not ok.any():
                continue

            if show_errorbars:
                ax.errorbar(
                    xs[ok],
                    means[ok],
                    yerr=stds[ok],
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=solvent,
                )
            else:
                ax.plot(
                    xs[ok],
                    means[ok],
                    marker="o",
                    linewidth=2,
                    label=solvent,
                )

            plotted = True

        ax.set_title(f"{func}% functionalization")
        ax.set_xlabel("Carbon chain length")
        ax.set_ylabel(f"⟨{ylab}⟩")

        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    window = ""
    if tmin is not None or tmax is not None:
        window = f" (window: {tmin if tmin is not None else '-inf'} to {tmax if tmax is not None else 'inf'})"

    fig.suptitle(
        f"Property vs Chain Length — {ylab}{window}",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[OK] Saved {outpath}")

    return fig, axes


    
def compute_mean_std_across_runs(
    base: Path,
    *,
    filename: str,
    carbon: int | list[int],
    func: int | list[int],
    runs: list[int],
    atol_grid: float = 1e-6,
    verbose: bool = True,
):
    """
    Compute mean/std across runs for multiple carbons and funcs.

    Returns
    -------
    results : dict
        results[(carbon, func)] = {
            "x": x_array,
            "mean": mean_array,
            "std": std_array,
            "n": number_of_runs_used
        }
    """

    # normalize inputs to lists
    carbons = carbon if isinstance(carbon, (list, tuple)) else [carbon]
    funcs = func if isinstance(func, (list, tuple)) else [func]

    def parse_xvg(path):
        xs, ys = [], []
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in ("#", "@"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
        return np.asarray(xs), np.asarray(ys)

    results = {}

    for c in carbons:
        for f in funcs:

            x_ref = None
            ys = []

            for run in runs:
                p = (
                    base
                    / f"C{c}"
                    / str(f)
                    / f"run{run}"
                    / "prod"
                    / filename
                )

                if not p.exists():
                    if verbose:
                        print(f"[MISS] {p}")
                    continue

                x, y = parse_xvg(p)

                if x_ref is None:
                    x_ref = x
                else:
                    if len(x) != len(x_ref) or not np.allclose(x, x_ref, atol=atol_grid):
                        if verbose:
                            print(f"[SKIP] grid mismatch: {p}")
                        continue

                ys.append(y)

            if len(ys) == 0:
                if verbose:
                    print(f"[WARN] No valid runs for C{c}, func={f}")
                continue

            Y = np.vstack(ys)
            mean_y = Y.mean(axis=0)
            std_y = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean_y)

            results[(c, f)] = {
                "x": x_ref,
                "mean": mean_y,
                "std": std_y,
                "n": len(ys),
            }

    return results



def plot_superplot_rdf(
    base: Path,
    *,
    solvent: str,
    carbons: list[int],
    funcs: list[int],
    runs: list[int],
    static_name: str | None = None,
    outpath: Path | None = None,
    ncols: int = 3,
    thin_runs: bool = True,
    title: str | None = None,
    dpi: int = 300,
    verbose: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    n = len(carbons)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, carbon in enumerate(carbons):
        ax = axes[i]

        funcs_use = [0] if carbon == 6 else funcs
        quiet_missing = carbon == 6

        plotted = False

        for func in funcs_use:
            x_ref, ys = load_runs_rdf(
                base, 
                solvent, 
                carbon, 
                func, 
                runs,
                static_name=static_name, 
                quiet_missing=quiet_missing, 
                verbose=verbose
            )
            if x_ref is None or len(ys) == 0:
                continue

            Y = np.vstack(ys)
            mean_y = Y.mean(axis=0)

            if thin_runs:
                for y in ys:
                    ax.plot(x_ref, y, alpha=0.2, linewidth=1)

            ax.plot(x_ref, mean_y, linewidth=2.5, label=f"{func}% (n={len(ys)})")
            plotted = True

        ax.set_title(f"C{carbon}")
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("g(r)")

        if plotted:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if title is None:
        title = f"RDF Superplot — {solvent} (thin = runs, thick = mean)"

    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[OK] Saved {outpath}")

    return fig, axes

def superplot_polystat_by_carbon(
    base: Path,
    *,
    metric: str,
    solvent: str | None = None,
    carbons: list[int],
    funcs: list[int],
    runs: list[int],
    ncols: int = 3,
    thin_runs: bool = True,
    outpath: Path | None = None,
    dpi: int = 300,
    verbose: bool = True,
    atol_grid: float = 1e-1,
    # NEW: choose ONE visualization mode
    mode: str = "time",                 # "time" or "kde"
    kde_mode: str = "mean",             # used only if mode="kde": "mean" or "runs"
    kde_bw: str | float = "scott",
    kde_points: int = 256,
    kde_orientation: str = "horizontal" # "horizontal" (density on x) or "vertical"
):
    """
    Superplot grid (one panel per carbon) for gmx polystat outputs.

    Layout assumed:
      base/C{carbon}/{func}/run{run}/prod/<file>

    Metrics:
      - "rg"         : from polystat.xvg (legend like <R\\sg\\N>)
      - "end-to-end" : from polystat.xvg (legend contains "end")
      - "persist"    : from persist.xvg (single series)
      - "intdist"    : from intdist.xvg (single series)

    mode="time" : plot time/trace (thin runs + thick mean)
    mode="kde"  : KDE-only (distribution of y-values; no time axis)
    """
    base = Path(base)

    if mode not in ("time", "kde"):
        raise ValueError("mode must be 'time' or 'kde'")
    kde_orientation = kde_orientation.lower().strip()
    if kde_orientation not in ("horizontal", "vertical"):
        raise ValueError("kde_orientation must be 'horizontal' or 'vertical'")

    def parse_xvg_xy(path):
        path = Path(path)
        lines = path.read_text(errors="replace").splitlines()

        legends = {}
        for ln in lines:
            m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', ln)
            if m:
                legends[int(m.group(1))] = m.group(2)

        data = []
        for ln in lines:
            if not ln or ln[0] in ("#", "@"):
                continue
            parts = ln.split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            data.append(row)

        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"No XY data found in {path}")

        x = arr[:, 0]
        ycols = arr[:, 1:]

        ys = {}
        for i in range(ycols.shape[1]):
            legend = legends.get(i, f"s{i}")
            ys[legend] = ycols[:, i]
        return x, ys

    def prod_dir(carbon: int, func: int, run: int) -> Path:
        return base / f"C{carbon}" / str(func) / f"run{run}" / "prod"

    def file_for_metric(metric_norm: str) -> str:
        if metric_norm in ("persist", "persistent"):
            return "persist.xvg"
        if metric_norm in ("intdist", "interatomic-distance", "interatomic", "distance"):
            return "intdist.xvg"
        if metric_norm in ("rg", "gyr", "gyration", "radius-of-gyration",
                           "end-to-end", "end2end", "endtoend", "ete"):
            return "polystat.xvg"
        raise ValueError("metric must be one of: 'rg', 'end-to-end', 'persist', 'intdist'")

    def pick_polystat_series(ys: dict[str, np.ndarray], metric_norm: str) -> tuple[str, np.ndarray]:
        items = [(k, k.lower()) for k in ys.keys()]

        def pick(pred):
            for k, kl in items:
                if pred(kl):
                    return k
            return None

        if metric_norm in ("rg", "gyr", "gyration", "radius-of-gyration"):
            # Prefer the non-eigen Rg if present
            key = pick(lambda s: ("r\\sg\\n" in s) and ("eig" not in s))  # <R\sg\N>
            if key is None:
                key = pick(lambda s: ("gyr" in s) or ("radius" in s) or ("rg" in s) or ("r\\sg\\n" in s))
            if key is None:
                raise KeyError(f"No Rg-like legend found. Available: {list(ys.keys())}")
            return key, ys[key]

        if metric_norm in ("end-to-end", "end2end", "endtoend", "ete"):
            key = pick(lambda s: "end" in s and "eig" not in s)
            if key is None:
                raise KeyError(f"No end-to-end-like legend found. Available: {list(ys.keys())}")
            return key, ys[key]

        raise ValueError("polystat series picker called with non-polystat metric")

    # KDE helpers (dependency-free)
    def _silverman_bw(samples: np.ndarray) -> float:
        x = np.asarray(samples, float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 2:
            return 1.0
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if sigma <= 0:
            sigma = std if std > 0 else 1.0
        return 0.9 * sigma * n ** (-1 / 5)

    def kde_1d(samples: np.ndarray, grid: np.ndarray, bw: str | float = "scott") -> np.ndarray:
        x = np.asarray(samples, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.zeros_like(grid)

        if isinstance(bw, str):
            h = _silverman_bw(x)
        else:
            h = float(bw)
        if h <= 0:
            h = 1.0

        diffs = (grid[:, None] - x[None, :]) / h
        dens = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (h * np.sqrt(2 * np.pi))
        return dens

    metric_norm = metric.strip().lower().replace("_", "-")

    n = len(carbons)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    # axis labels
    if metric_norm in ("persist", "persistent"):
        y_label = "Persistence"
        x_label = "time (ps)"
    elif metric_norm in ("intdist", "interatomic-distance", "interatomic", "distance"):
        y_label = "Interatomic distance"
        x_label = "time (ps)"
    elif metric_norm in ("rg", "gyr", "gyration", "radius-of-gyration"):
        y_label = "Rg"
        x_label = "time (ps)"
    else:
        y_label = "End-to-end"
        x_label = "time (ps)"

    # Per-carbon panels
    for i, carbon in enumerate(carbons):
        ax = axes[i]
        plotted = False

        funcs_use = [0] if carbon == 6 else funcs
        quiet_missing = (carbon == 6)

        # For KDE mode: collect samples per func
        kde_payload: dict[int, np.ndarray] = {}
        color_map: dict[int, str] = {}

        for func in funcs_use:
            x_ref = None
            ys_runs: list[np.ndarray] = []

            for run in runs:
                prod = prod_dir(carbon, func, run)
                fname = file_for_metric(metric_norm)
                p = prod / fname

                if not p.exists():
                    if verbose and not quiet_missing:
                        print(f"[MISS] {p}")
                    continue

                x, ys = parse_xvg_xy(p)

                # select series
                if fname == "polystat.xvg":
                    _, y = pick_polystat_series(ys, metric_norm)
                else:
                    y = next(iter(ys.values()))

                if x_ref is None:
                    x_ref = x
                else:
                    if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=0, atol=atol_grid):
                        if verbose:
                            print(f"[SKIP] x-grid mismatch: {p}")
                        continue

                ys_runs.append(y)

            if x_ref is None or len(ys_runs) == 0:
                continue

            Y = np.vstack(ys_runs)
            mean_y = Y.mean(axis=0)

            if mode == "time":
                if thin_runs:
                    for y in ys_runs:
                        ax.plot(x_ref, y, alpha=0.2, linewidth=1)

                line, = ax.plot(x_ref, mean_y, linewidth=2.5, label=f"{func}% (n={len(ys_runs)})")
                color_map[func] = line.get_color()
                plotted = True

            else:  # mode == "kde"
                # grab a consistent color per func from mpl cycle
                dummy, = ax.plot([], [], linewidth=2.5, label=f"{func}% (n={len(ys_runs)})")
                color_map[func] = dummy.get_color()
                dummy.remove()

                if kde_mode == "mean":
                    kde_payload[func] = mean_y
                elif kde_mode == "runs":
                    kde_payload[func] = Y.ravel()
                else:
                    raise ValueError("kde_mode must be 'mean' or 'runs'")
                plotted = True

        ax.set_title(f"C{carbon}")

        if mode == "time":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if plotted:
                ax.legend(frameon=False, fontsize=9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        else:
            # KDE-only plot
            if kde_orientation == "horizontal":
                ax.set_xlabel("density")
                ax.set_ylabel(y_label)
            else:
                ax.set_xlabel(y_label)
                ax.set_ylabel("density")

            if plotted and kde_payload:
                all_samples = np.concatenate([v[np.isfinite(v)] for v in kde_payload.values() if v.size > 0])
                if all_samples.size == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ylo, yhi = float(np.min(all_samples)), float(np.max(all_samples))
                    if np.isclose(ylo, yhi):
                        ylo -= 1.0
                        yhi += 1.0
                    y_grid = np.linspace(ylo, yhi, kde_points)

                    for func, samples in kde_payload.items():
                        d = kde_1d(samples, y_grid, bw=kde_bw)
                        if kde_orientation == "horizontal":
                            ax.plot(d, y_grid, linewidth=2.5, color=color_map.get(func), label=f"{func}%")
                        else:
                            ax.plot(y_grid, d, linewidth=2.5, color=color_map.get(func), label=f"{func}%")

                    ax.legend(frameon=False, fontsize=9)
                    ax.grid(True, alpha=0.2)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    title_solvent = f" — {solvent}" if solvent else ""
    fig.suptitle(
        f"{metric.upper()} Superplot{title_solvent} ({mode})",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[OK] Saved {outpath}")

    return fig, axes