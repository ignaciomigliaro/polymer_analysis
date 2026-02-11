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

def rdf_path_resolve(base: Path, solvent: str, carbon: int, func: int, run: int) -> Path | None:
    prod = base / f"C{carbon}" / str(func) / f"run{run}" / "prod"

    candidates = [
        prod / f"rdf_Cpoly_PDCmolCOM_{solvent}_C{carbon}_{func}_run{run}.xvg",
        prod / f"rdf_Cpoly_DCBmolCOM_{solvent}_C{carbon}_{func}_run{run}.xvg",
    ]

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
    quiet_missing: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    x_ref = None
    ys: list[np.ndarray] = []

    for run in runs:
        p = rdf_path_resolve(base, solvent, carbon, func, run)

        if p is None:
            if not quiet_missing and verbose:
                prod = base / f"C{carbon}" / str(func) / f"run{run}" / "prod"
                print(f"[MISS] no PDC/DCB RDF in {prod}")
            continue

        x, y = parse_xvg(p)

        if x_ref is None:
            x_ref = x
        else:
            # keep your original tolerance
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


def plot_superplot_rdf(
    base: Path,
    *,
    solvent: str,
    carbons: list[int],
    funcs: list[int],
    runs: list[int],
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
                base, solvent, carbon, func, runs, quiet_missing=quiet_missing, verbose=verbose
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
    atol_grid: float = 1e-6,
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