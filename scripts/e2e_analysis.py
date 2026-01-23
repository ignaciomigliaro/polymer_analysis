import os
import pickle
import matplotlib.pyplot as plt
import re

# ============================================================
# Cache utilities for E2E time series
# ============================================================

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# ---------- flexible topology ----------
def _find_topology_in_prod_dir(prod_dir, prefer_prefixes=("prod",), prefer_exts=(".tpr", ".gro")):
    cands = []
    for ext in prefer_exts:
        cands.extend(glob.glob(os.path.join(prod_dir, f"*{ext}")))
    if not cands:
        raise FileNotFoundError(f"No topology (.tpr/.gro) found in {prod_dir}")

    def score(p):
        base = os.path.basename(p).lower()
        st = os.stat(p)
        is_tpr = int(base.endswith(".tpr"))
        has_prefix = int(any(base.startswith(pref.lower()) for pref in prefer_prefixes))
        return (is_tpr, has_prefix, st.st_mtime, st.st_size)

    return sorted(cands, key=score, reverse=True)[0]

def _load_universe(prod_dir, xtc_name="prod.xtc"):
    xtc = os.path.join(prod_dir, xtc_name)
    if not os.path.isfile(xtc):
        raise FileNotFoundError(f"Missing {xtc_name} in {prod_dir}")
    topo = _find_topology_in_prod_dir(prod_dir)
    u = mda.Universe(topo, xtc)
    return u, topo, xtc

# ---------- polymer + end atoms ----------
def _polymer_as_largest_fragment(u):
    frags = u.atoms.fragments
    if len(frags) == 0:
        raise RuntimeError("No fragments found. Use a .tpr topology with bonds.")
    return max(frags, key=len)

def _heavy_atoms(poly, heavy_sel="not name H*"):
    heavy = poly.select_atoms(heavy_sel)
    if len(heavy) < 2:
        raise RuntimeError(f"Heavy selection '{heavy_sel}' returned <2 atoms. Adjust heavy_sel.")
    return heavy

def _terminals_by_bonds(group):
    idx_set = set(group.indices.tolist())
    terminals = []
    for a in group:
        deg = 0
        for b in getattr(a, "bonds", []):
            a1, a2 = b.atoms
            other = a2 if a1.index == a.index else a1
            if other.index in idx_set:
                deg += 1
        if deg == 1:
            terminals.append(a)
    return terminals

def _pick_end_atoms(u, heavy_sel="not name H*"):
    u.trajectory[0]
    poly = _polymer_as_largest_fragment(u)
    heavy = _heavy_atoms(poly, heavy_sel=heavy_sel)

    terms = _terminals_by_bonds(heavy)
    if len(terms) == 2:
        return poly, terms[0], terms[1], "bond_terminals_heavy"

    # If branching or selection ambiguity: choose farthest pair among heavy atoms (PBC-aware on *this* frame)
    coords = heavy.positions
    box = u.trajectory.ts.dimensions
    D = distance_array(coords, coords, box=box)
    np.fill_diagonal(D, -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return poly, heavy[i], heavy[j], f"geometry_farthest_heavy (terminals_found={len(terms)})"

# ---------- unwrap + compute time series ----------
def compute_e2e_timeseries_nm(prod_dir, xtc_name="prod.xtc", heavy_sel="not name H*"):
    u, topo, xtc = _load_universe(prod_dir, xtc_name=xtc_name)

    # pick ends once (on frame 0)
    poly, a1, a2, method = _pick_end_atoms(u, heavy_sel=heavy_sel)

    # UNWRAP polymer across PBC so intramolecular distance is continuous
    # This is the key fix for "crazy high" distances.
    from MDAnalysis.transformations import unwrap
    u.trajectory.add_transformations(unwrap(poly))

    times_ps = []
    dists_nm = []

    for ts in u.trajectory:
        # After unwrapping, compute simple Euclidean distance
        d_ang = np.linalg.norm(a1.position - a2.position)   # Å
        d_nm = d_ang / 10.0                                 # convert Å -> nm
        times_ps.append(ts.time)
        dists_nm.append(d_nm)

    times_ps = np.asarray(times_ps, float)
    dists_nm = np.asarray(dists_nm, float)

    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(len(times_ps)),
        "mean_nm": float(np.mean(dists_nm)),
        "std_nm": float(np.std(dists_nm, ddof=1)) if len(dists_nm) > 1 else 0.0,
        "box_firstframe": u.trajectory[0].dimensions,
    }
    return times_ps, dists_nm, meta

# ---------- simple KDE ----------
def gaussian_kde_1d(x, gridsize=400, bw_method="scott", bw=None, cut=3.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need >=2 samples for KDE.")
    std = np.std(x, ddof=1)

    if bw is None:
        if bw_method == "silverman":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            s = min(std, iqr / 1.34) if np.isfinite(iqr) and iqr > 0 else std
            bw = 0.9 * s * n ** (-1/5)
        else:
            bw = std * n ** (-1/5)
        if not np.isfinite(bw) or bw <= 0:
            bw = max(std, 1e-6) * 0.1

    lo = np.min(x) - cut * bw
    hi = np.max(x) + cut * bw
    grid = np.linspace(lo, hi, gridsize)
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))
    return grid, dens, bw

# ---------- plot: time or KDE ----------
def plot_e2e(prod_dir, mode="time", burn_in=0.0, downsample=1,
             rolling_window=None, kde_bw_method="scott", kde_bw=None,
             save_plot=None, figsize=(12,5)):
    t, d, meta = compute_e2e_timeseries_nm(prod_dir)

    # burn-in fraction
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(d) * burn_in)
        t = t[start:]
        d = d[start:]

    downsample = int(max(1, downsample))
    t_plot = t[::downsample]
    d_plot = d[::downsample]

    plt.figure(figsize=figsize)

    if mode.lower() == "time":
        plt.plot(t_plot, d_plot, lw=1.2, label="E2E (nm)")
        if rolling_window is not None and rolling_window >= 2 and len(d_plot) >= rolling_window:
            w = int(rolling_window)
            kernel = np.ones(w) / w
            d_roll = np.convolve(d_plot, kernel, mode="valid")
            t_roll = t_plot[w-1:]
            plt.plot(t_roll, d_roll, lw=2.0, label=f"Rolling mean (w={w})")
        plt.xlabel("Time (ps)")
        plt.ylabel("End-to-End Distance (nm)")
        plt.title(f"E2E vs Time | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

    elif mode.lower() == "kde":
        grid, dens, bw = gaussian_kde_1d(d_plot, bw_method=kde_bw_method, bw=kde_bw)
        plt.plot(grid, dens, lw=2.0, label=f"KDE (bw={bw:.3g} nm)")
        plt.xlabel("End-to-End Distance (nm)")
        plt.ylabel("Density")
        plt.title(f"E2E KDE | burn_in={burn_in} | downsample={downsample} | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

    else:
        raise ValueError("mode must be 'time' or 'kde'")

    plt.tight_layout()
    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
    plt.show()

    print("end atoms (0-based):", meta["end_atoms_0based"], "| names:", meta["end_atom_names"])
    print("mean E2E (nm):", meta["mean_nm"], "| std (nm):", meta["std_nm"])
    print("box (first frame):", meta["box_firstframe"])

    return t, d, meta


def _cache_path(prod_dir):
    """
    Returns path to cache file for a given /prod directory.
    """
    cache_dir = os.path.join(prod_dir, ".cache_e2e")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "e2e_timeseries.pkl")

def _file_sig(path):
    """
    Lightweight file signature used to invalidate cache
    if topology or trajectory changes.
    """
    if path is None or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime, st.st_size)

def _load_cache(path):
    """
    Load cache dict from disk if it exists.
    """
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(path, obj):
    """
    Atomically save cache dict to disk.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)



# ----------------------------
# loader + end picking (same logic as before)
# ----------------------------
def _find_topology_in_prod_dir(prod_dir, prefer_prefixes=("prod",), prefer_exts=(".tpr", ".gro")):
    candidates = []
    for ext in prefer_exts:
        candidates.extend(glob.glob(os.path.join(prod_dir, f"*{ext}")))
    if not candidates:
        raise FileNotFoundError(f"No topology files found in {prod_dir} (expected .tpr or .gro)")

    def score(p):
        base = os.path.basename(p).lower()
        st = os.stat(p)
        is_tpr = int(base.endswith(".tpr"))
        has_prefix = int(any(base.startswith(pref.lower()) for pref in prefer_prefixes))
        return (is_tpr, has_prefix, st.st_mtime, st.st_size)

    return sorted(candidates, key=score, reverse=True)[0]

def _load_universe_from_prod_dir(prod_dir, xtc_name="prod.xtc"):
    xtc = os.path.join(prod_dir, xtc_name)
    if not os.path.isfile(xtc):
        raise FileNotFoundError(f"Missing {xtc_name} in {prod_dir}")
    topo = _find_topology_in_prod_dir(prod_dir)
    u = mda.Universe(topo, xtc)
    return u, topo, xtc

def _polymer_as_largest_fragment(u):
    frags = u.atoms.fragments
    if len(frags) == 0:
        raise RuntimeError("No fragments found; ensure topology has bonds (TPR recommended).")
    return max(frags, key=len)

def _heavy_atoms(polymer_atoms, heavy_sel="not name H*"):
    heavy = polymer_atoms.select_atoms(heavy_sel)
    if len(heavy) < 2:
        raise RuntimeError(f"Heavy selection '{heavy_sel}' returned <2 atoms. Adjust heavy_sel.")
    return heavy

def _terminals_by_bonds(group):
    idx_set = set(group.indices.tolist())
    terminals = []
    for a in group:
        deg = 0
        for b in getattr(a, "bonds", []):
            a1, a2 = b.atoms
            other = a2 if a1.index == a.index else a1
            if other.index in idx_set:
                deg += 1
        if deg == 1:
            terminals.append(a)
    return terminals

def _pick_end_atoms_auto(u, heavy_sel="not name H*"):
    u.trajectory[0]
    poly = _polymer_as_largest_fragment(u)
    heavy = _heavy_atoms(poly, heavy_sel=heavy_sel)

    terms = _terminals_by_bonds(heavy)
    if len(terms) == 2:
        return poly, heavy, terms[0], terms[1], "bond_terminals_heavy"

    coords = heavy.positions
    box = u.trajectory.ts.dimensions
    D = distance_array(coords, coords, box=box)
    np.fill_diagonal(D, -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return poly, heavy, heavy[i], heavy[j], f"geometry_farthest_heavy (terminals_found={len(terms)})"

def compute_e2e_timeseries(prod_dir, xtc_name="prod.xtc", heavy_sel="not name H*"):
    u, topo, xtc = _load_universe_from_prod_dir(prod_dir, xtc_name=xtc_name)
    poly, heavy, a1, a2, method = _pick_end_atoms_auto(u, heavy_sel=heavy_sel)

    times = []
    dists = []
    for ts in u.trajectory:
        d = calc_bonds(a1.position[None, :], a2.position[None, :], box=ts.dimensions)[0]
        times.append(ts.time)
        dists.append(d)

    times = np.asarray(times, float)
    dists = np.asarray(dists, float)

    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "polymer_natoms": int(len(poly)),
        "heavy_natoms": int(len(heavy)),
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(len(times)),
        "mean_nm": float(np.mean(dists)),
        "std_nm": float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
    }
    return times, dists, meta

# ----------------------------
# simple Gaussian KDE (no seaborn)
# ----------------------------
def gaussian_kde_1d(
    x,
    grid=None,
    gridsize=400,
    bw_method="scott",
    bw=None,
    cut=3.0
):
    """
    1D Gaussian KDE with optional externally supplied grid.
    Returns: grid, density, bandwidth
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need >=2 samples for KDE.")

    std = np.std(x, ddof=1)

    # --- bandwidth selection ---
    if bw is None:
        if bw_method == "silverman":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            s = min(std, iqr / 1.34) if np.isfinite(iqr) and iqr > 0 else std
            bw = 0.9 * s * n ** (-1/5)
        else:  # scott
            bw = std * n ** (-1/5)

        if not np.isfinite(bw) or bw <= 0:
            bw = max(std, 1e-6) * 0.1

    # --- grid handling ---
    if grid is None:
        lo = np.min(x) - cut * bw
        hi = np.max(x) + cut * bw
        grid = np.linspace(lo, hi, gridsize)
    else:
        grid = np.asarray(grid, float)

    # --- KDE evaluation ---
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))

    return grid, dens, bw

# ----------------------------
# plot wrapper: time OR kde
# ----------------------------
def plot_e2e(
    prod_dir,
    mode="time",                 # 'time' or 'kde'
    xtc_name="prod.xtc",
    heavy_sel="not name H*",
    downsample=1,
    burn_in=0.0,                 # fraction of trajectory to discard from start, e.g. 0.1
    rolling_window=None,         # only for time mode
    kde_bw_method="scott",
    kde_bw=None,
    kde_gridsize=400,
    kde_cut=3.0,
    save_plot=None,
    save_dat=None,               # saves either time series or KDE curve
    figsize=(12, 5),
    dpi=300,
):
    times, dists, meta = compute_e2e_timeseries(prod_dir, xtc_name=xtc_name, heavy_sel=heavy_sel)

    # burn-in discard
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(dists) * burn_in)
        times = times[start:]
        dists = dists[start:]

    # downsample
    downsample = int(max(1, downsample))
    t_plot = times[::downsample]
    d_plot = dists[::downsample]

    plt.figure(figsize=figsize)

    if mode.lower() == "time":
        plt.plot(t_plot, d_plot, lw=1.2, label="E2E")

        if rolling_window is not None and rolling_window >= 2 and len(d_plot) >= rolling_window:
            w = int(rolling_window)
            kernel = np.ones(w) / w
            d_roll = np.convolve(d_plot, kernel, mode="valid")
            t_roll = t_plot[w-1:]
            plt.plot(t_roll, d_roll, lw=2.0, label=f"Rolling mean (w={w})")

        plt.xlabel("Time (ps)")
        plt.ylabel("End-to-End Distance (nm)")
        plt.title(f"E2E vs Time | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dat:
            os.makedirs(os.path.dirname(save_dat), exist_ok=True)
            np.savetxt(save_dat, np.column_stack([times, dists]), header="time_ps  end_to_end_nm")

    elif mode.lower() == "kde":
        grid, dens, bw = gaussian_kde_1d(
            d_plot,
            bw=kde_bw,
            bw_method=kde_bw_method,
            gridsize=kde_gridsize,
            cut=kde_cut,
        )
        plt.plot(grid, dens, lw=2.0, label=f"KDE (bw={bw:.4g} nm)")
        plt.xlabel("End-to-End Distance (nm)")
        plt.ylabel("Density")
        plt.title(f"E2E KDE | burn_in={burn_in} | downsample={downsample} | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dat:
            os.makedirs(os.path.dirname(save_dat), exist_ok=True)
            np.savetxt(save_dat, np.column_stack([grid, dens]), header="e2e_nm  density")

    else:
        raise ValueError("mode must be 'time' or 'kde'")

    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=dpi)

    plt.show()

    print("prod_dir:", meta["prod_dir"])
    print("end atoms (0-based):", meta["end_atoms_0based"], "| method:", meta["end_method_used"])
    print("frames used:", len(d_plot), "(after burn-in/downsample)")
    print("mean E2E (nm):", float(np.mean(d_plot)), "| std (nm):", float(np.std(d_plot, ddof=1)) if len(d_plot) > 1 else 0.0)

    return (times, dists, meta)

def compute_e2e_timeseries_nm(
    prod_dir,
    xtc_name="prod.xtc",
    heavy_sel="not name H*",
    use_cache=True,
    force_recompute=False,
):
    cache_file = _cache_path(prod_dir)
    cache = _load_cache(cache_file) if use_cache else {}

    u, topo, xtc = _load_universe(prod_dir, xtc_name=xtc_name)

    signature = {
        "topology": _file_sig(topo),
        "trajectory": _file_sig(xtc),
        "heavy_sel": heavy_sel,
        "version": "e2e_unwrap_v1",
    }

    # ---------------- cache hit ----------------
    if (
        use_cache
        and not force_recompute
        and cache.get("signature") == signature
    ):
        out = cache["data"]
        return (
            out["times_ps"],
            out["dists_nm"],
            out["meta"],
        )

    # ---------------- compute ----------------
    poly, a1, a2, method = _pick_end_atoms(u, heavy_sel=heavy_sel)

    from MDAnalysis.transformations import unwrap
    u.trajectory.add_transformations(unwrap(poly))

    times_ps = []
    dists_nm = []

    for ts in u.trajectory:
        d_ang = np.linalg.norm(a1.position - a2.position)
        dists_nm.append(d_ang / 10.0)
        times_ps.append(ts.time)

    times_ps = np.asarray(times_ps, float)
    dists_nm = np.asarray(dists_nm, float)

    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(len(times_ps)),
        "mean_nm": float(np.mean(dists_nm)),
        "std_nm": float(np.std(dists_nm, ddof=1)) if len(dists_nm) > 1 else 0.0,
    }

    if use_cache:
        _save_cache(cache_file, {
            "signature": signature,
            "data": {
                "times_ps": times_ps,
                "dists_nm": dists_nm,
                "meta": meta,
            },
        })

    return times_ps, dists_nm, meta

def expand_run_prod_dirs(run1_prod_dir, runs=(1, 2, 3)):
    """
    Given a prod_dir that contains '/run1/prod', return a list of prod_dirs for runs.
    Example input:
      /.../C200/0/run1/prod
    Output for runs=(1,2,3):
      [/.../C200/0/run1/prod, /.../C200/0/run2/prod, /.../C200/0/run3/prod]
    """
    # replace /run{n}/prod safely
    m = re.search(r"/run(\d+)/prod/?$", run1_prod_dir)
    if not m:
        raise ValueError("run1_prod_dir must end with /runX/prod, e.g. .../run1/prod")
    prefix = run1_prod_dir[:m.start()]
    suffix = "/prod"
    return [f"{prefix}/run{r}{suffix}" for r in runs]

def _apply_burnin_downsample(times, dists, burn_in=0.0, downsample=1):
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(dists) * burn_in)
        times = times[start:]
        dists = dists[start:]
    downsample = int(max(1, downsample))
    return times[::downsample], dists[::downsample]

def compute_e2e_run_means(run1_prod_dir, runs=(1,2,3), burn_in=0.0, downsample=1,
                          heavy_sel="not name H*", xtc_name="prod.xtc",
                          use_cache=True, force_recompute=False, verbose=True):
    """
    Computes per-run mean E2E (nm) for a condition defined by run1_prod_dir.
    Returns dict with:
      prod_dirs, per_run_means_nm, mean_of_means_nm, std_across_runs_nm, sem_across_runs_nm, n_runs_used
    """
    prod_dirs = expand_run_prod_dirs(run1_prod_dir, runs=runs)

    per_run_means = []
    used = []

    for d in prod_dirs:
        if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, xtc_name))):
            if verbose:
                print(f"[skip] missing {xtc_name} or dir: {d}")
            continue

        t, dist_nm, meta = compute_e2e_timeseries_nm(
            d, xtc_name=xtc_name, heavy_sel=heavy_sel,
            use_cache=use_cache, force_recompute=force_recompute
        )
        t2, dist2 = _apply_burnin_downsample(t, dist_nm, burn_in=burn_in, downsample=downsample)
        if len(dist2) < 2:
            if verbose:
                print(f"[skip] not enough frames after burn-in/downsample: {d}")
            continue

        per_run_means.append(float(np.mean(dist2)))
        used.append(d)

    per_run_means = np.array(per_run_means, float)
    n = len(per_run_means)
    if n == 0:
        raise RuntimeError("No valid runs found for averaging.")

    mean_of_means = float(np.mean(per_run_means))
    std_runs = float(np.std(per_run_means, ddof=1)) if n > 1 else 0.0
    sem_runs = float(std_runs / np.sqrt(n)) if n > 1 else 0.0

    return {
        "run1_prod_dir": run1_prod_dir,
        "prod_dirs_used": used,
        "per_run_means_nm": per_run_means,
        "n_runs_used": n,
        "mean_of_means_nm": mean_of_means,
        "std_across_runs_nm": std_runs,
        "sem_across_runs_nm": sem_runs,
    }

def plot_e2e_overlay(
    prod_dirs_or_run1_dirs,
    labels=None,
    mode="kde",                 # "kde" or "time"
    burn_in=0.0,
    downsample=1,
    rolling_window=None,
    kde_bw_method="scott",
    kde_bw=None,
    kde_shared_grid=True,
    gridsize=400,
    cut=3.0,
    units="A",                  # "nm" or "A"
    save_plot=None,
    figsize=(24, 10),
    dpi=300,
    # run expansion
    runs=None,                  # e.g., (1,2,3) means inputs are run1/prod and we auto-expand
    # NEW: combine behavior
    combine="average",          # "average" | "pool" | "both"
    show_individual_runs=False, # if True, plots each run faintly
    errorbars="sem",            # 'sem' or 'std' for reporting average-mode uncertainty
    heavy_sel="not name H*",
    xtc_name="prod.xtc",
    use_cache=True,
    force_recompute=False,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if combine not in ("average", "pool", "both"):
        raise ValueError("combine must be 'average', 'pool', or 'both'")

    # units scaling
    if units.lower() in ["a", "å", "angstrom", "angstroms"]:
        scale = 10.0
        x_unit = "Å"
    else:
        scale = 1.0
        x_unit = "nm"

    downsample = int(max(1, downsample))

    inputs = list(prod_dirs_or_run1_dirs)
    if labels is None:
        labels = [f"cond_{i+1}" for i in range(len(inputs))]
    if len(labels) != len(inputs):
        raise ValueError("labels must match length of prod_dirs_or_run1_dirs")

    def _apply_burnin_downsample(times, dists):
        if burn_in and 0.0 < burn_in < 1.0:
            start = int(len(dists) * burn_in)
            times = times[start:]
            dists = dists[start:]
        return times[::downsample], dists[::downsample]

    def _get_series(prod_dir):
        t, d_nm, meta = compute_e2e_timeseries_nm(
            prod_dir, xtc_name=xtc_name, heavy_sel=heavy_sel,
            use_cache=use_cache, force_recompute=force_recompute
        )
        t_plot, d_plot_nm = _apply_burnin_downsample(t, d_nm)
        return t_plot, d_plot_nm, meta

    # ---- solvent classifier (uses label first, then falls back to path) ----
    def _solvent_kind(label, base):
        s = (label or "").lower()
        b = str(base).lower()

        if ("1-2-dcb" in s) or ("1_2_dcb" in s) or ("1-2-dcb" in b) or ("1_2_dcb" in b):
            return "12"
        if ("1-4-dcb" in s) or ("1_4_dcb" in s) or ("1-4-dcb" in b) or ("1_4_dcb" in b):
            return "14"
        return "other"

    # ======================================================================
    # PASS 1: count how many *conditions* per solvent we will actually plot
    # (i.e., inputs that have at least one valid run/prod we can load)
    # ======================================================================
    cond_kind = []  # one entry per input: "12" | "14" | "other"
    will_plot = []  # bool per input
    for base, lab in zip(inputs, labels):
        kind = _solvent_kind(lab, base)
        prod_list = [base] if runs is None else expand_run_prod_dirs(base, runs=runs)

        any_valid = False
        for pd in prod_list:
            if not (os.path.isdir(pd) and os.path.isfile(os.path.join(pd, xtc_name))):
                continue
            try:
                _, d_nm, _ = _get_series(pd)
                if len(d_nm) >= 2:
                    any_valid = True
                    break
            except Exception:
                continue

        cond_kind.append(kind)
        will_plot.append(any_valid)

    n_12 = sum(1 for k, ok in zip(cond_kind, will_plot) if ok and k == "12")
    n_14 = sum(1 for k, ok in zip(cond_kind, will_plot) if ok and k == "14")
    n_other = sum(1 for k, ok in zip(cond_kind, will_plot) if ok and k == "other")

    # create palettes sized to detected conditions
    # (minimum 1 so iterators don't error)
    pal_12 = sns.color_palette("rocket_r")
    pal_14 = sns.color_palette("viridis")
    pal_ot = sns.color_palette("deep")

    # iterators we’ll consume as we encounter each condition of that type
    i12 = iter(pal_12)
    i14 = iter(pal_14)
    iot = iter(pal_ot)

    def _next_cond_color(kind):
        if kind == "12":
            return next(i12)
        if kind == "14":
            return next(i14)
        return next(iot)

    # ======================================================================
    # Optional shared KDE grid across ALL curves
    # ======================================================================
    kde_grid_nm = None
    if mode.lower() == "kde" and kde_shared_grid:
        pooled_all = []
        for base in inputs:
            prod_list = [base] if runs is None else expand_run_prod_dirs(base, runs=runs)
            for pd in prod_list:
                if os.path.isdir(pd) and os.path.isfile(os.path.join(pd, xtc_name)):
                    try:
                        _, d_nm, _ = _get_series(pd)
                        if len(d_nm) > 1:
                            pooled_all.append(d_nm)
                    except Exception:
                        pass
        if len(pooled_all) > 0:
            pooled_all = np.concatenate(pooled_all)
            _, _, bw_ref = gaussian_kde_1d(pooled_all, gridsize=50, bw_method=kde_bw_method, bw=kde_bw, cut=cut)
            lo = float(np.min(pooled_all) - cut * bw_ref)
            hi = float(np.max(pooled_all) + cut * bw_ref)
            kde_grid_nm = np.linspace(lo, hi, gridsize)

    # ======================================================================
    # PASS 2: plotting (same logic, now with dynamic palettes)
    # ======================================================================
    plt.figure(figsize=figsize)
    summaries = []

    for base, lab, kind, ok in zip(inputs, labels, cond_kind, will_plot):
        if not ok:
            continue

        # one "base color" per condition (distinct shades per solvent family)
        base_color = _next_cond_color(kind)

        # for pool vs average, use two closely related tones derived from base_color
        # (keeps family consistent but lets you see solid vs dashed easier)
        c_pool = base_color
        c_avg = base_color

        prod_list = [base] if runs is None else expand_run_prod_dirs(base, runs=runs)

        series_list = []
        for pd in prod_list:
            if not (os.path.isdir(pd) and os.path.isfile(os.path.join(pd, xtc_name))):
                print(f"[skip] missing {xtc_name} or dir: {pd}")
                continue
            try:
                t_plot, d_plot_nm, meta = _get_series(pd)
            except Exception as e:
                print(f"[skip] {pd}: {e}")
                continue
            if len(d_plot_nm) < 2:
                print(f"[skip] {pd}: not enough frames after burn-in/downsample")
                continue
            series_list.append((pd, t_plot, d_plot_nm, meta))

        if len(series_list) == 0:
            continue

        # Optionally show each run faintly (same condition color, just alpha)
        if show_individual_runs:
            if mode.lower() == "kde":
                for (pd, _, d_nm, _) in series_list:
                    g, dens, _ = gaussian_kde_1d(
                        d_nm, grid=kde_grid_nm, gridsize=gridsize,
                        bw_method=kde_bw_method, bw=kde_bw, cut=cut
                    )
                    plt.plot(g * scale, dens, lw=1.0, alpha=0.20, color=base_color)
            else:
                for (pd, t_plot, d_nm, _) in series_list:
                    plt.plot(t_plot, d_nm * scale, lw=1.0, alpha=0.20, color=base_color)

        # ----------------------------
        # combine logic
        # ----------------------------
        run_means = np.array([float(np.mean(d_nm)) for (_, _, d_nm, _) in series_list])
        n_runs = len(run_means)
        mean_of_means = float(np.mean(run_means))
        std_runs = float(np.std(run_means, ddof=1)) if n_runs > 1 else 0.0
        sem_runs = float(std_runs / np.sqrt(n_runs)) if n_runs > 1 else 0.0
        err = sem_runs if errorbars == "sem" else std_runs

        pooled = np.concatenate([d_nm for (_, _, d_nm, _) in series_list])

        # ----- KDE mode -----
        if mode.lower() == "kde":
            # pooled KDE
            if combine in ("pool", "both"):
                g, dens, bw = gaussian_kde_1d(
                    pooled, grid=kde_grid_nm, gridsize=gridsize,
                    bw_method=kde_bw_method, bw=kde_bw, cut=cut
                )
                plt.plot(
                    g * scale, dens, lw=2.6, color=c_pool,
                    label=f"{lab} pool (n_runs={n_runs})"
                )

            # average-of-KDEs
            if combine in ("average", "both"):
                dens_list = []
                for (_, _, d_nm, _) in series_list:
                    g, dens_r, _ = gaussian_kde_1d(
                        d_nm, grid=kde_grid_nm, gridsize=gridsize,
                        bw_method=kde_bw_method, bw=kde_bw, cut=cut
                    )
                    dens_list.append(dens_r)
                dens_avg = np.mean(np.vstack(dens_list), axis=0)
                plt.plot(
                    g * scale, dens_avg, lw=2.2, ls="--", color=c_avg,
                    label=f"{lab} average (μ={mean_of_means*scale:.1f}{x_unit}, {errorbars}={err*scale:.1f}{x_unit})"
                )

        # ----- TIME mode -----
        else:
            if combine in ("average", "both"):
                t0 = series_list[0][1]
                same_time = all(len(t0) == len(ti) and np.allclose(t0, ti) for (_, ti, _, _) in series_list[1:])
                if same_time:
                    D = np.vstack([d_nm for (_, _, d_nm, _) in series_list])
                    d_mean = np.mean(D, axis=0)
                    plt.plot(t0, d_mean * scale, lw=2.6, color=c_avg,
                             label=f"{lab} avg (μ={mean_of_means*scale:.1f}{x_unit})")
                else:
                    plt.plot([], [], color=c_avg,
                             label=f"{lab} avg (μ={mean_of_means*scale:.1f}{x_unit})")

            if combine in ("pool", "both"):
                pooled_mean = float(np.mean(pooled))
                plt.plot([], [], color=c_pool,
                         label=f"{lab} pool (μ={pooled_mean*scale:.1f}{x_unit})")

        # record summary
        pooled_mean = float(np.mean(pooled))
        pooled_std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        summaries.append({
            "label": lab,
            "n_runs": n_runs,
            "run_means_nm": run_means,
            "mean_of_means_nm": mean_of_means,
            "std_across_runs_nm": std_runs,
            "sem_across_runs_nm": sem_runs,
            "pooled_mean_nm": pooled_mean,
            "pooled_std_nm": pooled_std,
        })

    # finalize plot
    if mode.lower() == "kde":
        plt.xlabel(f"End-to-End Distance ({x_unit})")
        plt.ylabel("Density")
        plt.title(f"E2E KDE overlay | combine={combine} | burn_in={burn_in} | downsample={downsample}")
    else:
        plt.xlabel("Time (ps)")
        plt.ylabel(f"End-to-End Distance ({x_unit})")
        plt.title(f"E2E time overlay | combine={combine} | burn_in={burn_in} | downsample={downsample}")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=dpi)
    plt.show()

    # print summary
    if len(summaries) > 0:
        print("\n=== Summary (per condition) ===")
        for s in summaries:
            errv = s["sem_across_runs_nm"] if errorbars == "sem" else s["std_across_runs_nm"]
            print(f"{s['label']}:")
            print(f"  average: mean_of_means={s['mean_of_means_nm']*scale:.3f}{x_unit}  {errorbars}={errv*scale:.3f}{x_unit}  n_runs={s['n_runs']}")
            print(f"  pool   : pooled_mean   ={s['pooled_mean_nm']*scale:.3f}{x_unit}  pooled_std={s['pooled_std_nm']*scale:.3f}{x_unit}  n_samples={len(s['run_means_nm'])} runs")

    return summaries
