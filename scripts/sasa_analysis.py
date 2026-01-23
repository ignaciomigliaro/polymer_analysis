from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def make_filename(c, d, solvent):
    """
    Build filename depending on solvent mode.
    solvent = "odcb", "pdcb", or "both"
    """
    if solvent == "odcb":
        return f"sasa_c{c}_{d}_odcb.xvg"
    elif solvent == "pdcb":
        return f"sasa_c{c}_{d}_pdcb.xvg"
    else:  # "both"
        # try: _odcb → _pdcb → no suffix
        return [
            f"sasa_c{c}_{d}_odcb.xvg",
            f"sasa_c{c}_{d}_pdcb.xvg",
            f"sasa_c{c}_{d}.xvg",
        ]

def parse_xvg(file_path):
    """Safe parser that avoids iterator exhaustion and uneven columns."""
    time, sasa = [], []
    with open(file_path) as f:
        for line in f:
            if line.startswith(("#", "@")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time.append(float(parts[0]))
                    sasa.append(float(parts[1]))
                except ValueError:
                    continue
    return time, sasa

def plot_kde_sasa(
    carbons,
    degrees,
    base_dir=".",
    solvent="both",                 # "both", "1-2-dcb", "1-4-dcb"
    runs=(1, 2, 3),                 # iterable of run indices
    title="SASA Distribution (KDE)",
    xlabel="SASA (nm²)",
    ylabel="Density",
    output_file="sasa_kde.png",
    linewidth=2.0,
):
    """
    KDE distributions of SASA for multiple carbon lengths, degrees, solvents, and runs.

    Expected layout:
      base_dir/
        xvg/sasa/1-2-dcb/
        xvg/sasa/1-4-dcb/

    Expected filenames (preferred):
      sasa_c<carbon>_<degree>_<solvent>_run<run>.xvg

    Note:
      If the exact filename doesn't exist (often due to solvent tag formatting),
      we fall back to a glob search within the solvent directory.
    """

    base_dir = Path(base_dir)

    solvent_dirs_all = ["1-2-dcb", "1-4-dcb"]
    if solvent == "both":
        solvent_dirs = solvent_dirs_all
    else:
        # allow user to pass "12dcb" etc. but still map to folder names if possible
        if solvent in solvent_dirs_all:
            solvent_dirs = [solvent]
        else:
            # best-effort mapping
            # (you can remove this if you only ever pass folder names)
            solvent_norm = solvent.replace("_", "-").lower()
            if solvent_norm in solvent_dirs_all:
                solvent_dirs = [solvent_norm]
            else:
                solvent_dirs = [solvent]  # will try as-is

    # Colors: keep distinct per solvent folder
    # Total lines per solvent = carbons * degrees * runs (upper bound)
    N = len(carbons) * len(degrees) * len(tuple(runs))
    palettes = {
        "1-2-dcb": sns.color_palette("Blues", N),
        "1-4-dcb": sns.color_palette("Reds", N),
    }
    color_index = {s: 0 for s in solvent_dirs}

    plt.figure(figsize=(9, 5))

    plotted_any = False

    for sdir in solvent_dirs:
        sasa_dir = base_dir / "xvg" / "sasa" / sdir

        if not sasa_dir.exists():
            print(f"⚠️ Missing directory: {sasa_dir}")
            continue

        for c in carbons:
            for d in degrees:
                for r in runs:
                    # Preferred exact filename (matches your spec)
                    preferred = sasa_dir / f"sasa_c{c}_{d}_{sdir}_run{r}.xvg"

                    candidates = []
                    if preferred.exists():
                        candidates = [preferred]
                    else:
                        # Robust fallback:
                        # Find anything like sasa_c{c}_{d}_*run{r}.xvg in this solvent folder.
                        # This covers cases where the solvent tag in the filename is "12dcb" etc.
                        pattern = f"sasa_c{c}_{d}_*run{r}.xvg"
                        candidates = sorted(sasa_dir.glob(pattern))

                    if not candidates:
                        continue

                    for file_path in candidates:
                        data = parse_xvg(file_path)
                        if not data:
                            print(f"❌ No valid data in {file_path}")
                            continue

                        sasa_values = data[1]

                        # pick next color for this solvent folder
                        if sdir in palettes:
                            idx = color_index[sdir] % len(palettes[sdir])
                            color = palettes[sdir][idx]
                            color_index[sdir] += 1
                        else:
                            # if user passed a custom solvent folder name not in palettes
                            color = None

                        # label: include run + folder solvent
                        label = f"C{c} - {d}% ({sdir}, run{r})"
                        sns.kdeplot(sasa_values, label=label, linewidth=linewidth, color=color)
                        plotted_any = True

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if plotted_any:
        plt.legend(title="System", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        print("⚠️ No datasets were found to plot (check paths/filenames).")

    plt.tight_layout()
    # plt.savefig(output_file, dpi=600)
    print(f"✅ KDE plot saved as: {output_file}")
    plt.show()
    
def plot_kde_sasa_pooled(
    carbons,
    degrees,
    base_dir=".",
    solvents=("1-2-dcb", "1-4-dcb"),
    runs=(1, 2, 3),
    title="SASA Distribution (Pooled Runs)",
    xlabel="SASA (nm²)",
    ylabel="Density",
    output_file="sasa_kde_pooled.png",
    linewidth=2.5,
    verbose=False,
):
    """
    Pool all runs together and compute ONE KDE per system.

    Expects:
      base_dir/xvg/sasa/<solvent>/
      sasa_c<carbon>_<degree>_<something>_run<r>.xvg
    """

    base_dir = Path(base_dir)

    # ✅ If user passed a single string, wrap it
    if isinstance(solvents, str):
        solvents = (solvents,)

    # quick sanity print
    if verbose:
        print("Base dir:", base_dir)
        for s in solvents:
            print("Solvent dir expected:", base_dir / "xvg" / "sasa" / s)

    # palette sized per solvent (upper bound)
    nlines = len(carbons) * len(degrees)
    palettes = {
        "1-2-dcb": sns.color_palette("Blues", nlines),
        "1-4-dcb": sns.color_palette("Reds",  nlines),
    }
    color_index = {s: 0 for s in solvents}

    plt.figure(figsize=(9, 5))
    plotted_any = False

    for solvent in solvents:
        sasa_dir = base_dir / "xvg" / "sasa" / solvent
        if not sasa_dir.exists():
            if verbose:
                print(f"⚠️ Missing directory: {sasa_dir}")
            continue

        for c in carbons:
            for d in degrees:
                pooled_sasa = []

                for r in runs:
                    pattern = f"sasa_c{c}_{d}_*run{r}.xvg"
                    matches = sorted(sasa_dir.glob(pattern))

                    if verbose:
                        print(f"Looking for: {sasa_dir / pattern}")
                        print(f"  Found {len(matches)} file(s)")

                    for file_path in matches:
                        data = parse_xvg(file_path)
                        if data:
                            pooled_sasa.extend(data[1])
                        elif verbose:
                            print(f"  ❌ parse_xvg returned no data: {file_path}")

                if not pooled_sasa:
                    if verbose:
                        print(f"⚠️ No pooled data for C{c} {d}% in {solvent}")
                    continue

                # choose color
                if solvent in palettes:
                    idx = color_index[solvent] % len(palettes[solvent])
                    color = palettes[solvent][idx]
                    color_index[solvent] += 1
                else:
                    color = None

                label = f"C{c} – {d}% ({solvent}, pooled)"
                sns.kdeplot(pooled_sasa, label=label, linewidth=linewidth, color=color)
                plotted_any = True

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
    title="System",
    bbox_to_anchor=(0.5, -0.25),  # below the plot
    loc="upper center",
    ncol=2,                       # try 2–4 depending on entries
    frameon=False,
)
    plt.subplots_adjust(bottom=0.30)

    if plotted_any:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        if verbose:
            print("❗ Nothing was plotted. This usually means: wrong base_dir, wrong folder name, or filename pattern mismatch.")

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.show()
    print(f"✅ Pooled KDE saved as {output_file}")

def plot_avg_delta_sasa(
    carbons,
    degrees,
    base_dir=".",
    solvent="both",  # "1-2-dcb", "1-4-dcb", or "both"
    runs=(1, 2, 3),
    show_errorbars=True,
    title="Average ΔSASA vs Carbon Length",
    xlabel="Carbon Chain Length (Cₙ)",
    ylabel="Average ΔSASA (nm²)",
    output_file="avg_delta_sasa_line.png",
    verbose=False,
):
    """
    Compute & plot average ΔSASA for each carbon length and degree
    for solvents in the new layout:

      base_dir/xvg/sasa/1-2-dcb/
      base_dir/xvg/sasa/1-4-dcb/

    Files:
      sasa_c<carbon>_<degree>_<solvent>_run<1,2,3>.xvg

    Metric:
      For each run:  mean(|SASA - mean(SASA)|)
      Across runs:   average of the per-run metric (optionally std as errorbar)

    solvent options:
      "1-2-dcb" → only that folder
      "1-4-dcb" → only that folder
      "both"    → plot both
    """

    base_dir = Path(base_dir)
    plt.figure(figsize=(18, 10))

    # --- palettes per solvent (one color per degree) ---
    palettes = {
        "1-2-dcb": sns.color_palette("Blues", len(degrees)),
        "1-4-dcb": sns.color_palette("Reds",  len(degrees)),
    }

    # Determine which solvents to plot
    solvent_order = []
    if solvent in ("1-2-dcb", "both"):
        solvent_order.append("1-2-dcb")
    if solvent in ("1-4-dcb", "both"):
        solvent_order.append("1-4-dcb")

    for solvent_group in solvent_order:
        sasa_dir = base_dir / "xvg" / "sasa" / solvent_group
        if not sasa_dir.exists():
            if verbose:
                print(f"⚠️ Missing directory: {sasa_dir}")
            continue

        palette = palettes.get(solvent_group, None)

        # One line per degree
        for i, d in enumerate(degrees):
            avg_deltas = []
            std_deltas = []

            for c in carbons:
                per_run_metrics = []

                for r in runs:
                    # Robust search (doesn't assume exact solvent tag inside filename)
                    pattern = f"sasa_c{c}_{d}_*run{r}.xvg"
                    matches = sorted(sasa_dir.glob(pattern))

                    if not matches:
                        if verbose:
                            print(f"⚠️ No match: {sasa_dir / pattern}")
                        continue

                    # If multiple match, take the first (or you could average them)
                    file_path = matches[0]

                    data = parse_xvg(file_path)
                    if not data:
                        if verbose:
                            print(f"❌ parse_xvg failed: {file_path}")
                        continue

                    sasa = np.asarray(data[1], dtype=float)
                    if sasa.size == 0:
                        continue

                    delta_sasa = sasa - sasa.mean()
                    per_run_metrics.append(float(np.mean(np.abs(delta_sasa))))

                if len(per_run_metrics) == 0:
                    avg_deltas.append(np.nan)
                    std_deltas.append(np.nan)
                else:
                    avg_deltas.append(float(np.mean(per_run_metrics)))
                    std_deltas.append(float(np.std(per_run_metrics, ddof=1)) if len(per_run_metrics) > 1 else 0.0)

            # Plot line for this solvent and degree
            label = f"{d}% ({solvent_group})"
            color = palette[i] if palette is not None else None

            if show_errorbars:
                plt.errorbar(
                    carbons, avg_deltas, yerr=std_deltas,
                    marker="o", linewidth=2,
                    capsize=4, color=color, label=label
                )
            else:
                plt.plot(
                    carbons, avg_deltas,
                    marker="o", linewidth=2,
                    color=color, label=label
                )

    # Styling
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Legend below to avoid shrinking plot
    plt.legend(
        title="Functionalization + Solvent",
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.30)

    outpath = Path(output_file).expanduser().resolve()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=600)
    print(f"✅ Line plot saved as: {outpath}")
    plt.show()

