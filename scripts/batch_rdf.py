#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def run(cmd, cwd: Path, stdin: str | None = None, verbose: bool = False) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd),
        input=stdin,
        text=True,
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )


def main():
    ap = argparse.ArgumentParser(description="Batch RDF in every prod/ directory using homogenized prod.tpr/prod.xtc.")
    ap.add_argument("--root", default=".", help="Root directory to search (default: .)")
    ap.add_argument("--gmx", default="gmx_mpi", help="GROMACS command (default: gmx_mpi)")
    ap.add_argument("--poly", default="LIG", help='Polymer group name (default: "LIG")')
    ap.add_argument("--solv", default="DCB", help='Solvent group name (default: "DCB")')
    ap.add_argument("--rmax", type=float, default=1.5, help="RDF max distance in nm (default: 1.5)")
    ap.add_argument("--skip", type=int, default=0, help="Analyze 1 out of N frames by thinning xtc (default: 0=off)")
    ap.add_argument("--out-rdf", default="rdf_LIG_DCB.xvg", help="Output RDF filename")
    ap.add_argument("--out-cn", default="cn_LIG_DCB.xvg", help="Output CN filename")
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    ap.add_argument("--verbose", action="store_true", help="Show GROMACS output")
    ap.add_argument("--ref-group", type=int, default=2)
    ap.add_argument("--sel-group", type=int, default=3)

    args = ap.parse_args()

    root = Path(args.root).resolve()
    prod_dirs = sorted([d for d in root.rglob("prod") if d.is_dir()])

    if not prod_dirs:
        print(f"No prod directories found under {root}")
        return

    print(f"Found {len(prod_dirs)} prod directories under {root}")

    for d in prod_dirs:
        tpr = d / "prod.tpr"
        xtc = d / "prod.xtc"
        if not tpr.exists() or not xtc.exists():
            print(f"[skip] {d} missing prod.tpr/prod.xtc")
            continue

        out_rdf = d / args.out_rdf
        out_cn = d / args.out_cn
        if (out_rdf.exists() or out_cn.exists()) and not args.force:
            print(f"[skip] {d} outputs exist (use --force to overwrite)")
            continue

        print(f"Processing: {d}")

        # 1) index
        run([args.gmx, "make_ndx", "-f", str(tpr), "-o", "index.ndx"], cwd=d, stdin="q\n", verbose=args.verbose)

        # 2) optional thinning
        xtc_use = xtc
        if args.skip > 0:
            thin = d / f"prod_skip{args.skip}.xtc"
            if not thin.exists() or args.force:
                # trjconv asks for a group; "0" (System) is safe
                run(
                    [args.gmx, "trjconv", "-s", str(tpr), "-f", str(xtc), "-o", str(thin), "-skip", str(args.skip)],
                    cwd=d,
                    stdin="0\n",
                    verbose=args.verbose,
                )
            xtc_use = thin

        # 3) RDF (your working non-interactive selection method)
        sel_stdin = f"{args.poly}\n{args.solv}\n"
      
        try:
            run(
                [
                    args.gmx, "rdf",
                    "-s", str(tpr),
                    "-f", str(xtc_use),
                    "-n", "index.ndx",
                    "-ref", "group 2",
                    "-sel", "group 3",
                    "-rmax", str(args.rmax),
                    "-o", str(out_rdf.name),
                    "-cn", str(out_cn.name),
                ],
                cwd=d,
                stdin=None,
                verbose=args.verbose,
            )
        except RuntimeError as e:
            msg = (
                f"\n=== RDF FAILED ===\n"
                f"Directory : {d}\n"
                f"TPR       : {tpr}\n"
                f"XTC       : {xtc_use}\n"
                f"Reason    : {e}\n"
            )
            print("  [fail] rdf failed (logged)")
            with open(fail_log, "a") as fh:
                fh.write(msg)
            continue

        print(f"  wrote {out_rdf.name}, {out_cn.name} (xtc used: {xtc_use.name})")

    print("Done.")


if __name__ == "__main__":
    main()
