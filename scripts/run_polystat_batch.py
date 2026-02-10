#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

REQUIRED = ("prod.tpr", "prod.xtc", "poly_LIG.ndx")


def run_cmd(cmd, cwd: Path, verbose: bool = False, input_text: str | None = None):
    if verbose:
        print(f"[cmd] (cwd={cwd}) {' '.join(cmd)}")
        if input_text is not None and input_text.strip():
            print(f"[stdin]\n{input_text}")

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def find_prod_dirs(base: Path):
    # robust: find directories literally named "prod"
    for d in base.rglob("prod"):
        if d.is_dir():
            yield d


def build_cmd(args, tpr: Path, xtc: Path, ndx: Path) -> list[str]:
    """
    Build the command for either:
      - gmx polystat
      - gmx rmsdist
    """
    base = [args.gmx]

    if args.mode == "polystat":
        # Keep your original behavior
        cmd = base + [
            "polystat",
            "-s", tpr.name,
            "-f", xtc.name,
            "-n", ndx.name,
            "-o", "-p", "-i",
            "-b", str(args.begin),
        ]
        # Optional common time flags if user provided them
        if args.end is not None:
            cmd += ["-e", str(args.end)]
        if args.dt is not None:
            cmd += ["-dt", str(args.dt)]

        if args.extra:
            cmd += shlex.split(args.extra)

        return cmd

    if args.mode == "rmsdist":
        # gmx rmsdist supports -f/-s/-n and outputs like -o/-rms/-mean/-scl...
        cmd = base + [
            "rmsdist",
            "-s", tpr.name,
            "-f", xtc.name,
            "-n", ndx.name,
        ]

        # Default output (can be overridden by --out)
        if args.out:
            cmd += ["-o", args.out]

        # Optional time controls
        if args.begin is not None:
            cmd += ["-b", str(args.begin)]
        if args.end is not None:
            cmd += ["-e", str(args.end)]
        if args.dt is not None:
            cmd += ["-dt", str(args.dt)]

        # Optional pbc toggle
        if args.pbc is True:
            cmd += ["-pbc"]
        elif args.pbc is False:
            cmd += ["-nopbc"]

        # Pass-through for any other rmsdist flags you want (-rms, -mean, -max, etc.)
        if args.extra:
            cmd += shlex.split(args.extra)

        return cmd

    raise ValueError(f"Unknown mode: {args.mode}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run GROMACS polystat or rmsdist across many /prod directories."
    )
    ap.add_argument("--base", default=".", help="Base directory to search (default: .)")
    ap.add_argument("--gmx", default="gmx_mpi", help="GROMACS executable (default: gmx_mpi)")
    ap.add_argument("--tpr", default="prod.tpr", help="TPR filename in each prod dir")
    ap.add_argument("--xtc", default="prod.xtc", help="XTC filename in each prod dir")
    ap.add_argument("--ndx", default="poly_LIG.ndx", help="Index filename in each prod dir")

    ap.add_argument(
        "--mode",
        choices=("polystat", "rmsdist"),
        default="polystat",
        help="Which GROMACS tool to run (default: polystat)",
    )

    ap.add_argument("-b", "--begin", type=float, default=40.0, help="Start time in ps (default: 40)")
    ap.add_argument("-e", "--end", type=float, default=None, help="End time in ps (optional)")
    ap.add_argument("--dt", type=float, default=None, help="Time step in ps (optional)")

    # rmsdist-specific convenience args (optional)
    ap.add_argument(
        "--out",
        default="rmsdist.xvg",
        help="Output filename for rmsdist -o (default: rmsdist.xvg). Ignored by polystat.",
    )
    ap.add_argument(
        "--pbc",
        dest="pbc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use/disable -pbc (rmsdist). Use --pbc or --no-pbc. Default: let gmx decide.",
    )

    # Feed interactive selections to gmx (common for rmsdist)
    ap.add_argument(
        "--stdin",
        default=None,
        help='Text to pipe to GROMACS stdin for interactive prompts (e.g. "1 1\\n").',
    )

    # Pass-through extra flags safely
    ap.add_argument(
        "--extra",
        default="",
        help='Extra flags passed to the selected tool (quoted). Example: --extra "-rms rms.xpm -mean mean.xpm -max 0.5"',
    )

    ap.add_argument("--dry-run", action="store_true", help="Print what would run, but do nothing")
    ap.add_argument("--verbose", action="store_true", help="Print commands + stdout/stderr on failure")
    ap.add_argument("--log", default="gmx_batch_failures.log", help="Failure log file")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    log_path = Path(args.log).resolve()

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = 0
    skipped = 0
    failed = 0

    with open(log_path, "w") as log:
        log.write(f"{args.mode} batch run\n")
        log.write(f"started: {started}\n")
        log.write(f"base: {base}\n")
        log.write(f"gmx: {args.gmx}\n")
        log.write(f"begin (ps): {args.begin}\n")
        if args.end is not None:
            log.write(f"end (ps): {args.end}\n")
        if args.dt is not None:
            log.write(f"dt (ps): {args.dt}\n")
        if args.mode == "rmsdist":
            log.write(f"rmsdist -o: {args.out}\n")
            if args.pbc is not None:
                log.write(f"pbc: {args.pbc}\n")
        if args.extra:
            log.write(f"extra: {args.extra}\n")
        if args.stdin:
            log.write("stdin: (provided)\n")
        log.write("=" * 80 + "\n")

        for prod_dir in sorted(find_prod_dirs(base)):
            tpr = prod_dir / args.tpr
            xtc = prod_dir / args.xtc
            ndx = prod_dir / args.ndx

            missing = [p.name for p in (tpr, xtc, ndx) if not p.exists()]
            if missing:
                skipped += 1
                log.write(f"[SKIP missing {missing}] {prod_dir}\n")
                continue

            cmd = build_cmd(args, tpr=tpr, xtc=xtc, ndx=ndx)

            if args.dry_run:
                print(f"[dry-run] {prod_dir} :: {' '.join(cmd)}")
                if args.stdin:
                    print(f"[dry-run stdin]\n{args.stdin}")
                ok += 1
                continue

            rc, out, err = run_cmd(cmd, cwd=prod_dir, verbose=args.verbose, input_text=args.stdin)
            if rc == 0:
                ok += 1
            else:
                failed += 1
                log.write(f"[FAIL rc={rc}] {prod_dir}\n")
                log.write("CMD: " + " ".join(cmd) + "\n")
                if args.stdin:
                    log.write("STDIN:\n" + args.stdin + "\n")

                if out.strip():
                    tail = out.strip().splitlines()[-50:]
                    log.write("STDOUT (last 50 lines):\n" + "\n".join(tail) + "\n")
                if err.strip():
                    log.write("STDERR:\n" + err + "\n")
                log.write("-" * 80 + "\n")

                if args.verbose:
                    print(f"[FAIL] {prod_dir} (rc={rc})")
                    print(err)

        ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write("=" * 80 + "\n")
        log.write(f"ended:   {ended}\n")
        log.write(f"ok:      {ok}\n")
        log.write(f"skipped: {skipped}\n")
        log.write(f"failed:  {failed}\n")

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}")
    print(f"Failure log: {log_path}")


if __name__ == "__main__":
    main()
