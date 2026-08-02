"""``cascader <experiment> [options]``.

Run ``cascader --list`` for the catalogue, or ``cascader <name> --help`` for one
experiment's options. Every experiment writes ``<stem>_raw.csv`` and
``<stem>_summary.csv`` into ``--out`` (default ``results/``), and the raw rows
are enough to recompute any summary without re-running anything.

Defaults are the paper's, but they are not small: the main table at full size is
five collections, twenty evaluation seeds, five tuning seeds and ten learning
rates. Start with ``--collections Syn-10 --eval-seeds 3 --tune-seeds 1``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, Optional, Sequence

__all__ = ["EXPERIMENTS", "main"]


def _lazy(module: str) -> Callable:
    def entry(argv: Sequence[str]) -> None:
        import importlib
        importlib.import_module(f"cascader.experiments.{module}").main(list(argv))
    return entry


#: Name on the command line -> (entry point, one-line description).
EXPERIMENTS: Dict[str, tuple] = {
    "main-table": (_lazy("main_table"),
                   "seven methods on five collections at three per-task budgets"),
    "allocation-regime": (_lazy("allocation_regime"),
                          "floor1 against exact, with the realized-spend check"),
    "root-fraction": (_lazy("root_fraction"),
                      "sweep the share of the budget reserved for the root"),
    "rooting": (_lazy("rooting"),
                "four rooting rules compared at fixed topology"),
    "tworegime": (_lazy("tworegime"),
                  "scatter against budget: where the tree overtakes the chain"),
    "online": (_lazy("online"),
               "tasks arriving one at a time, no tree and no allocation"),
    "scaling": (_lazy("scaling"),
                "task count against gain, with construction cost"),
}


def _catalogue() -> str:
    width = max(len(k) for k in EXPERIMENTS)
    lines = [f"  {k.ljust(width)}  {d}" for k, (_, d) in EXPERIMENTS.items()]
    return "available experiments:\n" + "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="cascader", description=__doc__.splitlines()[0],
        epilog=_catalogue(), formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True)
    parser.add_argument("experiment", nargs="?", choices=sorted(EXPERIMENTS))
    parser.add_argument("--list", action="store_true", help="print the catalogue")
    parser.add_argument("--version", action="store_true")

    head, rest = [], []
    for i, token in enumerate(argv):
        if token in EXPERIMENTS:
            head, rest = argv[:i + 1], argv[i + 1:]
            break
    else:
        head = argv

    args = parser.parse_args(head)
    if args.version:
        from cascader import __version__
        print(__version__)
        return 0
    if args.list or not args.experiment:
        print(_catalogue())
        return 0 if args.list else 2

    EXPERIMENTS[args.experiment][0](rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
