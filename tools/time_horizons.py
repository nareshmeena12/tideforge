"""Copy TIME's frequency -> horizon convention into catalog/time_horizons.csv (run on Perlmutter, where the TIME repo and data
are; minutes: it opens each dataset to read its frequency; no GPU).

    python tools/time_horizons.py --tm-eval $TM_EVAL            # writes catalog/time_horizons.csv, prints the full listing

TIME sets prediction_length per (dataset, term) (timebench get_dataset_settings). Per canonical frequency and term this writes
the most common value over TIME's datasets (ties: the smaller), the number of datasets, and every value seen, so the choice
is visible. Paste the printed listing back with the file.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]

import pandas as pd  # noqa: E402

from tideforge.devdata.freqs import TIME_HORIZONS, canonical  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-eval", required=True, help="tidemark's fetched eval folder (holds data/TIME and data/TIME_data)")
    ap.add_argument("--out", default=str(TIME_HORIZONS))
    a = ap.parse_args(argv)
    import tidemark.eval.score_twins as S
    paths = S.Paths(Path(a.tm_eval).parent)          # Paths(root) uses root/eval
    if paths.data.resolve() != (Path(a.tm_eval) / "data").resolve():
        raise SystemExit(f"--tm-eval must be the folder that holds data/ (got {a.tm_eval})")
    tb = S.TB(paths)
    config = tb.load_dataset_config()
    seen = defaultdict(list)
    for ds_name in config["datasets"]:
        for term in tb.get_available_terms(ds_name, config):
            st = tb.get_dataset_settings(ds_name, term, config)
            ds = tb.Dataset(name=ds_name, term=term, to_univariate=False, prediction_length=st["prediction_length"],
                            test_length=st["test_length"], val_length=st["val_length"])
            f = canonical(ds.freq)
            seen[(f, term)].append((int(st["prediction_length"]), ds_name))
            print(f"{ds_name:40s} {term:6s} freq {ds.freq:8s} -> {f:6s} H {st['prediction_length']}")
    rows = []
    for (f, term), v in sorted(seen.items()):
        c = Counter(h for h, _ in v)
        best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rows.append(dict(freq=f, term=term, prediction_length=best, n_datasets=len(v),
                         values=" ".join(f"{h}x{n}" for h, n in sorted(c.items())),
                         source="TIME (zqiao11/TIME) get_dataset_settings, mode over datasets"))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(df.to_string(index=False))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
