import argparse, pathlib
from src.perf_viz import concat_runs, plot_core_metrics, save_csv, summarize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="JSONL log files (e.g., perf/*.jsonl)")
    ap.add_argument("--out", default="perf/out", help="Output dir for PNG/CSV")
    args = ap.parse_args()

    perf = concat_runs(args.logs)
    save_csv(perf, args.out)
    plot_core_metrics(perf, args.out)

    summ = summarize(perf)
    if not summ.empty:
        summ.to_csv(f"{args.out}/summary_agg.csv", index=False)
        print(summ.to_string(index=False))
    print(f"Saved figures & CSVs to: {pathlib.Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
