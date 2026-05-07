import sys
import pandas as pd


def safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def main():

    val_path = "validation_data.csv"

    if len(sys.argv) > 1:
        val_path = sys.argv[1]

    val = pd.read_csv(val_path)

    print(f"Loaded {len(val)} validation rows from {val_path}")

    val = val.dropna(subset=["external_price"]).copy()

    if val.empty:
        print(f"\nNo external_price values filled in yet. Edit {val_path} and add prices in the external_price column.")
        return

    val["date"] = pd.to_datetime(val["date"]).dt.strftime("%Y-%m-%d")

    hist = pd.read_csv("cleaned_data/price_history.csv")

    rows = []

    print("\nValidating filled-in rows:")

    for _, row in val.iterrows():

        q = str(row["query"]).lower().strip()
        target_date = row["date"]
        ext = float(row["external_price"])

        matches = hist[
            hist["component_id"].astype(str).str.lower().str.contains(q, na=False)
            & (hist["date"] == target_date)
        ]

        n_matches = len(matches)

        if n_matches == 0:
            print(f"  NO MATCH  '{safe(q)}' on {target_date}  (try a broader query like the chipset name)")
            continue

        synth_med = float(matches["price"].median())
        synth_min = float(matches["price"].min())
        synth_max = float(matches["price"].max())

        abs_err = abs(synth_med - ext)
        pct_err = abs_err / ext * 100

        rows.append({
            "query": q,
            "date": target_date,
            "external_price": round(ext, 2),
            "synthetic_median": round(synth_med, 2),
            "synthetic_min": round(synth_min, 2),
            "synthetic_max": round(synth_max, 2),
            "n_matched_components": n_matches,
            "abs_error": round(abs_err, 2),
            "pct_error": round(pct_err, 2),
            "source": str(row.get("source", "")),
            "notes": str(row.get("notes", "")),
        })

        flag = ""

        if n_matches > 10:
            flag = "  [WARN: query matches >10 components, error may be noisy — narrow the query]"

        print(f"  {safe(q)[:30]:<30} {target_date}  external=${ext:>8.2f}  synthetic=${synth_med:>8.2f}  err=${abs_err:>6.2f} ({pct_err:>5.1f}%)  n={n_matches}{flag}")

    if not rows:
        print("\nNo valid matches found — check query strings and dates.")
        return

    out = pd.DataFrame(rows)
    out.to_csv("cleaned_data/validation_results.csv", index=False)

    clean_match = out[out["n_matched_components"] <= 5]
    noisy_match = out[out["n_matched_components"] > 5]

    print(f"\n--- Summary across {len(out)} validated queries ---")
    print(f"Mean abs % error (all):           {out['pct_error'].mean():.2f}%")
    print(f"Median abs % error (all):         {out['pct_error'].median():.2f}%")

    if not clean_match.empty:
        print(f"Mean abs % error (n<=5 matches):  {clean_match['pct_error'].mean():.2f}%   ({len(clean_match)} queries)")
        print(f"Median abs % error (n<=5):        {clean_match['pct_error'].median():.2f}%")

    if not noisy_match.empty:
        print(f"Mean abs % error (n>5 matches):   {noisy_match['pct_error'].mean():.2f}%   ({len(noisy_match)} queries — likely noisy)")

    worst = out.loc[out["pct_error"].idxmax()]
    print(f"\nWorst case: '{safe(worst['query'])}' on {worst['date']}, error {worst['pct_error']:.1f}% (n={worst['n_matched_components']})")
    print(f"\nSaved: cleaned_data/validation_results.csv")


if __name__ == "__main__":
    main()
