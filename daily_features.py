# daily_features.py
import pandas as pd
from datetime import datetime, timedelta
from utils.db import supabase  # Supabase client from your utils/db.py

def fetch_day(date_str):
    """Fetch all half-hourly trends for a given date from Supabase."""
    resp = supabase.table("half_hourly_trends").select("*").eq("trend_date", date_str).execute()
    return pd.DataFrame(resp.data or [])

def build_daily_features(df):
    """Aggregate half-hourly data into daily-level features."""
    if df.empty:
        return pd.DataFrame()

    df["trend_name"] = df["trend_name"].astype(str).str.strip()
    grp = df.groupby(["trend_date", "trend_name"])

    out = grp.agg({
        "trend_rank": ["mean", "min", "max"],
        "volume": ["sum"],
    })

    out.columns = ["avg_rank", "min_rank", "max_rank", "total_volume"]
    out = out.reset_index()
    return out

def generate_labels(daily_df):
    """
    For each day, mark will_trend_tomorrow = 1
    if the same trend_name appears on the next day.
    """
    if daily_df.empty:
        return daily_df

    daily_df = daily_df.sort_values(["trend_date", "trend_name"]).reset_index(drop=True)
    dates = sorted(daily_df["trend_date"].unique())
    labeled_parts = []

    for i, d in enumerate(dates):
        today = daily_df[daily_df["trend_date"] == d].copy()
        if i + 1 < len(dates):
            tomorrow = daily_df[daily_df["trend_date"] == dates[i+1]]
            tomorrow_tags = set(tomorrow["trend_name"].unique())
        else:
            tomorrow_tags = set()

        today["will_trend_tomorrow"] = today["trend_name"].apply(
            lambda x: 1 if x in tomorrow_tags else 0
        )
        labeled_parts.append(today)

    return pd.concat(labeled_parts, ignore_index=True)

def run(start_date=None, end_date=None):
    """
    Build cleaned.csv between given date range.
    If not provided, uses 'today - 7 days' to 'today - 1 day'.
    """
    if start_date is None or end_date is None:
        end = datetime.utcnow().date() - timedelta(days=1)
        start = end - timedelta(days=6)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

    frames = []

    cur = start
    while cur <= end:
        ds = str(cur)
        print(f"📅 Processing {ds}")
        df_today = fetch_day(ds)
        if df_today.empty:
            print(f"  ℹ️ No data for {ds}, skipping.")
            cur += timedelta(days=1)
            continue

        daily = build_daily_features(df_today)
        frames.append(daily)
        cur += timedelta(days=1)

    if not frames:
        print("❌ No daily data collected. Nothing to save.")
        return

    combined = pd.concat(frames, ignore_index=True)
    labeled = generate_labels(combined)

    labeled.to_csv("cleaned.csv", index=False)
    print(f"✅ Saved cleaned.csv with {len(labeled)} rows.")

if __name__ == "__main__":
    # You scraped 2025-10-01 to 2025-10-05 → pass that range
    run("2025-09-19", "2025-11-18")
