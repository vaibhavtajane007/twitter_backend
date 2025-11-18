# live_scraper.py
import time
import datetime as dt
import pandas as pd
import snscrape.modules.twitter as sntwitter
from pathlib import Path

# Hashtags you want to track
WATCHLIST = ["#AI", "#ChatGPT", "#Budget2025", "#StockMarket"]

OUT_FILE = "half_hourly_trends.csv"

def fetch_for_hashtag(hashtag, limit=200):
    """Fetch recent tweets containing the hashtag."""
    query = f"{hashtag} lang:en"
    rows = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        rows.append({
            "scrape_time": dt.datetime.utcnow().isoformat(),
            "trend_name": hashtag,
            "tweet_id": tweet.id,
            "date": tweet.date.isoformat(),
            "user": tweet.user.username,
            "content": tweet.content,
            "retweets": tweet.retweetCount,
            "likes": tweet.likeCount,
            "replies": tweet.replyCount,
            "quotes": tweet.quoteCount,
        })
        if i + 1 >= limit:
            break
    return rows

def run_once():
    all_rows = []
    for tag in WATCHLIST:
        print(f"Fetching tweets for {tag} ...")
        rows = fetch_for_hashtag(tag, limit=200)
        all_rows.extend(rows)
        print(f"  got {len(rows)} tweets")

    if not all_rows:
        print("No tweets fetched.")
        return

    df_new = pd.DataFrame(all_rows)
    if Path(OUT_FILE).exists():
        df_new.to_csv(OUT_FILE, mode="a", index=False, header=False)
    else:
        df_new.to_csv(OUT_FILE, index=False)

    print(f"Appended {len(df_new)} rows to {OUT_FILE}")

if __name__ == "__main__":
    # Run once. Schedule via Windows Task Scheduler every 30 minutes.
    run_once()
