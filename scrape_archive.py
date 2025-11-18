import asyncio
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
from supabase import create_client, Client


SUPABASE_URL = "https://fymvqsdxdpzsqimflrmd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5bXZxc2R4ZHB6c3FpbWZscm1kIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ1NjAzNTgsImV4cCI6MjA3MDEzNjM1OH0.d2qtlBkvEhzdHAW2XieZSmjm0-fvZE-HoIoZjfyhWf4"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Parse Half-hourly trends
def parse_half_hourly(soup, trend_date):
    results = []
    for block in soup.find_all("div", class_="tek_tablo"):
        time_div = block.find("div", class_="trend_baslik611")
        time_str = time_div.text.strip() if time_div else "Unknown"

        rows = block.find_all("tr")
        i = 0
        while i < len(rows):
            rank_td = rows[i].find("td", class_="sira611")
            trend_td = rows[i].find("td", class_="trend611")

            rank = int(rank_td.text.strip()) if rank_td and rank_td.text.strip().isdigit() else None
            trend = trend_td.text.strip() if trend_td else None

            volume = None
            if i + 1 < len(rows):
                vol_td = rows[i + 1].find("td", class_="trend611_s")
                if vol_td:
                    vol_text = vol_td.text.strip()
                    match = re.search(r"([\d.,]+)", vol_text)
                    if match:
                        volume = float(match.group(1).replace(".", "").replace(",", ""))

            if trend:
                results.append({
                    "trend_date": trend_date,
                    "trend_time": time_str,
                    "trend_rank": rank,
                    "trend_name": trend,
                    "volume": volume
                })

            i += 2

    return results

# ✅ Parse Most Tweeted & Longest Trending
def parse_summary(soup, trend_date):
    most_tweeted, longest_trending = [], []

    # Most tweeted
    most_section = soup.find("div", id="en_volume_bb")
    if most_section:
        table = most_section.find_next("div", class_="table_bb")
        if table:
            for item in table.find_all("span", class_="table_bbi"):
                rank_span = item.find("span", class_="table_bbs")
                trend_span = item.find("span", class_="table_bbk")
                if not rank_span or not trend_span:
                    continue
                rank_text = rank_span.get_text(strip=True)
                trend_text = trend_span.get_text(strip=True)
                try:
                    rank_int = int(rank_text.replace(")", "").replace("(", ""))
                except ValueError:
                    rank_int = None
                most_tweeted.append({
                    "trend_date": trend_date,
                    "trend_rank": rank_int,
                    "trend_name": trend_text,
                })

    # Longest trending
    long_section = soup.find("div", id="en_volume_bbo")
    if long_section:
        table = long_section.find_next("div", class_="table_bb")
        if table:
            for item in table.find_all("span", class_="table_bbir"):
                rank_span = item.find("span", class_="table_bbs")
                trend_span = item.find("span", class_="table_bbk")
                if not rank_span or not trend_span:
                    continue
                rank_text = rank_span.get_text(strip=True)
                trend_text = trend_span.get_text(strip=True)
                try:
                    rank_int = int(rank_text.replace(")", "").replace("(", ""))
                except ValueError:
                    rank_int = None
                longest_trending.append({
                    "trend_date": trend_date,
                    "trend_rank": rank_int,
                    "trend_name": trend_text,
                })

    return most_tweeted, longest_trending

# ✅ Insert in bulk to Supabase
def insert_batch(table, rows, date_str):
    if not rows:
        print(f"ℹ️ No rows for {table} on {date_str}, skipping.")
        return
    try:
        # delete existing rows for that date to avoid duplicates
        supabase.table(table).delete().eq("trend_date", date_str).execute()
        supabase.table(table).insert(rows).execute()
        print(f"✅ Inserted {len(rows)} rows into {table} for {date_str}")
    except Exception as e:
        print(f"❌ Error inserting batch to {table}: {e}")

# ✅ Process a single day (Scrape + Parse + Insert)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

async def process_day(page, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://archive.twitter-trending.com/india/{date.strftime('%d-%m-%Y')}"
    print(f"🌐 Fetching {url}")
    try:
        # Try to load page fully
        await page.goto(url, timeout=60000, wait_until="networkidle")

        # Try to wait for the table, but don't die if it doesn't appear
        try:
            await page.wait_for_selector(".tek_tablo", timeout=15000)
        except PlaywrightTimeoutError:
            print(f"⚠️ No .tek_tablo selector visible for {date_str}, parsing anyway...")

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        half = parse_half_hourly(soup, date_str)
        most, longest = parse_summary(soup, date_str)

        if not half and not most and not longest:
            print(f"ℹ️ No trend data found on page for {date_str}, skipping inserts.")
            return

        insert_batch("half_hourly_trends", half, date_str)
        insert_batch("most_tweeted", most, date_str)
        insert_batch("longest_trending", longest, date_str)

    except Exception as e:
        print(f"⚠️ Failed on {date_str}: {e}")


# ✅ Main runner for a range of dates with concurrency
async def run_range(start_date="2025-10-01", end_date="2025-10-05"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # reuse up to 5 pages
        pages = [await context.new_page() for _ in range(5)]
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        for i in range(0, len(dates), 5):
            chunk = dates[i:i+5]
            tasks = []
            for j, date in enumerate(chunk):
                tasks.append(process_day(pages[j], date))
            await asyncio.gather(*tasks)

        await browser.close()

if __name__ == "__main__":
    from datetime import date, timedelta

    # 📅 Choose how many days back you want. Example: last 60 days.
    DAYS_BACK = 60

    today = date(2025, 11, 18)  # fix it to "today" for your project
    start = today - timedelta(days=DAYS_BACK)

    start_str = start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    print(f"🚀 Scraping from {start_str} to {end_str}")
    asyncio.run(run_range(start_str, end_str))

