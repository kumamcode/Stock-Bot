from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import praw
import json
import requests
import schedule
import time
import sys
import logging
import smtplib
import yfinance as yf
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
import warnings

# Suppress yfinance warnings and urllib3 warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()  # loads .env into os.environ

# ---------- LLM (local via Ollama) ----------
# Default model can be overridden via OLLAMA_MODEL env variable
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

def ask_llm(prompt: str, model: str = None) -> str:
    if model is None:
        model = DEFAULT_MODEL
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()

# ---------- Data (simple, free sources) ----------
def fetch_headlines():
    # Google News RSS (free)
    rss_url = "https://news.google.com/rss/search?q=stock+market+OR+fed+OR+inflation&hl=en-US&gl=US&ceid=US:en"
    xml = requests.get(rss_url, timeout=30).text
    # lightweight parse without extra deps
    items = xml.split("<item>")[1:6]
    titles = []
    for it in items:
        t = it.split("<title>")[1].split("</title>")[0]
        titles.append(t)
    return titles

def stooq_last_two_closes(symbol: str):
    # Stooq daily CSV, free no key
    # Try a few symbol variants (with and without leading ^). Provide clear errors when data is missing.
    candidates = [symbol]
    if symbol.startswith("^"):
        candidates.append(symbol.lstrip("^"))

    last_err = None
    for s in candidates:
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        try:
            resp = requests.get(url, timeout=30)
            text = resp.text.strip()
        except Exception as e:
            last_err = e
            continue

        # quick sanity checks
        if not text or text.lower().startswith("<html"):
            last_err = RuntimeError(f"Stooq returned no CSV for {s} (url: {url})")
            continue

        csv = text.splitlines()
        if len(csv) < 3:
            last_err = RuntimeError(f"Not enough CSV rows for {s} (url: {url})")
            continue

        try:
            prev = csv[-2].split(",")
            last = csv[-1].split(",")
            # Date,Open,High,Low,Close,Volume
            return {
                "date": last[0],
                "prev_close": float(prev[4]),
                "close": float(last[4]),
            }
        except Exception as e:
            last_err = e
            continue

    # If we exhaust candidates, raise a helpful error
    # Try a yfinance fallback for VIX-like symbols (more reliable for VIX)
    if "vix" in symbol.lower():
        try:
            #import yfinance as yf

            ticker = "^VIX" if symbol.startswith("^") else symbol
            hist = yf.Ticker(ticker).history(period="5d")
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
                close = float(closes.iloc[-1])
                date = closes.index[-1].strftime("%Y-%m-%d")
                return {"date": date, "prev_close": prev_close, "close": close}
        except ModuleNotFoundError:
            last_err = "yfinance not installed (pip install yfinance)"
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to fetch recent closes for {symbol}: {last_err}")

def pct_change(a, b):
    try:
        if a is None or b is None:
            return None
        if b == 0:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None
def summarize_reddit_sentiment(results: dict) -> str:
    """
    Turns per-subreddit pos/neg/neu into a compact summary for the LLM.
    Includes an overall weighted net sentiment and top +/- subreddits.
    """
    items = []
    total = 0
    w_pos = w_neg = w_neu = 0.0

    for sub, r in results.items():
        if not isinstance(r, dict) or "error" in r:
            continue
        c = int(r.get("count", 0) or 0)
        if c <= 0:
            continue
        pos = float(r.get("pos", 0.0))
        neg = float(r.get("neg", 0.0))
        neu = float(r.get("neu", 0.0))
        net = pos - neg
        items.append((sub, net, pos, neg, neu, c))
        total += c
        w_pos += pos * c
        w_neg += neg * c
        w_neu += neu * c

    if total == 0:
        return "Reddit sentiment: unavailable"

    overall_pos = w_pos / total
    overall_neg = w_neg / total
    overall_neu = w_neu / total
    overall_net = overall_pos - overall_neg

    items.sort(key=lambda x: x[1], reverse=True)
    top_pos = items[:3]
    top_neg = items[-3:][::-1]

    def fmt_row(t):
        sub, net, pos, neg, neu, c = t
        return f"{sub}: net={net:+.2f} (pos={pos:.2f}, neg={neg:.2f}, n={c})"

    lines = [
        "Reddit sentiment (VADER on hot posts; noisy/contrarian context, not a signal):",
        f"- Overall: net={overall_net:+.2f} (pos={overall_pos:.2f}, neg={overall_neg:.2f}, neu={overall_neu:.2f}, n={total})",
        "- Most positive subs: " + " | ".join(fmt_row(x) for x in top_pos),
        "- Most negative subs: " + " | ".join(fmt_row(x) for x in top_neg),
    ]
    return "\n".join(lines)

def is_valid_ticker(ticker: str) -> bool:
    """
    Quick validation to check if a ticker exists and has data.
    Returns False for invalid/delisted tickers.
    Suppresses yfinance warnings during validation.
    """
    import io
    import contextlib
    
    # Suppress stderr during validation to avoid yfinance error messages
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            t = yf.Ticker(ticker)
            # Try to get a small amount of history data
            hist = t.history(period="5d")
            # If we get empty data, ticker is likely invalid
            if hist is None or len(hist) == 0:
                return False
            # Check if we have price data
            close = hist["Close"].dropna()
            return len(close) > 0
        except Exception:
            return False

def yf_snapshot(ticker: str) -> dict | None:
    """
    Lightweight snapshot: price/returns + a few fundamentals if available.
    Returns None if ticker is invalid or data cannot be fetched.
    Suppresses yfinance warnings during data fetching.
    """
    import io
    import contextlib
    
    # Suppress stderr during fetch to avoid yfinance error messages
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            t = yf.Ticker(ticker)

            # Price history (works reliably)
            hist = t.history(period="6mo")
            if hist is None or len(hist) == 0:
                return None
            
            close = hist["Close"].dropna()
            if len(close) == 0:
                return None
                
            price = float(close.iloc[-1]) if len(close) else None
            ret_1m = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1.0) if len(close) >= 22 else None
            ret_3m = (float(close.iloc[-1]) / float(close.iloc[-63]) - 1.0) if len(close) >= 64 else None

            # Fundamentals (sometimes missing / flaky—guard heavily)
            info = {}
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}

            def safe_get(k):
                v = info.get(k)
                return v if v is not None else None

            return {
                "ticker": ticker,
                "price": price,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "marketCap": safe_get("marketCap"),
                "trailingPE": safe_get("trailingPE"),
                "forwardPE": safe_get("forwardPE"),
                "dividendYield": safe_get("dividendYield"),
                "beta": safe_get("beta"),
                "sector": safe_get("sector"),
                "industry": safe_get("industry"),
                "longName": safe_get("longName"),
            }
        except Exception:
            return None

def format_snapshots(snaps: list[dict]) -> str:
    lines = ["yfinance snapshots (for context; may be incomplete):"]
    for s in snaps:
        def fmt_pct(x):
            return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"
        def fmt_num(x):
            if x is None: return "n/a"
            if isinstance(x, (int, float)) and x > 1e9: return f"{x/1e9:.1f}B"
            if isinstance(x, (int, float)) and x > 1e6: return f"{x/1e6:.1f}M"
            return str(x)

        lines.append(
            f"- {s['ticker']} ({s.get('sector','n/a')}): price={fmt_num(s['price'])}, "
            f"1m={fmt_pct(s['ret_1m'])}, 3m={fmt_pct(s['ret_3m'])}, "
            f"mktcap={fmt_num(s['marketCap'])}, "
            f"P/E(TTM)={fmt_num(s['trailingPE'])}, Fwd P/E={fmt_num(s['forwardPE'])}, "
            f"div_yield={fmt_pct(s['dividendYield'])}, beta={fmt_num(s['beta'])}"
        )
    return "\n".join(lines)

def sanitize_and_extract_ticker(raw: str) -> str | None:
    """Sanitize LLM token and try to extract a plausible short ticker.
    Returns None if the token looks invalid and should be skipped.
    """
    if not raw or not isinstance(raw, str):
        return None
    tt = raw.strip().upper()
    tt = tt.lstrip("$")
    # keep only letters, numbers, dot, dash
    tt = re.sub(r"[^A-Z0-9\.\-]", "", tt)
    # If short enough, accept
    if 1 <= len(tt) <= 7:
        return tt

    # Try to extract trailing short ticker (1-5 letters/digits, optionally . or - parts)
    m = re.search(r"([A-Z0-9]{1,5}(?:[\.\-][A-Z0-9]{1,4})?)$", tt)
    if m:
        cand = m.group(1)
        if 1 <= len(cand) <= 7:
            return cand
    # As a last resort, query Yahoo Finance search endpoint to resolve fuzzy names
    resolved = resolve_ticker_via_yahoo(raw)
    if resolved:
        return resolved

    logging.info("Skipping improbable ticker token: %s", raw)
    return None

def resolve_ticker_via_yahoo(query: str) -> str | None:
    """Best-effort resolve a fuzzy name to a ticker using Yahoo Finance's search API.
    Returns an uppercase ticker symbol (e.g. 'SPY') or None if not found.
    """
    # Simple in-memory cache for this run
    if not hasattr(resolve_ticker_via_yahoo, "_cache"):
        resolve_ticker_via_yahoo._cache = {}
    cache = resolve_ticker_via_yahoo._cache
    if query in cache:
        return cache[query]

    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": 10, "newsCount": 0}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("quotes") or []
        if not quotes:
            cache[query] = None
            return None

        # Prefer results that are ETFs or EQUITY and on major US exchanges
        preferred = []
        for qd in quotes:
            sym = qd.get("symbol")
            qtype = (qd.get("quoteType") or "").upper()
            exch = (qd.get("exchDisp") or qd.get("exchange") or "").upper()
            if not sym:
                continue
            sym_up = sym.upper()
            if not re.fullmatch(r"[A-Z0-9\.\-]{1,7}", sym_up):
                continue
            score = 0
            if qtype in ("ETF", "EQUITY", "MUTUALFUND"):
                score += 10
            if exch in ("NMS", "NASDAQ", "NYQ", "NYSE", "AMEX") or exch.startswith("NYSE"):
                score += 5
            preferred.append((score, sym_up))

        preferred.sort(reverse=True)

        # Validate candidates by asking yfinance for a tiny history slice
        for _, sym in preferred + [(0, q.get("symbol").upper() if q.get("symbol") else None) for q in quotes]:
            if not sym:
                continue
            try:
                hist = yf.Ticker(sym).history(period="1d")
                if hist is not None and len(hist) > 0:
                    cache[query] = sym
                    return sym
            except Exception:
                continue

        cache[query] = None
        return None
    except Exception as e:
        logging.info("Yahoo resolve failed for %s: %s", query, e)
        cache[query] = None
        return None
# ---------- Email ----------
def send_email(subject: str, body: str):
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")

    # TO_EMAIL, CC_EMAIL, BCC_EMAIL may be comma-separated lists in the env
    to_emails = [e.strip() for e in os.environ.get("TO_EMAIL", "").split(",") if e.strip()]
    cc_emails = [e.strip() for e in os.environ.get("CC_EMAIL", "").split(",") if e.strip()]
    bcc_emails = [e.strip() for e in os.environ.get("BCC_EMAIL", "").split(",") if e.strip()]

    if not gmail_user or not gmail_app_password:
        raise RuntimeError("GMAIL_USER and GMAIL_APP_PASSWORD must be set in environment")

    if not to_emails:
        raise RuntimeError("No recipient found. Set TO_EMAIL in environment (comma-separated for multiple).")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)

    # Actual delivery list includes To + Cc + Bcc
    all_recipients = to_emails + cc_emails + bcc_emails

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, all_recipients, msg.as_string())

def fetch_reddit_sentiment(subreddits: list, limit=50):
    """
    Fetch Reddit sentiment using Reddit's public JSON API (no authentication required).
    Falls back to PRAW if credentials are provided and API fails.
    """
    # Try using Reddit's public JSON API first (no auth needed)
    analyzer = SentimentIntensityAnalyzer()
    results = {}
    user_agent = os.environ.get("REDDIT_USER_AGENT", "stock-bot/0.1 by /u/stockbot")
    
    for sub in subreddits:
        pos = neg = neu = cnt = 0
        try:
            # Use Reddit's JSON API (public, no auth required)
            url = f"https://www.reddit.com/r/{sub}/hot.json"
            headers = {"User-Agent": user_agent}
            params = {"limit": min(limit, 100)}  # Reddit API max is 100
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract posts from JSON response
            posts = data.get("data", {}).get("children", [])
            for post_data in posts:
                post = post_data.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text = f"{title} {selftext}".strip()
                
                if text:
                    score = analyzer.polarity_scores(text)["compound"]
                    if score >= 0.05:
                        pos += 1
                    elif score <= -0.05:
                        neg += 1
                    else:
                        neu += 1
                    cnt += 1
            
            if cnt:
                results[sub] = {"pos": pos / cnt, "neg": neg / cnt, "neu": neu / cnt, "count": cnt}
            else:
                results[sub] = {"error": "No posts found"}
                
        except Exception as e:
            # Try fallback to PRAW if credentials are available
            client_id = os.environ.get("REDDIT_CLIENT_ID")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
            
            if client_id and client_secret:
                try:
                    reddit = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent,
                    )
                    pos = neg = neu = cnt = 0
                    for submission in reddit.subreddit(sub).hot(limit=limit):
                        text = (submission.title or "") + " " + (submission.selftext or "")
                        score = analyzer.polarity_scores(text)["compound"]
                        if score >= 0.05:
                            pos += 1
                        elif score <= -0.05:
                            neg += 1
                        else:
                            neu += 1
                        cnt += 1
                    if cnt:
                        results[sub] = {"pos": pos / cnt, "neg": neg / cnt, "neu": neu / cnt, "count": cnt}
                    else:
                        results[sub] = {"error": "No posts found (PRAW fallback)"}
                except Exception as praw_error:
                    results[sub] = {"error": f"JSON API failed: {e}, PRAW fallback failed: {praw_error}"}
            else:
                results[sub] = {"error": str(e)}
    
    return results

def main():
    # Pull a few market proxies
    spx = stooq_last_two_closes("^spx")  # S&P 500
    # VIX sometimes fails on Stooq; handle errors gracefully
    try:
        vix = stooq_last_two_closes("^vix")  # VIX
    except Exception as e:
        print(f"Warning: failed to fetch VIX data: {e}")
        vix = None

    spx_chg = pct_change(spx["close"], spx["prev_close"])
    vix_chg = None
    if vix:
        vix_chg = pct_change(vix["close"], vix["prev_close"])

    headlines = fetch_headlines()
    reddit_sentiment = fetch_reddit_sentiment(["wallstreetbets", "investing", "stocks", "ETFs", "Bogleheads", "dividends", "SecurityAnalysis", "ValueInvesting", "finance"], limit=100)
    reddit_block = summarize_reddit_sentiment(reddit_sentiment)
    # Personal config (edit these)
    user_profile = {
        "risk_tolerance": "medium-high",
        "time_horizon": "5+ years",
        "style": "mostly ETFs / long-term, however, hoping to make some strategic bets on individual stocks that are a bit riskier",
        "constraints": [
            "No leverage",
            "No day trading",
            "Prefer simple rules and explainable reasoning",
        ],
    }

    # Build a safe VIX line for the prompt (handle missing data)
    if vix is None or vix_chg is None:
        vix_line = "VIX: data unavailable"
    else:
        vix_line = f"VIX: {vix['close']:.2f} ({vix_chg:+.2f}%)"

        # ---------- PASS 1: ask for tickers JSON only ----------
    proposal_prompt = f"""
You are selecting candidate tickers for a personal investing memo.
Return ONLY valid JSON. No markdown, no commentary.

Schema:
{{
  "core_stocks": ["TICKER","TICKER","TICKER"],
  "etfs": ["TICKER","TICKER","TICKER"],
  "ai_software": ["TICKER","TICKER","TICKER"],
  "robotics": ["TICKER","TICKER","TICKER"],
  "international": ["TICKER","TICKER","TICKER"]
}}

Constraints:
- Prefer large/mid-cap, liquid names. Avoid penny stocks.
- ETFs must be real tickers.
- International can be ADRs or major non-US listings; keep widely traded.
- Avoid duplicates across categories.
- User constraints: {", ".join(user_profile["constraints"])}

Market snapshot:
- S&P 500: {spx["close"]:.2f} ({spx_chg:+.2f}%)
- {vix_line}

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

{reddit_block}
"""
    proposal_raw = ask_llm(proposal_prompt)

    try:
        proposal = json.loads(proposal_raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Ticker proposal was not valid JSON:\n{proposal_raw}")

    # Flatten + de-dupe tickers (sanitize tokens returned by the LLM)
    tickers = []
    for key in ["core_stocks", "etfs", "ai_software", "robotics", "international"]:
        for t in proposal.get(key, []):
            if isinstance(t, str):
                tt = sanitize_and_extract_ticker(t)
                if tt and tt not in tickers:
                    tickers.append(tt)

    # ---------- Fetch yfinance context ----------
    # Filter out invalid tickers first to avoid noisy errors
    valid_tickers = []
    invalid_tickers = []
    for tk in tickers:
        if is_valid_ticker(tk):
            valid_tickers.append(tk)
        else:
            invalid_tickers.append(tk)
    
    if invalid_tickers:
        logging.info(f"Skipping {len(invalid_tickers)} invalid ticker(s): {', '.join(invalid_tickers)}")
    
    snaps = []
    for tk in valid_tickers:
        snapshot = yf_snapshot(tk)
        if snapshot is not None:
            snaps.append(snapshot)

    yf_block = format_snapshots(snaps)

    # ---------- PASS 2: final memo using yfinance facts ----------
    final_prompt = f"""
You are my personal investing memo writer. Be cautious, avoid certainty, avoid hype.
Use Reddit sentiment only as context (noisy/contrarian), not predictive.
Use the yfinance context below as factual input; do NOT invent metrics not shown.

User context:
- Risk tolerance: {user_profile["risk_tolerance"]}
- Horizon: {user_profile["time_horizon"]}
- Style: {user_profile["style"]}
- Constraints: {", ".join(user_profile["constraints"])}

Market snapshot:
- S&P 500: {spx["close"]:.2f} ({spx_chg:+.2f}%)
- {vix_line}

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

{reddit_block}

{yf_block}

Write the memo around 1000 words with:
1) 2 biggest market takeaways (tie to headlines)
2) Risk regime (bullish/neutral/bearish) + 1 sentence justification (reference SPX/VIX + sentiment)
3) Suggested actions (allocation/DCA/rebalancing/cash buffer)
4) Picks: use ONLY the selected tickers below. For each category: 3 tickers + 2 quick bullets each.
Selected tickers:
- Core Stocks: {", ".join(proposal.get("core_stocks", []))}
- ETFs: {", ".join(proposal.get("etfs", []))}
- AI/Software: {", ".join(proposal.get("ai_software", []))}
- Robotics: {", ".join(proposal.get("robotics", []))}
- International: {", ".join(proposal.get("international", []))}
5) One thing to watch that could change the view
"""
    brief = ask_llm(final_prompt)


    subject = f"Daily Market Updates and Stock Picks({spx['date']})"
    body = brief + f"\n\n Not Financial Advice -> Stock Bot created by Kuma McCraw leveraging {DEFAULT_MODEL} locally."
    send_email(subject, body)

if __name__ == "__main__":
    # Manual run options for testing:
    #   python Stock_Bot_V1_main_py --run-now   (or -r)  -> run once and exit
    #   python Stock_Bot_V1_main_py --prompt    (or -p)  -> ask interactively whether to run now
    if "--run-now" in sys.argv or "-r" in sys.argv:
        print("Manual run requested (--run-now). Running once and exiting.")
        main()
        sys.exit(0)

    if "--prompt" in sys.argv or "-p" in sys.argv:
        try:
            ans = input("Run Stock Bot now? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans == "y":
            print("Running now...")
            main()
            sys.exit(0)
        else:
            print("Continuing to scheduler...")

    # Schedule the main function to run weekdays at 9:15 AM (before market open at 9:30 AM ET)
    schedule.every().monday.at("09:15").do(main)
    schedule.every().tuesday.at("09:15").do(main)
    schedule.every().wednesday.at("09:15").do(main)
    schedule.every().thursday.at("09:15").do(main)
    schedule.every().friday.at("09:15").do(main)
    
    print("Stock Bot scheduler started. Running daily at 09:15 AM (weekdays only).")
    print("Press Ctrl+C to stop.")
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute if a job needs to run