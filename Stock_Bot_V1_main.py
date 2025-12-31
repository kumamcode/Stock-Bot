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
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
import warnings
import subprocess

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

# ---------- LLM (Cloud via Groq) ----------
# API STATUS: ✅ WORKING (free tier available, requires GROQ_API_KEY in .env)
# Default model can be overridden via GROQ_MODEL env variable
# Available models: llama-3.3-70b-versatile, llama-3.1-70b-instant, llama-3.1-8b-instant, mixtral-8x7b-32768
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq API - No local setup needed!
def ask_llm(prompt: str, model: str = None) -> str:
    """
    Calls Groq cloud LLM API (free tier available).
    Much faster and more powerful than local models.
    No local setup needed - runs entirely in the cloud!
    """
    if model is None:
        model = DEFAULT_MODEL
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set in .env file. "
            "Get a free API key at https://console.groq.com"
        )
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        logging.info(f"Calling Groq API with model: {model}")
        logging.info(f"Prompt length: {len(prompt)} characters")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        result = response.choices[0].message.content.strip()
        logging.info(f"Received response (length: {len(result)} characters)")
        return result
        
    except ImportError:
        # Fallback to requests if groq library not installed
        logging.warning("groq library not installed, using requests fallback")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

# ---------- Data (simple, free sources) ----------
def fetch_headlines():
    """
    Fetches stock market headlines from Google News RSS feed.
    API: https://news.google.com/rss
    Status: ✅ WORKING (free, no authentication required)
    Returns: List of headline titles (max 5)
    """
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
    """
    Fetches last two closing prices from Stooq.com CSV API.
    API: https://stooq.com/q/d/l/?s={symbol}&i=d
    Status: ✅ WORKING (free, no authentication required)
    Falls back to yfinance for VIX if Stooq fails.
    Returns: dict with date, prev_close, close
    """
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
    Fetches stock/ETF snapshot data from Yahoo Finance via yfinance library.
    API: Yahoo Finance (via yfinance Python library)
    Status: ✅ WORKING (free, no authentication required)
    Returns: dict with price, returns, fundamentals, or None if ticker invalid.
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
    """
    Only accept things that look like real ticker symbols.
    If it's a company name (e.g., NVIDIA), try to resolve via Yahoo search.
    """
    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip().upper()

    # Strip leading $ and punctuation
    s = s.lstrip("$")
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)

    # Reject empty
    if not s:
        return None

    # Heuristic: Real US tickers are usually 1–5 chars (sometimes 6, rarely 7).
    # BUT words like NVIDIA/MICRO will slip in. So we only accept as "direct tickers"
    # if it matches a ticker-like pattern AND is not a long word.
    direct_ok = re.fullmatch(r"[A-Z]{1,5}([.\-][A-Z0-9]{1,2})?", s) is not None

    if direct_ok:
        return s

    # If it doesn't look like a ticker symbol, try resolving via Yahoo search
    resolved = resolve_ticker_via_yahoo(raw)
    if resolved:
        return resolved

    logging.info("Skipping invalid ticker token: %s", raw)
    return None


def resolve_ticker_via_yahoo(query: str) -> str | None:
    """
    Resolves fuzzy company names to ticker symbols using Yahoo Finance search API.
    API: https://query1.finance.yahoo.com/v1/finance/search
    Status: ✅ WORKING (free, no authentication required)
    Returns: Uppercase ticker symbol (e.g. 'SPY') or None if not found.
    Uses caching to avoid repeated API calls for the same query.
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
def markdown_to_html(text: str) -> str:
    """
    Convert markdown-style text to HTML for email formatting.
    Converts # headers, * bullet points, + sub-bullets, **bold**, numbered lists, etc. to proper HTML.
    """
    lines = text.split('\n')
    result_lines = []
    in_ul = False
    in_ol = False
    in_sub_ul = False  # Track nested sub-bullets
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check for leading spaces to detect indentation level
        leading_spaces = len(line) - len(line.lstrip())
        
        # Skip empty lines (will add spacing later)
        if not stripped:
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append('<br>')
            continue
        
        # Headers
        if stripped.startswith('###'):
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(f'<h3>{stripped[3:].strip()}</h3>')
        elif stripped.startswith('##'):
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(f'<h2>{stripped[2:].strip()}</h2>')
        elif stripped.startswith('#'):
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(f'<h2>{stripped[1:].strip()}</h2>')
        # Sub-bullet points (+ or indented bullets)
        elif stripped.startswith('+') or (leading_spaces > 2 and re.match(r'^[\*\-\•\+]\s+', stripped)):
            # Close numbered lists if open
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            # If we're not in a main list, start one (treat as main bullet)
            if not in_ul:
                result_lines.append('<ul>')
                in_ul = True
                in_sub_ul = False
            # If we're in a main list but not in a sub-list, start nested list
            elif in_ul and not in_sub_ul:
                # Check if last item was a main bullet - nest the sub-bullet under it
                if result_lines and result_lines[-1].endswith('</li>'):
                    # Remove the closing </li> and start nested list
                    result_lines[-1] = result_lines[-1].replace('</li>', '')
                    result_lines.append('<ul>')
                    in_sub_ul = True
                else:
                    # Start a new nested list
                    result_lines.append('<ul>')
                    in_sub_ul = True
            # Remove the + or bullet and formatting
            content = re.sub(r'^[\+\*\-\•]\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        # Numbered lists (1) or 1. format
        elif re.match(r'^\d+[\)\.]\s+', stripped):
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if not in_ol:
                result_lines.append('<ol>')
                in_ol = True
            # Remove the number and formatting
            content = re.sub(r'^\d+[\)\.]\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        # Main bullet points (*, -, •) - but not if it's indented (that's a sub-bullet)
        elif re.match(r'^[\*\-\•]\s+', stripped) and leading_spaces == 0:
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            if not in_ul:
                result_lines.append('<ul>')
                in_ul = True
            # Remove the bullet and formatting
            content = re.sub(r'^[\*\-\•]\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        # Regular paragraph
        else:
            if in_sub_ul:
                result_lines.append('</ul>')
                in_sub_ul = False
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(f'<p>{stripped}</p>')
    
    # Close any open lists
    if in_sub_ul:
        result_lines.append('</ul>')
    if in_ul:
        result_lines.append('</ul>')
    if in_ol:
        result_lines.append('</ol>')
    
    html = '\n'.join(result_lines)
    
    # Convert bold (**text** -> <strong>text</strong>)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italic (_text_ -> <em>text</em>)
    html = re.sub(r'_(.*?)_', r'<em>\1</em>', html)
    
    return html

def send_email(subject: str, body: str):
    """
    Sends email via Gmail SMTP with HTML formatting.
    API: smtp.gmail.com:465 (SMTP_SSL)
    Status: ✅ WORKING (requires GMAIL_USER and GMAIL_APP_PASSWORD in .env)
    Supports TO, CC, and BCC recipients (comma-separated in env vars).
    Converts markdown-style formatting to HTML for proper rendering in Gmail.
    """
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

    # Convert markdown to HTML
    html_body = markdown_to_html(body)
    
    # Create HTML email with proper styling
    html_email = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
                margin-top: 20px;
            }}
            h3 {{
                color: #34495e;
                margin-top: 15px;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 30px;
            }}
            ul ul {{
                margin: 5px 0;
                padding-left: 25px;
                list-style-type: circle;
            }}
            li {{
                margin: 5px 0;
            }}
            p {{
                margin: 10px 0;
            }}
            strong {{
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # Create multipart message (HTML with plain text fallback)
    msg = MIMEMultipart('alternative')
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)

    # Add both plain text and HTML versions
    part1 = MIMEText(body, "plain", "utf-8")
    part2 = MIMEText(html_email, "html", "utf-8")
    
    msg.attach(part1)
    msg.attach(part2)

    # Actual delivery list includes To + Cc + Bcc
    all_recipients = to_emails + cc_emails + bcc_emails

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, all_recipients, msg.as_string())

def extract_tickers_from_text(text: str) -> set:
    """
    Extract potential stock tickers from text.
    Looks for patterns like $TICKER, TICKER, or common ticker formats.
    Filters out common words that aren't tickers.
    """
    tickers = set()
    
    # Pattern 1: $TICKER (e.g., $AAPL, $TSLA) - most reliable indicator
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
    tickers.update(dollar_tickers)
    
    # Pattern 2: TICKER in all caps (common in finance discussions)
    # Look for 2-5 letter sequences that are all caps and standalone
    caps_tickers = re.findall(r'\b([A-Z]{2,5})\b', text.upper())
    
    # Expanded list of common words that aren't tickers
    common_words = {
        # Common articles/prepositions
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 
        'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 
        'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'SHE', 'PUT', 'END', 'DID', 'SET', 'OFF', 
        'TRY', 'TOO', 'ANY', 'OWN', 'ASK', 'YES', 'LET', 'RUN', 'FAR', 'TOP',
        # The problematic ones
        'TO', 'IN', 'OF', 'ON', 'AT', 'BY', 'AS', 'IS', 'IT', 'BE', 'DO', 'OR', 'IF', 'UP',
        # Finance terms that aren't tickers
        'BUY', 'SELL', 'HOLD', 'ETF', 'SPY', 'QQQ', 'DIA', 'VIX', 'SPX', 'DJI', 'USD', 'EUR',
        # Other common words
        'VIA', 'DUE', 'PAY', 'TAX', 'FEE', 'NET', 'GAP', 'IPO', 'EPS', 'PE', 'ROI', 'YTD',
        'CEO', 'CFO', 'SEC', 'FED', 'IRS', 'GDP', 'CPI', 'PMI'
    }
    
    # Only include caps tickers that aren't common words
    filtered_tickers = [t for t in caps_tickers if t not in common_words and len(t) >= 2]
    tickers.update(filtered_tickers)
    
    return tickers

def fetch_reddit_sentiment(subreddits: list, limit=50):
    """
    Fetches Reddit post sentiment from specified subreddits.
    Also extracts and tracks stock tickers mentioned in posts.
    API: https://www.reddit.com/r/{subreddit}/hot.json (primary, free, no auth)
    Fallback: PRAW (Reddit API wrapper) if REDDIT_CLIENT_ID/SECRET provided
    Status: ✅ WORKING (public JSON API works without credentials)
    Uses VADER sentiment analysis to classify posts as positive/negative/neutral.
    Returns: dict mapping subreddit names to sentiment metrics, and stock mentions.
    """
    # Try using Reddit's public JSON API first (no auth needed)
    analyzer = SentimentIntensityAnalyzer()
    results = {}
    stock_mentions = {}  # Track ticker: {count, total_sentiment, positive_count, negative_count}
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
                    
                    # Extract tickers from this post
                    tickers = extract_tickers_from_text(text)
                    for ticker in tickers:
                        ticker_clean = sanitize_and_extract_ticker(ticker)
                        # Validate ticker exists before tracking it (filters out common words like TO, IN, OF)
                        if ticker_clean and len(ticker_clean) >= 2 and is_valid_ticker(ticker_clean):
                            if ticker_clean not in stock_mentions:
                                stock_mentions[ticker_clean] = {
                                    "count": 0,
                                    "total_sentiment": 0.0,
                                    "positive_count": 0,
                                    "negative_count": 0,
                                    "mentions": []
                                }
                            stock_mentions[ticker_clean]["count"] += 1
                            stock_mentions[ticker_clean]["total_sentiment"] += score
                            if score >= 0.05:
                                stock_mentions[ticker_clean]["positive_count"] += 1
                            elif score <= -0.05:
                                stock_mentions[ticker_clean]["negative_count"] += 1
                            stock_mentions[ticker_clean]["mentions"].append({
                                "subreddit": sub,
                                "sentiment": score,
                                "title": title[:100]  # First 100 chars
                            })
            
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
                        
                        # Extract tickers from this post
                        tickers = extract_tickers_from_text(text)
                        for ticker in tickers:
                            ticker_clean = sanitize_and_extract_ticker(ticker)
                            # Validate ticker exists before tracking it (filters out common words like TO, IN, OF)
                            if ticker_clean and len(ticker_clean) >= 2 and is_valid_ticker(ticker_clean):
                                if ticker_clean not in stock_mentions:
                                    stock_mentions[ticker_clean] = {
                                        "count": 0,
                                        "total_sentiment": 0.0,
                                        "positive_count": 0,
                                        "negative_count": 0,
                                        "mentions": []
                                    }
                                stock_mentions[ticker_clean]["count"] += 1
                                stock_mentions[ticker_clean]["total_sentiment"] += score
                                if score >= 0.05:
                                    stock_mentions[ticker_clean]["positive_count"] += 1
                                elif score <= -0.05:
                                    stock_mentions[ticker_clean]["negative_count"] += 1
                    if cnt:
                        results[sub] = {"pos": pos / cnt, "neg": neg / cnt, "neu": neu / cnt, "count": cnt}
                    else:
                        results[sub] = {"error": "No posts found (PRAW fallback)"}
                except Exception as praw_error:
                    results[sub] = {"error": f"JSON API failed: {e}, PRAW fallback failed: {praw_error}"}
            else:
                results[sub] = {"error": str(e)}
    
    # Store stock mentions in results for later analysis
    results["_stock_mentions"] = stock_mentions
    return results

def get_top_reddit_stocks(reddit_sentiment: dict, top_n: int = 3) -> list:
    """
    Analyze Reddit stock mentions and return top N stocks by mention count and sentiment.
    Returns list of dicts with ticker, mention_count, avg_sentiment, and positive_ratio.
    """
    stock_mentions = reddit_sentiment.get("_stock_mentions", {})
    
    if not stock_mentions:
        return []
    
    # Calculate scores for each stock
    scored_stocks = []
    for ticker, data in stock_mentions.items():
        if data["count"] < 2:  # Need at least 2 mentions to be significant
            continue
        
        avg_sentiment = data["total_sentiment"] / data["count"]
        positive_ratio = data["positive_count"] / data["count"] if data["count"] > 0 else 0
        
        # Score = mention_count * (1 + avg_sentiment) * positive_ratio
        # This favors stocks with many mentions, positive sentiment, and high positive ratio
        score = data["count"] * (1 + avg_sentiment) * (1 + positive_ratio)
        
        scored_stocks.append({
            "ticker": ticker,
            "mention_count": data["count"],
            "avg_sentiment": avg_sentiment,
            "positive_ratio": positive_ratio,
            "positive_count": data["positive_count"],
            "negative_count": data["negative_count"],
            "score": score
        })
    
    # Sort by score (highest first) and return top N
    scored_stocks.sort(key=lambda x: x["score"], reverse=True)
    return scored_stocks[:top_n]

def format_top_reddit_stocks(top_stocks: list) -> str:
    """
    Format top Reddit stocks for inclusion in prompts.
    """
    if not top_stocks:
        return "Top Reddit stocks: No significant stock mentions found in recent Reddit posts."
    
    lines = ["Top 3 stocks trending on Reddit (based on mention frequency and sentiment):"]
    for i, stock in enumerate(top_stocks, 1):
        sentiment_emoji = "📈" if stock["avg_sentiment"] > 0.1 else "📉" if stock["avg_sentiment"] < -0.1 else "➡️"
        lines.append(
            f"{i}. {stock['ticker']} {sentiment_emoji} - "
            f"Mentioned {stock['mention_count']}x, "
            f"avg sentiment: {stock['avg_sentiment']:+.2f}, "
            f"{stock['positive_count']} positive / {stock['negative_count']} negative mentions"
        )
    
    return "\n".join(lines)

def main():
    """
    Main function that orchestrates data collection from all APIs and generates investment memo.
    
    API Usage Flow:
    1. ✅ Stooq API: Fetch S&P 500 and VIX data
    2. ✅ Google News RSS: Fetch market headlines
    3. ✅ Reddit JSON API: Fetch sentiment from 9 finance subreddits
    4. ✅ Groq Cloud LLM: Generate ticker suggestions (Pass 1)
    5. ✅ Yahoo Finance (yfinance): Fetch data for suggested tickers
    6. ✅ Groq Cloud LLM: Generate final investment memo (Pass 2)
    7. ✅ Gmail SMTP: Email the memo
    
    All APIs are properly integrated into both LLM prompts.
    """
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

    # Fetch headlines from Google News RSS (✅ WORKING - included in prompts)
    headlines = fetch_headlines()
    
    # Fetch Reddit sentiment from 9 finance subreddits (✅ WORKING - included in prompts)
    reddit_sentiment = fetch_reddit_sentiment(["wallstreetbets", "investing", "stocks", "ETFs", "Bogleheads", "dividends", "SecurityAnalysis", "ValueInvesting", "finance"], limit=100)
    reddit_block = summarize_reddit_sentiment(reddit_sentiment)
    
    # Extract top 3 stocks from Reddit analysis
    top_reddit_stocks = get_top_reddit_stocks(reddit_sentiment, top_n=3)
    top_stocks_block = format_top_reddit_stocks(top_reddit_stocks)
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
    # PROMPT INCLUDES: ✅ SPX/VIX data, ✅ Headlines, ✅ Reddit sentiment, ✅ Top Reddit stocks
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
- Make sure that the tickers provided are actually in the industries for which they are selected.
- IMPORTANT: Consider the top Reddit stocks below - these are trending based on Reddit analysis.
- User constraints: {", ".join(user_profile["constraints"])}

Market snapshot:
- S&P 500: {spx["close"]:.2f} ({spx_chg:+.2f}%)
- {vix_line}

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

{reddit_block}

{top_stocks_block}
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
    # ✅ Yahoo Finance API: Fetch data for LLM-suggested tickers (included in Pass 2 prompt)
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
    # PROMPT INCLUDES: ✅ SPX/VIX data, ✅ Headlines, ✅ Reddit sentiment, ✅ Top Reddit stocks, ✅ yfinance ticker data
    final_prompt = f"""
You are my personal investing memo writer. Be cautious, avoid certainty, avoid hype.
Use Reddit sentiment only as context (noisy/contrarian), not predictive.
Use the yfinance context below as factual input; do NOT invent metrics not shown.
Pay special attention to the top Reddit stocks - these are trending based on Reddit analysis and may be worth considering.

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

{top_stocks_block}

{yf_block}

Write the memo around 1500 words with:
1) 2 biggest market takeaways (tie to headlines)
2) Risk regime (bullish/neutral/bearish) + 1 sentence justification (reference SPX/VIX + reddit sentiment)
3) Suggested actions (allocation/DCA/rebalancing/cash buffer)
4) Picks: use ONLY the selected tickers below. For each category: 3 tickers + 2 quick bullets each.
Selected tickers:
- Core Stocks: {", ".join(proposal.get("core_stocks", []))}
- ETFs: {", ".join(proposal.get("etfs", []))}
- AI/Software: {", ".join(proposal.get("ai_software", []))}
- Robotics: {", ".join(proposal.get("robotics", []))}
- International: {", ".join(proposal.get("international", []))}
5) Top Reddit stocks analysis: Provide a brief analysis of the top 3 Reddit stocks mentioned above, including why they're trending and whether they're worth considering (be cautious - Reddit sentiment can be contrarian/noisy).
6) One thing to watch that could change the view
"""
    brief = ask_llm(final_prompt)

    # ✅ Gmail SMTP API: Send the generated memo via email
    subject = f"Daily Market Updates and Stock Picks({spx['date']})"
    body = brief + f"\n\n Not Financial Advice -> Stock Bot created by Kuma McCraw leveraging {DEFAULT_MODEL} locally."
    send_email(subject, body)

def should_run_on_startup():
    """
    Check if we should run on startup (missed the scheduled time).
    Returns True if it's a weekday and we're past 9:15 AM but before 4:00 PM.
    """
    now = datetime.now()
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    
    # Only run on weekdays (Monday=0 to Friday=4)
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's after 9:15 AM and before 4:00 PM (market hours)
    current_time = now.time()
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    if market_open <= current_time <= market_close:
        return True
    
    return False

def mark_as_run_today():
    """Mark that we've run today by creating a flag file with today's date."""
    flag_file = os.path.join(os.path.dirname(__file__), ".stockbot_run_today")
    today = datetime.now().strftime("%Y-%m-%d")
    with open(flag_file, "w") as f:
        f.write(today)

def has_run_today():
    """Check if we've already run today."""
    flag_file = os.path.join(os.path.dirname(__file__), ".stockbot_run_today")
    if not os.path.exists(flag_file):
        return False
    
    try:
        with open(flag_file, "r") as f:
            last_run = f.read().strip()
        today = datetime.now().strftime("%Y-%m-%d")
        return last_run == today
    except:
        return False

if __name__ == "__main__":
    # Manual run options for testing:
    #   python Stock_Bot_V1_main_py --run-now   (or -r)  -> run once and exit
    #   python Stock_Bot_V1_main_py --prompt    (or -p)  -> ask interactively whether to run now
    if "--run-now" in sys.argv or "-r" in sys.argv:
        print("Manual run requested (--run-now). Running once and exiting.")
        main()
        mark_as_run_today()
        sys.exit(0)

    if "--prompt" in sys.argv or "-p" in sys.argv:
        try:
            ans = input("Run Stock Bot now? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans == "y":
            print("Running now...")
            main()
            mark_as_run_today()
            sys.exit(0)
        else:
            print("Continuing to scheduler...")

    # Check if we should run on startup (missed scheduled time)
    if "--startup-check" in sys.argv:
        if should_run_on_startup() and not has_run_today():
            logging.info("Startup check: Running bot (missed scheduled time)")
            try:
                main()
                mark_as_run_today()
            except Exception as e:
                logging.error(f"Error running bot on startup: {e}")
        else:
            if has_run_today():
                logging.info("Startup check: Already ran today, skipping")
            else:
                logging.info("Startup check: Not a weekday or outside market hours, skipping")
        sys.exit(0)

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