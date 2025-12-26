# Stock Bot - Run Instructions

## Quick Start

### 1. Install Dependencies

Run this command in your terminal:

```bash
cd /Users/kuma/Desktop/Python/Stock_Bot
pip3 install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):

```bash
cd /Users/kuma/Desktop/Python/Stock_Bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create `.env` File

Create a `.env` file in the project directory with the following required variables:

```bash
# Required for email functionality
GMAIL_USER=your-email@gmail.com
GMAIL_APP_PASSWORD=your-app-password
TO_EMAIL=recipient@example.com

# Optional: For multiple recipients
CC_EMAIL=cc1@example.com,cc2@example.com
BCC_EMAIL=bcc@example.com

# Optional: For Reddit API (falls back to public API if not provided)
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-secret
REDDIT_USER_AGENT=stock-bot/0.1 by /u/yourusername
```

**Note:** To get a Gmail App Password:
1. Go to your Google Account settings
2. Enable 2-Step Verification
3. Generate an App Password for "Mail"
4. Use that password (not your regular Gmail password)

### 3. Ensure Ollama is Running

The script uses Ollama (local LLM) which must be running on `http://localhost:11434`.

If you haven't installed Ollama:
- Download from: https://ollama.ai
- Install and start the service
- Pull the required model: `ollama pull llama3.2:3b`

Check if Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 4. Run the Script

**For testing (run once):**
```bash
python3 Stock_Bot_V1_main.py --run-now
```

Or with the short flag:
```bash
python3 Stock_Bot_V1_main.py -r
```

**Interactive prompt:**
```bash
python3 Stock_Bot_V1_main.py --prompt
```

**Run with scheduler (runs weekdays at 9:15 AM):**
```bash
python3 Stock_Bot_V1_main.py
```

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed (`pip3 list` to verify)
- **Ollama connection errors**: Ensure Ollama is running (`ollama serve` if needed)
- **Email errors**: Verify Gmail credentials and App Password in `.env`
- **Reddit API errors**: The script will try public API first, then fall back to PRAW if credentials are provided

## File Structure

- `Stock_Bot_V1_main.py` - Main script
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create this file)
- `.gitignore` - Git ignore file (`.env` is already ignored)

