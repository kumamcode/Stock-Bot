# Stock Bot Automatic Scheduler Setup

## ✅ Setup Complete!

Your stock bot is now configured to run automatically every weekday at 9:15 AM (before market open at 9:30 AM ET).

## How It Works

The bot uses macOS `launchd` to run automatically. It will:
- Run every Monday-Friday at 9:15 AM
- Run even if you're not logged in (if your Mac is on)
- Log output to `stockbot.log` and errors to `stockbot.error.log`

## Managing the Service

### Check if service is loaded:
```bash
launchctl list | grep stockbot
```

### Unload the service (to stop automatic runs):
```bash
launchctl unload ~/Library/LaunchAgents/com.stockbot.plist
```

### Reload the service (after making changes):
```bash
launchctl unload ~/Library/LaunchAgents/com.stockbot.plist
launchctl load ~/Library/LaunchAgents/com.stockbot.plist
```

### Check service status:
```bash
launchctl list com.stockbot
```

### View logs:
```bash
# View output log
tail -f /Users/kuma/Desktop/Python/Stock_Bot/stockbot.log

# View error log
tail -f /Users/kuma/Desktop/Python/Stock_Bot/stockbot.error.log
```

## Important Notes

1. **Time Zone**: The service uses your Mac's system time zone. Make sure it's set correctly.

2. **.env File**: The script will automatically load your `.env` file from the project directory.

3. **Internet Required**: The bot needs internet access to:
   - Fetch market data
   - Call Groq API
   - Send emails

4. **Mac Must Be On**: For the bot to run automatically, your Mac needs to be powered on (can be sleeping, but must wake up).

## Testing

To test if it's working, you can manually trigger a run:
```bash
python3 /Users/kuma/Desktop/Python/Stock_Bot/Stock_Bot_V1_main.py --run-now
```

## Troubleshooting

If the bot doesn't run:
1. Check the error log: `cat stockbot.error.log`
2. Check the output log: `cat stockbot.log`
3. Verify the service is loaded: `launchctl list | grep stockbot`
4. Make sure your `.env` file has all required variables (GROQ_API_KEY, GMAIL_USER, etc.)

## Changing the Schedule

To change when it runs, edit the plist file:
```bash
nano ~/Library/LaunchAgents/com.stockbot.plist
```

Then reload:
```bash
launchctl unload ~/Library/LaunchAgents/com.stockbot.plist
launchctl load ~/Library/LaunchAgents/com.stockbot.plist
```

