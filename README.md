# TradeSentry Bot

**TradeSentry Bot** is a Discord bot that provides real-time stock/some mainstream crypto market signals, portfolio tracking, and technical analysis. It uses [yfinance](https://pypi.org/project/yfinance/) for market data, [mplfinance](https://github.com/matplotlib/mplfinance) for charting, and the Discord API for interactive commands.

## Features

- **Real-Time Market Data:** Fetches stock data and displays current prices.
- **Technical Analysis:** Uses RSI, MACD, and SMA indicators to generate trading advice.
- **Portfolio Management:** Add, view, and close positions.
- **Interactive Commands:** Slash commands and buttons for a user-friendly experience.
- **Price Alerts:** Set alerts to get notified when a stock reaches your target price.

## Prerequisites

- **Python 3.8+**
- A Discord Bot Token (get one from the [Discord Developer Portal](https://discord.com/developers/applications))
- Your Discord Channel ID and Guild (Server) ID

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/TradeSentry-Bot.git
   cd TradeSentry-Bot


2. **Create a virtual environment:**

   python -m venv venv
   source venv/bin/activate

   On Windows: venv\Scripts\activate


4. **Install Dependencies:**

   pip install -r Requirements.txt


5. **Set up environment variables:**

   cp .env.example .env

Open .env and fill in your DISCORD_TOKEN, DISCORD_CHANNEL_ID, and DISCORD_GUILD_ID.


**Running the Bot:**

Start the bot with:
python stock_signal_bot.py

## Inviting the Bot to Your Server

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and select your application.
2. Navigate to the **OAuth2 > URL Generator** section.
3. Under **Scopes**, check both:
   - **bot**
   - **applications.commands**

4. Under **Bot Permissions**, select the following:
   - **Send Messages**
   - **Embed Links**
   - **Attach Files**
   - **Read Message History**

   These permissions ensure the bot can send messages (including embeds and attachments) and view message history in channels where it's active.

5. A URL will be generated at the bottom of the page. Copy this URL and open it in your browser to invite the bot to your server.


**Usage:**
Once the bot is running and added to your server, you can use the following commands, or buttons provided:
   /update - Force refresh market data (May be needed if Check Position button not working)

   /set_ticker - Change the default tracked ticker signal bot displays when refreshing in your Discord Channel. 

   /early_advice - Get early entry techincal advice for any stock or crypto. 


You can also interact with the bot using the on-screen buttons displayed in the discord channel you add bot to. You can add positions or set price alerts with /early_advice command. 




