import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent Tkinter errors

import os
import io
import uuid
import asyncio
import pytz
import json
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import discord
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from collections import defaultdict
from discord import app_commands
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import logging
import time

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("bot_errors.log"),
        logging.StreamHandler()
    ]
)

# Initialize environment variables from .env file
load_dotenv()

# Discord client setup
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Configuration - sensitive values are now loaded from environment variables
bot_token = os.getenv('DISCORD_TOKEN')
channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))  # Set your Discord channel ID in .env
TICKER = 'SPY'
POSITIONS_FILE = 'user_positions.json'
TRACKED_TICKER_FILE = 'tracked_ticker.json'
ALERTS_FILE = 'price_alerts.json'

# Data storage
user_portfolios = defaultdict(list)
user_update_intervals = defaultdict(lambda: 5)
user_last_update = {}
price_alerts = defaultdict(list)

# Define your guild object using your guild ID from .env
GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', '0'))  # Set your Discord guild (server) ID in .env
guild_obj = discord.Object(id=GUILD_ID)

# -----------------------
# Helper: Extract Current Price
# -----------------------
def get_current_price(data):
    try:
        close = data['Close']
        # If "Close" is a DataFrame, extract the first column's last value
        if isinstance(close, pd.DataFrame):
            return float(close.iloc[-1, 0])
        else:
            return float(close.iloc[-1])
    except Exception as e:
        logging.error(f"Error extracting current price: {e}")
        return None

# -----------------------
# Background Tasks
# -----------------------
async def monitor_positions():
    await client.wait_until_ready()
    cst = pytz.timezone('America/Chicago')
    while not client.is_closed():
        try:
            now = datetime.now(cst)
            if now.weekday() >= 5:  # Skip weekends
                await asyncio.sleep(3600)
                continue

            for user_id in list(user_portfolios.keys()):
                interval = user_update_intervals[user_id]
                last_update = user_last_update.get(user_id, datetime.min.replace(tzinfo=cst))
                if (now - last_update).total_seconds() >= interval * 60:
                    try:
                        user = await client.fetch_user(user_id)
                        positions = user_portfolios[user_id]
                        updates = []
                        for pos in positions:
                            if pos.exit_price is not None:
                                continue
                            try:
                                data = await get_stock_data(pos.ticker)
                                if data is not None and not data.empty:
                                    current_price = get_current_price(data)
                                    rec, analysis = await analyze_position(pos, current_price, data)
                                    updates.append(f"**{pos.ticker.upper()} Update**\n{analysis}\nRecommendation: {rec}")
                                else:
                                    updates.append(f"⚠️ {pos.ticker.upper()} - Market data unavailable")
                            except Exception as pos_error:
                                logging.error(f"Position {pos.ticker} analysis error: {pos_error}")
                                updates.append(f"⚠️ {pos.ticker.upper()} - Analysis failed")
                        if updates:
                            async def interval_callback(interaction):
                                await interaction.response.send_modal(IntervalModal())
                            view = discord.ui.View()
                            interval_btn = discord.ui.Button(label="Adjust Interval", style=discord.ButtonStyle.secondary)
                            interval_btn.callback = interval_callback
                            view.add_item(interval_btn)
                            await user.send("\n\n".join(updates), view=view)
                            user_last_update[user_id] = now
                    except discord.Forbidden:
                        logging.error(f"Can't DM user {user_id}")
                    except Exception as e:
                        logging.error(f"User {user_id} update error: {e}")

            for user_id, alerts in price_alerts.items():
                try:
                    user = await client.fetch_user(user_id)
                    for alert in alerts.copy():
                        try:
                            data = await get_stock_data(alert['ticker'])
                            if data is not None and not data.empty:
                                current_price = get_current_price(data)
                                if current_price >= alert['price']:
                                    await user.send(
                                        f"🚨 Price Alert: {alert['ticker']} reached ${alert['price']:.2f}\nCurrent Price: ${current_price:.2f}"
                                    )
                                    price_alerts[user_id].remove(alert)
                                    save_alerts()
                        except Exception as alert_error:
                            logging.error(f"Alert processing error for {alert['ticker']}: {alert_error}")
                except Exception as e:
                    logging.error(f"Alert check error for user {user_id}: {e}")

            await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"Monitor error: {e}")
            await asyncio.sleep(300)

# -----------------------
# Bot Events
# -----------------------
@client.event
async def on_ready():
    try:
        # Clear and re-register commands for your guild
        tree.clear_commands(guild=guild_obj)
        tree.add_command(app_commands.Command(
            name="update",
            description="Force refresh with command buttons",
            callback=update_command
        ))
        tree.add_command(app_commands.Command(
            name="set_ticker",
            description="Change the tracked ticker symbol",
            callback=set_ticker_command
        ))
        tree.add_command(app_commands.Command(
            name="early_advice",
            description="Get early indicator advice for a ticker",
            callback=early_advice_command
        ))
        await tree.sync(guild=guild_obj)

        load_data()
        client.loop.create_task(monitor_positions())

        channel = client.get_channel(channel_id)
        if channel:
            performers = await get_top_performers()
            ticker_info = f"**Ticker:** {TICKER}\nNo data available"
            chart_file = None
            data = await get_stock_data(TICKER)
            if data is not None and not data.empty:
                current_price = get_current_price(data)
                ticker_info = f"**Ticker:** {TICKER}\n**Current Price:** ${current_price:.2f}"
                chart_buffer = generate_chart_with_entry(data, TICKER, current_price)
                if chart_buffer:
                    chart_file = discord.File(chart_buffer, filename="chart.png")

            embed = discord.Embed(title="✅ Bot Online - Market Snapshot", color=0x00ff00)
            if performers:
                embed.add_field(name="🏆 Today's Top Performers", value=performers, inline=False)
            embed.add_field(name="Tracked Ticker", value=ticker_info, inline=True)
            if chart_file:
                embed.set_image(url="attachment://chart.png")

            if chart_file:
                await channel.send(embed=embed, view=CommandView(), file=chart_file)
            else:
                await channel.send(embed=embed, view=CommandView())

        logging.info(f"Successfully logged in as {client.user}")
    except Exception as e:
        logging.error(f"Startup error: {e}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user in message.mentions and "update" in message.content.lower():
        channel = client.get_channel(channel_id)
        if message.channel == channel:
            view = CommandView()
            await send_signal(channel, view)

# -----------------------
# Data Management Functions
# -----------------------
def load_data():
    global user_portfolios, TICKER, price_alerts
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                positions_data = json.load(f)
                for user_id_str, positions in positions_data.items():
                    user_id = int(user_id_str)
                    user_portfolios[user_id] = [
                        UserPosition.from_dict(pos) for pos in positions if isinstance(pos, dict)
                    ]
        except Exception as e:
            logging.error(f"Error loading positions: {e}")
    
    if os.path.exists(TRACKED_TICKER_FILE):
        try:
            with open(TRACKED_TICKER_FILE, 'r') as f:
                TICKER = json.load(f).get('ticker', 'SPY')
        except Exception as e:
            logging.error(f"Error loading ticker: {e}")
    
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                alerts_data = json.load(f)
                for user_id_str, alerts in alerts_data.items():
                    price_alerts[int(user_id_str)] = alerts
        except Exception as e:
            logging.error(f"Error loading alerts: {e}")

def save_positions():
    positions_data = {str(k): [pos.to_dict() for pos in v] for k, v in user_portfolios.items()}
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions_data, f, indent=2, default=str)
    except Exception as e:
        logging.error(f"Error saving positions: {e}")

def save_tracked_ticker():
    try:
        with open(TRACKED_TICKER_FILE, 'w') as f:
            json.dump({'ticker': TICKER}, f)
    except Exception as e:
        logging.error(f"Error saving tracked ticker: {e}")

def save_alerts():
    alerts_data = {str(k): v for k, v in price_alerts.items()}
    try:
        with open(ALERTS_FILE, 'w') as f:
            json.dump(alerts_data, f)
    except Exception as e:
        logging.error(f"Error saving alerts: {e}")

# -----------------------
# Helper Functions
# -----------------------
async def get_top_performers():
    try:
        url = "https://finance.yahoo.com/screener/predefined/day_gainers"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'data-test': 'historical-data-table'})
        if table is None or table.tbody is None:
            return None
        gainers = []
        for row in table.tbody.find_all('tr')[:5]:
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            symbol = cells[0].text.strip()
            name = cells[1].text.strip()
            change = cells[4].text.strip()
            gainers.append(f"**{symbol}** ({name}): {change}")
        return "\n".join(gainers)
    except Exception as e:
        logging.error(f"Performance check error: {e}")
        return None

async def get_stock_data(ticker):
    try:
        data = await asyncio.to_thread(
            yf.download,
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            prepost=False,
            threads=True,
        )
        return validate_ohlc_data(data)
    except Exception as e:
        logging.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

def validate_ohlc_data(df):
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.map(str).str.strip()
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None
    df_clean = df[required].copy()
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        df_clean = df_clean.astype(float)
    except Exception as e:
        logging.error(f"Error casting df_clean to float: {e}")
        return None
    if len(df_clean) < 10:
        return None
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index, errors='coerce')
    df_clean.index = df_clean.index.tz_localize(None)
    return df_clean

def generate_chart_with_entry(data, ticker, entry_price):
    try:
        clean_data = validate_ohlc_data(data)
        if clean_data is None or clean_data.empty:
            return None
        buffer = io.BytesIO()
        mpf.plot(
            clean_data,
            type='candle',
            style='charles',
            title=f'{ticker} Price',
            ylabel='Price',
            volume=True,
            hlines=dict(
                hlines=[entry_price],
                colors=['blue'],
                linestyle=['--'],
                linewidths=[1.5],
                alpha=[0.7]
            ),
            savefig=dict(fname=buffer, format='png', dpi=100)
        )
        buffer.seek(0)
        return buffer
    except Exception as e:
        logging.error(f"Chart Error for {ticker}: {e}")
        return None

# -----------------------
# UserPosition Class
# -----------------------
class UserPosition:
    def __init__(self, entry_price, quantity, entry_time, ticker, exit_price=None, exit_time=None):
        self.id = str(uuid.uuid4())[:8]
        self.entry_price = float(entry_price)
        self.quantity = int(quantity)
        self.entry_time = entry_time.replace(tzinfo=pytz.UTC) if entry_time.tzinfo is None else entry_time
        self.exit_price = float(exit_price) if exit_price is not None else None
        self.exit_time = exit_time.replace(tzinfo=pytz.UTC) if exit_time and exit_time.tzinfo is None else exit_time
        self.ticker = ticker

    def to_dict(self):
        return {
            'id': self.id,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'ticker': self.ticker
        }

    @classmethod
    def from_dict(cls, data):
        entry_time = datetime.fromisoformat(data['entry_time'])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=pytz.UTC)
        exit_time = None
        if data.get('exit_time'):
            exit_time = datetime.fromisoformat(data['exit_time'])
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=pytz.UTC)
        return cls(
            entry_price=data['entry_price'],
            quantity=data['quantity'],
            entry_time=entry_time,
            ticker=data.get('ticker', TICKER),
            exit_price=data.get('exit_price'),
            exit_time=exit_time
        )

# -----------------------
# Analysis Functions
# -----------------------
async def analyze_position(position, current_price, data):
    try:
        if isinstance(data['Close'], pd.DataFrame):
            close_prices = data['Close'].iloc[:, 0]
        else:
            close_prices = data['Close'].squeeze()
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
        
        price_change = ((current_price - position.entry_price) / position.entry_price) * 100

        rsi_indicator = RSIIndicator(close=close_prices, window=14)
        macd_indicator = MACD(close=close_prices)
        sma_indicator = SMAIndicator(close=close_prices, window=20)

        rsi_series = rsi_indicator.rsi().dropna()
        macd_series = macd_indicator.macd().dropna()
        signal_series = macd_indicator.macd_signal().dropna()
        macd_diff_series = macd_indicator.macd_diff().dropna()
        sma_series = sma_indicator.sma_indicator().dropna()

        if len(rsi_series) == 0 or len(macd_series) == 0 or len(signal_series) == 0 or len(macd_diff_series) == 0 or len(sma_series) == 0:
            raise ValueError("Insufficient data for technical indicators")

        rsi_value = float(rsi_series.values[-1])
        macd_value = float(macd_series.values[-1])
        signal_value = float(signal_series.values[-1])
        macd_hist = float(macd_diff_series.values[-1])
        sma_value = float(sma_series.values[-1])
        
        recommendation = "Hold"
        reasons = []

        if price_change >= 20:
            recommendation = "Sell"
            reasons.append(f"Substantial profit ({price_change:.2f}%)")
        elif price_change <= -10:
            recommendation = "Sell"
            reasons.append(f"Significant loss ({price_change:.2f}%)")
        else:
            tech_reasons = []
            if rsi_value > 70:
                tech_reasons.append("RSI overbought")
            elif rsi_value < 30:
                tech_reasons.append("RSI oversold")
            else:
                tech_reasons.append("RSI neutral")

            if current_price > sma_value:
                tech_reasons.append("Price above SMA20")
            else:
                tech_reasons.append("Price below SMA20")

            if macd_value > signal_value:
                tech_reasons.append("MACD bullish")
            else:
                tech_reasons.append("MACD bearish")

            strong_sell_signals = sum(1 for r in tech_reasons if "overbought" in r or "bearish" in r)
            strong_buy_signals = sum(1 for r in tech_reasons if "oversold" in r or "bullish" in r)
            
            if strong_sell_signals >= 2:
                recommendation = "Sell"
                reasons.extend(tech_reasons)
            elif strong_buy_signals >= 2:
                recommendation = "Buy" 
                reasons.extend(tech_reasons)
            else:
                reasons.append("Hold: Balanced technical signals")
                reasons.append(f"Moderate price movement ({price_change:.2f}%)")
                reasons.extend(tech_reasons)

        analysis_details = [
            f"Entry Price: ${position.entry_price:.2f}",
            f"Current Price: ${current_price:.2f}",
            f"Price Change: {price_change:.2f}%",
            "Reasons: " + "; ".join(reasons)
        ]
        
        return recommendation, "\n".join(analysis_details)
    except Exception as e:
        logging.error(f"Error analyzing position for {position.ticker}: {e}")
        return "Hold", f"Analysis Error: {str(e)}"

async def analyze_ticker(ticker, data):
    try:
        if isinstance(data['Close'], pd.DataFrame):
            close_prices = data['Close'].iloc[:, 0]
        else:
            close_prices = data['Close'].squeeze()
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
        
        current_price = get_current_price(data)
        
        rsi_indicator = RSIIndicator(close=close_prices, window=14)
        macd_indicator = MACD(close=close_prices)
        sma_indicator = SMAIndicator(close=close_prices, window=20)
        
        rsi_value = float(rsi_indicator.rsi().iloc[-1])
        macd_value = float(macd_indicator.macd().iloc[-1])
        signal_value = float(macd_indicator.macd_signal().iloc[-1])
        macd_hist = float(macd_indicator.macd_diff().iloc[-1])
        sma_value = float(sma_indicator.sma_indicator().iloc[-1])
        
        analysis_details = [
            f"Current Price: ${current_price:.2f}",
            f"RSI (14): {rsi_value:.2f} " + ("(Overbought)" if rsi_value > 70 else ("(Oversold)" if rsi_value < 30 else "(Neutral)")),
            f"MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, Hist: {macd_hist:.2f}",
            f"20-day SMA: {sma_value:.2f}"
        ]
        
        recommendation = "Hold"
        reasons = []
        if rsi_value > 70:
            recommendation = "Sell"
            reasons.append("RSI indicates overbought")
        elif rsi_value < 30:
            recommendation = "Buy"
            reasons.append("RSI indicates oversold")
        
        if current_price > sma_value:
            reasons.append("Price is above the 20-day SMA (uptrend)")
        else:
            reasons.append("Price is below the 20-day SMA (downtrend)")
        
        if macd_value > signal_value:
            reasons.append("MACD is bullish")
        else:
            reasons.append("MACD is bearish")
        
        analysis_details.append("Reasons: " + "; ".join(reasons))
        return recommendation, "\n".join(analysis_details)
    except Exception as e:
        logging.error(f"Ticker analysis error for {ticker}: {e}")
        return "Hold", "Technical analysis unavailable"

# -----------------------
# Slash Command Handlers
# -----------------------
async def update_command(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        view = CommandView()
        data = await get_stock_data(TICKER)
        if data is None or data.empty:
            await interaction.followup.send(f"⚠️ No data for {TICKER}", view=view)
            return

        current_price = get_current_price(data)
        chart_buffer = generate_chart_with_entry(data, TICKER, current_price)
        message_content = f"**{TICKER} Update**\nCurrent Price: ${current_price:.2f}"
        
        if chart_buffer:
            await interaction.followup.send(content=message_content, file=discord.File(chart_buffer, filename="chart.png"), view=view)
        else:
            await interaction.followup.send(message_content, view=view)
    except Exception as e:
        logging.error(f"Update command error: {e}")
        await interaction.followup.send(f"⚠️ Update error: {e}")

async def set_ticker_command(interaction: discord.Interaction, ticker: str):
    global TICKER
    if len(ticker) > 10 or not ticker.isalnum():
        await interaction.response.send_message("❌ Invalid ticker symbol. Use 1-10 alphanumeric characters.", ephemeral=True)
        return
    try:
        data = await get_stock_data(ticker)
        if data is None or data.empty:
            await interaction.response.send_message(f"❌ No market data found for {ticker}", ephemeral=True)
            return
        TICKER = ticker.upper()
        save_tracked_ticker()
        await interaction.response.send_message(f"✅ Successfully changed tracked ticker to {TICKER}", ephemeral=True)
    except Exception as e:
        logging.error(f"Set ticker error for {ticker}: {e}")
        await interaction.response.send_message(f"❌ Failed to set ticker: {e}", ephemeral=True)

async def early_advice_command(interaction: discord.Interaction, ticker: str):
    try:
        ticker_input = ticker.upper().strip()
        data = await get_stock_data(ticker_input)
        if data is None or data.empty:
            await interaction.response.send_message(f"⚠️ No data available for {ticker_input}", ephemeral=True)
            return

        recommendation, analysis_details = await analyze_ticker(ticker_input, data)
        current_price = get_current_price(data)
        chart_buffer = generate_chart_with_entry(data, ticker_input, current_price)

        # Calculate indicators for entry suggestions
        close_series = pd.Series(data['Close']).squeeze()
        sma_20 = float(SMAIndicator(close=close_series, window=20).sma_indicator().iloc[-1])
        rsi_val = float(RSIIndicator(close=close_series, window=14).rsi().iloc[-1])
        entry_suggestions = []
        if recommendation == "Buy":
            entry_suggestions.append(f"• Consider entries below 20-day SMA (${sma_20:.2f})")
            if rsi_val < 40:
                entry_suggestions.append("• Look for RSI crosses above 30")
        else:
            entry_suggestions.append("• Wait for clearer market signals")

        response_text = (
            f"**Advice for {ticker_input}**\n"
            f"{analysis_details}\n"
            f"Recommendation: {recommendation}\n\n"
            f"Entry Suggestions:\n" + "\n".join(entry_suggestions)
        )

        if chart_buffer:
            await interaction.response.send_message(response_text, file=discord.File(chart_buffer, filename=f"{ticker_input}_chart.png"), ephemeral=True)
        else:
            await interaction.response.send_message(response_text, ephemeral=True)
    except Exception as e:
        logging.error(f"Early advice command error for {ticker}: {e}")
        await interaction.response.send_message(f"Error analyzing {ticker}: {e}", ephemeral=True)


# -----------------------
# Discord Components (Modals, Views, etc.)
# -----------------------
class IntervalModal(discord.ui.Modal, title="Adjust Update Interval"):
    minutes = discord.ui.TextInput(
        label="Update Frequency (Minutes)",
        placeholder="Enter 5-240",
        min_length=1,
        max_length=3
    )
    async def on_submit(self, interaction: discord.Interaction):
        try:
            interval = int(self.minutes.value)
            if 5 <= interval <= 240:
                user_update_intervals[interaction.user.id] = interval
                await interaction.response.send_message(f"✅ Update interval set to {interval} minutes", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Interval must be between 5-240 minutes", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("❌ Please enter a valid number (e.g. '30')", ephemeral=True)

class AlertModal(discord.ui.Modal, title="Set Price Alert"):
    def __init__(self, ticker: str):
        super().__init__()
        self.ticker = ticker
        self.price = discord.ui.TextInput(label="Alert Price", placeholder="Enter target price")
        self.add_item(self.price)
    async def on_submit(self, interaction: discord.Interaction):
        try:
            alert_price = float(self.price.value)
            user_id = interaction.user.id
            price_alerts[user_id].append({
                "ticker": self.ticker,
                "price": alert_price,
                "timestamp": datetime.now().isoformat()
            })
            save_alerts()
            await interaction.response.send_message(f"✅ Alert set for {self.ticker} at ${alert_price:.2f}", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("❌ Invalid price format", ephemeral=True)

class AdviceView(discord.ui.View):
    def __init__(self, ticker: str):
        super().__init__(timeout=None)
        self.ticker = ticker
        alert_button = discord.ui.Button(label="Set Price Alert", style=discord.ButtonStyle.primary)
        alert_button.callback = self.alert_callback
        self.add_item(alert_button)

    async def alert_callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(AlertModal(self.ticker))

class AdviceModal(discord.ui.Modal, title="Check Indicator Advice"):
    ticker = discord.ui.TextInput(label="Ticker Symbol", placeholder="AAPL", max_length=10)
    async def on_submit(self, interaction: discord.Interaction):
        try:
            ticker_input = self.ticker.value.upper().strip()
            data = await get_stock_data(ticker_input)
            if data is None or data.empty:
                await interaction.response.send_message(f"⚠️ No data available for {ticker_input}", ephemeral=True)
                return

            recommendation, analysis_details = await analyze_ticker(ticker_input, data)
            current_price = get_current_price(data)
            chart_buffer = generate_chart_with_entry(data, ticker_input, current_price)

            sma_20 = float(SMAIndicator(pd.Series(data['Close']).squeeze(), window=20).sma_indicator().iloc[-1])
            rsi_val = float(RSIIndicator(pd.Series(data['Close']).squeeze(), window=14).rsi().iloc[-1])
            entry_suggestions = []
            if recommendation == "Buy":
                entry_suggestions.append(f"• Consider entries below 20-day SMA (${sma_20:.2f})")
                if rsi_val < 40:
                    entry_suggestions.append("• Look for RSI crosses above 30")
            else:
                entry_suggestions.append("• Wait for clearer market signals")
            
            response_text = (
                f"**Advice for {ticker_input}**\n"
                f"{analysis_details}\n"
                f"Recommendation: {recommendation}\n\n"
                f"Entry Suggestions:\n" + "\n".join(entry_suggestions)
            )

            view = AdviceView(ticker_input)
            if chart_buffer:
                await interaction.response.send_message(response_text, file=discord.File(chart_buffer, filename=f"{ticker_input}_chart.png"), view=view, ephemeral=True)
            else:
                await interaction.response.send_message(response_text, view=view, ephemeral=True)
        except Exception as e:
            logging.error(f"Advice modal error for {self.ticker.value}: {e}")
            await interaction.response.send_message(f"Error analyzing {self.ticker.value}: {e}", ephemeral=True)

class CommandView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Check Position", style=discord.ButtonStyle.primary)
    async def check_position_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_position_check(interaction)

    @discord.ui.button(label="View Portfolio", style=discord.ButtonStyle.primary)
    async def view_portfolio_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_portfolio_view(interaction)

    @discord.ui.button(label="Add Position", style=discord.ButtonStyle.green)
    async def add_position_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AddPositionModal())

    @discord.ui.button(label="Close Position", style=discord.ButtonStyle.red)
    async def close_position_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_position_close(interaction)

    @discord.ui.button(label="Get Ticker Advice", style=discord.ButtonStyle.secondary)
    async def check_advice_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AdviceModal())

    @discord.ui.button(label="Help", style=discord.ButtonStyle.secondary)
    async def help_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_help(interaction)

    @discord.ui.button(label="Adjust Interval", style=discord.ButtonStyle.secondary)
    async def interval_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(IntervalModal())

    @discord.ui.button(label="Update", style=discord.ButtonStyle.primary)
    async def update_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        try:
            view = CommandView()
            data = await get_stock_data(TICKER)
            if data is None or data.empty:
                await interaction.followup.send(f"⚠️ No data for {TICKER}", view=view)
                return
            current_price = get_current_price(data)
            chart_buffer = generate_chart_with_entry(data, TICKER, current_price)
            message_content = f"**{TICKER} Update**\nCurrent Price: ${current_price:.2f}"
            if chart_buffer:
                await interaction.followup.send(content=message_content, file=discord.File(chart_buffer, filename="chart.png"), view=view)
            else:
                await interaction.followup.send(message_content, view=view)
        except Exception as e:
            logging.error(f"Update button error: {e}")
            await interaction.followup.send(f"⚠️ Update error: {e}")

    async def handle_position_check(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            user_id = interaction.user.id
            positions = user_portfolios.get(user_id, [])
            if not positions:
                await interaction.followup.send("You don't have any active positions.", ephemeral=True)
                return
            messages = []
            files = []
            for idx, pos in enumerate(positions, 1):
                try:
                    if pos.exit_price is not None:
                        continue
                    data = await get_stock_data(pos.ticker)
                    if data is None or data.empty:
                        messages.append(f"⚠️ {pos.ticker} - Market data unavailable")
                        continue
                    current_price = get_current_price(data)
                    rec, analysis_details = await analyze_position(pos, current_price, data)
                    chart_buffer = generate_chart_with_entry(data, pos.ticker, pos.entry_price)
                    if chart_buffer:
                        files.append(discord.File(chart_buffer, filename=f"{pos.ticker}_chart.png"))
                    position_value = current_price * pos.quantity
                    pnl = position_value - (pos.entry_price * pos.quantity)
                    pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100
                    summary = (
                        f"**Position #{idx} - {pos.ticker.upper()}**\n"
                        f"• Shares: {pos.quantity}\n"
                        f"• Entry: ${pos.entry_price:.2f}\n"
                        f"• Current: ${current_price:.2f}\n"
                        f"• Value: ${position_value:.2f}\n"
                        f"• P/L: ${pnl:.2f} ({pnl_pct:.2f}%)\n"
                        f"{analysis_details}\nRecommendation: {rec}"
                    )
                    messages.append(summary)
                except Exception as pos_error:
                    logging.error(f"Error checking position for {pos.ticker}: {pos_error}")
                    messages.append(f"⚠️ {pos.ticker} analysis failed: {str(pos_error)}")
                    continue

            if messages:
                content = "\n\n".join(messages)
                if len(content) <= 2000:
                    await interaction.followup.send(content, files=files, ephemeral=True)
                else:
                    chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                    await interaction.followup.send(chunks[0], files=files, ephemeral=True)
                    for chunk in chunks[1:]:
                        await interaction.followup.send(chunk, ephemeral=True)
            else:
                await interaction.followup.send("No analyzable positions found.", ephemeral=True)
        except Exception as e:
            logging.error(f"Critical position check error: {e}")
            await interaction.followup.send("❌ Failed to check positions due to system error", ephemeral=True)

    async def handle_portfolio_view(self, interaction: discord.Interaction):
        try:
            user_id = interaction.user.id
            positions = user_portfolios.get(user_id, [])
            total_value = 0.0
            position_details = []
            files = []
            for pos in positions:
                data = await get_stock_data(pos.ticker)
                if data is not None and not data.empty:
                    price = get_current_price(data)
                    total_value += price * pos.quantity
                    position_details.append(f"- {pos.ticker}: {pos.quantity} shares @ ${pos.entry_price:.2f} (Current: ${price:.2f})")
                    chart_buffer = generate_chart_with_entry(data, pos.ticker, pos.entry_price)
                    if chart_buffer:
                        files.append(discord.File(chart_buffer, filename=f"{pos.ticker}_chart.png"))
            response = (
                f"**Portfolio Summary**\n"
                f"Total Value: ${total_value:.2f}\n"
                f"Positions ({len(positions)}):\n" +
                ("\n".join(position_details) if position_details else "No positions")
            )
            if files:
                await interaction.response.send_message(response, files=files, ephemeral=True)
            else:
                await interaction.response.send_message(response, ephemeral=True)
        except Exception as e:
            logging.error(f"Portfolio view error: {e}")
            await interaction.response.send_message("❌ Error viewing portfolio", ephemeral=True)

    async def handle_position_close(self, interaction: discord.Interaction):
        user_id = interaction.user.id
        if user_id in user_portfolios:
            user_portfolios[user_id] = []
            save_positions()
            await interaction.response.send_message("✅ Closed all positions", ephemeral=True)
        else:
            await interaction.response.send_message("ℹ️ No positions to close", ephemeral=True)

    async def handle_help(self, interaction: discord.Interaction):
        help_text = (
            "**Stock Bot Commands**\n"
            "/help - Show this menu\n"
            "/check_position - View all positions\n"
            "/set_ticker - Change tracked stock\n"
            "/close_position - Close positions\n"
            "/view_portfolio - Portfolio summary\n"
            "/update - Force refresh\n"
            "/early_advice - Get early ticker advice\n"
            "Use buttons below to:\n"
            "- Add/check positions\n"
            "- Get technical analysis\n"
            "- Set price alerts\n"
            "- Update chart for current ticker"
        )
        await interaction.response.send_message(help_text, ephemeral=True)

class AddPositionModal(discord.ui.Modal, title="Add New Position"):
    ticker = discord.ui.TextInput(label="Ticker Symbol", placeholder="AAPL", max_length=10)
    price = discord.ui.TextInput(label="Entry Price", placeholder="150.50")
    quantity = discord.ui.TextInput(label="Quantity", placeholder="100")
    async def on_submit(self, interaction: discord.Interaction):
        try:
            new_position = UserPosition(
                entry_price=float(self.price.value),
                quantity=int(self.quantity.value),
                entry_time=datetime.now(),
                ticker=self.ticker.value.upper().strip()
            )
            user_portfolios[interaction.user.id].append(new_position)
            save_positions()
            await interaction.response.send_message(
                f"✅ Added {new_position.quantity} shares of {new_position.ticker} @ ${new_position.entry_price:.2f}",
                ephemeral=True
            )
        except ValueError:
            await interaction.response.send_message("❌ Invalid input values", ephemeral=True)

async def send_signal(channel, view=None):
    try:
        data = await get_stock_data(TICKER)
        if data is None or data.empty:
            return
        current_price = get_current_price(data)
        chart = generate_chart_with_entry(data, TICKER, current_price)
        content = f"**{TICKER} Update**\nCurrent Price: ${current_price:.2f}"
        if chart:
            await channel.send(content, file=discord.File(chart, filename="chart.png"), view=view)
        else:
            await channel.send(content, view=view)
    except Exception as e:
        logging.error(f"Signal error for {TICKER}: {e}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    while True:
        try:
            client.run(bot_token)
        except Exception as e:
            logging.error(f"Fatal error: {e}")
            save_positions()
            save_tracked_ticker()
            save_alerts()
            logging.info("Restarting bot in 10 seconds...")
            time.sleep(10)
        else:
            break
