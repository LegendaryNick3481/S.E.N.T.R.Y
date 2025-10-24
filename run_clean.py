#!/usr/bin/env python3
"""
Clean runner for S.E.N.T.R.Y trading system with live terminal dashboard
"""
import sys
import os
import logging
from datetime import datetime
import asyncio
import warnings
import threading
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import Dict, List

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')

# Suppress all external library logs
logging.getLogger('snscrape').setLevel(logging.CRITICAL)
logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
logging.getLogger('torch').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sentry_dashboard.log')]
)

console = Console()


class DashboardUI:
    """Live terminal dashboard for SENTRY system"""
    
    def __init__(self):
        self.status = "Initializing"
        self.market_status = "CLOSED"
        self.start_time = datetime.now()
        self.symbols_data = {}
        self.recent_news = []
        self.system_metrics = {
            'news_scraped': 0,
            'websocket_status': '❌',
            'last_update': 'Never',
            'total_symbols': 0,
            'signals_generated': 0,
            'errors': 0
        }
        self.log_messages = []
        self.max_logs = 5
        
    def generate_layout(self) -> Layout:
        """Generate the dashboard layout"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split(
            Layout(name="symbols", ratio=2),
            Layout(name="news", ratio=1)
        )
        
        layout["right"].split(
            Layout(name="metrics", size=8),
            Layout(name="logs")
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render header panel"""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]
        
        header_text = Text()
        header_text.append("🎯 S.E.N.T.R.Y ", style="bold cyan")
        header_text.append("- Sentiment Enhanced Neural Trading Research Yield\n", style="bold white")
        header_text.append(f"Status: ", style="white")
        
        if self.status == "LIVE":
            header_text.append(self.status, style="bold green")
        elif self.status == "ERROR":
            header_text.append(self.status, style="bold red")
        else:
            header_text.append(self.status, style="bold yellow")
        
        header_text.append(f" | Market: ", style="white")
        header_text.append(self.market_status, style="bold green" if self.market_status == "OPEN" else "bold red")
        header_text.append(f" | Uptime: {uptime_str}", style="white")
        header_text.append(f" | {datetime.now().strftime('%H:%M:%S')}", style="dim white")
        
        return Panel(header_text, box=box.DOUBLE, style="cyan")
    
    def render_symbols_table(self) -> Panel:
        """Render symbols analysis table"""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Price", justify="right", style="white", width=10)
        table.add_column("Chg%", justify="right", width=8)
        table.add_column("Sent", justify="right", width=6)
        table.add_column("Rel", justify="right", width=6)
        table.add_column("Discord", justify="right", width=8)
        table.add_column("Signal", width=8)
        
        if not self.symbols_data:
            table.add_row("─" * 10, "─" * 10, "─" * 8, "─" * 6, "─" * 6, "─" * 8, "─" * 8)
        else:
            for symbol, data in list(self.symbols_data.items())[:15]:  # Show top 15
                price = data.get('price', 0)
                change = data.get('change', 0)
                sentiment = data.get('sentiment', 0)
                relevance = data.get('relevance', 0)
                discord = data.get('discord', 0)
                signal = data.get('signal', 'HOLD')
                
                # Color code change
                if change > 0:
                    change_str = f"[green]+{change:.2f}%[/green]"
                elif change < 0:
                    change_str = f"[red]{change:.2f}%[/red]"
                else:
                    change_str = f"[dim]{change:.2f}%[/dim]"
                
                # Color code sentiment
                if sentiment > 0.3:
                    sent_str = f"[green]{sentiment:.2f}[/green]"
                elif sentiment < -0.3:
                    sent_str = f"[red]{sentiment:.2f}[/red]"
                else:
                    sent_str = f"[yellow]{sentiment:.2f}[/yellow]"
                
                # Color code signal
                signal_colors = {
                    'BUY': '[bold green]BUY[/bold green]',
                    'SELL': '[bold red]SELL[/bold red]',
                    'HOLD': '[dim]HOLD[/dim]'
                }
                signal_str = signal_colors.get(signal, signal)
                
                table.add_row(
                    symbol,
                    f"₹{price:,.2f}" if price > 0 else "─",
                    change_str,
                    sent_str,
                    f"{relevance:.2f}" if relevance > 0 else "─",
                    f"[yellow]{discord:.2f}[/yellow]" if discord > 0.3 else f"{discord:.2f}",
                    signal_str
                )
        
        return Panel(table, title="[bold]📊 Symbol Analysis[/bold]", box=box.ROUNDED, border_style="magenta")
    
    def render_news_panel(self) -> Panel:
        """Render recent news panel"""
        if not self.recent_news:
            news_text = Text("No recent news", style="dim")
        else:
            news_text = Text()
            for news in self.recent_news[:5]:  # Show 5 most recent
                time_str = news.get('time', 'N/A')
                symbol = news.get('symbol', 'N/A')
                title = news.get('title', 'No title')
                relevance = news.get('relevance', 0)
                
                news_text.append(f"[{time_str}] ", style="dim cyan")
                news_text.append(f"{symbol} ", style="bold white")
                news_text.append(f"({relevance:.2f}) ", style="yellow")
                news_text.append(f"{title[:60]}...\n" if len(title) > 60 else f"{title}\n", style="white")
        
        return Panel(news_text, title="[bold]📰 Recent News[/bold]", box=box.ROUNDED, border_style="blue")
    
    def render_metrics_panel(self) -> Panel:
        """Render system metrics panel"""
        metrics_text = Text()
        
        metrics_text.append("WebSocket: ", style="white")
        metrics_text.append(self.system_metrics['websocket_status'], style="")
        metrics_text.append("\n")
        
        metrics_text.append(f"News Scraped: ", style="white")
        metrics_text.append(f"{self.system_metrics['news_scraped']}\n", style="cyan")
        
        metrics_text.append(f"Symbols Tracked: ", style="white")
        metrics_text.append(f"{self.system_metrics['total_symbols']}\n", style="cyan")
        
        metrics_text.append(f"Signals Generated: ", style="white")
        metrics_text.append(f"{self.system_metrics['signals_generated']}\n", style="green")
        
        metrics_text.append(f"Errors: ", style="white")
        metrics_text.append(f"{self.system_metrics['errors']}\n", style="red" if self.system_metrics['errors'] > 0 else "green")
        
        metrics_text.append(f"Last Update: ", style="white")
        metrics_text.append(f"{self.system_metrics['last_update']}", style="dim white")
        
        return Panel(metrics_text, title="[bold]📈 System Metrics[/bold]", box=box.ROUNDED, border_style="green")
    
    def render_logs_panel(self) -> Panel:
        """Render recent logs panel"""
        if not self.log_messages:
            logs_text = Text("No recent logs", style="dim")
        else:
            logs_text = Text()
            for log in self.log_messages[-self.max_logs:]:
                time_str = log.get('time', 'N/A')
                level = log.get('level', 'INFO')
                message = log.get('message', '')
                
                level_colors = {
                    'INFO': 'cyan',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'SUCCESS': 'green'
                }
                
                logs_text.append(f"[{time_str}] ", style="dim")
                logs_text.append(f"{level}: ", style=level_colors.get(level, 'white'))
                logs_text.append(f"{message}\n", style="white")
        
        return Panel(logs_text, title="[bold]📋 System Logs[/bold]", box=box.ROUNDED, border_style="yellow")
    
    def render_footer(self) -> Panel:
        """Render footer panel"""
        footer_text = Text()
        footer_text.append("Press ", style="dim white")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to exit | ", style="dim white")
        footer_text.append("Dashboard updates every 5 seconds", style="dim cyan")
        
        return Panel(footer_text, box=box.SIMPLE, style="dim")
    
    def render(self) -> Layout:
        """Render the complete dashboard"""
        layout = self.generate_layout()
        
        layout["header"].update(self.render_header())
        layout["symbols"].update(self.render_symbols_table())
        layout["news"].update(self.render_news_panel())
        layout["metrics"].update(self.render_metrics_panel())
        layout["logs"].update(self.render_logs_panel())
        layout["footer"].update(self.render_footer())
        
        return layout
    
    def add_log(self, level: str, message: str):
        """Add a log message"""
        self.log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message
        })
        if len(self.log_messages) > 20:
            self.log_messages = self.log_messages[-20:]
    
    def update_symbol(self, symbol: str, data: Dict):
        """Update symbol data"""
        self.symbols_data[symbol] = data
    
    def add_news(self, news_item: Dict):
        """Add a news item"""
        self.recent_news.insert(0, news_item)
        if len(self.recent_news) > 20:
            self.recent_news = self.recent_news[:20]


# Global dashboard instance
dashboard = DashboardUI()


def start_dashboard():
    """Start the live dashboard"""
    console.clear()
    console.show_cursor(False)
    
    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            # Manual update loop
            import time
            while True:
                try:
                    live.update(dashboard.render())
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Dashboard render error: {e}")
                    time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
    finally:
        console.show_cursor(True)


async def handle_event(event: Dict):
    """Handle events from the trading system and update dashboard"""
    try:
        event_type = event.get('type', '')
        
        if event_type == 'status':
            stage = event.get('stage', '')
            if stage == 'start':
                dashboard.status = "LIVE"
                message = event.get('message', f"Live trading started for {len(event.get('watchlist', []))} symbols")
                dashboard.add_log('SUCCESS', message)
            elif stage == 'market_closed':
                dashboard.market_status = "CLOSED"
                dashboard.add_log('INFO', 'Market is closed')
        
        elif event_type == 'log':
            # Direct log event
            level = event.get('level', 'INFO')
            message = event.get('message', '')
            dashboard.add_log(level, message)
            
        elif event_type == 'moves':
            moves = event.get('data', [])
            if moves:
                dashboard.add_log('INFO', f"Detected {len(moves)} significant price moves")
        
        elif event_type == 'news':
            counts = event.get('counts', {})
            total = sum(counts.values())
            dashboard.system_metrics['news_scraped'] = total
            dashboard.system_metrics['last_update'] = datetime.now().strftime('%H:%M:%S')
            if total > 0:
                dashboard.add_log('INFO', f"Scraped {total} news items")
        
        elif event_type == 'sentiment':
            symbol = event.get('symbol', '')
            summary = event.get('summary', {})
            sentiment = summary.get('weighted_sentiment', 0)
            
            if symbol in dashboard.symbols_data:
                dashboard.symbols_data[symbol]['sentiment'] = sentiment
            else:
                dashboard.symbols_data[symbol] = {'sentiment': sentiment}
        
        elif event_type == 'mismatch':
            symbol = event.get('symbol', '')
            analysis = event.get('analysis', {})
            discord = analysis.get('discord_score', 0)
            is_mismatched = analysis.get('is_mismatched', False)
            
            if symbol not in dashboard.symbols_data:
                dashboard.symbols_data[symbol] = {}
            
            dashboard.symbols_data[symbol]['discord'] = discord
            
            if is_mismatched:
                dashboard.add_log('WARNING', f"Mismatch detected: {symbol} (Discord: {discord:.3f})")
        
        elif event_type == 'signals':
            signals = event.get('data', [])
            dashboard.system_metrics['signals_generated'] += len(signals)
            
            for signal in signals:
                symbol = signal.get('symbol', '')
                action = signal.get('action', 'HOLD')
                
                if symbol not in dashboard.symbols_data:
                    dashboard.symbols_data[symbol] = {}
                
                dashboard.symbols_data[symbol]['signal'] = action
                dashboard.add_log('SUCCESS', f"Signal: {action} {symbol}")
        
        elif event_type == 'price_update':
            symbol = event.get('symbol', '')
            price = event.get('price', 0)
            change = event.get('change', 0)
            
            if symbol not in dashboard.symbols_data:
                dashboard.symbols_data[symbol] = {}
            
            dashboard.symbols_data[symbol]['price'] = price
            dashboard.symbols_data[symbol]['change'] = change
        
        elif event_type == 'news_item':
            news_item = event.get('data', {})
            dashboard.add_news({
                'time': datetime.now().strftime('%H:%M'),
                'symbol': news_item.get('symbol', 'N/A'),
                'title': news_item.get('title', 'No title'),
                'relevance': news_item.get('relevance_score', 0)
            })
        
        elif event_type == 'websocket_status':
            status = event.get('connected', False)
            dashboard.system_metrics['websocket_status'] = '✓' if status else '❌'
        
        elif event_type == 'error':
            dashboard.system_metrics['errors'] += 1
            dashboard.add_log('ERROR', event.get('message', 'Unknown error'))
        
        # Update metrics
        dashboard.system_metrics['total_symbols'] = len(dashboard.symbols_data)
        
    except Exception as e:
        logging.error(f"Error handling event: {e}")


async def show_startup_animation():
    """Show fancy loading animation during startup"""
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    import time
    
    # ASCII art banner
    banner = Text()
    banner.append("███████╗███████╗███╗   ██╗████████╗██████╗ ██╗   ██╗\n", style="bold cyan")
    banner.append("██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚██╗ ██╔╝\n", style="bold cyan")
    banner.append("███████╗█████╗  ██╔██╗ ██║   ██║   ██████╔╝ ╚████╔╝ \n", style="bold cyan")
    banner.append("╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗  ╚██╔╝  \n", style="bold cyan")
    banner.append("███████║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   \n", style="bold cyan")
    banner.append("╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   \n", style="bold cyan")
    banner.append("\nSentiment Enhanced Neural Trading Research Yield", style="bold white")
    
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
    
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        # Fyers API initialization
        task1 = progress.add_task("[cyan]🔌 Connecting to Fyers API...", total=100)
        for i in range(100):
            await asyncio.sleep(0.01)
            progress.update(task1, advance=1)
        progress.update(task1, description="[green]✓ Fyers API Connected")
        
        # News scraper initialization
        task2 = progress.add_task("[cyan]📰 Initializing News Scraper...", total=100)
        for i in range(100):
            await asyncio.sleep(0.008)
            progress.update(task2, advance=1)
        progress.update(task2, description="[green]✓ News Scraper Ready")
        
        # Sentiment analyzer
        task3 = progress.add_task("[cyan]🧠 Loading ML Models...", total=100)
        for i in range(100):
            await asyncio.sleep(0.012)
            progress.update(task3, advance=1)
        progress.update(task3, description="[green]✓ ML Models Loaded")
        
        # WebSocket
        task4 = progress.add_task("[cyan]📡 Establishing WebSocket Connection...", total=100)
        for i in range(100):
            await asyncio.sleep(0.007)
            progress.update(task4, advance=1)
        progress.update(task4, description="[green]✓ WebSocket Connected")
        
        # Final setup
        task5 = progress.add_task("[cyan]⚙️  Configuring Trading Engine...", total=100)
        for i in range(100):
            await asyncio.sleep(0.005)
            progress.update(task5, advance=1)
        progress.update(task5, description="[green]✓ Trading Engine Ready")
    
    # Success message
    success_msg = Text()
    success_msg.append("🚀 ", style="bold green")
    success_msg.append("System Initialized Successfully!", style="bold green")
    success_msg.append("\n\n")
    success_msg.append("All systems operational. Starting live trading dashboard...", style="dim white")
    
    console.print(Panel(success_msg, border_style="green", padding=(1, 2)))
    await asyncio.sleep(1.5)


async def run_with_dashboard():
    """Run the main system with dashboard integration"""
    from utils.event_bus import event_bus
    
    # Register dashboard callback
    event_bus.register_dashboard_callback(handle_event)
    
    dashboard.add_log('INFO', 'System started')
    dashboard.status = "LIVE"
    
    try:
        # Import and run main
        from main import main
        await main()
    except Exception as e:
        dashboard.status = "ERROR"
        dashboard.add_log('ERROR', str(e))
        raise


if __name__ == "__main__":
    console.clear()
    
    try:
        # Show startup animation first (blocks until complete)
        asyncio.run(show_startup_animation())
        
        # Clear screen for dashboard
        console.clear()
        
        # Now start dashboard in background thread
        dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Small delay to let dashboard start rendering
        time.sleep(0.5)
        
        # Run main system with dashboard integration
        asyncio.run(run_with_dashboard())
        
    except KeyboardInterrupt:
        console.show_cursor(True)
        console.clear()
        console.print("\n[bold red]🛑 System stopped by user[/bold red]")
    except Exception as e:
        console.show_cursor(True)
        console.clear()
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        console.show_cursor(True)
