"""
Test the dashboard with simulated data
"""
import asyncio
import sys
import time
from datetime import datetime
from run_clean import dashboard, start_dashboard, handle_event
import threading
from rich.console import Console

console = Console()


async def simulate_trading():
    """Simulate trading events to test the dashboard"""
    
    # Initial setup
    await handle_event({"type": "websocket_status", "connected": True})
    await asyncio.sleep(1)
    
    dashboard.status = "LIVE"
    dashboard.market_status = "OPEN"
    
    # Simulate some symbols
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    await handle_event({
        "type": "status",
        "stage": "start",
        "watchlist": symbols
    })
    
    # Simulate price updates
    for i in range(30):
        for symbol in symbols:
            import random
            price = 1000 + random.randint(-100, 100)
            change = random.uniform(-3, 3)
            
            await handle_event({
                "type": "price_update",
                "symbol": symbol,
                "price": price,
                "change": change,
                "volume": random.randint(10000, 100000),
                "high": price + 10,
                "low": price - 10,
                "time": datetime.now().strftime('%H:%M:%S')
            })
        
        await asyncio.sleep(0.5)
    
    # Simulate news scraping
    await handle_event({
        "type": "news",
        "symbols": symbols,
        "counts": {s: random.randint(1, 5) for s in symbols}
    })
    
    await asyncio.sleep(2)
    
    # Simulate sentiment analysis
    for symbol in symbols:
        import random
        await handle_event({
            "type": "sentiment",
            "symbol": symbol,
            "summary": {
                "weighted_sentiment": random.uniform(-1, 1),
                "news_count": random.randint(1, 5)
            }
        })
        
        await asyncio.sleep(0.3)
    
    # Simulate news items
    for symbol in symbols[:3]:
        await handle_event({
            "type": "news_item",
            "data": {
                "symbol": symbol,
                "title": f"Breaking: {symbol} announces major update",
                "relevance_score": random.uniform(0.5, 1.0)
            }
        })
        await asyncio.sleep(0.5)
    
    # Simulate mismatch detection
    await asyncio.sleep(2)
    
    for symbol in symbols[:2]:
        await handle_event({
            "type": "mismatch",
            "symbol": symbol,
            "analysis": {
                "discord_score": random.uniform(0.3, 0.8),
                "is_mismatched": True,
                "confidence": random.uniform(0.6, 0.9)
            }
        })
        await asyncio.sleep(1)
    
    # Simulate signals
    await asyncio.sleep(1)
    
    await handle_event({
        "type": "signals",
        "data": [
            {"symbol": "RELIANCE", "action": "BUY"},
            {"symbol": "TCS", "action": "SELL"}
        ]
    })
    
    # Keep updating prices
    while True:
        for symbol in symbols:
            price = dashboard.symbols_data.get(symbol, {}).get('price', 1000)
            new_price = price * (1 + random.uniform(-0.01, 0.01))
            change = ((new_price - price) / price) * 100
            
            await handle_event({
                "type": "price_update",
                "symbol": symbol,
                "price": new_price,
                "change": change,
                "volume": random.randint(10000, 100000),
                "time": datetime.now().strftime('%H:%M:%S')
            })
        
        await asyncio.sleep(2)


if __name__ == "__main__":
    console.clear()
    
    try:
        # Import startup animation
        from run_clean import show_startup_animation
        import asyncio
        
        # Show fancy startup animation
        asyncio.run(show_startup_animation())
        
        # Clear screen for dashboard
        console.clear()
        
        # Start dashboard rendering in background thread
        dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Small delay to let dashboard start
        time.sleep(0.5)
        
        # Run simulation
        asyncio.run(simulate_trading())
        
    except KeyboardInterrupt:
        console.show_cursor(True)
        console.clear()
        console.print("\n[bold red]🛑 Test stopped[/bold red]")
