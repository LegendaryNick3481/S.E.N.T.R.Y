from flask import Flask, render_template, jsonify, Response, request
import asyncio
import os
import logging
from datetime import datetime
import threading
import queue

# Core system imports
from main import MismatchedEnergySystem
from config import Config
from utils.event_bus import event_bus

# Suppress Flask and external library logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('snscrape').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

app = Flask(__name__)

# Setup a long-running asyncio event loop in a separate thread
loop = asyncio.new_event_loop()

def run_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=run_async_loop, args=(loop,), daemon=True).start()

def run_async(coro, timeout=30):
    """Run async coroutine with timeout protection"""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except asyncio.TimeoutError:
        logging.error(f"Async operation timed out after {timeout}s")
        future.cancel()
        raise


def format_error(message):
    return {"ok": False, "error": message}


# Singleton system instance - created once and reused
_system_instance = None
_system_lock = threading.Lock()
_system_init_lock = asyncio.Lock()

async def get_system():
    """Get or create the singleton system instance (thread-safe)"""
    global _system_instance
    
    # Double-checked locking pattern for async
    if _system_instance is None:
        async with _system_init_lock:
            if _system_instance is None:
                _system_instance = MismatchedEnergySystem()
                await _system_instance.initialize()
                logging.info("System initialized and ready")
    
    return _system_instance

def get_watchlist():
    """Helper to load watchlist from tickers"""
    try:
        from data.tickers import get_tickers
        return get_tickers()
    except Exception as e:
        logging.error(f"Error loading tickers: {e}")
        return []

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/overview')
def api_overview():
    try:
        async def _run():
            system = await get_system()
            overview = await system.get_market_overview()
            return overview

        data = run_async(_run())
        
        # Return empty data structure if system fails
        if not data:
            data = {
                'fyers_connected': False,
                'market_summary': {
                    'total_symbols': 0,
                    'avg_discord_score': 0.0,
                    'max_discord_score': 0.0,
                    'mismatched_symbols': 0
                },
                'symbols': {},
                'news_data': {},
                'portfolio_history': [],
                'win_rate': 0.0,
                'total_trades': 0,
                'portfolio_value': 0
            }
        
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error in /api/overview: {e}")
        return jsonify(format_error(str(e))), 500


@app.route('/api/signals')
def api_signals():
    try:
        watchlist = get_watchlist()
        if not watchlist:
            return jsonify(format_error("Could not load tickers.")), 500

        async def _run():
            system = await get_system()
            system.live_executor.watchlist = watchlist
            signals = await system.live_executor.get_trading_signals()
            return signals

        signals = run_async(_run())
        return jsonify({"ok": True, "signals": signals or []})
    except asyncio.TimeoutError:
        return jsonify(format_error("Request timed out")), 504
    except Exception as e:
        logging.error(f"Error in /api/signals: {e}")
        return jsonify(format_error(str(e))), 500


def get_mock_portfolio():
    return {
        "portfolio": {
            "portfolio_value": 100000,
            "available_cash": 50000,
            "exposure": 0.5,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "positions": {
                "RELIANCE": {"pnl": 1000},
                "TCS": {"pnl": -500},
            },
        }
    }


@app.route('/api/portfolio')
def api_portfolio():
    try:
        async def _run():
            system = await get_system()
            portfolio = await system.live_executor.get_portfolio_status()
            return portfolio

        portfolio = run_async(_run())
        
        # Use mock data if real portfolio is empty
        if not portfolio:
            portfolio = get_mock_portfolio()
        
        return jsonify({"ok": True, "portfolio": portfolio.get('portfolio', {})})
    except asyncio.TimeoutError:
        return jsonify(format_error("Request timed out")), 504
    except Exception as e:
        logging.error(f"Error in /api/portfolio: {e}")
        return jsonify(format_error(str(e))), 500


@app.route('/events')
def events():
    """Server-Sent Events endpoint - reuses main event loop"""
    message_queue = queue.Queue(maxsize=100)

    async def subscribe_and_put():
        try:
            async for payload in event_bus.subscribe():
                try:
                    message_queue.put_nowait(f"data: {payload}\n\n")
                except queue.Full:
                    # Drop oldest message if queue full
                    try:
                        message_queue.get_nowait()
                        message_queue.put_nowait(f"data: {payload}\n\n")
                    except:
                        pass
        except Exception as e:
            logging.error(f"Error in SSE subscription: {e}")

    # Reuse existing event loop instead of creating new one
    asyncio.run_coroutine_threadsafe(subscribe_and_put(), loop)

    def generate():
        try:
            while True:
                msg = message_queue.get(timeout=30)
                yield msg
        except queue.Empty:
            yield "data: {\"type\": \"keepalive\"}\n\n"
        except GeneratorExit:
            # Client disconnected
            pass

    return Response(generate(), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    })

@app.route('/api/run-cycle', methods=['POST'])
def api_run_cycle():
    try:
        watchlist = request.json.get('watchlist') if request.is_json else None
        if not watchlist:
            watchlist = get_watchlist()
            if not watchlist:
                return jsonify(format_error("Could not load tickers.")), 500

        async def _run():
            system = await get_system()
            system.live_executor.watchlist = watchlist
            # Run a single processing cycle
            await system.live_executor._process_trading_cycle()
            return {"ok": True}

        result = run_async(_run(), timeout=60)  # Longer timeout for trading cycle
        return jsonify(result)
    except asyncio.TimeoutError:
        return jsonify(format_error("Trading cycle timed out")), 504
    except Exception as e:
        logging.error(f"Error in /api/run-cycle: {e}")
        return jsonify(format_error(str(e))), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system_initialized": _system_instance is not None
        }
        
        if _system_instance:
            status["components"] = {
                "fyers": _system_instance.fyers_client.is_connected,
                "news_scraper": _system_instance.news_scraper.session is not None,
            }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 500


@app.teardown_appcontext
def shutdown_system(exception=None):
    """Cleanup system on app shutdown"""
    global _system_instance
    if _system_instance is not None:
        logging.info("Shutting down system...")
        try:
            run_async(_system_instance.shutdown(), timeout=10)
            logging.info("System shutdown complete")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            _system_instance = None


if __name__ == '__main__':
    # Production-ready settings
    import sys
    
    # Check if running in production mode
    is_production = '--production' in sys.argv or os.getenv('FLASK_ENV') == 'production'
    
    if is_production:
        logging.info("Starting Flask in PRODUCTION mode")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logging.info("Starting Flask in DEVELOPMENT mode")
        app.run(debug=True, host='0.0.0.0', port=5000)