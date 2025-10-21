## Project Overview

This project is a sophisticated trading system named "Mismatched Energy" that operates in the Indian stock market. The core strategy is to identify and exploit discrepancies between news sentiment and price action, referred to as "mismatched energy." For instance, if a stock's price is dropping despite positive news, the system identifies this as a potential buying opportunity.

The backend is built with Python and utilizes a rich set of libraries for data analysis, machine learning, and trading:
- **Data Processing:** `pandas`, `numpy`, `scipy`
- **Trading:** `fyers-apiv3` for integration with the Fyers trading platform.
- **NLP:** `sentence-transformers` and `vaderSentiment` for news sentiment analysis.
- **Web Scraping:** `feedparser`, `snscrape`, and `beautifulsoup4` for gathering news from various sources.

The frontend is a React and TypeScript application, providing a dashboard to visualize market data, trading signals, and system performance. It uses `vite` for the build tooling and includes libraries like `recharts` for charting and `tailwindcss` for styling.

## Building and Running

### Backend

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the environment:**
    Copy the `env_example.txt` to `.env` and add your Fyers API credentials.

3.  **Run the system:**
    The system can be run in different modes:

    *   **Backtesting:**
        ```bash
        python main.py --mode backtest --symbols RELIANCE TCS INFY --start-date 2023-01-01 --end-date 2024-01-01
        ```
    *   **Live Trading:**
        ```bash
        python main.py --mode live --watchlist RELIANCE TCS INFY HDFC ICICIBANK
        ```
    *   **Analysis:**
        ```bash
        python main.py --mode analyze --symbols RELIANCE TCS
        ```

### Frontend

1.  **Install dependencies:**
    ```bash
    npm install
    ```

2.  **Run the development server:**
    ```bash
    npm run dev
    ```

3.  **Build for production:**
    ```bash
    npm run build
    ```

## Development Conventions

- **Python:** The project uses `pylint` for linting.
- **TypeScript/React:** The project uses `eslint` for linting.
- **Logging:** The backend uses Python's `logging` module, configured in `main.py`, to log to both a file (`mismatched_energy.log`) and the console.

## Directory Overview

- **`data/`**: Contains the Fyers API client for market data.
- **`news/`**: Includes the news scraper for gathering news from various sources.
- **`nlp/`**: Houses the sentiment analyzer for processing news data.
- **`analysis/`**: Contains the cross-modal analyzer to detect "mismatched energy."
- **`scoring/`**: Includes the capital allocator for position sizing and risk management.
- **`backtesting/`**: Contains the backtesting engine for strategy validation.
- **`trading/`**: Includes the live executor for real-time trading.
- **`src/`**: Contains the source code for the React frontend.
- **`static/`**: Contains static assets for the web interface.
- **`templates/`**: Contains HTML templates for the web interface.

## Key Files

- **`main.py`**: The main entry point for the backend application.
- **`config.py`**: Configuration settings for the system.
- **`data/fyers_client.py`**: The Fyers API client.
- **`news/news_scraper.py`**: The news scraping system.
- **`nlp/sentiment_analyzer.py`**: The NLP processing and sentiment analysis module.
- **`analysis/cross_modal_analyzer.py`**: The cross-modal analysis module.
- **`scoring/capital_allocator.py`**: The capital allocation module.
- **`backtesting/backtest_engine.py`**: The backtesting framework.
- **`trading/live_executor.py`**: The live trading execution module.
- **`src/App.tsx`**: The main component for the React frontend.
- **`web_app.py`**: A Flask application to serve the web interface.
