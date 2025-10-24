# S.E.N.T.R.Y Architecture Redesign

## Current Problems
1. Flask (sync) + asyncio (async) = impedance mismatch
2. Heavy operations block request threads
3. 30-90 second response times
4. Session management issues
5. Race conditions with shared event loop

## New Architecture: Microservices Approach

### Components:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER / BROWSER                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Web Server (Port 8000)                 │
│  - Serves frontend                                          │
│  - WebSocket for real-time updates                          │
│  - REST API (reads from Redis)                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Redis (In-Memory DB)                     │
│  - market_data (overview, symbols)                          │
│  - signals (trading signals)                                │
│  - portfolio (positions, PnL)                               │
│  - websocket_feed (price updates)                           │
└───────────────────────────┬─────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   Trading    │  │  Market Data     │  │  News & NLP     │
│   Engine     │  │  Collector       │  │  Processor      │
│  (main.py)   │  │  (Fyers WS)      │  │  (async)        │
│              │  │                  │  │                 │
│ - Strategy   │  │ - Websocket      │  │ - News scraping │
│ - Execution  │  │ - Price data     │  │ - Sentiment     │
│ - Risk Mgmt  │  │ - Write to Redis │  │ - Write to Redis│
└──────────────┘  └──────────────────┘  └─────────────────┘
```

### Benefits:
- ✅ Each component runs independently
- ✅ No blocking - all async
- ✅ Horizontal scaling
- ✅ Easy to debug/monitor
- ✅ <10ms API responses
- ✅ Real-time WebSocket updates

