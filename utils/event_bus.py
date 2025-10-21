"""
Simple async event bus for broadcasting backend events to the web UI via SSE.
"""
import asyncio
import json
from typing import AsyncIterator, Dict, Any


class EventBus:
    def __init__(self):
        self._subscribers = set()

    def _new_queue(self) -> asyncio.Queue:
        return asyncio.Queue(maxsize=1000)

    async def publish(self, event: Dict[str, Any]):
        # Non-blocking put: drop if full to avoid backpressure issues
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest and enqueue
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    async def subscribe(self) -> AsyncIterator[str]:
        q = self._new_queue()
        self._subscribers.add(q)
        try:
            while True:
                event = await q.get()
                yield json.dumps(event)
        finally:
            self._subscribers.discard(q)


event_bus = EventBus()


