"""
Event bus for broadcasting events to the terminal dashboard.
"""
import asyncio
from typing import Dict, Any, Callable


class EventBus:
    def __init__(self):
        self._dashboard_callbacks = []

    def register_dashboard_callback(self, callback: Callable):
        """Register a callback for terminal dashboard updates"""
        self._dashboard_callbacks.append(callback)

    async def publish(self, event: Dict[str, Any]):
        """Publish event to all registered dashboard callbacks"""
        for callback in self._dashboard_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception:
                pass


event_bus = EventBus()


