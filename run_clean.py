#!/usr/bin/env python3
"""
Clean runner for S.E.N.T.R.Y trading system with minimal logging
"""
import sys
import os
import logging
from datetime import datetime
import asyncio
import warnings

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

# Custom formatter for desired timestamp format
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3] # Default to milliseconds with colon

# Only show our system logs
formatter = CustomFormatter(fmt='%(asctime)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)

def print_banner():
    print("=" * 60)
    print("🚀 S.E.N.T.R.Y - Advanced Trading Intelligence")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]}")
    print("🎯 System Status: Initializing...")
    print()

if __name__ == "__main__":
    print_banner()
    
    try:
        from main import main
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
