#!/usr/bin/env python3
"""
Clean runner for S.E.N.T.R.Y trading system with minimal logging
"""
import sys
import os
import logging
from datetime import datetime

# Suppress all external library logs
logging.getLogger('snscrape').setLevel(logging.CRITICAL)
logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
logging.getLogger('torch').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Only show our system logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def print_banner():
    print("=" * 60)
    print("🚀 S.E.N.T.R.Y - Advanced Trading Intelligence")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 System Status: Initializing...")
    print()

if __name__ == "__main__":
    print_banner()
    
    try:
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
