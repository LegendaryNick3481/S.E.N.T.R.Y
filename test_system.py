"""
Test script to verify Mismatched Energy Trading System components
"""
import asyncio
import sys
import os
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from data.fyers_client import FyersClient
        print("✅ FyersClient imported")
    except Exception as e:
        print(f"❌ FyersClient import failed: {e}")
        return False
    
    try:
        from news.news_scraper import NewsScraper
        print("✅ NewsScraper imported")
    except Exception as e:
        print(f"❌ NewsScraper import failed: {e}")
        return False
    
    try:
        from nlp.sentiment_analyzer import SentimentAnalyzer
        print("✅ SentimentAnalyzer imported")
    except Exception as e:
        print(f"❌ SentimentAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.cross_modal_analyzer import CrossModalAnalyzer
        print("✅ CrossModalAnalyzer imported")
    except Exception as e:
        print(f"❌ CrossModalAnalyzer import failed: {e}")
        return False
    
    try:
        from scoring.capital_allocator import CapitalAllocator
        print("✅ CapitalAllocator imported")
    except Exception as e:
        print(f"❌ CapitalAllocator import failed: {e}")
        return False
    
    try:
        from backtesting.backtest_engine import BacktestEngine
        print("✅ BacktestEngine imported")
    except Exception as e:
        print(f"❌ BacktestEngine import failed: {e}")
        return False
    
    try:
        from trading.live_executor import LiveExecutor
        print("✅ LiveExecutor imported")
    except Exception as e:
        print(f"❌ LiveExecutor import failed: {e}")
        return False
    
    return True

def test_sentiment_analyzer():
    """Test sentiment analyzer functionality"""
    print("\n🧪 Testing SentimentAnalyzer...")
    
    try:
        from nlp.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        asyncio.run(analyzer.initialize())
        
        # Test sentiment analysis
        test_text = "This is a positive news about the company's growth"
        sentiment = analyzer.analyze_sentiment(test_text)
        
        print(f"✅ Sentiment analysis: {sentiment['label']} (score: {sentiment['compound']:.3f})")
        
        # Test text embedding
        embedding = analyzer.get_text_embedding(test_text)
        print(f"✅ Text embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SentimentAnalyzer test failed: {e}")
        return False

def test_cross_modal_analyzer():
    """Test cross-modal analyzer functionality"""
    print("\n🧪 Testing CrossModalAnalyzer...")
    
    try:
        from analysis.cross_modal_analyzer import CrossModalAnalyzer
        
        analyzer = CrossModalAnalyzer()
        
        # Test mismatched energy detection
        mismatch = analyzer.detect_mismatched_energy(
            news_sentiment=0.5,  # Positive news
            price_direction=-0.03,  # Price down 3%
            volume_anomaly=0.8,  # High volume
            volatility_spike=0.6  # High volatility
        )
        
        print(f"✅ Mismatch detected: {mismatch['is_mismatched']}")
        print(f"✅ Discord score: {mismatch['discord_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CrossModalAnalyzer test failed: {e}")
        return False

def test_capital_allocator():
    """Test capital allocator functionality"""
    print("\n🧪 Testing CapitalAllocator...")
    
    try:
        from scoring.capital_allocator import CapitalAllocator
        
        allocator = CapitalAllocator()
        
        # Test position sizing
        position_info = allocator.calculate_position_size(
            symbol="RELIANCE",
            discord_score=0.7,
            current_price=2500.0,
            volatility=0.02
        )
        
        print(f"✅ Position size: {position_info['num_shares']} shares")
        print(f"✅ Position value: ₹{position_info['position_value']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CapitalAllocator test failed: {e}")
        return False

def test_backtest_engine():
    """Test backtest engine functionality"""
    print("\n🧪 Testing BacktestEngine...")
    
    try:
        from backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Test mock data generation
        price_data = engine._generate_mock_price_data("RELIANCE", "2023-01-01", "2023-01-02")
        news_data = engine._generate_mock_news_data("RELIANCE", "2023-01-01", "2023-01-02")
        
        print(f"✅ Generated {len(price_data)} price records")
        print(f"✅ Generated {len(news_data)} news items")
        
        return True
        
    except Exception as e:
        print(f"❌ BacktestEngine test failed: {e}")
        return False

async def test_async_components():
    """Test async components"""
    print("\n🧪 Testing async components...")
    
    try:
        from news.news_scraper import NewsScraper
        from nlp.sentiment_analyzer import SentimentAnalyzer
        
        # Test news scraper initialization
        news_scraper = NewsScraper()
        await news_scraper.initialize()
        print("✅ NewsScraper initialized")
        
        # Test sentiment analyzer initialization
        sentiment_analyzer = SentimentAnalyzer()
        if await sentiment_analyzer.initialize():
            print("✅ SentimentAnalyzer initialized")
        else:
            print("❌ SentimentAnalyzer initialization failed")
            return False
        
        # Cleanup
        await news_scraper.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Async components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Mismatched Energy Trading System - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test sentiment analyzer
    if test_sentiment_analyzer():
        tests_passed += 1
    
    # Test cross-modal analyzer
    if test_cross_modal_analyzer():
        tests_passed += 1
    
    # Test capital allocator
    if test_capital_allocator():
        tests_passed += 1
    
    # Test backtest engine
    if test_backtest_engine():
        tests_passed += 1
    
    # Test async components
    if asyncio.run(test_async_components()):
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Configure Fyers API credentials in .env")
        print("2. Run demo: python examples/demo.py")
        print("3. Start backtesting: python main.py --mode backtest")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
