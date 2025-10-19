"""
Setup script for Mismatched Energy Trading System
"""
import os
import sys
import subprocess
import shutil

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data/cache',
        'backtest_results',
        'examples'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def setup_environment():
    """Setup environment file"""
    env_file = '.env'
    env_example = 'env_example.txt'
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print(f"✅ Created {env_file} from template")
            print(f"📝 Please edit {env_file} with your Fyers API credentials")
        else:
            print(f"❌ {env_example} not found")
            return False
    else:
        print(f"✅ {env_file} already exists")
    
    return True

def download_nlp_models():
    """Download NLP models"""
    try:
        print("🧠 Downloading NLP models...")
        from sentence_transformers import SentenceTransformer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        # Download sentence transformer model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Sentence transformer model downloaded")
        
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER sentiment analyzer initialized")
        
    except Exception as e:
        print(f"❌ Failed to download NLP models: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Mismatched Energy Trading System")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed at environment setup")
        return
    
    # Download NLP models
    if not download_nlp_models():
        print("❌ Setup failed at NLP model download")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your Fyers API credentials")
    print("2. Run demo: python examples/demo.py")
    print("3. Run backtest: python main.py --mode backtest")
    print("4. Start live trading: python main.py --mode live")
    
    print("\n⚠️  Important Notes:")
    print("- Always start with paper trading")
    print("- Monitor system performance carefully")
    print("- Use proper risk management")

if __name__ == "__main__":
    main()
