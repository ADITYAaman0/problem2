#!/usr/bin/env python3
"""
Main Execution Script for Wallet Risk Scoring System
Demonstrates the complete workflow from data collection to risk assessment
"""

import os
import json
import logging
from datetime import datetime
from data_collector import AaveDataCollector
from enhanced_data_collector import EnhancedDataCollector
from risk_scoring import WalletRiskScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wallet_risk_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 WALLET RISK SCORING SYSTEM                   ║
    ║                                                              ║
    ║              DeFi Lending Protocol Risk Assessment           ║
    ║                     Based on Aave V2/V3                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required files and dependencies exist"""
    logger.info("Checking dependencies...")
    
    required_files = ['Walletid.xlsx', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    try:
        import pandas
        import numpy
        import requests
        import sklearn
        logger.info("All dependencies satisfied")
        return True
    except ImportError as e:
        logger.error(f"Missing Python dependency: {e}")
        return False

def collect_data(use_live_data=False):
    """
    Collect transaction data either from live APIs or generate synthetic data
    
    Args:
        use_live_data: If True, attempt to collect from Aave APIs
    
    Returns:
        bool: Success status
    """
    logger.info("Starting data collection phase...")
    
    if use_live_data:
        logger.info("Attempting to collect live data from Aave APIs...")
        collector = AaveDataCollector()
        collector.collect_all_transactions('Walletid.xlsx', 'user-wallet-transactions.json')
        
        # Check if we got any data
        try:
            with open('user-wallet-transactions.json', 'r') as f:
                data = json.load(f)
            if len(data) == 0:
                logger.warning("No live data collected, falling back to synthetic data generation")
                use_live_data = False
            else:
                logger.info(f"Successfully collected {len(data)} live transactions")
                return True
        except Exception as e:
            logger.error(f"Error reading collected data: {e}")
            use_live_data = False
    
    if not use_live_data:
        logger.info("Generating synthetic transaction data for demonstration...")
        enhanced_collector = EnhancedDataCollector()
        enhanced_collector.create_sample_data_with_known_scores('user-wallet-transactions.json')
        return True
    
    return False

def analyze_risk():
    """
    Perform risk scoring analysis on collected data
    
    Returns:
        bool: Success status
    """
    logger.info("Starting risk analysis phase...")
    
    # Check if transaction data exists
    if not os.path.exists('user-wallet-transactions.json'):
        logger.error("Transaction data file not found. Please run data collection first.")
        return False
    
    # Initialize and run risk scorer
    scorer = WalletRiskScorer('user-wallet-transactions.json')
    success = scorer.run_complete_analysis('wallet_scores.csv')
    
    if success:
        logger.info("Risk analysis completed successfully")
        return True
    else:
        logger.error("Risk analysis failed")
        return False

def generate_summary():
    """Generate and display analysis summary"""
    logger.info("Generating analysis summary...")
    
    try:
        import pandas as pd
        
        # Read results
        df = pd.read_csv('wallet_scores.csv')
        
        print("\\n" + "="*80)
        print("                        ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"📊 Total Wallets Analyzed: {len(df)}")
        print(f"📈 Score Range: {df['score'].min()} - {df['score'].max()}")
        print(f"📉 Average Score: {df['score'].mean():.1f}")
        print(f"📋 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\\n🎯 Risk Distribution:")
        risk_counts = df['risk_category'].value_counts()
        for category, count in risk_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} wallets ({percentage:.1f}%)")
        
        print("\\n🏆 Top 5 Lowest Risk Wallets:")
        top_5 = df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. {row['wallet_id'][:20]}... - Score: {row['score']} ({row['risk_category']})")
        
        print("\\n⚠️  Bottom 5 Highest Risk Wallets:")
        bottom_5 = df.tail(5)
        for i, (_, row) in enumerate(bottom_5.iterrows(), len(df)-4):
            print(f"   {i}. {row['wallet_id'][:20]}... - Score: {row['score']} ({row['risk_category']})")
        
        print("\\n📁 Output Files Generated:")
        output_files = [
            'wallet_scores.csv - Detailed scores and features',
            'analysis_report.txt - Comprehensive analysis report',
            'user-wallet-transactions.json - Raw transaction data',
            'wallet_risk_analysis.log - Execution log'
        ]
        for file in output_files:
            print(f"   ✓ {file}")
        
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return False

def main():
    """Main execution function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages:")
        print("   pip install -r requirements.txt")
        return 1
    
    print("✅ All dependencies satisfied")
    
    # Data collection phase
    print("\\n" + "="*60)
    print("PHASE 1: DATA COLLECTION")
    print("="*60)
    
    if not collect_data(use_live_data=False):  # Set to True to attempt live data collection
        print("❌ Data collection failed")
        return 1
    
    print("✅ Data collection completed")
    
    # Risk analysis phase
    print("\\n" + "="*60)
    print("PHASE 2: RISK ANALYSIS")
    print("="*60)
    
    if not analyze_risk():
        print("❌ Risk analysis failed")
        return 1
    
    print("✅ Risk analysis completed")
    
    # Summary generation
    print("\\n" + "="*60)
    print("PHASE 3: RESULTS SUMMARY")
    print("="*60)
    
    if not generate_summary():
        print("❌ Summary generation failed")
        return 1
    
    print("\\n🎉 Analysis completed successfully!")
    print("\\n📖 For detailed methodology and explanations, see:")
    print("   • PROJECT_DOCUMENTATION.md")
    print("   • analysis_report.txt")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
