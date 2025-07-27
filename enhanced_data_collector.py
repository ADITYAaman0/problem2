"""
Enhanced Data Collector with Sample Data Generation
Creates realistic transaction data for wallet risk scoring demonstration
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """
    Enhanced data collector with sample data generation capabilities
    """
    
    def __init__(self):
        self.assets = ['USDC', 'USDT', 'DAI', 'WETH', 'WBTC', 'LINK', 'UNI', 'COMP', 'AAVE']
        self.asset_prices = {
            'USDC': 1.0, 'USDT': 1.0, 'DAI': 1.0,
            'WETH': 2500.0, 'WBTC': 45000.0, 'LINK': 15.0,
            'UNI': 8.0, 'COMP': 80.0, 'AAVE': 120.0
        }
        self.asset_decimals = {
            'USDC': 6, 'USDT': 6, 'DAI': 18,
            'WETH': 18, 'WBTC': 8, 'LINK': 18,
            'UNI': 18, 'COMP': 18, 'AAVE': 18
        }
        
    def load_wallet_addresses(self, file_path: str) -> List[str]:
        """Load wallet addresses from Excel file"""
        try:
            df = pd.read_excel(file_path)
            wallets = df.iloc[:, 0].dropna().astype(str).tolist()
            logger.info(f"Loaded {len(wallets)} wallet addresses from {file_path}")
            return wallets
        except Exception as e:
            logger.error(f"Error loading wallet addresses: {e}")
            return []
    
    def generate_realistic_transaction_data(self, wallets: List[str], num_transactions_per_wallet: tuple = (5, 50)) -> List[Dict]:
        """
        Generate realistic transaction data for demonstration purposes
        
        Args:
            wallets: List of wallet addresses
            num_transactions_per_wallet: Min and max transactions per wallet
            
        Returns:
            List of transaction dictionaries
        """
        logger.info("Generating realistic transaction data for demonstration...")
        
        all_transactions = []
        current_time = int(datetime.now().timestamp())
        
        # Define wallet behavior patterns
        patterns = {
            'conservative': {'deposit_prob': 0.4, 'borrow_prob': 0.2, 'repay_prob': 0.25, 'redeem_prob': 0.14, 'liquidation_prob': 0.01},
            'moderate': {'deposit_prob': 0.3, 'borrow_prob': 0.3, 'repay_prob': 0.25, 'redeem_prob': 0.13, 'liquidation_prob': 0.02},
            'aggressive': {'deposit_prob': 0.25, 'borrow_prob': 0.4, 'repay_prob': 0.2, 'redeem_prob': 0.1, 'liquidation_prob': 0.05},
            'very_risky': {'deposit_prob': 0.2, 'borrow_prob': 0.5, 'repay_prob': 0.15, 'redeem_prob': 0.05, 'liquidation_prob': 0.1}
        }
        
        for wallet in wallets:
            # Assign random behavior pattern
            pattern_name = random.choices(
                list(patterns.keys()), 
                weights=[0.4, 0.35, 0.2, 0.05]  # Most wallets are conservative/moderate
            )[0]
            pattern = patterns[pattern_name]
            
            # Determine number of transactions for this wallet
            num_txs = random.randint(*num_transactions_per_wallet)
            
            # Generate transaction timeline (last 2 years)
            start_time = current_time - (2 * 365 * 24 * 60 * 60)  # 2 years ago
            transaction_times = sorted([
                random.randint(start_time, current_time) for _ in range(num_txs)
            ])
            
            wallet_transactions = []
            wallet_balance = {}  # Track balances for realistic behavior
            
            for i, timestamp in enumerate(transaction_times):
                # Choose action based on pattern and current state
                actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
                probabilities = [
                    pattern['deposit_prob'],
                    pattern['borrow_prob'],
                    pattern['repay_prob'],
                    pattern['redeem_prob'],
                    pattern['liquidation_prob']
                ]
                
                # Adjust probabilities based on wallet state
                if not wallet_balance:  # First transaction should likely be deposit
                    probabilities = [0.8, 0.1, 0.05, 0.04, 0.01]
                
                action = random.choices(actions, weights=probabilities)[0]
                asset = random.choice(self.assets)
                
                # Generate realistic amounts based on action and asset
                amount = self._generate_realistic_amount(action, asset, pattern_name)
                
                transaction = {
                    'userWallet': wallet,
                    'action': action,
                    'timestamp': timestamp,
                    'actionData': {
                        'amount': str(amount),
                        'assetSymbol': asset,
                        'assetPriceUSD': self.asset_prices[asset] * random.uniform(0.95, 1.05),  # Price fluctuation
                        'decimals': self.asset_decimals[asset]
                    }
                }
                
                # Add specific fields for certain actions
                if action == 'borrow':
                    transaction['actionData']['borrowRate'] = str(random.uniform(0.02, 0.15))  # 2-15% APR
                    transaction['actionData']['borrowRateMode'] = random.choice(['1', '2'])  # Stable or variable
                elif action == 'liquidationcall':
                    collateral_asset = random.choice([a for a in self.assets if a != asset])
                    transaction['actionData'] = {
                        'collateralAmount': str(amount),
                        'principalAmount': str(amount * random.uniform(0.8, 1.2)),
                        'collateralAsset': collateral_asset,
                        'principalAsset': asset
                    }
                
                wallet_transactions.append(transaction)
                
                # Update wallet balance tracking
                if action == 'deposit':
                    wallet_balance[asset] = wallet_balance.get(asset, 0) + amount
                elif action == 'redeem':
                    wallet_balance[asset] = max(0, wallet_balance.get(asset, 0) - amount)
            
            all_transactions.extend(wallet_transactions)
            
            if len(all_transactions) % 1000 == 0:
                logger.info(f"Generated {len(all_transactions)} transactions so far...")
        
        logger.info(f"Generated {len(all_transactions)} total transactions for {len(wallets)} wallets")
        return all_transactions
    
    def _generate_realistic_amount(self, action: str, asset: str, pattern: str) -> int:
        """Generate realistic transaction amounts based on action and risk pattern"""
        
        # Base amounts (in token units before decimal adjustment)
        base_amounts = {
            'USDC': {'small': 1000, 'medium': 10000, 'large': 100000},
            'USDT': {'small': 1000, 'medium': 10000, 'large': 100000},
            'DAI': {'small': 1000, 'medium': 10000, 'large': 100000},
            'WETH': {'small': 1, 'medium': 5, 'large': 20},
            'WBTC': {'small': 0.1, 'medium': 0.5, 'large': 2},
            'LINK': {'small': 100, 'medium': 1000, 'large': 5000},
            'UNI': {'small': 200, 'medium': 2000, 'large': 10000},
            'COMP': {'small': 20, 'medium': 100, 'large': 500},
            'AAVE': {'small': 50, 'medium': 200, 'large': 1000}
        }
        
        # Choose size based on pattern
        if pattern == 'conservative':
            size = random.choices(['small', 'medium', 'large'], weights=[0.6, 0.3, 0.1])[0]
        elif pattern == 'moderate':
            size = random.choices(['small', 'medium', 'large'], weights=[0.4, 0.4, 0.2])[0]
        elif pattern == 'aggressive':
            size = random.choices(['small', 'medium', 'large'], weights=[0.2, 0.4, 0.4])[0]
        else:  # very_risky
            size = random.choices(['small', 'medium', 'large'], weights=[0.1, 0.3, 0.6])[0]
        
        base_amount = base_amounts[asset][size]
        
        # Add randomness
        multiplier = random.uniform(0.5, 2.0)
        amount = base_amount * multiplier
        
        # Convert to wei/smallest unit
        decimals = self.asset_decimals[asset]
        amount_wei = int(amount * (10 ** decimals))
        
        return amount_wei
    
    def create_sample_data_with_known_scores(self, output_file: str = 'user-wallet-transactions.json') -> None:
        """
        Create sample data with wallets designed to have specific risk profiles
        """
        logger.info("Creating sample data with known risk profiles...")
        
        # Load wallet addresses
        wallets = self.load_wallet_addresses('Walletid.xlsx')
        if not wallets:
            logger.error("No wallet addresses loaded. Cannot proceed.")
            return
        
        # Take first 100 wallets for demonstration
        wallets = wallets[:100]
        
        all_transactions = []
        current_time = int(datetime.now().timestamp())
        
        # Create different risk categories
        categories = {
            'low_risk': wallets[:25],      # First 25 wallets - low risk
            'medium_risk': wallets[25:50], # Next 25 wallets - medium risk  
            'high_risk': wallets[50:75],   # Next 25 wallets - high risk
            'very_high_risk': wallets[75:100]  # Last 25 wallets - very high risk
        }
        
        for category, wallet_list in categories.items():
            logger.info(f"Generating {category} transactions for {len(wallet_list)} wallets...")
            
            for wallet in wallet_list:
                transactions = self._generate_category_transactions(wallet, category, current_time)
                all_transactions.extend(transactions)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(all_transactions, f, indent=2)
        
        logger.info(f"Sample data created with {len(all_transactions)} transactions saved to {output_file}")
        
        # Print summary
        print("\\n" + "="*60)
        print("SAMPLE DATA GENERATION COMPLETE")
        print("="*60)
        print(f"Total Transactions: {len(all_transactions)}")
        print(f"Total Wallets: {len(wallets)}")
        print("\\nWallet Categories:")
        for category, wallet_list in categories.items():
            print(f"  {category.replace('_', ' ').title()}: {len(wallet_list)} wallets")
        print(f"\\nData saved to: {output_file}")
        print("="*60)
    
    def _generate_category_transactions(self, wallet: str, category: str, current_time: int) -> List[Dict]:
        """Generate transactions for a specific risk category"""
        
        transactions = []
        start_time = current_time - (365 * 24 * 60 * 60)  # 1 year ago
        
        if category == 'low_risk':
            # Conservative behavior: Many deposits, few borrows, good repayment
            num_deposits = random.randint(15, 25)
            num_borrows = random.randint(2, 8)
            num_repays = random.randint(num_borrows, num_borrows + 3)  # Always repay well
            num_redeems = random.randint(5, 10)
            liquidations = 0  # No liquidations
            
        elif category == 'medium_risk':
            # Moderate behavior: Balanced activity
            num_deposits = random.randint(10, 20)
            num_borrows = random.randint(8, 15)
            num_repays = random.randint(max(1, num_borrows - 2), num_borrows + 1)
            num_redeems = random.randint(5, 12)
            liquidations = random.randint(0, 1)  # Maybe 1 liquidation
            
        elif category == 'high_risk':
            # Risky behavior: Heavy borrowing, poor repayment
            num_deposits = random.randint(5, 15)
            num_borrows = random.randint(15, 25)
            num_repays = random.randint(max(1, num_borrows - 5), num_borrows - 1)  # Poor repayment
            num_redeems = random.randint(3, 8)
            liquidations = random.randint(1, 3)  # Multiple liquidations
            
        else:  # very_high_risk
            # Very risky behavior: Minimal deposits, excessive borrowing
            num_deposits = random.randint(2, 8)
            num_borrows = random.randint(20, 35)
            num_repays = random.randint(max(1, num_borrows - 10), num_borrows - 5)  # Very poor repayment
            num_redeems = random.randint(1, 5)
            liquidations = random.randint(3, 8)  # Many liquidations
        
        # Generate all transaction types
        all_actions = (
            ['deposit'] * num_deposits +
            ['borrow'] * num_borrows +
            ['repay'] * num_repays +
            ['redeemunderlying'] * num_redeems +
            ['liquidationcall'] * liquidations
        )
        
        # Create timestamps
        num_total = len(all_actions)
        timestamps = sorted([random.randint(start_time, current_time) for _ in range(num_total)])
        
        # Generate transactions
        for i, action in enumerate(all_actions):
            asset = random.choice(self.assets)
            amount = self._generate_realistic_amount(action, asset, category)
            
            transaction = {
                'userWallet': wallet,
                'action': action,
                'timestamp': timestamps[i],
                'actionData': {
                    'amount': str(amount),
                    'assetSymbol': asset,
                    'assetPriceUSD': self.asset_prices[asset] * random.uniform(0.95, 1.05),
                    'decimals': self.asset_decimals[asset]
                }
            }
            
            # Add specific fields for certain actions
            if action == 'borrow':
                transaction['actionData']['borrowRate'] = str(random.uniform(0.02, 0.15))
                transaction['actionData']['borrowRateMode'] = random.choice(['1', '2'])
            elif action == 'liquidationcall':
                collateral_asset = random.choice([a for a in self.assets if a != asset])
                transaction['actionData'] = {
                    'collateralAmount': str(amount),
                    'principalAmount': str(amount * random.uniform(0.8, 1.2)),
                    'collateralAsset': collateral_asset,
                    'principalAsset': asset
                }
            
            transactions.append(transaction)
        
        return transactions

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    collector.create_sample_data_with_known_scores()
