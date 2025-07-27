"""
Enhanced Risk Scoring Module for DeFi Wallets
Analyzes Aave transaction data to create comprehensive risk profiles
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WalletRiskScorer:
    """
    Comprehensive wallet risk scoring system based on DeFi transaction analysis
    """
    
    def __init__(self, transaction_file: str = 'user-wallet-transactions.json'):
        """
        Initialize the risk scorer
        
        Args:
            transaction_file: Path to the transaction data JSON file
        """
        self.transaction_file = transaction_file
        self.transactions = []
        self.wallet_features = defaultdict(dict)
        self.feature_weights = {
            'volume_score': 0.25,
            'frequency_score': 0.20,
            'diversity_score': 0.15,
            'liquidity_score': 0.15,
            'risk_behavior_score': 0.25
        }
    
    def load_transactions(self) -> bool:
        """
        Load transaction data from JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.transaction_file, 'r') as f:
                self.transactions = json.load(f)
            logger.info(f"Loaded {len(self.transactions)} transactions")
            return True
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            return False
    
    def extract_basic_features(self) -> Dict:
        """
        Extract basic transaction features for each wallet
        
        Returns:
            Dictionary with wallet features
        """
        wallet_features = defaultdict(lambda: {
            'deposit_count': 0, 'deposit_usd': 0.0,
            'borrow_count': 0, 'borrow_usd': 0.0,
            'repay_count': 0, 'repay_usd': 0.0,
            'redeem_count': 0, 'redeem_usd': 0.0,
            'liquidation_count': 0,
            'first_transaction': None,
            'last_transaction': None,
            'unique_assets': set(),
            'transaction_days': set(),
            'avg_transaction_size': 0.0,
            'total_volume': 0.0
        })
        
        for tx in self.transactions:
            wallet = tx['userWallet']
            action = tx['action'].lower()
            timestamp = tx.get('timestamp', 0)
            action_data = tx.get('actionData', {})
            
            # Calculate USD amount
            usd_amount = self._calculate_usd_amount(action_data)
            
            # Update basic counters
            if action == 'deposit':
                wallet_features[wallet]['deposit_count'] += 1
                wallet_features[wallet]['deposit_usd'] += usd_amount
            elif action == 'borrow':
                wallet_features[wallet]['borrow_count'] += 1
                wallet_features[wallet]['borrow_usd'] += usd_amount
            elif action == 'repay':
                wallet_features[wallet]['repay_count'] += 1
                wallet_features[wallet]['repay_usd'] += usd_amount
            elif action == 'redeemunderlying':
                wallet_features[wallet]['redeem_count'] += 1
                wallet_features[wallet]['redeem_usd'] += usd_amount
            elif action == 'liquidationcall':
                wallet_features[wallet]['liquidation_count'] += 1
            
            # Update temporal and diversity features
            if timestamp:
                tx_date = datetime.fromtimestamp(timestamp)
                wallet_features[wallet]['transaction_days'].add(tx_date.date())
                
                if not wallet_features[wallet]['first_transaction']:
                    wallet_features[wallet]['first_transaction'] = timestamp
                    wallet_features[wallet]['last_transaction'] = timestamp
                else:
                    wallet_features[wallet]['first_transaction'] = min(
                        wallet_features[wallet]['first_transaction'], timestamp
                    )
                    wallet_features[wallet]['last_transaction'] = max(
                        wallet_features[wallet]['last_transaction'], timestamp
                    )
            
            # Track unique assets
            asset_symbol = action_data.get('assetSymbol', '')
            if asset_symbol:
                wallet_features[wallet]['unique_assets'].add(asset_symbol)
            
            # Update total volume
            wallet_features[wallet]['total_volume'] += usd_amount
        
        # Calculate derived features
        for wallet, features in wallet_features.items():
            total_transactions = (
                features['deposit_count'] + features['borrow_count'] + 
                features['repay_count'] + features['redeem_count'] + 
                features['liquidation_count']
            )
            
            if total_transactions > 0:
                features['avg_transaction_size'] = features['total_volume'] / total_transactions
            
            features['unique_assets_count'] = len(features['unique_assets'])
            features['active_days'] = len(features['transaction_days'])
            
            # Convert sets to counts for JSON serialization
            features['unique_assets'] = features['unique_assets_count']
            features['transaction_days'] = features['active_days']
        
        return dict(wallet_features)
    
    def extract_advanced_features(self, basic_features: Dict) -> Dict:
        """
        Extract advanced risk-related features
        
        Args:
            basic_features: Dictionary of basic wallet features
            
        Returns:
            Dictionary with advanced features added
        """
        advanced_features = {}
        
        for wallet, features in basic_features.items():
            advanced = features.copy()
            
            # 1. Borrowing Behavior Analysis
            if features['borrow_count'] > 0:
                advanced['borrow_to_deposit_ratio'] = features['borrow_usd'] / max(features['deposit_usd'], 1)
                advanced['repay_to_borrow_ratio'] = features['repay_usd'] / features['borrow_usd']
            else:
                advanced['borrow_to_deposit_ratio'] = 0
                advanced['repay_to_borrow_ratio'] = 1.0  # No borrowing is good
            
            # 2. Activity Patterns
            if features['first_transaction'] and features['last_transaction']:
                days_active = (features['last_transaction'] - features['first_transaction']) / 86400  # seconds to days
                advanced['activity_span_days'] = max(days_active, 1)
                advanced['activity_frequency'] = features['active_days'] / advanced['activity_span_days'] if days_active > 0 else 0
            else:
                advanced['activity_span_days'] = 1
                advanced['activity_frequency'] = 1
            
            # 3. Liquidity Management
            total_inflow = features['deposit_usd']
            total_outflow = features['redeem_usd'] + features['borrow_usd']
            advanced['net_position'] = total_inflow - total_outflow
            advanced['liquidity_ratio'] = total_inflow / max(total_outflow, 1)
            
            # 4. Risk Indicators
            advanced['liquidation_rate'] = features['liquidation_count'] / max(features['borrow_count'], 1)
            advanced['transaction_diversity'] = features['unique_assets_count']
            
            # 5. Volume-based indicators
            if features['total_volume'] > 0:
                advanced['deposit_dominance'] = features['deposit_usd'] / features['total_volume']
                advanced['borrow_dominance'] = features['borrow_usd'] / features['total_volume']
            else:
                advanced['deposit_dominance'] = 0
                advanced['borrow_dominance'] = 0
            
            # 6. Behavioral consistency
            total_count = (features['deposit_count'] + features['borrow_count'] + 
                          features['repay_count'] + features['redeem_count'])
            if total_count > 0:
                advanced['action_balance_score'] = 1 - abs(
                    (features['deposit_count'] + features['repay_count']) - 
                    (features['borrow_count'] + features['redeem_count'])
                ) / total_count
            else:
                advanced['action_balance_score'] = 1
            
            advanced_features[wallet] = advanced
        
        return advanced_features
    
    def calculate_component_scores(self, features_dict: Dict) -> Dict:
        """
        Calculate individual component scores for risk assessment
        
        Args:
            features_dict: Dictionary of wallet features
            
        Returns:
            Dictionary with component scores
        """
        scored_wallets = {}
        
        # Collect all values for normalization
        all_features = pd.DataFrame.from_dict(features_dict, orient='index').fillna(0)
        
        # Initialize scalers
        scaler = MinMaxScaler()
        
        for wallet, features in features_dict.items():
            scores = {}
            
            # 1. Volume Score (0-200 points)
            volume_factors = [
                min(features['deposit_usd'] / 10000, 1.0) * 50,  # Max 50 points for high deposits
                min(features['repay_usd'] / 5000, 1.0) * 30,     # Max 30 points for repayments
                min(features['total_volume'] / 20000, 1.0) * 20   # Max 20 points for total volume
            ]
            # Penalty for excessive borrowing
            borrow_penalty = min(features['borrow_usd'] / 15000, 0.3) * 100
            scores['volume_score'] = max(sum(volume_factors) - borrow_penalty, 0)
            
            # 2. Frequency Score (0-200 points)
            activity_points = min(features['active_days'] * 2, 50)  # Max 50 for activity
            frequency_points = min(features['activity_frequency'] * 150, 100)  # Max 100 for consistency
            span_points = min(features['activity_span_days'] / 30, 1.0) * 50  # Max 50 for longevity
            scores['frequency_score'] = activity_points + frequency_points + span_points
            
            # 3. Diversity Score (0-200 points)
            asset_diversity = min(features['unique_assets_count'] * 20, 80)  # Max 80 points
            action_balance = features['action_balance_score'] * 60  # Max 60 points
            liquidity_management = min(features['liquidity_ratio'], 2.0) / 2.0 * 60  # Max 60 points
            scores['diversity_score'] = asset_diversity + action_balance + liquidity_management
            
            # 4. Liquidity Score (0-200 points)
            net_position_score = min(max(features['net_position'] / 5000, -1), 2) * 50 + 50  # -50 to +150
            repay_ratio_score = min(features['repay_to_borrow_ratio'], 2.0) * 50  # Max 100 points
            scores['liquidity_score'] = max(net_position_score + repay_ratio_score, 0)
            
            # 5. Risk Behavior Score (0-200 points) - Higher is better (lower risk)
            liquidation_penalty = features['liquidation_count'] * 40  # 40 points penalty per liquidation
            borrow_ratio_penalty = max(features['borrow_to_deposit_ratio'] - 0.5, 0) * 60  # Penalty for high leverage
            consistency_bonus = features['activity_frequency'] * 30  # Bonus for consistent activity
            
            base_risk_score = 200
            scores['risk_behavior_score'] = max(
                base_risk_score - liquidation_penalty - borrow_ratio_penalty + consistency_bonus, 0
            )
            
            scored_wallets[wallet] = {**features, **scores}
        
        return scored_wallets
    
    def calculate_final_scores(self, scored_wallets: Dict) -> pd.DataFrame:
        """
        Calculate final risk scores and create output DataFrame
        
        Args:
            scored_wallets: Dictionary with component scores
            
        Returns:
            DataFrame with final scores and rankings
        """
        results = []
        
        for wallet, data in scored_wallets.items():
            # Calculate weighted final score
            final_score = (
                data['volume_score'] * self.feature_weights['volume_score'] +
                data['frequency_score'] * self.feature_weights['frequency_score'] +
                data['diversity_score'] * self.feature_weights['diversity_score'] +
                data['liquidity_score'] * self.feature_weights['liquidity_score'] +
                data['risk_behavior_score'] * self.feature_weights['risk_behavior_score']
            )
            
            # Scale to 0-1000 range
            final_score = int(np.clip(final_score * 5, 0, 1000))  # Multiply by 5 to reach 1000
            
            # Risk category
            if final_score >= 800:
                risk_category = 'Low Risk'
            elif final_score >= 600:
                risk_category = 'Medium Risk'
            elif final_score >= 400:
                risk_category = 'High Risk'
            else:
                risk_category = 'Very High Risk'
            
            results.append({
                'wallet_id': wallet,
                'score': final_score,
                'risk_category': risk_category,
                'volume_score': round(data['volume_score'], 2),
                'frequency_score': round(data['frequency_score'], 2),
                'diversity_score': round(data['diversity_score'], 2),
                'liquidity_score': round(data['liquidity_score'], 2),
                'risk_behavior_score': round(data['risk_behavior_score'], 2),
                'total_volume_usd': round(data['total_volume'], 2),
                'deposit_usd': round(data['deposit_usd'], 2),
                'borrow_usd': round(data['borrow_usd'], 2),
                'repay_usd': round(data['repay_usd'], 2),
                'liquidation_count': data['liquidation_count'],
                'unique_assets': data['unique_assets_count'],
                'active_days': data['active_days'],
                'borrow_to_deposit_ratio': round(data['borrow_to_deposit_ratio'], 3),
                'repay_to_borrow_ratio': round(data['repay_to_borrow_ratio'], 3)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        return df
    
    def _calculate_usd_amount(self, action_data: Dict) -> float:
        """
        Calculate USD amount from action data
        
        Args:
            action_data: Transaction action data
            
        Returns:
            USD amount
        """
        try:
            amount = float(action_data.get('amount', 0))
            price = float(action_data.get('assetPriceUSD', 1.0))
            decimals = int(action_data.get('decimals', 18))
            asset_symbol = action_data.get('assetSymbol', '')
            
            # Adjust for token decimals
            if 'USD' in asset_symbol:
                divisor = 10**6  # USDC, USDT typically have 6 decimals
            elif asset_symbol.startswith('W'):
                divisor = 10**18  # WETH, WBTC typically have 18 decimals
            else:
                divisor = 10**decimals
                
            usd_amount = (amount * price) / divisor
            return max(usd_amount, 0)
            
        except Exception:
            return 0.0
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            df: DataFrame with scored wallets
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("WALLET RISK SCORING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Wallets Analyzed: {len(df)}")
        report.append("")
        
        # Score distribution
        report.append("SCORE DISTRIBUTION:")
        report.append("-" * 40)
        risk_dist = df['risk_category'].value_counts()
        for category, count in risk_dist.items():
            pct = (count / len(df)) * 100
            report.append(f"{category}: {count} wallets ({pct:.1f}%)")
        report.append("")
        
        # Statistical summary
        report.append("STATISTICAL SUMMARY:")
        report.append("-" * 40)
        report.append(f"Mean Score: {df['score'].mean():.1f}")
        report.append(f"Median Score: {df['score'].median():.1f}")
        report.append(f"Standard Deviation: {df['score'].std():.1f}")
        report.append(f"Min Score: {df['score'].min()}")
        report.append(f"Max Score: {df['score'].max()}")
        report.append("")
        
        # Top 10 wallets
        report.append("TOP 10 HIGHEST SCORING WALLETS:")
        report.append("-" * 40)
        for i, row in df.head(10).iterrows():
            report.append(f"{row['rank']:2d}. {row['wallet_id'][:20]}... - Score: {row['score']} ({row['risk_category']})")
        report.append("")
        
        # Bottom 10 wallets
        report.append("BOTTOM 10 LOWEST SCORING WALLETS:")
        report.append("-" * 40)
        for i, row in df.tail(10).iterrows():
            report.append(f"{row['rank']:2d}. {row['wallet_id'][:20]}... - Score: {row['score']} ({row['risk_category']})")
        report.append("")
        
        # Feature importance insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)
        high_risk_wallets = df[df['risk_category'].isin(['High Risk', 'Very High Risk'])]
        if len(high_risk_wallets) > 0:
            avg_liquidations = high_risk_wallets['liquidation_count'].mean()
            avg_borrow_ratio = high_risk_wallets['borrow_to_deposit_ratio'].mean()
            report.append(f"High-risk wallets have an average of {avg_liquidations:.1f} liquidations")
            report.append(f"High-risk wallets have an average borrow-to-deposit ratio of {avg_borrow_ratio:.2f}")
        
        low_risk_wallets = df[df['risk_category'] == 'Low Risk']
        if len(low_risk_wallets) > 0:
            avg_volume = low_risk_wallets['total_volume_usd'].mean()
            avg_assets = low_risk_wallets['unique_assets'].mean()
            report.append(f"Low-risk wallets have an average volume of ${avg_volume:,.0f}")
            report.append(f"Low-risk wallets interact with an average of {avg_assets:.1f} unique assets")
        
        report.append("=" * 80)
        
        return "\\n".join(report)
    
    def run_complete_analysis(self, output_file: str = 'wallet_scores.csv') -> bool:
        """
        Run the complete risk scoring analysis
        
        Args:
            output_file: Output CSV file path
            
        Returns:
            True if successful
        """
        try:
            logger.info("Starting wallet risk analysis...")
            
            # Load and process data
            if not self.load_transactions():
                return False
            
            # Extract features
            logger.info("Extracting basic features...")
            basic_features = self.extract_basic_features()
            
            logger.info("Extracting advanced features...")
            advanced_features = self.extract_advanced_features(basic_features)
            
            # Calculate scores
            logger.info("Calculating component scores...")
            scored_wallets = self.calculate_component_scores(advanced_features)
            
            logger.info("Calculating final scores...")
            results_df = self.calculate_final_scores(scored_wallets)
            
            # Save results
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Generate and save report
            report = self.generate_report(results_df)
            with open('analysis_report.txt', 'w') as f:
                f.write(report)
            logger.info("Analysis report saved to analysis_report.txt")
            
            # Print summary
            print("\\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            print(f"Analyzed {len(results_df)} wallets")
            print(f"Score range: {results_df['score'].min()} - {results_df['score'].max()}")
            print(f"Average score: {results_df['score'].mean():.1f}")
            print(f"Results saved to: {output_file}")
            print("="*50)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return False

if __name__ == "__main__":
    # Run the analysis
    scorer = WalletRiskScorer()
    scorer.run_complete_analysis()
