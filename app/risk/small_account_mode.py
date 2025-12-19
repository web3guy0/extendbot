#!/usr/bin/env python3
"""
Small Account Mode
==================

Optimized settings for trading with small capital ($20-$100).

Challenges with small accounts:
1. Minimum order sizes (HyperLiquid has minimums per asset)
2. Fees eat into profits faster
3. Need higher leverage but must manage risk tightly
4. Can't afford many losing trades

Solutions:
1. Higher leverage (10x-20x) to meet minimums
2. Fewer but higher quality trades
3. Wider stops to avoid noise (ATR-based)
4. Larger profit targets to offset fees
5. Only trade most liquid assets
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


# Minimum notional values per asset on HyperLiquid (approximate)
ASSET_MINIMUMS = {
    'BTC': 0.001,   # ~$100 at $100k
    'ETH': 0.01,    # ~$40 at $4k
    'SOL': 0.1,     # ~$14 at $140
    'DOGE': 50,     # ~$10 at $0.20
    'WIF': 10,      # ~$15 at $1.50
    'PEPE': 500000, # ~$10 at $0.00002
    'BONK': 500000, # ~$10 at $0.00002
    'ARB': 5,       # ~$5 at $1
    'OP': 5,        # ~$10 at $2
    'SUI': 5,       # ~$10 at $2
}

# Recommended assets for small accounts (low minimums, good volatility)
SMALL_ACCOUNT_ASSETS = ['HYPE', 'SOL', 'DOGE', 'WIF', 'SUI']


class SmallAccountMode:
    """
    Optimized trading parameters for small accounts.
    
    Key principles:
    1. Quality over quantity - fewer trades, higher win rate
    2. Larger moves - 2-3% targets instead of 0.5% scalps
    3. Wider stops - avoid noise, use ATR-based stops
    4. Higher leverage - meet minimums, but strict position limits
    5. One position at a time - no diversification risk
    """
    
    def __init__(self, account_balance: float):
        """
        Initialize small account mode.
        
        Args:
            account_balance: Current account balance in USD
        """
        self.balance = Decimal(str(account_balance))
        self.is_small_account = account_balance < 100
        
        # Determine tier
        if account_balance < 50:
            self.tier = 'micro'
            self.recommended_leverage = 10  # High enough to meet minimums, not crazy
            self.max_positions = 1
            self.min_signal_score = 18  # Only 72%+ setups (18/25)
            self.target_pct = Decimal('2.5')  # Realistic targets
            self.stop_pct = Decimal('1.0')  # Tight but not too tight
        elif account_balance < 100:
            self.tier = 'small'
            self.recommended_leverage = 7
            self.max_positions = 1
            self.min_signal_score = 16  # 64%
            self.target_pct = Decimal('2.0')
            self.stop_pct = Decimal('0.8')
        elif account_balance < 500:
            self.tier = 'starter'
            self.recommended_leverage = 5
            self.max_positions = 2
            self.min_signal_score = 15  # 60%
            self.target_pct = Decimal('2.0')
            self.stop_pct = Decimal('0.8')
        else:
            self.tier = 'normal'
            self.recommended_leverage = 5
            self.max_positions = 3
            self.min_signal_score = 12  # 48% - default threshold
            self.target_pct = Decimal('2.0')
            self.stop_pct = Decimal('0.8')
        
        logger.info(f"💰 Small Account Mode: {self.tier.upper()}")
        logger.info(f"   Balance: ${account_balance:.2f}")
        logger.info(f"   Leverage: {self.recommended_leverage}x")
        logger.info(f"   Max Positions: {self.max_positions}")
        logger.info(f"   Min Signal Score: {self.min_signal_score}/25")
        logger.info(f"   Target: {self.target_pct}% | Stop: {self.stop_pct}%")
    
    def get_tradeable_assets(self) -> List[str]:
        """Get list of assets suitable for this account size."""
        if self.tier == 'micro':
            # Only lowest minimum assets
            return ['SOL', 'SUI', 'WIF']
        elif self.tier == 'small':
            return ['SOL', 'ETH', 'SUI', 'WIF', 'DOGE']
        elif self.tier == 'starter':
            return ['SOL', 'ETH', 'SUI', 'WIF', 'DOGE', 'ARB', 'OP']
        else:
            return list(ASSET_MINIMUMS.keys())
    
    def can_trade_asset(self, symbol: str, price: float) -> Tuple[bool, str]:
        """
        Check if we can trade this asset with current balance.
        
        Args:
            symbol: Asset symbol
            price: Current price
            
        Returns:
            (can_trade, reason)
        """
        min_size = ASSET_MINIMUMS.get(symbol, 1.0)
        min_notional = min_size * price
        
        # Calculate max position we can take
        max_margin = float(self.balance) * 0.9  # Keep 10% buffer
        max_notional = max_margin * self.recommended_leverage
        
        if min_notional > max_notional:
            return False, f"Min order ${min_notional:.2f} > max position ${max_notional:.2f}"
        
        # Check if position would be too large relative to account
        position_pct = (min_notional / self.recommended_leverage) / float(self.balance) * 100
        if position_pct > 80:
            return False, f"Position too large ({position_pct:.0f}% of account)"
        
        return True, f"OK - min ${min_notional:.2f}, max ${max_notional:.2f}"
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_score: int,
        max_score: int = 10,
    ) -> Tuple[Decimal, Decimal, str]:
        """
        Calculate optimal position size for small account.
        
        Args:
            symbol: Asset symbol
            price: Current price
            signal_score: Signal quality score
            max_score: Maximum possible score
        
        Returns:
            (size_tokens, size_usd, reason)
        """
        min_size = Decimal(str(ASSET_MINIMUMS.get(symbol, 1.0)))
        
        # Base: Use 50% of max possible position
        max_margin = self.balance * Decimal('0.5')
        max_notional = max_margin * self.recommended_leverage
        
        # Adjust based on signal quality (prevent divide by zero)
        if max_score > 0:
            score_mult = Decimal(str(signal_score / max_score))
        else:
            score_mult = Decimal('0.5')  # Default to 50% if no max_score
        if score_mult >= Decimal('0.8'):
            size_mult = Decimal('1.0')  # Full size for A+ setups
        elif score_mult >= Decimal('0.7'):
            size_mult = Decimal('0.75')
        else:
            size_mult = Decimal('0.5')  # Half size for min quality signals
        
        # Calculate size
        target_notional = max_notional * size_mult
        size_tokens = target_notional / Decimal(str(price))
        
        # Ensure meets minimum
        if size_tokens < min_size:
            size_tokens = min_size
            target_notional = size_tokens * Decimal(str(price))
        
        # Calculate actual margin used
        margin_used = target_notional / self.recommended_leverage
        margin_pct = margin_used / self.balance * 100
        
        reason = f"Size: {size_tokens:.4f} ({margin_pct:.0f}% margin) - Score: {signal_score}/{max_score}"
        
        return size_tokens, target_notional, reason
    
    def get_config_overrides(self) -> Dict[str, Any]:
        """Get environment variable overrides for small account mode."""
        return {
            'MAX_LEVERAGE': str(self.recommended_leverage),
            'MAX_POSITIONS': str(self.max_positions),
            'MIN_SIGNAL_SCORE': str(self.min_signal_score),
            'SWING_TARGET_PCT': str(self.target_pct),
            'SWING_STOP_PCT': str(self.stop_pct),
            'SWING_COOLDOWN': '600',  # 10 min cooldown - be patient
            'POSITION_SIZE_PCT': '50',  # Conservative position size
        }
    
    def apply_config(self):
        """
        Apply small account config overrides to environment.
        
        Note: This modifies global os.environ. It's called once at bot startup
        to configure the trading parameters for small accounts.
        """
        overrides = self.get_config_overrides()
        for key, value in overrides.items():
            os.environ[key] = value
            logger.info(f"   Override: {key}={value}")
    
    def get_risk_per_trade(self) -> Tuple[Decimal, Decimal]:
        """
        Calculate risk per trade in USD and %.
        
        For small accounts, we risk 3-5% per trade.
        This allows for 20-30 losing trades before wipeout.
        """
        if self.tier == 'micro':
            risk_pct = Decimal('5')  # 5% risk for micro accounts
        elif self.tier == 'small':
            risk_pct = Decimal('4')  # 4% for small
        else:
            risk_pct = Decimal('3')  # 3% for normal
        
        risk_usd = self.balance * risk_pct / 100
        
        return risk_usd, risk_pct
    
    def get_summary(self) -> str:
        """Get human-readable summary of small account settings."""
        risk_usd, risk_pct = self.get_risk_per_trade()
        
        # Calculate breakeven trades needed
        # With 3% target, 1% stop, 3:1 R:R
        # Win 1 trade = +3%, Lose 1 trade = -1%
        # Breakeven at 25% win rate (1 win covers 3 losses)
        
        return (
            f"💰 **SMALL ACCOUNT MODE: {self.tier.upper()}**\n\n"
            f"Balance: ${float(self.balance):.2f}\n"
            f"Leverage: {self.recommended_leverage}x\n"
            f"Max Positions: {self.max_positions}\n"
            f"Min Signal Score: {self.min_signal_score}/25\n\n"
            f"**Risk Management:**\n"
            f"Risk per Trade: ${float(risk_usd):.2f} ({float(risk_pct)}%)\n"
            f"Target: {self.target_pct}% price move\n"
            f"Stop: {self.stop_pct}% price move\n"
            f"R:R Ratio: {float(self.target_pct/self.stop_pct):.1f}:1\n\n"
            f"**Tradeable Assets:**\n"
            f"{', '.join(self.get_tradeable_assets())}\n\n"
            f"_Quality over quantity - wait for A+ setups!_"
        )


def get_small_account_mode(balance: float) -> SmallAccountMode:
    """Get small account mode configuration."""
    return SmallAccountMode(balance)
