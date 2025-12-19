#!/usr/bin/env python3
"""
Position Calculator with Percentage-Based Risk Management
========================================================

Calculates position sizes based on account equity percentages,
not fixed USD amounts. Supports dynamic symbol switching.

Key Features:
- Percentage-based position sizing
- Dynamic symbol support via HyperLiquid API
- Risk-adjusted position calculations
- Account equity-based limits
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketInfo:
    """Market information from HyperLiquid"""
    symbol: str
    name: str
    asset_id: int
    sz_decimals: int
    max_leverage: float
    only_isolated: bool

@dataclass
class PositionSize:
    """Calculated position size"""
    symbol: str
    size: Decimal
    leverage: float
    notional_value: Decimal
    risk_amount: Decimal
    max_loss: Decimal
    entry_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None

class PositionCalculator:
    """
    Calculate position sizes based on percentage of account equity
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize position calculator
        
        Args:
            config: Configuration with percentage-based limits
        """
        self.config = config
        self.markets: Dict[str, MarketInfo] = {}
        self.account_equity = Decimal('0')
        self.available_margin = Decimal('0')
        
        # Position sizing parameters (percentages)
        self.position_size_pct = Decimal(str(config.get('position_size_pct', 20.0)))
        self.max_position_pct = Decimal(str(config.get('max_position_pct', 15.0)))
        self.risk_per_trade_pct = Decimal(str(config.get('risk_per_trade_pct', 2.0)))
        self.max_leverage = Decimal(str(config.get('max_leverage', 5.0)))
        
        logger.info(f"📊 Position Calculator initialized:")
        logger.info(f"   Position Size: {self.position_size_pct}% of equity")
        logger.info(f"   Max Position: {self.max_position_pct}% of equity")
        logger.info(f"   Risk per Trade: {self.risk_per_trade_pct}% of equity")
        logger.info(f"   Max Leverage: {self.max_leverage}x")
    
    def load_markets(self, meta_data: Dict[str, Any]) -> None:
        """Load available markets from HyperLiquid meta data"""
        try:
            self.markets = {}
            universe = meta_data.get('universe', [])
            
            for asset in universe:
                market_info = MarketInfo(
                    symbol=asset['name'],
                    name=asset['name'], 
                    asset_id=asset.get('assetId', 0),
                    sz_decimals=asset.get('szDecimals', 4),
                    max_leverage=float(asset.get('maxLeverage', 50)),
                    only_isolated=asset.get('onlyIsolated', False)
                )
                self.markets[asset['name']] = market_info
            
            logger.info(f"✅ Loaded {len(self.markets)} markets from HyperLiquid")
            
        except Exception as e:
            logger.error(f"❌ Failed to load markets: {e}")
    
    def update_account_state(self, equity: Decimal, available_margin: Decimal) -> None:
        """Update current account state"""
        self.account_equity = equity
        self.available_margin = available_margin
        
        logger.debug(f"💰 Account updated - Equity: ${equity:.2f}, Available: ${available_margin:.2f}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        return list(self.markets.keys())
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available for trading"""
        return symbol in self.markets
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None
    ) -> Optional[PositionSize]:
        """Calculate position size based on risk percentage"""
        try:
            if not self.validate_symbol(symbol):
                logger.error(f"❌ Invalid symbol: {symbol}")
                return None
            
            market_info = self.markets[symbol]
            
            if self.account_equity <= 0:
                logger.error("❌ Invalid account equity")
                return None
            
            # Calculate position sizing
            risk_amount = self.account_equity * (self.risk_per_trade_pct / Decimal('100'))
            max_position_value = self.account_equity * (self.max_position_pct / Decimal('100'))
            
            # Default position size based on percentage
            position_notional = self.account_equity * (self.position_size_pct / Decimal('100'))
            position_size = position_notional / entry_price
            
            # Ensure position doesn't exceed maximum
            if position_notional > max_position_value:
                position_size = max_position_value / entry_price
                position_notional = max_position_value
            
            # Calculate leverage
            max_leverage_allowed = min(self.max_leverage, Decimal(str(market_info.max_leverage)))
            margin_required = position_notional / max_leverage_allowed
            
            # Check margin availability
            if margin_required > self.available_margin:
                position_size = (self.available_margin * max_leverage_allowed) / entry_price
                position_notional = position_size * entry_price
            
            # Round to market decimals
            position_size = self._round_to_decimals(position_size, market_info.sz_decimals)
            final_notional = position_size * entry_price
            
            # Calculate stops
            if not stop_loss_price:
                stop_loss_pct = Decimal(str(self.config.get('stop_loss_pct', 0.7))) / Decimal('100')
                stop_loss_price = entry_price * (Decimal('1') - stop_loss_pct)
            
            take_profit_pct = Decimal(str(self.config.get('take_profit_pct', 1.5))) / Decimal('100')
            take_profit_price = entry_price * (Decimal('1') + take_profit_pct)
            
            max_loss = position_size * abs(entry_price - stop_loss_price)
            leverage_used = final_notional / (final_notional / max_leverage_allowed)
            
            position = PositionSize(
                symbol=symbol,
                size=position_size,
                leverage=float(leverage_used),
                notional_value=final_notional,
                risk_amount=risk_amount,
                max_loss=max_loss,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            return position
            
        except Exception as e:
            logger.error(f"❌ Position calculation failed: {e}")
            return None
    
    def _round_to_decimals(self, value: Decimal, decimals: int) -> Decimal:
        """Round value to specified decimal places"""
        quantize_value = Decimal('0.1') ** decimals
        return value.quantize(quantize_value)