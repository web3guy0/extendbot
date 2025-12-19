"""
ExtendBot Configuration Validation Module

Validates required environment variables on startup to prevent
runtime errors from missing configuration.

Extended Exchange requires:
- API_KEY: API key for REST calls
- PRIVATE_KEY: Stark private key for signing
- PUBLIC_KEY: Stark public key
- VAULT_ID: Account vault ID
"""

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# Required environment variables (bot will not start without these)
REQUIRED_VARS = [
    'API_KEY',          # Extended Exchange API key
    'PRIVATE_KEY',      # Stark private key for signing orders
    'PUBLIC_KEY',       # Stark public key
    'VAULT_ID',         # Account vault ID
]

# Optional variables with defaults
OPTIONAL_VARS: Dict[str, Any] = {
    # Exchange settings
    'TESTNET': 'false',  # Use testnet endpoints
    
    # Trading settings
    'SYMBOL': 'SOL-USD',
    'MULTI_ASSET_MODE': 'false',
    'MULTI_ASSETS': 'BTC-USD,ETH-USD,SOL-USD',
    'TIMEFRAME': '1m',
    'MAX_LEVERAGE': '5',
    'MAX_POSITIONS': '3',
    'BASE_POSITION_SIZE_PCT': '25',
    
    # Strategy settings
    'SIGNAL_THRESHOLD': '12',
    'BOT_MODE': 'rule_based',
    
    # Risk management
    'MAX_DAILY_LOSS_PCT': '5.0',
    'MAX_DRAWDOWN_PCT': '10.0',
    'KELLY_FRACTION': '0.5',
    
    # TP/SL defaults
    'DEFAULT_TP_PCT': '3.0',
    'DEFAULT_SL_PCT': '1.5',
    
    # Paper trading
    'PAPER_TRADING': 'false',
    'PAPER_TRADING_BALANCE': '1000',
    
    # Telegram (optional but recommended)
    'TELEGRAM_BOT_TOKEN': '',
    'TELEGRAM_CHAT_ID': '',
    
    # Database (optional but recommended)
    'DATABASE_URL': '',
    
    # Extended specific
    'EXTENDED_TIMEOUT': '30',
}


class ConfigError(Exception):
    """Configuration error"""
    pass


def validate_config() -> Dict[str, Any]:
    """
    Validate all required environment variables are set.
    Returns dict of all config values.
    Raises ConfigError if required variables are missing.
    """
    errors = []
    config = {}
    
    # Check required variables
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if not value:
            errors.append(var)
        else:
            config[var] = value
    
    if errors:
        error_msg = f"Missing required environment variables: {', '.join(errors)}"
        logger.critical(f"❌ CONFIG ERROR: {error_msg}")
        raise ConfigError(error_msg)
    
    # Load optional variables with defaults
    for var, default in OPTIONAL_VARS.items():
        config[var] = os.getenv(var, default)
    
    return config


def get_config_summary() -> str:
    """Get a summary of current configuration for logging."""
    lines = ["📋 Configuration Summary:"]
    
    # Exchange mode
    testnet = os.getenv('TESTNET', 'false').lower() == 'true'
    lines.append(f"   Exchange: Extended Exchange {'(TESTNET)' if testnet else '(MAINNET)'}")
    
    # Trading mode
    multi_asset = os.getenv('MULTI_ASSET_MODE', 'false').lower() == 'true'
    if multi_asset:
        assets = os.getenv('MULTI_ASSETS', 'BTC-USD,ETH-USD,SOL-USD')
        lines.append(f"   Mode: Multi-Asset ({assets})")
    else:
        lines.append(f"   Mode: Single Asset ({os.getenv('SYMBOL', 'SOL-USD')})")
    
    lines.append(f"   Timeframe: {os.getenv('TIMEFRAME', '1m')}")
    lines.append(f"   Leverage: {os.getenv('MAX_LEVERAGE', '5')}x")
    
    # Risk settings
    lines.append(f"   Max Daily Loss: {os.getenv('MAX_DAILY_LOSS_PCT', '5.0')}%")
    lines.append(f"   Max Drawdown: {os.getenv('MAX_DRAWDOWN_PCT', '10.0')}%")
    
    # Paper trading
    if os.getenv('PAPER_TRADING', 'false').lower() == 'true':
        lines.append(f"   📝 Paper Trading: ENABLED")
    
    # Database
    if os.getenv('DATABASE_URL'):
        lines.append("   💾 Database: Configured")
    else:
        lines.append("   ⚠️ Database: Not configured (trades will only be logged to JSONL)")
    
    # Telegram
    if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
        lines.append("   📱 Telegram: Configured")
    else:
        lines.append("   ⚠️ Telegram: Not configured (no notifications)")
    
    return "\n".join(lines)


def check_credentials() -> bool:
    """
    Quick check if credentials are set.
    Returns True if basic credentials are present.
    """
    return all([
        os.getenv('API_KEY'),
        os.getenv('PRIVATE_KEY'),
        os.getenv('PUBLIC_KEY'),
        os.getenv('VAULT_ID'),
    ])


def get_testnet_mode() -> bool:
    """Check if testnet mode is enabled."""
    return os.getenv('TESTNET', 'false').lower() == 'true'


def get_exchange_endpoint() -> str:
    """Get the appropriate API endpoint based on mode."""
    if get_testnet_mode():
        return "https://api.testnet.extended.exchange"
    return "https://api.starknet.extended.exchange"


def get_ws_endpoint() -> str:
    """Get the appropriate WebSocket endpoint based on mode."""
    if get_testnet_mode():
        return "wss://api.testnet.extended.exchange/stream.extended.exchange/v1"
    return "wss://api.starknet.extended.exchange/stream.extended.exchange/v1"
