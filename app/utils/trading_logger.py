"""
Trading Logger - Structured logging and audit trail
Comprehensive logging system for trading operations
"""

import logging
import logging.handlers
import json
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Trading event types"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    TRADE_SIGNAL = "trade_signal"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_VIOLATION = "risk_violation"
    KILL_SWITCH = "kill_switch"
    DRAWDOWN_ALERT = "drawdown_alert"
    BALANCE_UPDATE = "balance_update"
    ERROR = "error"


class TradingLogger:
    """
    Enterprise trading logger with structured logging
    """
    
    def __init__(self, log_dir: str = "logs", component_name: str = "HyperBot"):
        """
        Initialize trading logger
        
        Args:
            log_dir: Directory for log files
            component_name: Name of component
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.component_name = component_name
        
        # Create component logger
        self.logger = logging.getLogger(component_name)
        
        # Create file handlers
        self._setup_file_handlers()
        
        # Event log
        self.event_log_path = self.log_dir / f"events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        
        # Statistics
        self.events_logged = 0
        self.errors_logged = 0
        
        self.logger.info(f"📝 Trading Logger initialized for {component_name}")
        self.logger.info(f"   Log directory: {self.log_dir}")
    
    # ==================== PASSTHROUGH METHODS ====================
    def debug(self, msg: str, *args, **kwargs):
        """Pass-through to logger.debug"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Pass-through to logger.info"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Pass-through to logger.warning"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Pass-through to logger.error"""
        self.errors_logged += 1
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Pass-through to logger.critical"""
        self.errors_logged += 1
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Pass-through to logger.exception"""
        self.errors_logged += 1
        self.logger.exception(msg, *args, **kwargs)
    
    def _setup_file_handlers(self):
        """Setup file handlers with rotation for different log levels"""
        # Main log file with rotation (10MB max, keep 5 backups)
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.component_name.lower()}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(main_handler)
        
        # Error log file with rotation
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
        ))
        self.logger.addHandler(error_handler)
    
    def log_event(self, event_type: EventType, data: Dict[str, Any],
                  level: LogLevel = LogLevel.INFO):
        """
        Log a trading event
        
        Args:
            event_type: Type of event
            data: Event data
            level: Log level
        """
        # Create event record
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'component': self.component_name,
            'event_type': event_type.value,
            'level': level.value,
            'data': self._serialize_data(data)
        }
        
        # Write to event log
        try:
            with open(self.event_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write event log: {e}")
        
        # Log to standard logger
        log_msg = f"[{event_type.value}] {self._format_data(data)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_msg)
        elif level == LogLevel.INFO:
            self.logger.info(log_msg)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_msg)
        elif level == LogLevel.ERROR:
            self.logger.error(log_msg)
            self.errors_logged += 1
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_msg)
            self.errors_logged += 1
        
        self.events_logged += 1
    
    def log_trade_signal(self, strategy: str, symbol: str, signal_type: str,
                        strength: str, price: Decimal):
        """Log trading signal"""
        self.log_event(EventType.TRADE_SIGNAL, {
            'strategy': strategy,
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength,
            'price': float(price)
        })
    
    def log_order_placed(self, symbol: str, side: str, size: Decimal,
                        price: Decimal, order_id: Optional[str] = None):
        """Log order placement"""
        self.log_event(EventType.ORDER_PLACED, {
            'symbol': symbol,
            'side': side,
            'size': float(size),
            'price': float(price),
            'order_id': order_id
        })
    
    def log_order_filled(self, symbol: str, side: str, size: Decimal,
                        fill_price: Decimal, order_id: Optional[str] = None):
        """Log order fill"""
        self.log_event(EventType.ORDER_FILLED, {
            'symbol': symbol,
            'side': side,
            'size': float(size),
            'fill_price': float(fill_price),
            'order_id': order_id
        })
    
    def log_order_rejected(self, symbol: str, side: str, size: Decimal,
                          reason: str):
        """Log order rejection"""
        self.log_event(EventType.ORDER_REJECTED, {
            'symbol': symbol,
            'side': side,
            'size': float(size),
            'reason': reason
        }, level=LogLevel.WARNING)
    
    def log_position_opened(self, symbol: str, side: str, size: Decimal,
                           entry_price: Decimal):
        """Log position opening"""
        self.log_event(EventType.POSITION_OPENED, {
            'symbol': symbol,
            'side': side,
            'size': float(size),
            'entry_price': float(entry_price)
        })
    
    def log_position_closed(self, symbol: str, side: str, size: Decimal,
                           exit_price: Decimal, pnl: Decimal):
        """Log position closing"""
        self.log_event(EventType.POSITION_CLOSED, {
            'symbol': symbol,
            'side': side,
            'size': float(size),
            'exit_price': float(exit_price),
            'pnl': float(pnl)
        })
    
    def log_risk_violation(self, violation_type: str, details: str):
        """Log risk violation"""
        self.log_event(EventType.RISK_VIOLATION, {
            'violation_type': violation_type,
            'details': details
        }, level=LogLevel.WARNING)
    
    def log_kill_switch(self, reason: str, details: str):
        """Log kill switch trigger"""
        self.log_event(EventType.KILL_SWITCH, {
            'reason': reason,
            'details': details
        }, level=LogLevel.CRITICAL)
    
    def log_drawdown_alert(self, drawdown_pct: Decimal, level: str):
        """Log drawdown alert"""
        self.log_event(EventType.DRAWDOWN_ALERT, {
            'drawdown_pct': float(drawdown_pct),
            'level': level
        }, level=LogLevel.WARNING if level != "emergency" else LogLevel.CRITICAL)
    
    def log_balance_update(self, balance: Decimal, equity: Decimal,
                          pnl: Decimal):
        """Log balance update"""
        self.log_event(EventType.BALANCE_UPDATE, {
            'balance': float(balance),
            'equity': float(equity),
            'pnl': float(pnl)
        }, level=LogLevel.DEBUG)
    
    def log_error(self, error: str, exception: Optional[Exception] = None):
        """Log error"""
        data = {'error': error}
        if exception:
            data['exception'] = str(exception)
            data['exception_type'] = type(exception).__name__
        
        self.log_event(EventType.ERROR, data, level=LogLevel.ERROR)
    
    def log_system_start(self, config: Dict[str, Any]):
        """Log system start"""
        self.log_event(EventType.SYSTEM_START, {
            'config': config
        })
    
    def log_system_stop(self, reason: str = "Normal shutdown"):
        """Log system stop"""
        self.log_event(EventType.SYSTEM_STOP, {
            'reason': reason
        })
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data to JSON-serializable format"""
        result = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format data for log message"""
        parts = []
        for key, value in data.items():
            if isinstance(value, float):
                if key in ['price', 'fill_price', 'entry_price', 'exit_price']:
                    parts.append(f"{key}=${value:.2f}")
                elif key in ['size']:
                    parts.append(f"{key}={value:.4f}")
                elif key in ['pnl']:
                    parts.append(f"{key}=${value:+.2f}")
                else:
                    parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'events_logged': self.events_logged,
            'errors_logged': self.errors_logged,
            'log_directory': str(self.log_dir),
            'event_log_path': str(self.event_log_path)
        }


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def get_logger(component_name: str = "HyperBot") -> TradingLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TradingLogger(component_name=component_name)
    return _global_logger
