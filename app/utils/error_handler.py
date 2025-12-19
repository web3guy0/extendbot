"""
Enhanced Error Handler with Telegram Notifications
Catches all critical errors and sends alerts
"""

import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Set
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

# Transient errors that should NOT trigger critical alerts
TRANSIENT_ERROR_PATTERNS = {
    '502', '503', '504', '520', '521', '522', '523', '524',
    'Bad Gateway', 'Service Unavailable', 'Gateway Timeout',
    'Connection reset', 'Connection refused', 'timed out',
    'ETIMEDOUT', 'ECONNRESET', 'ECONNREFUSED'
}


class ErrorHandler:
    """
    Centralized error handling with Telegram notifications
    """
    
    def __init__(self, telegram_bot=None):
        """
        Initialize error handler
        
        Args:
            telegram_bot: Optional Telegram bot for notifications
        """
        self.telegram_bot = telegram_bot
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10  # Increased from 5
        
        # Transient error tracking
        self.transient_error_count = 0
        self.transient_error_start: Optional[datetime] = None
        self.last_transient_notification: Optional[datetime] = None
        self.transient_notification_interval = timedelta(minutes=30)  # Only notify every 30 min
        
        logger.info("🛡️ Error handler initialized")
    
    def _is_transient_error(self, error: Exception) -> bool:
        """Check if error is transient (API outage, network issues)"""
        error_str = str(error)
        return any(pattern in error_str for pattern in TRANSIENT_ERROR_PATTERNS)
    
    async def handle_critical_error(self, error: Exception, context: str = "Unknown"):
        """
        Handle critical errors with logging and Telegram notification
        
        Args:
            error: The exception that occurred
            context: Description of where the error occurred
        """
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error_time = datetime.now(timezone.utc)
        
        # Check if this is a transient error (API down, network issues)
        if self._is_transient_error(error):
            await self._handle_transient_error(error, context)
            return
        
        # Reset transient counter on non-transient error
        self.transient_error_count = 0
        self.transient_error_start = None
        
        # Log full traceback
        error_trace = traceback.format_exc()
        logger.error(
            f"❌ CRITICAL ERROR in {context}\n"
            f"Error #{self.error_count} (consecutive: {self.consecutive_errors})\n"
            f"{error_trace}"
        )
        
        # Send Telegram alert
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(
                    f"🚨 *CRITICAL ERROR*\n\n"
                    f"📍 Context: {context}\n"
                    f"❌ Error: {str(error)[:200]}\n"
                    f"🔢 Count: {self.error_count} (consecutive: {self.consecutive_errors})\n"
                    f"⏰ Time: {self.last_error_time.strftime('%H:%M:%S UTC')}\n\n"
                    f"{'⚠️ Bot may auto-restart if errors continue' if self.consecutive_errors >= 3 else '✅ Bot continuing'}"
                )
            except Exception as e:
                logger.error(f"Failed to send error notification: {e}")
        
        # Check if too many consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical(f"🚨 Too many consecutive errors ({self.consecutive_errors}). Bot may need manual intervention.")
            if self.telegram_bot:
                try:
                    await self.telegram_bot.send_message(
                        f"🚨 *CRITICAL: TOO MANY ERRORS*\n\n"
                        f"Consecutive errors: {self.consecutive_errors}\n"
                        f"Bot stability compromised.\n\n"
                        f"⚠️ Check logs immediately!\n"
                        f"Consider manual restart if issues persist."
                    )
                except Exception:
                    pass
    
    async def _handle_transient_error(self, error: Exception, context: str):
        """Handle transient errors (502, network issues) with less noise"""
        self.transient_error_count += 1
        now = datetime.now(timezone.utc)
        
        if not self.transient_error_start:
            self.transient_error_start = now
        
        outage_duration = now - self.transient_error_start
        
        # Log at debug level (less noise)
        logger.debug(f"🌐 Transient error #{self.transient_error_count} in {context}: {str(error)[:100]}")
        
        # Only notify Telegram periodically during extended outages
        should_notify = (
            self.last_transient_notification is None or
            (now - self.last_transient_notification) >= self.transient_notification_interval
        )
        
        # First notification after 5 errors
        if self.transient_error_count == 5 and self.telegram_bot:
            try:
                await self.telegram_bot.send_message(
                    f"🌐 *API CONNECTIVITY ISSUE*\n\n"
                    f"Exchange API appears to be down.\n"
                    f"Errors: {self.transient_error_count}\n"
                    f"Bot is waiting for recovery...\n\n"
                    f"ℹ️ This is usually temporary."
                )
                self.last_transient_notification = now
            except Exception:
                pass
        
        # Periodic updates during long outages (every 30 min)
        elif should_notify and outage_duration > timedelta(minutes=5) and self.telegram_bot:
            try:
                await self.telegram_bot.send_message(
                    f"🌐 *API OUTAGE CONTINUES*\n\n"
                    f"Duration: {outage_duration.total_seconds() / 60:.0f} minutes\n"
                    f"Errors: {self.transient_error_count}\n\n"
                    f"Bot waiting for recovery..."
                )
                self.last_transient_notification = now
            except Exception:
                pass
    
    def reset_transient_errors(self):
        """Reset transient error counter after successful API call"""
        if self.transient_error_count > 0:
            duration = ""
            if self.transient_error_start:
                duration = f" (outage lasted {(datetime.now(timezone.utc) - self.transient_error_start).total_seconds() / 60:.1f} min)"
            logger.info(f"✅ API recovered after {self.transient_error_count} transient errors{duration}")
            self.transient_error_count = 0
            self.transient_error_start = None
    
    async def handle_recoverable_error(self, error: Exception, context: str = "Unknown"):
        """
        Handle recoverable errors (warnings, not critical)
        
        Args:
            error: The exception that occurred
            context: Description of where the error occurred
        """
        logger.warning(f"⚠️ Recoverable error in {context}: {error}")
        
        # Send warning to Telegram (less alarming)
        if self.telegram_bot and self.error_count % 10 == 0:  # Only every 10th warning
            try:
                await self.telegram_bot.send_message(
                    f"⚠️ *Warning*\n\n"
                    f"Context: {context}\n"
                    f"Issue: {str(error)[:150]}\n\n"
                    f"Bot is handling this automatically."
                )
            except Exception:
                pass
    
    def reset_consecutive_errors(self):
        """Reset consecutive error counter after successful operation"""
        if self.consecutive_errors > 0:
            logger.info(f"✅ Recovered from {self.consecutive_errors} consecutive errors")
            self.consecutive_errors = 0
    
    def get_stats(self) -> dict:
        """Get error statistics"""
        return {
            'total_errors': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'health_status': 'healthy' if self.consecutive_errors == 0 else 'warning' if self.consecutive_errors < 3 else 'critical'
        }


def with_error_handling(context: str = "Operation"):
    """
    Decorator for automatic error handling
    
    Args:
        context: Description of the operation
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                result = await func(self, *args, **kwargs)
                # Reset consecutive errors on success
                if hasattr(self, 'error_handler'):
                    self.error_handler.reset_consecutive_errors()
                return result
            except Exception as e:
                if hasattr(self, 'error_handler'):
                    await self.error_handler.handle_critical_error(e, context)
                else:
                    logger.error(f"Error in {context}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
