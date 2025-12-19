"""
Portfolio Management Module

This module handles:
- Position tracking and management
- Manual order detection and auto TP/SL
- Entry validation and early exit logic
- Account and portfolio state management
"""

from .position_manager import PositionManager

__all__ = ['PositionManager']
