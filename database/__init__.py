"""
Database module for Bank Reviews Analysis.

This module provides database connection management and data insertion utilities
for the PostgreSQL bank_reviews database.
"""

from .db_connection import DatabaseManager

__all__ = ['DatabaseManager']
