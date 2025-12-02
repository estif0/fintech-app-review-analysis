"""
Database connection manager for Bank Reviews Analysis.

This module provides a DatabaseManager class for managing PostgreSQL connections
using credentials from environment variables.
"""

import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database connections with connection pooling.

    Attributes:
        host (str): Database host
        port (str): Database port
        database (str): Database name
        user (str): Database user
        password (str): Database password
        connection_pool: psycopg2 connection pool
    """

    def __init__(self, use_admin=False):
        """
        Initialize database manager with credentials from environment variables.

        Args:
            use_admin (bool): If True, use admin credentials (postgres user)
        """
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "bank_reviews")

        if use_admin:
            self.user = os.getenv("DB_ADMIN_USER", "postgres")
            self.password = os.getenv("DB_ADMIN_PASSWORD", "")
        else:
            self.user = os.getenv("DB_USER", "user")
            self.password = os.getenv("DB_PASSWORD", "password")

        self.connection_pool = None
        logger.info(f"DatabaseManager initialized for {self.database} as {self.user}")

    def create_connection_pool(self, minconn=1, maxconn=5):
        """
        Create a connection pool for efficient connection management.

        Args:
            minconn (int): Minimum number of connections in pool
            maxconn (int): Maximum number of connections in pool
        """
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn,
                maxconn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            logger.info("Connection pool created successfully")
        except psycopg2.Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise

    def get_connection(self):
        """
        Get a connection from the pool or create a new one.

        Returns:
            psycopg2.connection: Database connection
        """
        try:
            if self.connection_pool:
                conn = self.connection_pool.getconn()
            else:
                conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                )
            logger.debug("Database connection established")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def return_connection(self, conn):
        """
        Return a connection to the pool.

        Args:
            conn: Database connection to return
        """
        if self.connection_pool:
            self.connection_pool.putconn(conn)
            logger.debug("Connection returned to pool")

    def close_connection(self, conn):
        """
        Close a database connection.

        Args:
            conn: Database connection to close
        """
        if conn:
            conn.close()
            logger.debug("Database connection closed")

    def close_all_connections(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All connections closed")

    def test_connection(self):
        """
        Test database connection and return database information.

        Returns:
            dict: Database connection information
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get PostgreSQL version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # Get current database and user
            cursor.execute("SELECT current_database(), current_user;")
            db_info = cursor.fetchone()

            # Get table count
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """
            )
            table_count = cursor.fetchone()[0]

            cursor.close()
            self.close_connection(conn)

            info = {
                "status": "success",
                "version": version,
                "database": db_info[0],
                "user": db_info[1],
                "table_count": table_count,
            }

            logger.info(f"Connection test successful: {info}")
            return info

        except psycopg2.Error as e:
            logger.error(f"Connection test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def execute_query(self, query, params=None, fetch=True):
        """
        Execute a SQL query and return results.

        Args:
            query (str): SQL query to execute
            params (tuple): Query parameters for parameterized queries
            fetch (bool): Whether to fetch results

        Returns:
            list: Query results if fetch=True, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params)

            if fetch:
                results = cursor.fetchall()
                cursor.close()
                self.close_connection(conn)
                return results
            else:
                conn.commit()
                cursor.close()
                self.close_connection(conn)
                return None

        except psycopg2.Error as e:
            if conn:
                conn.rollback()
                self.close_connection(conn)
            logger.error(f"Error executing query: {e}")
            raise

    def execute_many(self, query, data_list):
        """
        Execute a query with multiple parameter sets (batch insert).

        Args:
            query (str): SQL query with placeholders
            data_list (list): List of tuples containing parameter values

        Returns:
            int: Number of rows affected
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.executemany(query, data_list)
            rowcount = cursor.rowcount

            conn.commit()
            cursor.close()
            self.close_connection(conn)

            logger.info(f"Batch insert successful: {rowcount} rows affected")
            return rowcount

        except psycopg2.Error as e:
            if conn:
                conn.rollback()
                self.close_connection(conn)
            logger.error(f"Error executing batch insert: {e}")
            raise


if __name__ == "__main__":
    # Test the database connection
    print("Testing Database Connection...")
    print("-" * 50)

    db = DatabaseManager()
    info = db.test_connection()

    if info["status"] == "success":
        print(f"✓ Connection successful!")
        print(f"  Database: {info['database']}")
        print(f"  User: {info['user']}")
        print(f"  Tables: {info['table_count']}")
        print(f"  Version: {info['version'][:50]}...")
    else:
        print(f"✗ Connection failed: {info['error']}")
